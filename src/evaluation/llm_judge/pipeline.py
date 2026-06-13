"""Main LLM judge evaluation pipeline."""

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseEvaluator
from ..exact_match import ExactMatchEvaluator
from ...config import ModelConfig
from ...data_utils import get_output_options
from ...infra.io import load_json, save_json
from ...infra.azure import AzureConfig, ContentFilterError

from .client import AzureJudgeClient, JudgeClient
from .metrics import (
    compute_metrics,
    _empty_metrics,
    build_summary,
)
from .parser import parse_judge_response
from .prompts import PROMPT_VERSION, JudgePromptConfig, format_judge_prompt

_EXACT_MATCH_EVALUATOR = ExactMatchEvaluator()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class JudgeDecision:
    """Result of a judge evaluation."""

    correct: str
    reasoning: str
    decision_method: str
    prompt_version: Optional[str] = None
    judge_prompt_sha256: Optional[str] = None
    raw_judge_response: Optional[str] = None
    judge_parse_status: Optional[str] = None
    judge_latency_ms: Optional[float] = None
    from_cache: bool = False


class LLMJudge:
    """LLM-based judge for evaluating ATL formula correctness."""

    def __init__(
        self,
        judge_model: str,
        no_llm: bool = False,
        prompt_version: str = PROMPT_VERSION,
        api_model: Optional[str] = None,
        provider: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        prompt_config: Optional[JudgePromptConfig] = None,
    ):
        self.judge_model = judge_model
        self.api_model = api_model or judge_model
        self.no_llm = no_llm
        self.prompt_version = prompt_version
        self.provider = (
            provider or (model_config.provider if model_config else "azure")
        ).lower()
        self.model_config = model_config
        self.client: Optional[JudgeClient] = None
        self.prompt_config = prompt_config
        self._decision_cache: Dict[str, JudgeDecision] = {}
        self.cache_hits = 0
        self.api_calls = 0

        if not self.no_llm:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider != "azure":
            raise ValueError(
                f"Unsupported judge provider: {self.provider}. "
                "LLM judge models must use provider='azure'."
            )

        azure_config = AzureConfig.from_env()
        self.client = AzureJudgeClient(azure_config, model=self.api_model)

    def _build_prompt(self, input_text: str, gold: Any, prediction: str) -> str:
        return format_judge_prompt(
            input_text=input_text,
            gold=gold,
            prediction=prediction,
            config=self.prompt_config,
        )

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        verdict = parse_judge_response(raw)
        return verdict.decision, verdict.reasoning or "No reasoning provided."

    def judge(self, input_text: str, gold: Any, prediction: str) -> JudgeDecision:
        """Judge a single prediction."""
        if self.no_llm:
            return JudgeDecision(
                correct="no",
                reasoning="LLM disabled; non-exact match treated as incorrect.",
                decision_method="no_llm",
                prompt_version=self.prompt_version,
            )

        if not self.client:
            raise RuntimeError("LLM client is not configured.")

        prompt = self._build_prompt(input_text, gold, prediction)
        prompt_hash = _sha256_text(prompt)
        # Tag the cache key with the judge identity so a cached verdict can never
        # be served for a different judge (the prompt itself does not encode which
        # model answers it). This preserves judge independence: dedup only avoids
        # asking the SAME judge the SAME (input, gold, prediction) twice.
        cache_key = "\x1f".join(
            [self.judge_model, self.api_model, self.prompt_version or "", prompt_hash]
        )

        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return replace(cached, from_cache=True)

        start_time = time.perf_counter()
        try:
            raw = self.client.complete(prompt, max_new_tokens=256)
        except ContentFilterError as exc:
            # The Azure Responsible AI filter deterministically rejected this
            # (input, gold, prediction) prompt. Retrying cannot succeed, so
            # record it as unjudgeable and let the batch continue.
            latency_ms = (time.perf_counter() - start_time) * 1000
            decision = JudgeDecision(
                correct="no",
                reasoning=f"Judge prompt blocked by content filter: {exc}",
                decision_method="content_filtered",
                prompt_version=self.prompt_version,
                judge_prompt_sha256=prompt_hash,
                raw_judge_response=None,
                judge_parse_status="content_filtered",
                judge_latency_ms=round(latency_ms, 2),
            )
            self._decision_cache[cache_key] = decision
            return decision

        latency_ms = (time.perf_counter() - start_time) * 1000
        correct, reasoning = self._parse_response(raw)

        decision = JudgeDecision(
            correct=correct,
            reasoning=reasoning,
            decision_method="llm",
            prompt_version=self.prompt_version,
            judge_prompt_sha256=prompt_hash,
            raw_judge_response=raw,
            judge_parse_status=(
                "invalid"
                if reasoning == "Judge response was not valid JSON."
                else "parsed"
            ),
            judge_latency_ms=round(latency_ms, 2),
        )
        self._decision_cache[cache_key] = decision
        self.api_calls += 1
        return decision


class LLMJudgeEvaluator(BaseEvaluator):
    """LLM-as-a-judge evaluation pipeline."""

    def __init__(
        self,
        client: JudgeClient,
        prompt_config: Optional[JudgePromptConfig] = None,
        judge_model: str = "llm",
        prompt_version: str = PROMPT_VERSION,
        no_llm: bool = False,
    ):
        self.client = client
        self.prompt_config = prompt_config
        self.judge_model = judge_model
        self.prompt_version = prompt_version
        self.no_llm = no_llm

    def evaluate_single(
        self, prediction: Dict[str, Any], reference: Dict[str, Any]
    ) -> Dict[str, Any]:
        input_text = prediction.get("input") or reference.get("input") or ""
        pred_text = (
            prediction.get("prediction")
            or prediction.get("generated")
            or prediction.get("output")
            or ""
        )
        gold_options = get_output_options(reference)
        gold_text = gold_options[0] if gold_options else ""

        if (
            pred_text
            and gold_options
            and matches_all_gold_outputs(pred_text, gold_options)
        ):
            return {
                "input": input_text,
                "gold": gold_text,
                "gold_options": gold_options,
                "prediction": pred_text,
                "correct": "yes",
                "reasoning": "Exact match against all required gold output formulas.",
                "decision_method": "exact",
                "prompt_version": self.prompt_version,
                "judge_prompt_sha256": None,
                "raw_judge_response": None,
                "judge_parse_status": "not_called_exact_match",
                "judge_latency_ms": None,
            }

        if self.no_llm:
            return {
                "input": input_text,
                "gold": gold_text,
                "gold_options": gold_options,
                "prediction": pred_text,
                "correct": "no",
                "reasoning": "LLM disabled; non-exact match treated as incorrect.",
                "decision_method": "no_llm",
                "prompt_version": self.prompt_version,
                "judge_prompt_sha256": None,
                "raw_judge_response": None,
                "judge_parse_status": "not_called_no_llm",
                "judge_latency_ms": None,
            }

        prompt = format_judge_prompt(
            input_text=input_text,
            gold=gold_options or gold_text,
            prediction=pred_text,
            config=self.prompt_config,
        )
        start_time = time.perf_counter()
        raw = self.client.complete(prompt, max_new_tokens=256)
        latency_ms = (time.perf_counter() - start_time) * 1000
        verdict = parse_judge_response(raw)

        result = {
            "input": input_text,
            "gold": gold_text,
            "gold_options": gold_options,
            "prediction": pred_text,
            "correct": verdict.decision,
            "reasoning": verdict.reasoning or "No reasoning provided.",
            "decision_method": "llm",
            "prompt_version": self.prompt_version,
            "judge_prompt_sha256": _sha256_text(prompt),
            "raw_judge_response": raw,
            "judge_parse_status": (
                "invalid"
                if verdict.reasoning == "Judge response was not valid JSON."
                else "parsed"
            ),
            "judge_latency_ms": round(latency_ms, 2),
        }

        return result

    def evaluate(
        self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        rows = [
            self.evaluate_single(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        return compute_metrics(rows)


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def normalize_formula_for_match(text: Optional[str]) -> str:
    """Normalize ATL formulas using the same rules as exact-match evaluation."""
    return _EXACT_MATCH_EVALUATOR.normalize(normalize_text(text))


def split_formula_lines(text: str) -> List[str]:
    """Split a formula-only model output into non-empty formula lines."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def matches_all_gold_outputs(prediction: str, gold_options: List[str]) -> bool:
    """Return whether prediction exactly matches all required gold outputs.

    The match is order-insensitive but multiplicity-sensitive. In QSA/multi-output
    cases, all gold readings are required; they are not alternatives.
    """
    predicted_outputs = split_formula_lines(prediction)
    if not predicted_outputs or not gold_options:
        return False

    normalized_prediction = Counter(
        normalize_formula_for_match(formula) for formula in predicted_outputs
    )
    normalized_gold = Counter(
        normalize_formula_for_match(gold) for gold in gold_options
    )
    return normalized_prediction == normalized_gold


def _gold_options_from_prediction_item(item: Dict[str, Any]) -> List[str]:
    return get_output_options(item)


def extract_prediction_items(prediction_data: Any) -> List[Dict[str, Any]]:
    """Extract prediction items from various input formats."""
    if isinstance(prediction_data, dict):
        items = prediction_data.get("detailed_results") or prediction_data.get(
            "predictions"
        )
        if not isinstance(items, list):
            items = []
    elif isinstance(prediction_data, list):
        items = prediction_data
    else:
        items = []

    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue

        prediction = (
            item.get("generated")
            or item.get("output")
            or item.get("prediction")
            or item.get("model_output")
        )
        gold_options = _gold_options_from_prediction_item(item)
        gold = gold_options[0] if gold_options else None

        parsed.append(
            {
                "input": item.get("input"),
                "prediction": prediction,
                "gold": gold,
                "gold_options": gold_options,
                "expected_outputs": item.get("expected_outputs") or gold_options,
                "exact_match": item.get("exact_match"),
                "id": item.get("id"),
            }
        )

    return parsed


def evaluate_prediction_file(
    prediction_path: Path,
    judge: LLMJudge,
    no_llm: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Evaluate a prediction file."""
    prediction_data = load_json(prediction_path)
    prediction_items = extract_prediction_items(prediction_data)

    stats = {
        "unmatched": 0,
        "auto_exact": 0,
        "llm_calls": 0,
        "no_llm": 0,
        "cached": 0,
        "content_filtered": 0,
    }
    rows = []

    def is_positive_exact_flag(flag: Any) -> bool:
        if isinstance(flag, str):
            return flag.strip().lower() in {"1", "true", "yes"}
        return bool(flag)

    def is_exact_match(
        pred: str, gold_options: List[str], flag: Optional[bool]
    ) -> bool:
        if flag is not None:
            return is_positive_exact_flag(flag)
        return matches_all_gold_outputs(pred, gold_options)

    for item in prediction_items:
        input_text = item.get("input") or ""
        prediction = item.get("prediction") or ""
        gold_options = item.get("gold_options") or []
        gold = item.get("gold") or (gold_options[0] if gold_options else "")
        exact_flag = item.get("exact_match")

        if not prediction or not gold_options:
            stats["unmatched"] += 1
            decision = JudgeDecision(
                correct="no",
                reasoning="Missing prediction or gold.",
                decision_method="unmatched",
                prompt_version=judge.prompt_version,
                judge_parse_status="not_called_missing_data",
            )
        elif is_exact_match(prediction, gold_options, exact_flag):
            stats["auto_exact"] += 1
            decision = JudgeDecision(
                correct="yes",
                reasoning="Exact match against all required gold output formulas.",
                decision_method="exact",
                prompt_version=judge.prompt_version,
                judge_parse_status="not_called_exact_match",
            )
        elif no_llm:
            stats["no_llm"] += 1
            decision = JudgeDecision(
                correct="no",
                reasoning="LLM disabled; non-exact match treated as incorrect.",
                decision_method="no_llm",
                prompt_version=judge.prompt_version,
                judge_parse_status="not_called_no_llm",
            )
        else:
            decision = judge.judge(input_text, gold_options, prediction)
            if decision.decision_method == "llm":
                stats["llm_calls"] += 1
                if decision.from_cache:
                    stats["cached"] += 1
            elif decision.decision_method == "content_filtered":
                stats["content_filtered"] += 1

        rows.append(
            {
                "input": input_text,
                "gold": gold,
                "gold_options": gold_options,
                "prediction": prediction,
                "correct": decision.correct,
                "reasoning": decision.reasoning,
                "decision_method": decision.decision_method,
                "prompt_version": decision.prompt_version or judge.prompt_version,
                "judge_prompt_sha256": decision.judge_prompt_sha256,
                "raw_judge_response": decision.raw_judge_response,
                "judge_parse_status": decision.judge_parse_status,
                "judge_latency_ms": decision.judge_latency_ms,
                "from_cache": decision.from_cache,
                "id": item.get("id"),
            }
        )

    return rows, stats


def build_summary_notebook(summary_path: Path, output_path: Path) -> None:
    """Generate a Jupyter notebook for exploring the summary."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "# ATL LLM Judge Summary\n",
                    f"Summary file: {summary_path.name}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    f'summary_path = Path("{summary_path.name}")\n',
                    "if not summary_path.exists():\n",
                    "    for parent in Path.cwd().parents:\n",
                    "        candidate = parent / summary_path.name\n",
                    "        if candidate.exists():\n",
                    "            summary_path = candidate\n",
                    "            break\n",
                    "\n",
                    "summary = json.loads(summary_path.read_text())\n",
                    "df = pd.DataFrame([\n",
                    "    {\n",
                    "        'source_file': item['source_file'],\n",
                    "        'accuracy': item['metrics']['accuracy'],\n",
                    "        'evaluated': item['metrics']['total_evaluated'],\n",
                    "        'correct': item['metrics']['correct'],\n",
                    "    }\n",
                    "    for item in summary['per_file']\n",
                    "])\n",
                    "display(pd.DataFrame([summary.get('overall', {})]))\n",
                    "display(df.sort_values('accuracy', ascending=False))\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, max(4, min(14, len(df) * 0.35))))\n",
                    "plt.bar(df['source_file'], df['accuracy'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.title('Accuracy by Model')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    save_json(notebook, output_path)


def run_llm_judge(
    prediction_path: Path,
    judge_model: str,
    *,
    no_llm: bool = False,
    prompt_version: str = PROMPT_VERSION,
    api_model: Optional[str] = None,
    provider: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Convenience wrapper to evaluate a single prediction file."""
    judge = LLMJudge(
        judge_model=judge_model,
        no_llm=no_llm,
        prompt_version=prompt_version,
        api_model=api_model,
        provider=provider,
        model_config=model_config,
    )
    rows, stats = evaluate_prediction_file(Path(prediction_path), judge, no_llm=no_llm)
    metrics = compute_metrics(rows)
    return {
        "rows": rows,
        "stats": stats,
        "metrics": metrics,
    }


__all__ = [
    "LLMJudge",
    "LLMJudgeEvaluator",
    "JudgeDecision",
    "normalize_text",
    "extract_prediction_items",
    "evaluate_prediction_file",
    "compute_metrics",
    "_empty_metrics",
    "build_summary",
    "build_summary_notebook",
    "PROMPT_VERSION",
    "run_llm_judge",
]
