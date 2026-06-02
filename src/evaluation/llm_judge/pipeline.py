"""Main LLM judge evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseEvaluator
from ..exact_match import ExactMatchEvaluator
from ...config import ModelConfig
from ...data_utils import get_output_options
from ...infra.io import load_json, save_json
from ...infra.azure import AzureConfig

from .client import AzureJudgeClient, LocalJudgeClient, JudgeClient, get_client
from .metrics import (
    compute_metrics,
    compute_metrics_with_difficulty,
    _empty_metrics,
    build_summary,
)
from .parser import JudgeVerdict, parse_judge_response
from .prompts import PROMPT_VERSION, JudgePromptConfig, format_judge_prompt

_EXACT_MATCH_EVALUATOR = ExactMatchEvaluator()


@dataclass
class JudgeDecision:
    """Result of a judge evaluation."""

    correct: str
    reasoning: str
    decision_method: str


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

        if not self.no_llm:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == "azure":
            azure_config = AzureConfig.from_env()
            self.client = AzureJudgeClient(azure_config, model=self.api_model)

        elif self.provider == "huggingface":
            if not self.model_config:
                raise ValueError("Local judge requires model_config.")
            self.client = LocalJudgeClient(self.model_config)

        else:
            raise ValueError(f"Unsupported judge provider: {self.provider}")

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
            )

        if not self.client:
            raise RuntimeError("LLM client is not configured.")

        prompt = self._build_prompt(input_text, gold, prediction)
        raw = self.client.complete(prompt, max_new_tokens=256)
        correct, reasoning = self._parse_response(raw)

        return JudgeDecision(
            correct=correct, reasoning=reasoning, decision_method="llm"
        )


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
            and matches_any_gold_option(pred_text, gold_options)
        ):
            return {
                "input": input_text,
                "gold": gold_text,
                "gold_options": gold_options,
                "prediction": pred_text,
                "correct": "yes",
                "reasoning": "Exact match against an accepted gold formula.",
                "decision_method": "exact",
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
            }

        prompt = format_judge_prompt(
            input_text=input_text,
            gold=gold_options or gold_text,
            prediction=pred_text,
            config=self.prompt_config,
        )
        raw = self.client.complete(prompt, max_new_tokens=256)
        verdict = parse_judge_response(raw)

        result = {
            "input": input_text,
            "gold": gold_text,
            "gold_options": gold_options,
            "prediction": pred_text,
            "correct": verdict.decision,
            "reasoning": verdict.reasoning or "No reasoning provided.",
            "decision_method": "llm",
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


def matches_any_gold_option(prediction: str, gold_options: List[str]) -> bool:
    """Return whether prediction exactly matches any accepted gold formula."""
    normalized_prediction = normalize_formula_for_match(prediction)
    return any(
        normalized_prediction == normalize_formula_for_match(gold)
        for gold in gold_options
    )


def _gold_options_from_prediction_item(item: Dict[str, Any]) -> List[str]:
    options: List[str] = []

    def append(value: Any) -> None:
        if isinstance(value, list):
            for entry in value:
                append(entry)
            return
        if isinstance(value, str) and value.strip() and value.strip() not in options:
            options.append(value.strip())

    for key in ("expected_options", "gold_options", "reference_options", "outputs"):
        append(item.get(key))
    for key in ("expected", "gold", "reference"):
        append(item.get(key))

    return options


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
                "exact_match": item.get("exact_match"),
                "id": item.get("id"),
                "difficulty": item.get("difficulty"),
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
    }
    rows = []

    def is_positive_exact_flag(flag: Any) -> bool:
        if isinstance(flag, str):
            return flag.strip().lower() in {"1", "true", "yes"}
        return bool(flag)

    def is_exact_match(
        pred: str, gold_options: List[str], flag: Optional[bool]
    ) -> bool:
        if flag is not None and is_positive_exact_flag(flag):
            return True
        return matches_any_gold_option(pred, gold_options)

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
            )
        elif is_exact_match(prediction, gold_options, exact_flag):
            stats["auto_exact"] += 1
            decision = JudgeDecision(
                correct="yes",
                reasoning="Exact match against an accepted gold formula.",
                decision_method="exact",
            )
        elif no_llm:
            stats["no_llm"] += 1
            decision = JudgeDecision(
                correct="no",
                reasoning="LLM disabled; non-exact match treated as incorrect.",
                decision_method="no_llm",
            )
        else:
            decision = judge.judge(input_text, gold_options, prediction)
            if decision.decision_method == "llm":
                stats["llm_calls"] += 1

        rows.append(
            {
                "input": input_text,
                "gold": gold,
                "gold_options": gold_options,
                "prediction": prediction,
                "correct": decision.correct,
                "reasoning": decision.reasoning,
                "decision_method": decision.decision_method,
                "id": item.get("id"),
                "difficulty": item.get("difficulty"),
            }
        )

    return rows, stats


def build_summary_notebook(summary_path: Path, output_path: Path) -> None:
    """Generate a Jupyter notebook for exploring the summary."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ATL LLM Judge Summary\n",
                    f"Summary file: {summary_path.name}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
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
                    "df.sort_values('accuracy', ascending=False)\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "plt.figure(figsize=(100, 40))\n",
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
    "compute_metrics_with_difficulty",
    "_empty_metrics",
    "build_summary",
    "build_summary_notebook",
    "PROMPT_VERSION",
    "run_llm_judge",
]
