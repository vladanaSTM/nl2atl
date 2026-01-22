"""
LLM-as-a-judge evaluator for ATL outputs.
"""

import ast
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .azure_utils import AzureClient, AzureConfig
from .config import ModelConfig
from .io_utils import load_json, load_json_safe, save_json

PROMPT_VERSION = "v1.0"


def _get_model_registry():
    """Lazy import to avoid heavy dependencies during module import."""
    from . import model_registry

    return model_registry


@dataclass
class JudgeDecision:
    """Result of a judge evaluation."""

    correct: str
    reasoning: str
    decision_method: str


class LocalJudgeClient:
    """Wrapper for local HuggingFace models as judges."""

    provider = "local"

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        mr = _get_model_registry()
        self.model, self.tokenizer = mr.load_model(model_config, for_training=False)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        mr = _get_model_registry()
        raw = mr.generate(
            self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        if raw.startswith(prompt):
            return raw[len(prompt) :].strip()
        return raw.strip()


class LLMJudge:
    """LLM-based judge for evaluating ATL formula correctness."""

    def __init__(
        self,
        judge_model: str,
        cache_path: Path,
        no_llm: bool = False,
        prompt_version: str = PROMPT_VERSION,
        api_model: Optional[str] = None,
        provider: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.judge_model = judge_model
        self.api_model = api_model or judge_model
        self.no_llm = no_llm
        self.prompt_version = prompt_version
        self.cache_path = Path(cache_path)
        self.cache = load_json_safe(self.cache_path, default={})
        self.provider = (
            provider or (model_config.provider if model_config else "azure")
        ).lower()
        self.model_config = model_config
        self.client = None

        if not self.no_llm:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == "azure":
            azure_config = AzureConfig.from_env()
            self.client = AzureClient.from_config(azure_config, model=self.api_model)

        elif self.provider == "huggingface":
            if not self.model_config:
                raise ValueError("Local judge requires model_config.")
            self.client = LocalJudgeClient(self.model_config)

        else:
            raise ValueError(f"Unsupported judge provider: {self.provider}")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        save_json(self.cache, self.cache_path)

    def _cache_key(
        self,
        input_text: str,
        gold: str,
        prediction: str,
        judge_model: str,
    ) -> str:
        """Generate cache key for a judgment."""
        key_payload = [input_text, gold, prediction, judge_model, self.prompt_version]
        return hashlib.sha256(str(key_payload).encode("utf-8")).hexdigest()

    def _build_prompt(self, input_text: str, gold: str, prediction: str) -> str:
        """Build the judge prompt."""
        return (
            "You are an expert judge for ATL (Alternating-time Temporal Logic) formulas.\n"
            "Decide whether the prediction is semantically correct ATL for the given natural-language input.\n"
            "Be strict about meaning: incorrect if coalition/agent set, temporal operator (X/F/G/U),\n"
            "polarity (!p vs p), or connective (|| vs &&) changes the expressed property.\n\n"
            "Return ONLY machine-parseable JSON with keys correct and reasoning:\n"
            '{ "correct": "yes" | "no", "reasoning": "..." }\n\n'
            "Few-shot examples:\n"
            "Example 1 (correct despite deviation)\n"
            "input: The collaborative robot can guarantee that it will keep running the cycle until a stop is requested.\n"
            "gold: <<Cobot>>(cycle_running U stop_requested)\n"
            "prediction: <<CollaborativeRobot>>(running_cycle U stop_requested)\n"
            'output: { "correct": "yes", "reasoning": "Same coalition intent and same until structure; predicates are clear aliases from the sentence." }\n\n'
            "Example 2 (correct despite deviation)\n"
            "input: The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.\n"
            "gold: <<Machine>>G (paid -> X ticket_printed)\n"
            "prediction: <<Machine>>G (!paid || X ticket_printed)\n"
            'output: { "correct": "yes", "reasoning": "Implication rewrite preserves meaning; same coalition and temporal structure." }\n\n'
            "Example 3 (correct despite deviation)\n"
            "input: The user can guarantee that at the next step either a card or cash will be inserted.\n"
            "gold: <<User>>X (card_inserted || cash_inserted)\n"
            "prediction: <<User>>X (cash_inserted || card_inserted)\n"
            'output: { "correct": "yes", "reasoning": "Disjunction order doesn\'t matter; same agent and X." }\n\n'
            "Example 4 (incorrect: wrong temporal operator)\n"
            "input: The user can guarantee that at the next step either a card or cash will be inserted.\n"
            "gold: <<User>>X (card_inserted || cash_inserted)\n"
            "prediction: <<User>>F (card_inserted || cash_inserted)\n"
            'output: { "correct": "no", "reasoning": "F allows it eventually, not necessarily next step X." }\n\n'
            "Example 5 (incorrect: wrong agent)\n"
            "input: The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.\n"
            "gold: <<Machine>>G (paid -> X ticket_printed)\n"
            "prediction: <<User>>G (paid -> X ticket_printed)\n"
            'output: { "correct": "no", "reasoning": "Coalition changed; ability attributed to wrong actor." }\n\n'
            "Example 6 (incorrect: polarity flipped)\n"
            "input: The controller can guarantee that the door is never open.\n"
            "gold: <<Controller>>G !door_open\n"
            "prediction: <<Controller>>G door_open\n"
            'output: { "correct": "no", "reasoning": "Negation flipped; expresses the opposite." }\n\n'
            "Now evaluate:\n"
            f"input: {input_text}\n"
            f"gold: {gold}\n"
            f"prediction: {prediction}\n"
            "output:"
        )

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        """Parse judge response to extract decision."""
        if "output:" in raw:
            raw = raw.split("output:")[-1]

        # Clean up common JSON issues
        def clean_json(text: str) -> str:
            cleaned = text.strip()
            cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
            cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            return cleaned

        # Try to parse JSON
        import json

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            candidates = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
            data = None

            for candidate in reversed(candidates):
                cleaned = clean_json(candidate)
                try:
                    data = json.loads(cleaned)
                    break
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(cleaned)
                        if isinstance(data, dict):
                            break
                        data = None
                    except Exception:
                        continue

            if data is None:
                return "no", "Judge response was not valid JSON."

        correct = str(data.get("correct", "no")).strip().lower()
        if correct not in ("yes", "no"):
            correct = "no"

        reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."
        return correct, reasoning

    def judge(self, input_text: str, gold: str, prediction: str) -> JudgeDecision:
        """Judge a single prediction."""
        if self.no_llm:
            return JudgeDecision(
                correct="no",
                reasoning="LLM disabled; non-exact match treated as incorrect.",
                decision_method="no_llm",
            )

        cache_key = self._cache_key(input_text, gold, prediction, self.judge_model)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return JudgeDecision(
                correct=cached.get("correct", "no"),
                reasoning=cached.get("reasoning", "Cached response."),
                decision_method=cached.get("decision_method", "llm"),
            )

        if not self.client:
            raise RuntimeError("LLM client is not configured.")

        prompt = self._build_prompt(input_text, gold, prediction)
        raw = self.client.generate(prompt, max_new_tokens=256)
        correct, reasoning = self._parse_response(raw)

        self.cache[cache_key] = {
            "correct": correct,
            "reasoning": reasoning,
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "decision_method": "llm",
            "cached_at": datetime.utcnow().isoformat() + "Z",
        }
        self._save_cache()

        return JudgeDecision(
            correct=correct, reasoning=reasoning, decision_method="llm"
        )


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_prediction_items(prediction_data: Any) -> List[Dict[str, Optional[str]]]:
    """Extract prediction items from various input formats."""
    if isinstance(prediction_data, dict) and "detailed_results" in prediction_data:
        items = prediction_data.get("detailed_results", [])
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
        gold = item.get("expected") or item.get("gold") or item.get("reference")

        parsed.append(
            {
                "input": item.get("input"),
                "prediction": prediction,
                "gold": gold,
                "exact_match": item.get("exact_match"),
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
        "cache_hits": 0,
        "no_llm": 0,
    }
    rows = []

    def is_exact_match(pred: str, gold: str, flag: Optional[bool]) -> bool:
        if flag is not None:
            return bool(flag)
        return normalize_text(pred) == normalize_text(gold)

    for item in prediction_items:
        input_text = item.get("input") or ""
        prediction = item.get("prediction") or ""
        gold = item.get("gold") or ""
        exact_flag = item.get("exact_match")

        if not prediction or not gold:
            stats["unmatched"] += 1
            decision = JudgeDecision(
                correct="no",
                reasoning="Missing prediction or gold.",
                decision_method="unmatched",
            )
        elif is_exact_match(prediction, gold, exact_flag):
            stats["auto_exact"] += 1
            decision = JudgeDecision(
                correct="yes",
                reasoning="Exact match (normalized).",
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
            decision = judge.judge(input_text, gold, prediction)
            if decision.decision_method == "llm":
                stats["llm_calls"] += 1
            else:
                stats["cache_hits"] += 1

        rows.append(
            {
                "input": input_text,
                "gold": gold,
                "prediction": prediction,
                "correct": decision.correct,
                "reasoning": decision.reasoning,
                "judge_model": judge.judge_model,
                "source_file": prediction_path.name,
                "decision_method": decision.decision_method,
            }
        )

    return rows, stats


def _safe_rate(numerator: int, denominator: int) -> float:
    """Compute rate safely, avoiding division by zero."""
    return numerator / denominator if denominator else 0.0


def compute_metrics(rows: List[Dict]) -> Dict[str, Any]:
    """Compute evaluation metrics from judged rows."""
    if not rows:
        return _empty_metrics()

    evaluated = [r for r in rows if r.get("decision_method") != "unmatched"]
    total = len(evaluated)

    if total == 0:
        return _empty_metrics()

    correct_count = sum(1 for r in evaluated if r.get("correct") == "yes")
    accuracy = _safe_rate(correct_count, total)

    # Breakdown by decision method
    exact_rows = [r for r in evaluated if r.get("decision_method") == "exact"]
    llm_rows = [r for r in evaluated if r.get("decision_method") == "llm"]
    no_llm_rows = [r for r in evaluated if r.get("decision_method") == "no_llm"]

    llm_approved = sum(1 for r in llm_rows if r.get("correct") == "yes")

    return {
        "accuracy": round(accuracy, 4),
        "total_evaluated": total,
        "evaluated": total,  # Alias for compatibility
        "correct": correct_count,
        "incorrect": total - correct_count,
        "exact_match": {
            "count": len(exact_rows),
            "rate": round(_safe_rate(len(exact_rows), total), 4),
        },
        "llm_judged": {
            "count": len(llm_rows),
            "rate": round(_safe_rate(len(llm_rows), total), 4),
            "approved": llm_approved,
            "rejected": len(llm_rows) - llm_approved,
            "approval_rate": round(_safe_rate(llm_approved, len(llm_rows)), 4),
        },
        "accuracy_from_exact_match": round(_safe_rate(len(exact_rows), total), 4),
        "accuracy_boost_from_llm": round(_safe_rate(llm_approved, total), 4),
        "no_llm_fallback_count": len(no_llm_rows),
    }


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure."""
    return {
        "accuracy": 0.0,
        "total_evaluated": 0,
        "evaluated": 0,
        "correct": 0,
        "incorrect": 0,
        "exact_match": {"count": 0, "rate": 0.0},
        "llm_judged": {
            "count": 0,
            "rate": 0.0,
            "approved": 0,
            "rejected": 0,
            "approval_rate": 0.0,
        },
        "accuracy_from_exact_match": 0.0,
        "accuracy_boost_from_llm": 0.0,
        "no_llm_fallback_count": 0,
    }


def compute_metrics_with_difficulty(rows: List[Dict]) -> Dict[str, Any]:
    """Compute metrics with breakdown by difficulty level."""
    base = compute_metrics(rows)

    by_difficulty = defaultdict(list)
    for r in rows:
        difficulty = r.get("difficulty", "unknown")
        by_difficulty[difficulty].append(r)

    breakdown = {}
    for difficulty, diff_rows in sorted(by_difficulty.items()):
        evaluated = [r for r in diff_rows if r.get("decision_method") != "unmatched"]
        if not evaluated:
            continue
        correct = sum(1 for r in evaluated if r.get("correct") == "yes")
        breakdown[difficulty] = {
            "count": len(evaluated),
            "correct": correct,
            "accuracy": round(correct / len(evaluated), 4),
        }

    base["by_difficulty"] = breakdown
    return base


def build_summary(
    results: List[Dict],
    totals: Dict,
    judge_model: str,
    prompt_version: str,
) -> Dict:
    """Build summary report."""
    all_rows = []
    for result in results:
        all_rows.extend(result["rows"])

    overall = compute_metrics(all_rows)

    per_file = [
        {
            "source_file": r["source_file"],
            "stem": r["stem"],
            "metrics": r["metrics"],
            "stats": r["stats"],
        }
        for r in results
    ]

    ranking = sorted(per_file, key=lambda x: -x["metrics"]["accuracy"])
    ranking_table = [
        {
            "rank": idx,
            "source_file": item["source_file"],
            "accuracy": item["metrics"]["accuracy"],
            "exact_match_rate": item["metrics"]["exact_match"]["rate"],
            "llm_approval_rate": item["metrics"]["llm_judged"]["approval_rate"],
            "total": item["metrics"]["total_evaluated"],
        }
        for idx, item in enumerate(ranking, start=1)
    ]

    return {
        "judge_model": judge_model,
        "prompt_version": prompt_version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "overall": overall,
        "per_file": per_file,
        "ranking": ranking_table,
        "totals": totals,
    }


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
                    "plt.figure(figsize=(10, 4))\n",
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
