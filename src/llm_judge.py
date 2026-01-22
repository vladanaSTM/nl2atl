"""
LLM-as-a-judge evaluator for ATL outputs.
"""

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ModelConfig
from .model_registry import AzureClient, generate, load_model

PROMPT_VERSION = "v1.0"


@dataclass
class JudgeDecision:
    correct: str
    reasoning: str
    decision_method: str


class LocalJudgeClient:
    """Lightweight wrapper for local HF models."""

    provider = "local"

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model, self.tokenizer = load_model(model_config, for_training=False)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        raw = generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
        if raw.startswith(prompt):
            return raw[len(prompt) :].strip()
        return raw.strip()


class LLMJudge:
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
        self.cache_path = cache_path
        self.cache = self._load_cache(cache_path)
        self.provider = (provider or (model_config.provider if model_config else "azure")).lower()
        self.model_config = model_config
        self.client = None

        if not self.no_llm:
            if self.provider == "azure":
                api_key = os.getenv("AZURE_API_KEY")
                endpoint = os.getenv("AZURE_INFER_ENDPOINT")
                api_version = os.getenv("AZURE_API_VERSION")
                verify_ssl_env = os.getenv("AZURE_VERIFY_SSL", "false").lower()
                verify_ssl = verify_ssl_env in ["1", "true", "yes"]

                if not api_key or not endpoint:
                    raise ValueError(
                        "Missing LLM credentials. Set AZURE_API_KEY and AZURE_INFER_ENDPOINT, "
                        "or pass --no_llm to run exact-match only."
                    )

                self.client = AzureClient(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=self.api_model,
                    api_version=api_version,
                    verify_ssl=verify_ssl,
                    use_cache=True,
                )
            elif self.provider == "huggingface":
                if not self.model_config:
                    raise ValueError("Local judge requires model_config.")
                self.client = LocalJudgeClient(self.model_config)
            else:
                raise ValueError(f"Unsupported judge provider: {self.provider}")

    def _load_cache(self, cache_path: Path) -> Dict[str, dict]:
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _cache_key(
        self, input_text: str, gold: str, prediction: str, judge_model: str
    ) -> str:
        key_payload = json.dumps(
            [input_text, gold, prediction, judge_model, self.prompt_version],
            ensure_ascii=False,
        )
        return hashlib.sha256(key_payload.encode("utf-8")).hexdigest()

    def _build_prompt(self, input_text: str, gold: str, prediction: str) -> str:
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
            'output: { "correct": "yes", "reasoning": "Disjunction order doesn’t matter; same agent and X." }\n\n'
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

    def _parse_judge_response(self, raw: str) -> Tuple[str, str]:
        def _clean_json_like(text: str) -> str:
            cleaned = text.strip()
            cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
            cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            return cleaned

        if "output:" in raw:
            raw = raw.split("output:")[-1]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            candidates = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
            data = None
            for candidate in reversed(candidates):
                cleaned = _clean_json_like(candidate)
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
                        data = None
                        continue
            if data is None:
                return "no", "Judge response was not valid JSON."

        correct = str(data.get("correct", "no")).strip().lower()
        if correct not in {"yes", "no"}:
            correct = "no"
        reasoning = str(data.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = "No reasoning provided."
        return correct, reasoning

    def judge(self, input_text: str, gold: str, prediction: str) -> JudgeDecision:
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

        prompt = self._build_prompt(input_text, gold, prediction)
        if not self.client:
            raise RuntimeError("LLM client is not configured.")
        raw = self.client.generate(prompt, max_new_tokens=256)
        correct, reasoning = self._parse_judge_response(raw)

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
    if text is None:
        return ""
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_prediction_items(prediction_data) -> List[Dict[str, Optional[str]]]:
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
        input_text = item.get("input")
        prediction = (
            item.get("generated")
            or item.get("output")
            or item.get("prediction")
            or item.get("model_output")
        )
        gold = item.get("expected") or item.get("gold") or item.get("reference")
        exact_match = item.get("exact_match")
        parsed.append(
            {
                "input": input_text,
                "prediction": prediction,
                "gold": gold,
                "exact_match": exact_match,
            }
        )
    return parsed


def evaluate_prediction_file(
    prediction_path: Path,
    judge: LLMJudge,
    no_llm: bool = False,
) -> Tuple[List[Dict], Dict]:
    prediction_data = load_json(prediction_path)
    prediction_items = extract_prediction_items(prediction_data)
    unmatched = 0
    auto_exact = 0
    llm_calls = 0
    no_llm_count = 0
    cache_hits = 0

    corrected_rows = []

    for i, pred_item in enumerate(prediction_items):
        input_text = pred_item.get("input")
        prediction = pred_item.get("prediction") or ""
        gold = pred_item.get("gold") or ""
        exact_match = pred_item.get("exact_match")
        match_method = "input" if gold else "unmatched"

        if not prediction or not gold:
            match_method = "unmatched"

        if match_method == "unmatched":
            unmatched += 1
            decision = JudgeDecision(
                correct="no",
                reasoning="Missing prediction or gold for input.",
                decision_method="unmatched",
            )
        else:
            if exact_match is not None:
                is_exact = bool(exact_match)
            else:
                is_exact = normalize_text(prediction) == normalize_text(gold)

            if is_exact:
                decision = JudgeDecision(
                    correct="yes",
                    reasoning="Exact match (normalized).",
                    decision_method="exact",
                )
                auto_exact += 1
            else:
                if no_llm:
                    decision = JudgeDecision(
                        correct="no",
                        reasoning="LLM disabled; non-exact match treated as incorrect.",
                        decision_method="no_llm",
                    )
                    no_llm_count += 1
                else:
                    decision = judge.judge(input_text or "", gold, prediction or "")
                    if decision.decision_method == "llm":
                        llm_calls += 1
                    elif decision.decision_method == "cache":
                        cache_hits += 1

        corrected_rows.append(
            {
                "input": input_text or "",
                "gold": gold,
                "prediction": prediction or "",
                "correct": decision.correct,
                "reasoning": decision.reasoning,
                "judge_model": judge.judge_model,
                "source_file": prediction_path.name,
                "decision_method": decision.decision_method,
            }
        )

    stats = {
        "unmatched": unmatched,
        "auto_exact": auto_exact,
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "no_llm": no_llm_count,
    }
    return corrected_rows, stats


def compute_metrics(rows: List[Dict]) -> Dict[str, Any]:
    """
    Compute meaningful metrics for ATL generation evaluation.
    
    Metrics focus on:
    - Overall correctness rate
    - Breakdown by decision method (exact match vs LLM judge)
    - Semantic flexibility (how often non-exact matches are still correct)
    """
    if not rows:
        return _empty_metrics()
    
    # Filter to evaluated rows only
    evaluated_rows = [r for r in rows if r.get("decision_method") != "unmatched"]
    total_evaluated = len(evaluated_rows)
    
    if total_evaluated == 0:
        return _empty_metrics()
    
    # === Core Correctness ===
    correct_count = sum(1 for r in evaluated_rows if r.get("correct") == "yes")
    incorrect_count = total_evaluated - correct_count
    accuracy = correct_count / total_evaluated
    
    # === Breakdown by Decision Method ===
    exact_match_rows = [r for r in evaluated_rows if r.get("decision_method") == "exact"]
    llm_judged_rows = [r for r in evaluated_rows if r.get("decision_method") == "llm"]
    no_llm_rows = [r for r in evaluated_rows if r.get("decision_method") == "no_llm"]
    
    exact_match_count = len(exact_match_rows)
    llm_judged_count = len(llm_judged_rows)
    
    # Exact matches are always correct by definition
    exact_match_correct = exact_match_count
    
    # LLM-judged: how many did the judge approve?
    llm_approved = sum(1 for r in llm_judged_rows if r.get("correct") == "yes")
    llm_rejected = llm_judged_count - llm_approved
    
    # === Derived Rates ===
    # What fraction of outputs matched gold exactly?
    exact_match_rate = exact_match_count / total_evaluated if total_evaluated else 0.0
    
    # What fraction required LLM judgment?
    llm_judge_rate = llm_judged_count / total_evaluated if total_evaluated else 0.0
    
    # Of those requiring LLM judgment, how many were approved?
    # This measures "semantic flexibility" - correct despite surface differences
    llm_approval_rate = llm_approved / llm_judged_count if llm_judged_count else 0.0
    
    # How much does LLM judging "rescue" the accuracy?
    # (accuracy with LLM judge) - (accuracy if we only counted exact matches)
    accuracy_boost_from_llm = (llm_approved / total_evaluated) if total_evaluated else 0.0
    
    return {
        # === Primary Metrics ===
        "accuracy": round(accuracy, 4),
        "total_evaluated": total_evaluated,
        "correct": correct_count,
        "incorrect": incorrect_count,
        
        # === Decision Method Breakdown ===
        "exact_match": {
            "count": exact_match_count,
            "rate": round(exact_match_rate, 4),
        },
        "llm_judged": {
            "count": llm_judged_count,
            "rate": round(llm_judge_rate, 4),
            "approved": llm_approved,
            "rejected": llm_rejected,
            "approval_rate": round(llm_approval_rate, 4),
        },
        
        # === Interpretable Summary ===
        "accuracy_from_exact_match": round(exact_match_rate, 4),
        "accuracy_boost_from_llm": round(accuracy_boost_from_llm, 4),
        
        # === No LLM fallback (if used) ===
        "no_llm_fallback_count": len(no_llm_rows),
    }


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure."""
    return {
        "accuracy": 0.0,
        "total_evaluated": 0,
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
    base_metrics = compute_metrics(rows)
    
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for r in rows:
        difficulty = r.get("difficulty", "unknown")
        by_difficulty[difficulty].append(r)
    
    difficulty_breakdown = {}
    for difficulty, diff_rows in sorted(by_difficulty.items()):
        evaluated = [r for r in diff_rows if r.get("decision_method") != "unmatched"]
        if not evaluated:
            continue
        correct = sum(1 for r in evaluated if r.get("correct") == "yes")
        difficulty_breakdown[difficulty] = {
            "count": len(evaluated),
            "correct": correct,
            "accuracy": round(correct / len(evaluated), 4),
        }
    
    base_metrics["by_difficulty"] = difficulty_breakdown
    return base_metrics

def build_summary(
    results: List[Dict],
    totals: Dict,
    judge_model: str,
    prompt_version: str,
) -> Dict:
    """Build summary with improved metrics."""
    
    # Aggregate all rows
    all_rows = []
    for result in results:
        all_rows.extend(result["rows"])
    
    overall_metrics = compute_metrics(all_rows)
    
    # Per-file breakdown
    per_file = []
    for result in results:
        per_file.append({
            "source_file": result["source_file"],
            "stem": result["stem"],
            "metrics": result["metrics"],
            "stats": result["stats"],
        })
    
    # Ranking by accuracy
    ranking = sorted(
        per_file,
        key=lambda r: -r["metrics"]["accuracy"],
    )
    
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
        "overall": overall_metrics,
        "per_file": per_file,
        "ranking": ranking_table,
        "totals": totals,
    }


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_summary_notebook(summary_path: Path, output_path: Path):
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
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "\n",
                    "# Resolve summary path from common locations\n",
                    f'summary_rel = Path("{summary_path.name}")\n',
                    "candidates = [Path.cwd() / summary_rel]\n",
                    "for parent in Path.cwd().parents:\n",
                    "    candidates.append(parent / summary_rel)\n",
                    "candidates.append(Path(__file__).resolve().parent / summary_rel if '__file__' in globals() else None)\n",
                    "candidates = [c for c in candidates if c is not None]\n",
                    "\n",
                    "summary_path = next((c for c in candidates if c.exists()), None)\n",
                    "if summary_path is None:\n",
                    "    raise FileNotFoundError(\n",
                    "        'Could not find summary__judge-*.json. Run run_llm_judge.py first.'\n",
                    "    )\n",
                    "\n",
                    "summary = json.loads(summary_path.read_text(encoding='utf-8'))\n",
                    "per_file = summary['per_file']\n",
                    "df = pd.DataFrame([\n",
                    "    {\n",
                    "        'source_file': item['source_file'],\n",
                    "        'accuracy': item.get('metrics', {}).get('accuracy', 0.0),\n",
                    "        'evaluated': item.get('metrics', {}).get('total_evaluated', item.get('metrics', {}).get('evaluated', 0)),\n",
                    "        'correct': item.get('metrics', {}).get('correct', 0),\n",
                    "        'incorrect': item.get('metrics', {}).get('incorrect', 0),\n",
                    "        'exact_match_rate': item.get('metrics', {}).get('exact_match', {}).get('rate', None),\n",
                    "        'llm_approval_rate': item.get('metrics', {}).get('llm_judged', {}).get('approval_rate', None),\n",
                    "    }\n",
                    "    for item in per_file\n",
                    "])\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "# Leaderboard table\n",
                    "# Sort primarily by accuracy, secondarily by correct count\n",
                    "leaderboard = df.sort_values(['accuracy', 'correct'], ascending=[False, False])\n",
                    "leaderboard[['source_file', 'accuracy', 'evaluated', 'correct', 'incorrect']]\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "# Accuracy by model file\n",
                    "plt.figure(figsize=(10, 4))\n",
                    "plt.bar(df['source_file'], df['accuracy'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.title('Accuracy by Model File')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "# Correct vs Incorrect bar chart for best model\n",
                    "best = summary['ranking'][0]['source_file'] if summary['ranking'] else None\n",
                    "if best is not None:\n",
                    "    best_row = df[df['source_file'] == best].iloc[0]\n",
                    "    plt.figure(figsize=(6, 4))\n",
                    "    plt.bar(['correct', 'incorrect'], [int(best_row.get('correct', 0)), int(best_row.get('incorrect', 0))])\n",
                    "    plt.title(f'Correct vs Incorrect: {best}')\n",
                    "    plt.ylabel('Count')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
