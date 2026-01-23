"""Metrics computation for LLM judge evaluations."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class JudgeMetrics:
    accuracy: float
    total_evaluated: int
    correct: int
    incorrect: int


def _safe_rate(numerator: int, denominator: int) -> float:
    """Compute rate safely, avoiding division by zero."""
    return numerator / denominator if denominator else 0.0


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


def compute_metrics_with_difficulty(rows: List[Dict]) -> Dict[str, Any]:
    """Compute metrics with breakdown by difficulty level."""
    base = compute_metrics(rows)

    by_difficulty: Dict[str, List[Dict]] = {}
    for r in rows:
        difficulty = r.get("difficulty", "unknown")
        by_difficulty.setdefault(difficulty, []).append(r)

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
