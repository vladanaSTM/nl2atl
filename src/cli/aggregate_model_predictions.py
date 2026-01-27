#!/usr/bin/env python3
"""Aggregate seed metrics from saved evaluation outputs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


NUMERIC_FIELDS = [
    "accuracy",
    "exact_match",
    "exact_match_rate",
    "llm_approval_rate",
    "total_evaluated",
    "evaluated",
    "correct",
    "incorrect",
    "accuracy_from_exact_match",
    "accuracy_boost_from_llm",
    "no_llm_fallback_count",
    "n_examples",
    "total_tokens_input",
    "total_tokens_output",
    "total_tokens",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_total_ms",
]


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _iter_prediction_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def _extract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)

    return {
        key: value
        for key, value in payload.items()
        if key not in {"predictions", "detailed_results"}
    }


def _flatten_judge_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    exact_match = metrics.get("exact_match") or {}
    llm_judged = metrics.get("llm_judged") or {}
    return {
        "accuracy": metrics.get("accuracy"),
        "exact_match_rate": exact_match.get("rate"),
        "llm_approval_rate": llm_judged.get("approval_rate"),
        "total_evaluated": metrics.get("total_evaluated"),
        "evaluated": metrics.get("evaluated"),
        "correct": metrics.get("correct"),
        "incorrect": metrics.get("incorrect"),
        "accuracy_from_exact_match": metrics.get("accuracy_from_exact_match"),
        "accuracy_boost_from_llm": metrics.get("accuracy_boost_from_llm"),
        "no_llm_fallback_count": metrics.get("no_llm_fallback_count"),
    }


def _extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    from ..evaluation.llm_judge import compute_metrics

    metadata = _extract_metadata(payload)
    metrics = (metadata.get("metrics") or {}).copy()

    for key in [
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_total_ms",
    ]:
        if key in metadata:
            metrics[key] = metadata.get(key)

    detailed = payload.get("detailed_results")
    if isinstance(detailed, list):
        judge_metrics = _flatten_judge_metrics(compute_metrics(detailed))
        metrics.update(judge_metrics)

    metrics["n_examples"] = metrics.get("n_examples") or metadata.get("total_samples")
    return metrics


def aggregate_predictions(input_dir: Path) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, bool, bool], List[Dict[str, Any]]] = defaultdict(list)

    for path in _iter_prediction_files(input_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        metadata = _extract_metadata(payload)
        model_short = metadata.get("model_short") or metadata.get("model") or "unknown"
        condition = metadata.get("condition") or "unknown"
        finetuned = bool(metadata.get("finetuned", False))
        few_shot = bool(metadata.get("few_shot", False))
        seed = metadata.get("seed")

        metrics = _extract_metrics(payload)

        grouped[(model_short, condition, finetuned, few_shot)].append(
            {
                "seed": seed,
                "model": metadata.get("model"),
                "model_short": model_short,
                "condition": condition,
                "finetuned": finetuned,
                "few_shot": few_shot,
                "metrics": metrics,
                "source": path.name,
            }
        )

    aggregates: List[Dict[str, Any]] = []
    for (model_short, condition, finetuned, few_shot), items in grouped.items():
        agg_metrics: Dict[str, Dict[str, float]] = {}
        for field in NUMERIC_FIELDS:
            values = [
                i["metrics"].get(field)
                for i in items
                if i["metrics"].get(field) is not None
            ]
            if values:
                agg_metrics[field] = {
                    "mean": _safe_mean(values),
                    "std": _safe_std(values),
                }

        aggregates.append(
            {
                "model_short": model_short,
                "condition": condition,
                "finetuned": finetuned,
                "few_shot": few_shot,
                "num_seeds": len(items),
                "metrics": agg_metrics,
                "per_seed": [
                    {
                        "seed": i["seed"],
                        "metrics": i["metrics"],
                        "source": i["source"],
                    }
                    for i in items
                ],
            }
        )

    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate seed metrics from saved evaluation outputs."
    )
    parser.add_argument(
        "--input_dir",
        default="outputs/LLM-evaluation/evaluated_datasets",
        help="Directory containing evaluated JSON files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/seed_aggregate_metrics_from_judged.json",
        help="Path to write aggregated metrics JSON.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    aggregates = aggregate_predictions(input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Aggregated {len(aggregates)} groups into {output_path}")


if __name__ == "__main__":
    main()
