"""Accuracy and latency reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..infra.io import load_json, save_json


@dataclass
class EfficiencyWeights:
    """Weights for the secondary accuracy-latency score."""

    accuracy: float = 0.5
    latency: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "latency": self.latency,
        }


def _numeric_values(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values if value is not None]


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    numeric = _numeric_values(values)
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _safe_sum(values: Iterable[Any]) -> Optional[float]:
    numeric = _numeric_values(values)
    if not numeric:
        return None
    return round(sum(numeric), 6)


def _normalize(
    values: List[Optional[float]], higher_is_better: bool = True
) -> List[Optional[float]]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return [None for _ in values]

    value_min = min(numeric)
    value_max = max(numeric)
    if value_min == value_max:
        return [1.0 if value is not None else None for value in values]

    normalized: List[Optional[float]] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        scaled = (value - value_min) / (value_max - value_min)
        normalized.append(scaled if higher_is_better else 1.0 - scaled)
    return normalized


def _weighted_score(
    accuracy_score: Optional[float],
    latency_score: Optional[float],
    weights: EfficiencyWeights,
) -> Optional[float]:
    score = 0.0
    weight_sum = 0.0

    if accuracy_score is not None:
        score += accuracy_score * weights.accuracy
        weight_sum += weights.accuracy
    if latency_score is not None:
        score += latency_score * weights.latency
        weight_sum += weights.latency

    if weight_sum == 0:
        return None
    return round(score / weight_sum, 6)


def _rank_by(
    entries: List[Dict[str, Any]],
    key: str,
    higher_is_better: bool = True,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    candidates = [entry for entry in entries if entry.get(key) is not None]
    candidates.sort(key=lambda entry: entry[key], reverse=higher_is_better)

    ranking = []
    for index, item in enumerate(candidates[:top_k]):
        confidence_score = item.get("confidence_score")
        if isinstance(confidence_score, dict):
            confidence_score = confidence_score.get("mean")

        ranking.append(
            {
                "rank": index + 1,
                "model": item.get("model_short"),
                "condition": item.get("condition"),
                "judge_model": item.get("judge_model"),
                key: item.get(key),
                "confidence_score": confidence_score,
            }
        )
    return ranking


def _metric_value(metrics: Dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _metric_std(metrics: Dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("std")
    return None


def _extract_metrics_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "accuracy": _metric_value(metrics, "accuracy"),
        "accuracy_std": _metric_std(metrics, "accuracy"),
        "exact_match_rate": _metric_value(metrics, "exact_match_rate"),
        "llm_approval_rate": _metric_value(metrics, "llm_approval_rate"),
        "latency_mean_ms": _metric_value(metrics, "latency_mean_ms"),
        "latency_total_ms": _metric_value(metrics, "latency_total_ms"),
        "latency_p50_ms": _metric_value(metrics, "latency_p50_ms"),
        "latency_p95_ms": _metric_value(metrics, "latency_p95_ms"),
        "latency_p99_ms": _metric_value(metrics, "latency_p99_ms"),
        "n_examples": _metric_value(metrics, "n_examples"),
    }


def _extract_seed_metrics(seed_entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = seed_entry.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "seed": seed_entry.get("seed"),
        "source": seed_entry.get("source"),
        "accuracy": metrics.get("accuracy"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "llm_approval_rate": metrics.get("llm_approval_rate"),
        "latency_mean_ms": metrics.get("latency_mean_ms"),
        "latency_total_ms": metrics.get("latency_total_ms"),
        "latency_p50_ms": metrics.get("latency_p50_ms"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "latency_p99_ms": metrics.get("latency_p99_ms"),
        "n_examples": metrics.get("n_examples"),
    }


def _calculate_derived_metrics(extracted: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = extracted.get("accuracy")
    latency_mean_ms = extracted.get("latency_mean_ms")
    latency_total_ms = extracted.get("latency_total_ms")
    n_examples = extracted.get("n_examples")

    duration_seconds = None
    if latency_total_ms is not None:
        duration_seconds = round(float(latency_total_ms) / 1000.0, 6)

    throughput = None
    if duration_seconds and n_examples and duration_seconds > 0:
        throughput = round(float(n_examples) / duration_seconds, 6)

    accuracy_per_second = None
    if accuracy is not None and latency_mean_ms and latency_mean_ms > 0:
        accuracy_per_second = round(
            float(accuracy) / (float(latency_mean_ms) / 1000.0), 6
        )

    return {
        "duration_seconds": duration_seconds,
        "throughput_samples_per_sec": throughput,
        "accuracy_per_second": accuracy_per_second,
    }


def _apply_scores(entries: List[Dict[str, Any]], weights: EfficiencyWeights) -> None:
    accuracy_norm = _normalize([entry.get("accuracy") for entry in entries], True)
    latency_norm = _normalize(
        [entry.get("latency_mean_ms") for entry in entries], False
    )

    for entry, accuracy_score, latency_score in zip(
        entries, accuracy_norm, latency_norm
    ):
        entry["accuracy_normalized"] = accuracy_score
        entry["latency_normalized"] = latency_score
        entry["efficiency_score"] = _weighted_score(
            accuracy_score, latency_score, weights
        )


def _pareto_frontier(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        entry
        for entry in entries
        if entry.get("accuracy") is not None
        and entry.get("latency_mean_ms") is not None
    ]
    frontier = []
    for entry in candidates:
        accuracy = float(entry["accuracy"])
        latency = float(entry["latency_mean_ms"])
        dominated = False
        for other in candidates:
            if other is entry:
                continue
            other_accuracy = float(other["accuracy"])
            other_latency = float(other["latency_mean_ms"])
            at_least_as_good = other_accuracy >= accuracy and other_latency <= latency
            strictly_better = other_accuracy > accuracy or other_latency < latency
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "model": entry.get("model_short"),
                    "condition": entry.get("condition"),
                    "judge_model": entry.get("judge_model"),
                    "accuracy": entry.get("accuracy"),
                    "latency_mean_ms": entry.get("latency_mean_ms"),
                    "confidence_score": entry.get("confidence_score"),
                }
            )

    frontier.sort(
        key=lambda item: (-float(item["accuracy"]), float(item["latency_mean_ms"]))
    )
    return frontier


def _load_agreement_scores(
    agreement_report_path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    if not agreement_report_path or not agreement_report_path.exists():
        return {}

    try:
        agreement_data = load_json(agreement_report_path)
    except Exception as exc:
        print(
            f"Warning: Could not load agreement report from {agreement_report_path}: {exc}"
        )
        return {}

    scores: Dict[str, Dict[str, Any]] = {}
    per_source = (
        agreement_data.get("per_source_file", {})
        if isinstance(agreement_data, dict)
        else {}
    )
    for source_file, metrics in per_source.items():
        if isinstance(metrics, dict):
            scores[source_file] = {
                "confidence_score": metrics.get("mean_kappa"),
                "mean_kappa": metrics.get("mean_kappa"),
                "min_kappa": metrics.get("min_kappa"),
                "max_kappa": metrics.get("max_kappa"),
            }
    return scores


def _confidence_from_entry(item: Dict[str, Any]) -> Any:
    judge_agreement = item.get("judge_agreement", {})
    if not isinstance(judge_agreement, dict):
        return None
    confidence = judge_agreement.get("confidence_score")
    if isinstance(confidence, dict):
        return confidence.get("mean")
    return confidence


def build_efficiency_report_from_aggregate(
    aggregate_path: Path,
    weights: Optional[EfficiencyWeights] = None,
    top_k: int = 5,
    include_per_seed: bool = False,
    agreement_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build an accuracy-latency report from seed aggregate metrics JSON."""
    weights = weights or EfficiencyWeights()
    data = load_json(aggregate_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {aggregate_path}, got {type(data)}")

    agreement_scores = _load_agreement_scores(agreement_report_path)
    entries: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        extracted = _extract_metrics_from_entry(item)
        derived = _calculate_derived_metrics(extracted)

        entry = {
            "model_short": item.get("model_short"),
            "condition": item.get("condition"),
            "judge_model": item.get("judge_model"),
            "judge_models": item.get("judge_models", []),
            "finetuned": item.get("finetuned", False),
            "few_shot": item.get("few_shot", False),
            "num_seeds": item.get("num_seeds", 0),
            "num_runs": item.get("num_runs", 0),
            **extracted,
            **derived,
            "confidence_score": _confidence_from_entry(item),
        }

        source_file = item.get("source")
        if source_file and source_file in agreement_scores:
            entry.update(agreement_scores[source_file])

        if include_per_seed:
            seed_entries = []
            for seed_data in item.get("per_seed", []):
                if not isinstance(seed_data, dict):
                    continue
                seed_extracted = _extract_seed_metrics(seed_data)
                seed_entries.append(
                    {**seed_extracted, **_calculate_derived_metrics(seed_extracted)}
                )
            _apply_scores(seed_entries, weights)
            entry["per_seed_efficiency"] = seed_entries

        entries.append(entry)

    _apply_scores(entries, weights)

    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(aggregate_path),
        "weights": weights.to_dict(),
        "include_per_seed": include_per_seed,
        "totals": {
            "model_conditions": len(entries),
            "unique_models": len(
                {
                    entry.get("model_short")
                    for entry in entries
                    if entry.get("model_short")
                }
            ),
            "unique_conditions": len(
                {entry.get("condition") for entry in entries if entry.get("condition")}
            ),
            "with_accuracy": sum(
                1 for entry in entries if entry.get("accuracy") is not None
            ),
            "with_latency": sum(
                1 for entry in entries if entry.get("latency_mean_ms") is not None
            ),
        },
        "models": entries,
        "pareto_frontier": _pareto_frontier(entries),
        "rankings": {
            "fastest": _rank_by(
                entries, "latency_mean_ms", higher_is_better=False, top_k=top_k
            ),
            "most_accurate": _rank_by(
                entries, "accuracy", higher_is_better=True, top_k=top_k
            ),
            "best_accuracy_per_second": _rank_by(
                entries, "accuracy_per_second", higher_is_better=True, top_k=top_k
            ),
            "highest_throughput": _rank_by(
                entries,
                "throughput_samples_per_sec",
                higher_is_better=True,
                top_k=top_k,
            ),
            "best_efficiency_score": _rank_by(
                entries, "efficiency_score", higher_is_better=True, top_k=top_k
            ),
        },
        "by_condition": _group_by_condition(entries),
        "by_model": _group_by_model(entries),
        "notes": {
            "score_definition": "Secondary score from normalized accuracy and inverse normalized mean latency.",
            "missing_accuracy": [
                f"{entry.get('model_short')}_{entry.get('condition')}"
                for entry in entries
                if entry.get("accuracy") is None
            ],
            "missing_latency": [
                f"{entry.get('model_short')}_{entry.get('condition')}"
                for entry in entries
                if entry.get("latency_mean_ms") is None
            ],
        },
    }


def _accuracy_from_judge_summary(
    prediction_path: Path,
    judge_summary: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not judge_summary:
        return None

    for item in judge_summary.get("per_file", []):
        if not isinstance(item, dict):
            continue
        source_file = item.get("source_file")
        if source_file not in {prediction_path.name, str(prediction_path)}:
            continue
        metrics = item.get("metrics", {})
        accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
        return float(accuracy) if accuracy is not None else None

    return None


def _accuracy_from_predictions(
    metadata: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Optional[float]:
    metrics = metadata.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("exact_match") is not None:
        return float(metrics["exact_match"])

    matches = [row.get("exact_match") for row in predictions if "exact_match" in row]
    if not matches:
        return None
    return round(sum(1 for value in matches if bool(value)) / len(matches), 6)


def _prediction_file_entry(
    prediction_path: Path,
    judge_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = load_json(prediction_path)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(predictions, list):
        predictions = []

    judge_accuracy = _accuracy_from_judge_summary(prediction_path, judge_summary)
    if judge_accuracy is not None:
        accuracy = judge_accuracy
        accuracy_source = "llm_judge"
    else:
        accuracy = _accuracy_from_predictions(metadata, predictions)
        accuracy_source = "exact_match" if accuracy is not None else None

    total_samples = metadata.get("total_samples") or len(predictions)
    latency_total_ms = metadata.get("latency_total_ms")
    if latency_total_ms is None and metadata.get("duration_seconds") is not None:
        latency_total_ms = float(metadata["duration_seconds"]) * 1000.0
    if latency_total_ms is None:
        latencies = [
            row.get("latency_ms")
            for row in predictions
            if row.get("latency_ms") is not None
        ]
        latency_total_ms = _safe_sum(latencies)

    latency_mean_ms = metadata.get("latency_mean_ms")
    if latency_mean_ms is None and latency_total_ms is not None and total_samples:
        latency_mean_ms = round(float(latency_total_ms) / float(total_samples), 6)

    extracted = {
        "accuracy": accuracy,
        "latency_mean_ms": latency_mean_ms,
        "latency_total_ms": latency_total_ms,
        "n_examples": total_samples,
    }

    return {
        "source_file": str(prediction_path),
        "run_id": metadata.get("run_id") or prediction_path.stem,
        "model": metadata.get("model"),
        "model_short": metadata.get("model_short") or metadata.get("model"),
        "condition": metadata.get("condition"),
        "accuracy": accuracy,
        "accuracy_source": accuracy_source,
        "latency_mean_ms": latency_mean_ms,
        "latency_total_ms": latency_total_ms,
        "n_examples": total_samples,
        **_calculate_derived_metrics(extracted),
    }


def build_efficiency_report(
    prediction_files: List[Path],
    judge_summary: Optional[Dict[str, Any]] = None,
    weights: Optional[EfficiencyWeights] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Build an accuracy-latency report directly from prediction JSON files."""
    weights = weights or EfficiencyWeights()
    entries = [
        _prediction_file_entry(Path(prediction_file), judge_summary)
        for prediction_file in prediction_files
    ]
    _apply_scores(entries, weights)

    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": weights.to_dict(),
        "totals": {
            "models": len(entries),
            "with_accuracy": sum(
                1 for entry in entries if entry.get("accuracy") is not None
            ),
            "with_latency": sum(
                1 for entry in entries if entry.get("latency_mean_ms") is not None
            ),
        },
        "models": entries,
        "pareto_frontier": _pareto_frontier(entries),
        "rankings": {
            "fastest": _rank_by(
                entries, "latency_mean_ms", higher_is_better=False, top_k=top_k
            ),
            "most_accurate": _rank_by(
                entries, "accuracy", higher_is_better=True, top_k=top_k
            ),
            "best_efficiency_score": _rank_by(
                entries, "efficiency_score", higher_is_better=True, top_k=top_k
            ),
        },
    }


def _group_by_condition(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        condition = entry.get("condition") or "unknown"
        groups.setdefault(condition, []).append(entry)

    result = {}
    for condition, items in groups.items():
        accuracies = _numeric_values(entry.get("accuracy") for entry in items)
        latencies = _numeric_values(entry.get("latency_mean_ms") for entry in items)
        scores = _numeric_values(entry.get("efficiency_score") for entry in items)
        result[condition] = {
            "count": len(items),
            "models": [entry.get("model_short") for entry in items],
            "accuracy_mean": _safe_mean(accuracies),
            "latency_mean_ms": _safe_mean(latencies),
            "efficiency_score_mean": _safe_mean(scores),
        }
    return result


def _group_by_model(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        model = entry.get("model_short") or "unknown"
        groups.setdefault(model, []).append(entry)

    result = {}
    for model, items in groups.items():
        accuracies = _numeric_values(entry.get("accuracy") for entry in items)
        latencies = _numeric_values(entry.get("latency_mean_ms") for entry in items)
        scores = _numeric_values(entry.get("efficiency_score") for entry in items)
        result[model] = {
            "count": len(items),
            "conditions": [entry.get("condition") for entry in items],
            "accuracy_mean": _safe_mean(accuracies),
            "accuracy_best": max(accuracies) if accuracies else None,
            "latency_mean_ms": _safe_mean(latencies),
            "latency_best_ms": min(latencies) if latencies else None,
            "efficiency_score_mean": _safe_mean(scores),
            "efficiency_score_best": max(scores) if scores else None,
        }
    return result


def build_efficiency_notebook(report_path: Path, output_path: Path) -> None:
    """Generate a small notebook for exploring the accuracy-latency report."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "# Model Accuracy-Latency Report\n",
                    f"Report file: `{report_path.name}`\n",
                    "\n",
                    "This notebook explores reproducible accuracy and latency metrics.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    f'report_path = Path("{report_path}")\n',
                    "if not report_path.exists():\n",
                    f'    report_path = Path("{report_path.name}")\n',
                    "report = json.loads(report_path.read_text())\n",
                    "df = pd.DataFrame(report['models'])\n",
                    "display(df)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["## Accuracy vs Latency\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6))\n",
                    "plt.scatter(df['latency_mean_ms'], df['accuracy'], s=80)\n",
                    "for _, row in df.iterrows():\n",
                    "    plt.annotate(row.get('model_short', ''), (row['latency_mean_ms'], row['accuracy']), fontsize=8)\n",
                    "plt.xlabel('Mean latency (ms)')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.title('Accuracy vs latency')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["## Pareto Frontier\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "display(pd.DataFrame(report.get('pareto_frontier', [])))\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["## Rankings\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "for name, rows in report['rankings'].items():\n",
                    "    print(f'\\n{name}')\n",
                    "    display(pd.DataFrame(rows))\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    save_json(notebook, output_path)


def resolve_prediction_files(predictions_dir: Path, datasets: List[str]) -> List[Path]:
    """Resolve prediction files from a directory."""
    if not datasets or (len(datasets) == 1 and str(datasets[0]).lower() == "all"):
        return sorted(predictions_dir.rglob("*.json"))

    resolved: List[Path] = []
    for entry in datasets:
        path = Path(entry)
        if path.exists():
            resolved.append(path)
            continue

        candidate = predictions_dir / entry
        if candidate.exists():
            resolved.append(candidate)
            continue

        if not str(entry).endswith(".json"):
            candidate = predictions_dir / f"{entry}.json"
            if candidate.exists():
                resolved.append(candidate)
                continue

        matches = list(predictions_dir.rglob(str(entry)))
        if matches:
            resolved.extend(matches)

    return resolved


def load_judge_summary(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a judge summary file."""
    if not path or not Path(path).exists():
        return None
    data = load_json(path)
    return data if isinstance(data, dict) else None
