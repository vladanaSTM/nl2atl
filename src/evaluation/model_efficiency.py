"""Model efficiency reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..infra.io import load_json, save_json


@dataclass
class EfficiencyWeights:
    accuracy: float = 0.5
    cost: float = 0.25
    latency: float = 0.25

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "cost": self.cost,
            "latency": self.latency,
        }


def resolve_prediction_files(predictions_dir: Path, datasets: List[str]) -> List[Path]:
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
    if not path:
        return None
    if not Path(path).exists():
        return None
    data = load_json(path)
    return data if isinstance(data, dict) else None


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _safe_sum(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values), 6)


def _extract_predictions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions = data.get("predictions")
    return predictions if isinstance(predictions, list) else []


def _extract_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = data.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _extract_accuracy(
    metadata: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    summary_accuracy: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:
    if summary_accuracy is not None:
        return float(summary_accuracy), "llm_judge"

    metrics = (
        metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    )
    if "exact_match" in metrics:
        return float(metrics["exact_match"]), "exact_match"

    exact_matches = [p.get("exact_match") for p in predictions if "exact_match" in p]
    if exact_matches:
        rate = sum(1 for v in exact_matches if v) / len(exact_matches)
        return float(rate), "exact_match"

    return None, None


def _extract_costs(
    metadata: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    metrics = (
        metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    )

    cost_total = (
        metadata.get("cost_total_usd")
        if metadata.get("cost_total_usd") is not None
        else metrics.get("total_cost_usd")
    )
    cost_input = (
        metadata.get("cost_input_usd")
        if metadata.get("cost_input_usd") is not None
        else metrics.get("total_cost_input_usd")
    )
    cost_output = (
        metadata.get("cost_output_usd")
        if metadata.get("cost_output_usd") is not None
        else metrics.get("total_cost_output_usd")
    )

    if cost_total is None:
        cost_vals = [p.get("cost_usd") for p in predictions if p.get("cost_usd")]
        cost_total = _safe_sum([float(v) for v in cost_vals])

    if cost_input is None:
        cost_vals = [
            p.get("cost_input_usd") for p in predictions if p.get("cost_input_usd")
        ]
        cost_input = _safe_sum([float(v) for v in cost_vals])

    if cost_output is None:
        cost_vals = [
            p.get("cost_output_usd") for p in predictions if p.get("cost_output_usd")
        ]
        cost_output = _safe_sum([float(v) for v in cost_vals])

    avg_cost = metadata.get("avg_cost_usd")
    if avg_cost is None and cost_total is not None:
        total_samples = metadata.get("total_samples") or len(predictions) or 0
        avg_cost = float(cost_total) / total_samples if total_samples else None

    return {
        "cost_total_usd": float(cost_total) if cost_total is not None else None,
        "cost_input_usd": float(cost_input) if cost_input is not None else None,
        "cost_output_usd": float(cost_output) if cost_output is not None else None,
        "avg_cost_usd": float(avg_cost) if avg_cost is not None else None,
    }


def _extract_latency(
    metadata: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    latency_mean = metadata.get("latency_mean_ms")
    latency_total = metadata.get("latency_total_ms")

    if latency_mean is None:
        latencies = [p.get("latency_ms") for p in predictions if p.get("latency_ms")]
        latency_mean = _safe_mean([float(v) for v in latencies])

    if latency_total is None:
        latencies = [p.get("latency_ms") for p in predictions if p.get("latency_ms")]
        latency_total = _safe_sum([float(v) for v in latencies])

    duration_seconds = metadata.get("duration_seconds")
    if duration_seconds is None and latency_total is not None:
        duration_seconds = float(latency_total) / 1000.0

    return {
        "latency_mean_ms": float(latency_mean) if latency_mean is not None else None,
        "latency_total_ms": float(latency_total) if latency_total is not None else None,
        "duration_seconds": (
            float(duration_seconds) if duration_seconds is not None else None
        ),
    }


def _extract_tokens(
    metadata: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Optional[int]]:
    metrics = (
        metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    )
    tokens_input = metrics.get("total_tokens_input")
    tokens_output = metrics.get("total_tokens_output")
    tokens_total = metrics.get("total_tokens")

    if tokens_input is None:
        vals = [p.get("tokens_input") for p in predictions if p.get("tokens_input")]
        tokens_input = int(sum(vals)) if vals else None

    if tokens_output is None:
        vals = [p.get("tokens_output") for p in predictions if p.get("tokens_output")]
        tokens_output = int(sum(vals)) if vals else None

    if tokens_total is None:
        vals = [p.get("tokens_input") for p in predictions if p.get("tokens_input")]
        outs = [p.get("tokens_output") for p in predictions if p.get("tokens_output")]
        if vals or outs:
            tokens_total = int(sum(vals) + sum(outs))

    return {
        "total_tokens_input": int(tokens_input) if tokens_input is not None else None,
        "total_tokens_output": (
            int(tokens_output) if tokens_output is not None else None
        ),
        "total_tokens": int(tokens_total) if tokens_total is not None else None,
    }


def _normalize(
    values: List[Optional[float]], higher_is_better: bool = True
) -> List[Optional[float]]:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return [None for _ in values]
    v_min, v_max = min(numeric), max(numeric)
    if v_min == v_max:
        return [1.0 if v is not None else None for v in values]
    normalized = []
    for v in values:
        if v is None:
            normalized.append(None)
            continue
        scale = (v - v_min) / (v_max - v_min)
        normalized.append(scale if higher_is_better else 1.0 - scale)
    return normalized


def _weighted_score(
    accuracy: Optional[float],
    cost_score: Optional[float],
    latency_score: Optional[float],
    weights: EfficiencyWeights,
) -> Optional[float]:
    score = 0.0
    weight_sum = 0.0

    if accuracy is not None:
        score += accuracy * weights.accuracy
        weight_sum += weights.accuracy
    if cost_score is not None:
        score += cost_score * weights.cost
        weight_sum += weights.cost
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
    candidates = [e for e in entries if e.get(key) is not None]
    candidates.sort(key=lambda x: x[key], reverse=higher_is_better)
    return [
        {
            "rank": idx + 1,
            "model": item.get("model_short") or item.get("model"),
            "source_file": item.get("source_file"),
            key: item.get(key),
        }
        for idx, item in enumerate(candidates[:top_k])
    ]


def build_efficiency_report(
    prediction_files: Iterable[Path],
    judge_summary: Optional[Dict[str, Any]] = None,
    weights: Optional[EfficiencyWeights] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    weights = weights or EfficiencyWeights()
    summary_map: Dict[str, float] = {}
    if judge_summary and isinstance(judge_summary.get("per_file"), list):
        for entry in judge_summary["per_file"]:
            if not isinstance(entry, dict):
                continue
            source_file = entry.get("source_file")
            metrics = (
                entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            )
            if source_file and "accuracy" in metrics:
                summary_map[str(source_file)] = float(metrics["accuracy"])

    entries: List[Dict[str, Any]] = []

    for path in prediction_files:
        data = load_json(path)
        if not isinstance(data, dict):
            continue

        metadata = _extract_metadata(data)
        predictions = _extract_predictions(data)
        total_samples = metadata.get("total_samples") or len(predictions)

        summary_accuracy = summary_map.get(path.name)
        accuracy, accuracy_source = _extract_accuracy(
            metadata, predictions, summary_accuracy
        )
        cost_stats = _extract_costs(metadata, predictions)
        latency_stats = _extract_latency(metadata, predictions)
        token_stats = _extract_tokens(metadata, predictions)

        gpu_hour_usd = metadata.get("gpu_hour_usd")
        total_tokens = token_stats.get("total_tokens")

        avg_cost = cost_stats.get("avg_cost_usd")
        latency_mean = latency_stats.get("latency_mean_ms")
        duration_seconds = latency_stats.get("duration_seconds")

        cost_per_1k_tokens = None
        tokens_per_hour = None

        if duration_seconds and total_tokens and duration_seconds > 0:
            tokens_per_hour = round(
                float(total_tokens) / (float(duration_seconds) / 3600.0), 6
            )

        if avg_cost is None and gpu_hour_usd and duration_seconds and total_tokens:
            hours_used = float(duration_seconds) / 3600.0
            if hours_used > 0:
                cost_total_gpu = round(float(gpu_hour_usd) * hours_used, 6)
                avg_cost = (
                    round(cost_total_gpu / float(total_samples), 6)
                    if total_samples
                    else None
                )
                cost_stats = {
                    **cost_stats,
                    "cost_total_usd": cost_total_gpu,
                    "avg_cost_usd": avg_cost,
                }

        if avg_cost is not None and total_tokens:
            cost_total = cost_stats.get("cost_total_usd")
            if cost_total is not None and total_tokens > 0:
                cost_per_1k_tokens = round(
                    (float(cost_total) / float(total_tokens)) * 1000.0, 6
                )

        throughput = None
        if duration_seconds and total_samples:
            throughput = round(float(total_samples) / float(duration_seconds), 6)

        accuracy_per_dollar = None
        if accuracy is not None and avg_cost and avg_cost > 0:
            accuracy_per_dollar = round(float(accuracy) / float(avg_cost), 6)

        accuracy_per_second = None
        if accuracy is not None and latency_mean and latency_mean > 0:
            accuracy_per_second = round(
                float(accuracy) / (float(latency_mean) / 1000.0), 6
            )

        entry = {
            "source_file": path.name,
            "run_id": metadata.get("run_id"),
            "model": metadata.get("model"),
            "model_short": metadata.get("model_short"),
            "condition": metadata.get("condition"),
            "seed": metadata.get("seed"),
            "finetuned": metadata.get("finetuned"),
            "few_shot": metadata.get("few_shot"),
            "total_samples": total_samples,
            "accuracy": accuracy,
            "accuracy_source": accuracy_source,
            "accuracy_per_dollar": accuracy_per_dollar,
            "accuracy_per_second": accuracy_per_second,
            "throughput_samples_per_sec": throughput,
            "gpu_hour_usd": float(gpu_hour_usd) if gpu_hour_usd is not None else None,
            "tokens_per_hour": tokens_per_hour,
            "cost_per_1k_tokens_usd": cost_per_1k_tokens,
            **cost_stats,
            **latency_stats,
            **token_stats,
        }
        entries.append(entry)

    accuracy_list = [e.get("accuracy") for e in entries]
    cost_list = [e.get("avg_cost_usd") for e in entries]
    latency_list = [e.get("latency_mean_ms") for e in entries]

    accuracy_norm = _normalize(accuracy_list, higher_is_better=True)
    cost_norm = _normalize(cost_list, higher_is_better=False)
    latency_norm = _normalize(latency_list, higher_is_better=False)

    for entry, acc_norm, cost_norm_val, lat_norm in zip(
        entries, accuracy_norm, cost_norm, latency_norm
    ):
        entry["accuracy_normalized"] = acc_norm
        entry["cost_normalized"] = cost_norm_val
        entry["latency_normalized"] = lat_norm
        entry["efficiency_score"] = _weighted_score(
            acc_norm, cost_norm_val, lat_norm, weights
        )

    report = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": weights.to_dict(),
        "totals": {
            "models": len(entries),
            "with_accuracy": sum(1 for e in entries if e.get("accuracy") is not None),
            "with_cost": sum(1 for e in entries if e.get("avg_cost_usd") is not None),
            "with_latency": sum(
                1 for e in entries if e.get("latency_mean_ms") is not None
            ),
        },
        "models": entries,
        "rankings": {
            "cheapest": _rank_by(
                entries, "avg_cost_usd", higher_is_better=False, top_k=top_k
            ),
            "fastest": _rank_by(
                entries, "latency_mean_ms", higher_is_better=False, top_k=top_k
            ),
            "most_accurate": _rank_by(
                entries, "accuracy", higher_is_better=True, top_k=top_k
            ),
            "best_accuracy_per_dollar": _rank_by(
                entries, "accuracy_per_dollar", higher_is_better=True, top_k=top_k
            ),
            "best_accuracy_per_second": _rank_by(
                entries, "accuracy_per_second", higher_is_better=True, top_k=top_k
            ),
            "best_efficiency_score": _rank_by(
                entries, "efficiency_score", higher_is_better=True, top_k=top_k
            ),
        },
        "notes": {
            "missing_accuracy": [
                e.get("source_file") for e in entries if e.get("accuracy") is None
            ],
            "missing_cost": [
                e.get("source_file") for e in entries if e.get("avg_cost_usd") is None
            ],
            "missing_latency": [
                e.get("source_file")
                for e in entries
                if e.get("latency_mean_ms") is None
            ],
        },
    }

    return report


def build_efficiency_notebook(report_path: Path, output_path: Path) -> None:
    """Generate a Jupyter notebook for exploring the efficiency report."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "# Model Efficiency Report\n",
                    f"Report file: {report_path.name}\n",
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
                    "\n",
                    f'report_path = Path("{report_path.name}")\n',
                    "if not report_path.exists():\n",
                    "    for parent in Path.cwd().parents:\n",
                    "        candidate = parent / report_path.name\n",
                    "        if candidate.exists():\n",
                    "            report_path = candidate\n",
                    "            break\n",
                    "\n",
                    "report = json.loads(report_path.read_text())\n",
                    "df = pd.DataFrame(report['models'])\n",
                    "display(df.sort_values('efficiency_score', ascending=False).head(10))\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "df_sorted = df.sort_values('accuracy', ascending=False)\n",
                    "plt.bar(df_sorted['model_short'].fillna(df_sorted['model']), df_sorted['accuracy'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.title('Accuracy by Model')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "df_sorted = df.sort_values('avg_cost_usd', ascending=True)\n",
                    "plt.bar(df_sorted['model_short'].fillna(df_sorted['model']), df_sorted['avg_cost_usd'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Average Cost (USD)')\n",
                    "plt.title('Cost per Sample by Model')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "df_sorted = df.sort_values('latency_mean_ms', ascending=True)\n",
                    "plt.bar(df_sorted['model_short'].fillna(df_sorted['model']), df_sorted['latency_mean_ms'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Latency Mean (ms)')\n",
                    "plt.title('Latency by Model')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "df_sorted = df.sort_values('efficiency_score', ascending=False)\n",
                    "plt.bar(df_sorted['model_short'].fillna(df_sorted['model']), df_sorted['efficiency_score'])\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Efficiency Score')\n",
                    "plt.title('Composite Efficiency Score by Model')\n",
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
