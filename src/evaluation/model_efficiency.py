"""Model efficiency reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _safe_sum(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values), 6)


def _normalize(
    values: List[Optional[float]], higher_is_better: bool = True
) -> List[Optional[float]]:
    """Normalize values to 0-1 range."""
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
    """Calculate weighted efficiency score."""
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
    """Rank entries by a specific key, including confidence scores."""
    candidates = [e for e in entries if e.get(key) is not None]
    candidates.sort(key=lambda x: x[key], reverse=higher_is_better)

    results = []
    for idx, item in enumerate(candidates[:top_k]):
        # Extract confidence score, handling both scalar and dict formats
        conf_score = item.get("confidence_score")
        if isinstance(conf_score, dict):
            # If it's aggregated (has mean/std), use the mean
            conf_score = conf_score.get("mean")

        results.append(
            {
                "rank": idx + 1,
                "model": item.get("model_short"),
                "condition": item.get("condition"),
                key: item.get(key),
                "confidence_score": conf_score,
            }
        )
    return results


def _extract_metrics_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant metrics from an aggregate entry."""
    metrics = entry.get("metrics", {})

    # Get mean values from metrics
    accuracy = None
    if "accuracy" in metrics:
        acc_data = metrics["accuracy"]
        if isinstance(acc_data, dict):
            accuracy = acc_data.get("mean")
        else:
            accuracy = acc_data

    latency_mean_ms = None
    if "latency_mean_ms" in metrics:
        lat_data = metrics["latency_mean_ms"]
        if isinstance(lat_data, dict):
            latency_mean_ms = lat_data.get("mean")
        else:
            latency_mean_ms = lat_data

    latency_total_ms = None
    if "latency_total_ms" in metrics:
        lat_total_data = metrics["latency_total_ms"]
        if isinstance(lat_total_data, dict):
            latency_total_ms = lat_total_data.get("mean")
        else:
            latency_total_ms = lat_total_data

    total_tokens = None
    if "total_tokens" in metrics:
        tok_data = metrics["total_tokens"]
        if isinstance(tok_data, dict):
            total_tokens = tok_data.get("mean")
        else:
            total_tokens = tok_data

    total_tokens_input = None
    if "total_tokens_input" in metrics:
        tok_in_data = metrics["total_tokens_input"]
        if isinstance(tok_in_data, dict):
            total_tokens_input = tok_in_data.get("mean")
        else:
            total_tokens_input = tok_in_data

    total_tokens_output = None
    if "total_tokens_output" in metrics:
        tok_out_data = metrics["total_tokens_output"]
        if isinstance(tok_out_data, dict):
            total_tokens_output = tok_out_data.get("mean")
        else:
            total_tokens_output = tok_out_data

    n_examples = None
    if "n_examples" in metrics:
        n_data = metrics["n_examples"]
        if isinstance(n_data, dict):
            n_examples = n_data.get("mean")
        else:
            n_examples = n_data

    exact_match_rate = None
    if "exact_match_rate" in metrics:
        em_data = metrics["exact_match_rate"]
        if isinstance(em_data, dict):
            exact_match_rate = em_data.get("mean")
        else:
            exact_match_rate = em_data

    llm_approval_rate = None
    if "llm_approval_rate" in metrics:
        llm_data = metrics["llm_approval_rate"]
        if isinstance(llm_data, dict):
            llm_approval_rate = llm_data.get("mean")
        else:
            llm_approval_rate = llm_data

    return {
        "accuracy": accuracy,
        "latency_mean_ms": latency_mean_ms,
        "latency_total_ms": latency_total_ms,
        "total_tokens": total_tokens,
        "total_tokens_input": total_tokens_input,
        "total_tokens_output": total_tokens_output,
        "n_examples": n_examples,
        "exact_match_rate": exact_match_rate,
        "llm_approval_rate": llm_approval_rate,
    }


def _extract_seed_metrics(seed_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from a per-seed entry."""
    metrics = seed_entry.get("metrics", {})

    return {
        "seed": seed_entry.get("seed"),
        "source": seed_entry.get("source"),
        "accuracy": metrics.get("accuracy"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "llm_approval_rate": metrics.get("llm_approval_rate"),
        "latency_mean_ms": metrics.get("latency_mean_ms"),
        "latency_total_ms": metrics.get("latency_total_ms"),
        "total_tokens": metrics.get("total_tokens"),
        "total_tokens_input": metrics.get("total_tokens_input"),
        "total_tokens_output": metrics.get("total_tokens_output"),
        "n_examples": metrics.get("n_examples"),
    }


def _calculate_derived_metrics(
    extracted: Dict[str, Any],
    gpu_hour_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate derived efficiency metrics."""
    accuracy = extracted.get("accuracy")
    latency_mean_ms = extracted.get("latency_mean_ms")
    latency_total_ms = extracted.get("latency_total_ms")
    total_tokens = extracted.get("total_tokens")
    n_examples = extracted.get("n_examples")

    # Calculate duration in seconds
    duration_seconds = None
    if latency_total_ms is not None:
        duration_seconds = float(latency_total_ms) / 1000.0

    # Throughput: samples per second
    throughput = None
    if duration_seconds and n_examples and duration_seconds > 0:
        throughput = round(float(n_examples) / float(duration_seconds), 6)

    # Tokens per hour
    tokens_per_hour = None
    if duration_seconds and total_tokens and duration_seconds > 0:
        tokens_per_hour = round(
            float(total_tokens) / (float(duration_seconds) / 3600.0), 6
        )

    # Cost estimation (if gpu_hour_usd provided)
    avg_cost_usd = None
    cost_total_usd = None
    cost_per_1k_tokens_usd = None

    if gpu_hour_usd and duration_seconds:
        hours_used = float(duration_seconds) / 3600.0
        if hours_used > 0:
            cost_total_usd = round(float(gpu_hour_usd) * hours_used, 6)
            if n_examples:
                avg_cost_usd = round(cost_total_usd / float(n_examples), 6)
            if total_tokens and total_tokens > 0:
                cost_per_1k_tokens_usd = round(
                    (cost_total_usd / float(total_tokens)) * 1000.0, 6
                )

    # Accuracy per dollar
    accuracy_per_dollar = None
    if accuracy is not None and avg_cost_usd and avg_cost_usd > 0:
        accuracy_per_dollar = round(float(accuracy) / float(avg_cost_usd), 6)

    # Accuracy per second
    accuracy_per_second = None
    if accuracy is not None and latency_mean_ms and latency_mean_ms > 0:
        accuracy_per_second = round(
            float(accuracy) / (float(latency_mean_ms) / 1000.0), 6
        )

    return {
        "duration_seconds": duration_seconds,
        "throughput_samples_per_sec": throughput,
        "tokens_per_hour": tokens_per_hour,
        "avg_cost_usd": avg_cost_usd,
        "cost_total_usd": cost_total_usd,
        "cost_per_1k_tokens_usd": cost_per_1k_tokens_usd,
        "accuracy_per_dollar": accuracy_per_dollar,
        "accuracy_per_second": accuracy_per_second,
    }


def build_efficiency_report_from_aggregate(
    aggregate_path: Path,
    weights: Optional[EfficiencyWeights] = None,
    top_k: int = 5,
    include_per_seed: bool = False,
    gpu_hour_usd: Optional[float] = None,
    agreement_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build efficiency report from seed aggregate metrics JSON.

    Args:
        aggregate_path: Path to seed_aggregate_metrics_from_judged.json
        weights: Efficiency weights for scoring
        top_k: Number of top entries to include in rankings
        include_per_seed: Whether to include per-seed efficiency scores
        gpu_hour_usd: GPU cost per hour for cost calculations
        agreement_report_path: Optional path to agreement_report.json for confidence scores
    """
    weights = weights or EfficiencyWeights()

    data = load_json(aggregate_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {aggregate_path}, got {type(data)}")

    # Load agreement scores if provided
    agreement_scores: Dict[str, Dict[str, Any]] = {}
    if agreement_report_path and agreement_report_path.exists():
        try:
            agreement_data = load_json(agreement_report_path)
            per_source = agreement_data.get("per_source_file", {})
            for source_file, metrics in per_source.items():
                if isinstance(metrics, dict):
                    # Use mean_kappa as primary confidence metric (pairwise agreement)
                    confidence = metrics.get("mean_kappa")
                    agreement_scores[source_file] = {
                        "confidence_score": confidence,
                        "mean_kappa": metrics.get("mean_kappa"),
                        "min_kappa": metrics.get("min_kappa"),
                        "max_kappa": metrics.get("max_kappa"),
                    }
        except Exception as e:
            print(
                f"Warning: Could not load agreement report from {agreement_report_path}: {e}"
            )

    entries: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        model_short = item.get("model_short")
        condition = item.get("condition")
        finetuned = item.get("finetuned", False)
        few_shot = item.get("few_shot", False)
        num_seeds = item.get("num_seeds", 0)

        # Extract aggregate metrics
        extracted = _extract_metrics_from_entry(item)
        derived = _calculate_derived_metrics(extracted, gpu_hour_usd)

        entry = {
            "model_short": model_short,
            "condition": condition,
            "finetuned": finetuned,
            "few_shot": few_shot,
            "num_seeds": num_seeds,
            "accuracy": extracted.get("accuracy"),
            "accuracy_std": None,
            "exact_match_rate": extracted.get("exact_match_rate"),
            "llm_approval_rate": extracted.get("llm_approval_rate"),
            "latency_mean_ms": extracted.get("latency_mean_ms"),
            "latency_total_ms": extracted.get("latency_total_ms"),
            "total_tokens": extracted.get("total_tokens"),
            "total_tokens_input": extracted.get("total_tokens_input"),
            "total_tokens_output": extracted.get("total_tokens_output"),
            "n_examples": extracted.get("n_examples"),
            **derived,
        }

        # Extract accuracy std if available
        metrics = item.get("metrics", {})
        if "accuracy" in metrics and isinstance(metrics["accuracy"], dict):
            entry["accuracy_std"] = metrics["accuracy"].get("std")

        # Extract confidence score from judge agreement data if available
        confidence_score = None
        judge_agreement = item.get("judge_agreement", {})
        if isinstance(judge_agreement, dict):
            conf = judge_agreement.get("confidence_score")
            # Extract mean if it's aggregated (has mean/std)
            if isinstance(conf, dict):
                confidence_score = conf.get("mean")
            else:
                confidence_score = conf
        entry["confidence_score"] = confidence_score

        # Process per-seed data if requested
        if include_per_seed:
            per_seed = item.get("per_seed", [])
            seed_efficiencies = []

            for seed_data in per_seed:
                if not isinstance(seed_data, dict):
                    continue

                seed_extracted = _extract_seed_metrics(seed_data)
                seed_derived = _calculate_derived_metrics(seed_extracted, gpu_hour_usd)

                seed_entry = {
                    **seed_extracted,
                    **seed_derived,
                }
                seed_efficiencies.append(seed_entry)

            entry["per_seed_efficiency"] = seed_efficiencies

        entries.append(entry)

    # Calculate normalized scores and efficiency
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

    # Calculate per-seed efficiency scores if included
    if include_per_seed:
        all_seed_entries = []
        for entry in entries:
            per_seed = entry.get("per_seed_efficiency", [])
            for seed_data in per_seed:
                seed_data["model_short"] = entry["model_short"]
                seed_data["condition"] = entry["condition"]
                all_seed_entries.append(seed_data)

        if all_seed_entries:
            seed_accuracy_list = [e.get("accuracy") for e in all_seed_entries]
            seed_cost_list = [e.get("avg_cost_usd") for e in all_seed_entries]
            seed_latency_list = [e.get("latency_mean_ms") for e in all_seed_entries]

            seed_accuracy_norm = _normalize(seed_accuracy_list, higher_is_better=True)
            seed_cost_norm = _normalize(seed_cost_list, higher_is_better=False)
            seed_latency_norm = _normalize(seed_latency_list, higher_is_better=False)

            for seed_entry, acc_norm, cost_norm_val, lat_norm in zip(
                all_seed_entries, seed_accuracy_norm, seed_cost_norm, seed_latency_norm
            ):
                seed_entry["accuracy_normalized"] = acc_norm
                seed_entry["cost_normalized"] = cost_norm_val
                seed_entry["latency_normalized"] = lat_norm
                seed_entry["efficiency_score"] = _weighted_score(
                    acc_norm, cost_norm_val, lat_norm, weights
                )

    # Build report
    report = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(aggregate_path),
        "weights": weights.to_dict(),
        "include_per_seed": include_per_seed,
        "totals": {
            "model_conditions": len(entries),
            "unique_models": len(
                set(e.get("model_short") for e in entries if e.get("model_short"))
            ),
            "unique_conditions": len(
                set(e.get("condition") for e in entries if e.get("condition"))
            ),
            "with_accuracy": sum(1 for e in entries if e.get("accuracy") is not None),
            "with_cost": sum(1 for e in entries if e.get("avg_cost_usd") is not None),
            "with_latency": sum(
                1 for e in entries if e.get("latency_mean_ms") is not None
            ),
        },
        "models": entries,
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
            "missing_accuracy": [
                f"{e.get('model_short')}_{e.get('condition')}"
                for e in entries
                if e.get("accuracy") is None
            ],
            "missing_latency": [
                f"{e.get('model_short')}_{e.get('condition')}"
                for e in entries
                if e.get("latency_mean_ms") is None
            ],
        },
    }

    return report


def _group_by_condition(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group entries by condition and calculate aggregate stats."""
    conditions: Dict[str, List[Dict[str, Any]]] = {}

    for entry in entries:
        condition = entry.get("condition", "unknown")
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(entry)

    result = {}
    for condition, items in conditions.items():
        accuracies = [e.get("accuracy") for e in items if e.get("accuracy") is not None]
        latencies = [
            e.get("latency_mean_ms")
            for e in items
            if e.get("latency_mean_ms") is not None
        ]
        efficiencies = [
            e.get("efficiency_score")
            for e in items
            if e.get("efficiency_score") is not None
        ]

        result[condition] = {
            "count": len(items),
            "models": [e.get("model_short") for e in items],
            "accuracy_mean": _safe_mean(accuracies),
            "latency_mean_ms": _safe_mean(latencies),
            "efficiency_score_mean": _safe_mean(efficiencies),
        }

    return result


def _group_by_model(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group entries by model and calculate aggregate stats."""
    models: Dict[str, List[Dict[str, Any]]] = {}

    for entry in entries:
        model = entry.get("model_short", "unknown")
        if model not in models:
            models[model] = []
        models[model].append(entry)

    result = {}
    for model, items in models.items():
        accuracies = [e.get("accuracy") for e in items if e.get("accuracy") is not None]
        latencies = [
            e.get("latency_mean_ms")
            for e in items
            if e.get("latency_mean_ms") is not None
        ]
        efficiencies = [
            e.get("efficiency_score")
            for e in items
            if e.get("efficiency_score") is not None
        ]

        result[model] = {
            "count": len(items),
            "conditions": [e.get("condition") for e in items],
            "accuracy_mean": _safe_mean(accuracies),
            "accuracy_best": max(accuracies) if accuracies else None,
            "latency_mean_ms": _safe_mean(latencies),
            "latency_best_ms": min(latencies) if latencies else None,
            "efficiency_score_mean": _safe_mean(efficiencies),
            "efficiency_score_best": max(efficiencies) if efficiencies else None,
        }

    return result


def build_efficiency_notebook(report_path: Path, output_path: Path) -> None:
    """Generate a Jupyter notebook for exploring the efficiency report."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Model Efficiency Report\n",
                    f"Report file: `{report_path.name}`\n",
                    "\n",
                    "This notebook explores model efficiency metrics including accuracy, latency, and derived efficiency scores.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "\n",
                    f'report_path = Path("{report_path}")\n',
                    "if not report_path.exists():\n",
                    f'    report_path = Path("{report_path.name}")\n',
                    "\n",
                    "report = json.loads(report_path.read_text())\n",
                    "df = pd.DataFrame(report['models'])\n",
                    'print(f"Loaded {len(df)} model-condition entries")\n',
                    "print(f\"Created: {report['created_at']}\")\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Top Models by Efficiency Score\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cols = ['model_short', 'condition', 'accuracy', 'latency_mean_ms', 'efficiency_score', 'confidence_score']\n",
                    "display(df[[c for c in cols if c in df.columns]].sort_values('efficiency_score', ascending=False).head(10))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Top Most Accurate Models (with Judge Confidence Scores)\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Show top accurate models with confidence scores\n",
                    "acc_cols = ['model_short', 'condition', 'accuracy', 'confidence_score', 'num_seeds', 'latency_mean_ms']\n",
                    "top_accurate = df[[c for c in acc_cols if c in df.columns]].sort_values('accuracy', ascending=False).head(10)\n",
                    "print('\\n=== Top 10 Most Accurate Models (with Judge Agreement Confidence) ===')\n",
                    "display(top_accurate)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Accuracy by Model and Condition\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, 5))\n",
                    "df_sorted = df.sort_values('accuracy', ascending=False)\n",
                    "labels = df_sorted['model_short'] + ' (' + df_sorted['condition'].fillna('') + ')'\n",
                    "colors = ['green' if x else 'blue' for x in df_sorted['finetuned']]\n",
                    "plt.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors)\n",
                    "plt.yticks(range(len(df_sorted)), labels, fontsize=8)\n",
                    "plt.xlabel('Accuracy')\n",
                    "plt.title('Accuracy by Model-Condition (Green=Finetuned, Blue=Baseline)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Latency Distribution\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, 5))\n",
                    "df_sorted = df.sort_values('latency_mean_ms', ascending=True)\n",
                    "labels = df_sorted['model_short'] + ' (' + df_sorted['condition'].fillna('') + ')'\n",
                    "plt.barh(range(len(df_sorted)), df_sorted['latency_mean_ms'])\n",
                    "plt.yticks(range(len(df_sorted)), labels, fontsize=8)\n",
                    "plt.xlabel('Mean Latency (ms)')\n",
                    "plt.title('Latency by Model-Condition (Lower is Better)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Efficiency Score Comparison\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, 5))\n",
                    "df_sorted = df.sort_values('efficiency_score', ascending=False)\n",
                    "labels = df_sorted['model_short'] + ' (' + df_sorted['condition'].fillna('') + ')'\n",
                    "plt.barh(range(len(df_sorted)), df_sorted['efficiency_score'])\n",
                    "plt.yticks(range(len(df_sorted)), labels, fontsize=8)\n",
                    "plt.xlabel('Efficiency Score')\n",
                    "plt.title('Composite Efficiency Score (Higher is Better)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Accuracy vs Latency Scatter\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6))\n",
                    "colors = ['green' if x else 'blue' for x in df['finetuned']]\n",
                    "plt.scatter(df['latency_mean_ms'], df['accuracy'], c=colors, alpha=0.7, s=100)\n",
                    "for idx, row in df.iterrows():\n",
                    "    plt.annotate(row['model_short'], (row['latency_mean_ms'], row['accuracy']), fontsize=7)\n",
                    "plt.xlabel('Mean Latency (ms)')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.title('Accuracy vs Latency (Green=Finetuned, Blue=Baseline)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Summary by Condition\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "by_condition = pd.DataFrame(report['by_condition']).T\n",
                    "display(by_condition)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Summary by Model\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "by_model = pd.DataFrame(report['by_model']).T\n",
                    "display(by_model.sort_values('accuracy_best', ascending=False))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Rankings Summary\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for ranking_name, ranking_data in report['rankings'].items():\n",
                    '    print(f"\\n=== {ranking_name.upper()} ===")\n',
                    "    for item in ranking_data:\n",
                    "        print(f\"  {item['rank']}. {item['model']} ({item.get('condition', 'N/A')})\")\n",
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


# Keep legacy functions for backward compatibility
def resolve_prediction_files(predictions_dir: Path, datasets: List[str]) -> List[Path]:
    """Resolve prediction files from directory (legacy support)."""
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
    """Load judge summary file (legacy support)."""
    if not path:
        return None
    if not Path(path).exists():
        return None
    data = load_json(path)
    return data if isinstance(data, dict) else None
