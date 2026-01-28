#!/usr/bin/env python3
"""Aggregate seed metrics from saved evaluation outputs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    # Search recursively so files inside dataset subdirectories are included
    return sorted(p for p in input_dir.rglob("*.json") if p.is_file())


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


def _load_agreement_scores(
    agreement_report_path: Optional[Path],
) -> Dict[str, Dict[str, float]]:
    """Load judge agreement scores from agreement report.

    Returns dict mapping source_file to agreement metrics dict.
    """
    if not agreement_report_path or not agreement_report_path.exists():
        return {}

    try:
        with open(agreement_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        per_source = report.get("per_source_file", {})

        # Extract agreement scores per source file
        agreement_scores = {}
        for source_file, metrics in per_source.items():
            if isinstance(metrics, dict):
                # Use mean_kappa as confidence metric (pairwise agreement between judges)
                # This represents how much judges agree on this specific source file
                confidence = metrics.get("mean_kappa")

                agreement_scores[source_file] = {
                    "confidence_score": confidence,
                    "mean_kappa": metrics.get("mean_kappa"),
                    "min_kappa": metrics.get("min_kappa"),
                    "max_kappa": metrics.get("max_kappa"),
                    "n_items": metrics.get("n_items"),
                }

        return agreement_scores
    except Exception as e:
        print(
            f"Warning: Could not load agreement scores from {agreement_report_path}: {e}"
        )
        return {}


def aggregate_predictions(
    input_dir: Path,
    agreement_report_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Aggregate predictions with optional agreement scores.

    Args:
        input_dir: Directory with prediction files
        agreement_report_path: Optional path to agreement_report.json
    """
    grouped: Dict[Tuple[str, str, bool, bool], List[Dict[str, Any]]] = defaultdict(list)

    # Load agreement scores if provided
    agreement_scores = _load_agreement_scores(agreement_report_path)

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

        # Add agreement scores if available for this source file
        # The agreement report keys use base filename (without judge suffix)
        # e.g., "ds-r1-qwen-32b_baseline_few_shot_seed42.json"
        # matches prediction file "ds-r1-qwen-32b_baseline_few_shot_seed42__judge-ds-v3.2.json"
        source_file = path.name
        base_source = (
            source_file.split("__judge-")[0] + ".json"
            if "__judge-" in source_file
            else source_file
        )

        if base_source in agreement_scores:
            metrics["judge_agreement"] = agreement_scores[base_source]

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

        # Aggregate judge agreement scores if present
        agg_agreement_temp: Dict[str, List[float]] = {}
        for item in items:
            agreement = item["metrics"].get("judge_agreement")
            if agreement and isinstance(agreement, dict):
                for key, val in agreement.items():
                    if val is not None:
                        if key not in agg_agreement_temp:
                            agg_agreement_temp[key] = []
                        agg_agreement_temp[key].append(val)

        # Compute mean/std for agreement metrics
        agg_agreement: Dict[str, Dict[str, float]] = {}
        for key in agg_agreement_temp:
            values = agg_agreement_temp[key]
            if values:
                agg_agreement[key] = {
                    "mean": _safe_mean(values),
                    "std": _safe_std(values),
                }

        aggregate_entry = {
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

        # Add aggregated agreement scores if available
        if agg_agreement:
            aggregate_entry["judge_agreement"] = agg_agreement

        aggregates.append(aggregate_entry)

    return aggregates


def _build_notebook(
    aggregates: List[Dict[str, Any]], agg_json_path: str
) -> Dict[str, Any]:
    # Build a simple notebook that loads the aggregated JSON and shows
    # comparison tables and a basic accuracy bar chart.
    nb_cells = []

    nb_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {"language": "markdown"},
            "source": [
                "# Aggregated Seed Metrics\n",
                f"Generated from `{agg_json_path}`.\n",
                "This notebook loads the aggregated JSON and shows comparison tables and plots.\n",
            ],
        }
    )

    nb_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {"language": "markdown"},
            "source": [
                "## Requirements\n",
                "Install the plotting and data libraries if needed:\n",
                "```\n",
                "pip install pandas matplotlib seaborn\n",
                "```\n",
            ],
        }
    )

    code_source = [
        "import json\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns  # seaborn should be installed in the notebook kernel\n",
        "from IPython.display import display\n",
        "# Use POSIX-style path to avoid Windows backslash escape warnings\n",
        # Embed an absolute POSIX path string so the generated notebook
        # doesn't contain Windows backslashes or produce escape warnings.
        f"PATH = '{Path(agg_json_path).resolve().as_posix()}'\n",
        "with open(PATH, 'r', encoding='utf-8') as fh:\n",
        "    aggregates = json.load(fh)\n",
        "rows = []\n",
        "for g in aggregates:\n",
        "    row = {k: g.get(k) for k in ('model_short','condition','finetuned','few_shot','num_seeds')}\n",
        "    for metric,vals in (g.get('metrics') or {}).items():\n",
        "        row[f'{metric}_mean'] = vals.get('mean')\n",
        "        row[f'{metric}_std'] = vals.get('std')\n",
        "    # Add confidence score (mean if aggregated)\n",
        "    judge_agreement = g.get('judge_agreement', {})\n",
        "    conf_score = judge_agreement.get('confidence_score')\n",
        "    if isinstance(conf_score, dict):\n",
        "        row['confidence_score'] = conf_score.get('mean')\n",
        "    else:\n",
        "        row['confidence_score'] = conf_score\n",
        "    rows.append(row)\n",
        "df = pd.DataFrame(rows)\n",
        "display(df.sort_values(by=['model_short','condition']).head(20))\n",
        "# Show top models by accuracy with confidence scores\n",
        "if 'accuracy_mean' in df.columns:\n",
        "    top_models = df.sort_values('accuracy_mean', ascending=False).head(10)\n",
        "    display_cols = ['model_short', 'condition', 'accuracy_mean', 'confidence_score', 'num_seeds']\n",
        "    print('\\n=== Top 10 Most Accurate Models (with Confidence Scores) ===')\n",
        "    display(top_models[[c for c in display_cols if c in top_models.columns]])\n",
        "    plot_df = top_models\n",
        "    plt.figure(figsize=(12,6))\n",
        "    sns.barplot(data=plot_df, x='accuracy_mean', y='model_short', hue='condition', dodge=False)\n",
        "    plt.title('Top 10 Groups by Accuracy (with Judge Agreement Scores)')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
    ]

    # Display table sorted by accuracy_mean if available, otherwise fallback
    # to the previous model/condition sort.
    # We insert a small wrapper code cell that performs the conditional display.
    display_wrapper = [
        "# Display aggregated table, ranking by accuracy_mean when available\n",
        "if 'accuracy_mean' in df.columns:\n",
        "    display(df.sort_values(by='accuracy_mean', ascending=False).head(20))\n",
        "else:\n",
        "    display(df.sort_values(by=['model_short','condition']).head(20))\n",
    ]

    nb_cells.append(
        {
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": code_source,
        }
    )

    nb_cells.append(
        {
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": display_wrapper,
        }
    )

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return nb


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
        "--agreement_report",
        default="outputs/LLM-evaluation/agreement_report.json",
        help="Optional path to agreement_report.json for judge agreement scores.",
    )
    parser.add_argument(
        "--output",
        default="outputs/seed_aggregate_metrics_from_judged.json",
        help="Path to write aggregated metrics JSON.",
    )
    # Notebook enabled by default; provide --no-notebook to disable
    parser.set_defaults(notebook=True)
    parser.add_argument(
        "--notebook",
        dest="notebook",
        action="store_true",
        help="Also write a Jupyter notebook with comparison tables and plots. (default)",
    )
    parser.add_argument(
        "--no-notebook",
        dest="notebook",
        action="store_false",
        help="Do not write a Jupyter notebook.",
    )
    parser.add_argument(
        "--notebook_output",
        default="outputs/seed_aggregate_metrics_from_judged.ipynb",
        help="Path to write the generated Jupyter notebook.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    agreement_report_path = (
        Path(args.agreement_report) if args.agreement_report else None
    )

    aggregates = aggregate_predictions(input_dir, agreement_report_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.notebook:
        nb = _build_notebook(aggregates, str(output_path))
        nb_path = Path(args.notebook_output)
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        print(f"Wrote notebook to {nb_path}")

    print(f"Aggregated {len(aggregates)} groups into {output_path}")


if __name__ == "__main__":
    main()
