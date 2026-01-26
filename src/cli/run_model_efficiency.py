#!/usr/bin/env python
"""Generate a model efficiency report from prediction outputs."""

import argparse
from pathlib import Path
from typing import Optional

from ..infra.io import save_json
from ..evaluation.model_efficiency import (
    EfficiencyWeights,
    build_efficiency_notebook,
    build_efficiency_report,
    load_judge_summary,
    resolve_prediction_files,
)


def _resolve_judge_summary(
    summary_path: Optional[str],
    llm_eval_dir: Path,
    judge_model: Optional[str],
) -> Optional[Path]:
    if summary_path:
        candidate = Path(summary_path)
        return candidate if candidate.exists() else None

    if judge_model:
        candidate = llm_eval_dir / f"summary__judge-{judge_model}.json"
        return candidate if candidate.exists() else None

    summaries = sorted(llm_eval_dir.glob("summary__judge-*.json"))
    if not summaries:
        return None

    summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return summaries[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Prediction files to analyze (default: all)",
    )
    parser.add_argument(
        "--predictions_dir",
        default="outputs/model_predictions",
        help="Directory with prediction JSON files",
    )
    parser.add_argument(
        "--llm_eval_dir",
        default="outputs/LLM-evaluation",
        help="Directory with LLM judge outputs",
    )
    parser.add_argument(
        "--judge_summary",
        default=None,
        help="Path to summary__judge-<judge>.json (optional)",
    )
    parser.add_argument(
        "--judge_model",
        default=None,
        help="Judge model name used to resolve summary file (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report JSON path (default: <llm_eval_dir>/efficiency_report.json)",
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="Output notebook path (default: <llm_eval_dir>/efficiency_report.ipynb)",
    )
    parser.add_argument(
        "--no_notebook",
        action="store_true",
        help="Disable notebook output",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top entries to include in rankings (default: 5)",
    )
    parser.add_argument(
        "--weight_accuracy",
        type=float,
        default=0.5,
        help="Weight for accuracy in efficiency score (default: 0.5)",
    )
    parser.add_argument(
        "--weight_cost",
        type=float,
        default=0.25,
        help="Weight for cost in efficiency score (default: 0.25)",
    )
    parser.add_argument(
        "--weight_latency",
        type=float,
        default=0.25,
        help="Weight for latency in efficiency score (default: 0.25)",
    )

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    llm_eval_dir = Path(args.llm_eval_dir)

    prediction_files = resolve_prediction_files(predictions_dir, args.datasets)
    if not prediction_files:
        raise ValueError("No prediction files found to analyze.")

    summary_path = _resolve_judge_summary(
        args.judge_summary, llm_eval_dir, args.judge_model
    )
    summary = load_judge_summary(summary_path)

    report = build_efficiency_report(
        prediction_files,
        judge_summary=summary,
        weights=EfficiencyWeights(
            accuracy=args.weight_accuracy,
            cost=args.weight_cost,
            latency=args.weight_latency,
        ),
        top_k=args.top_k,
    )

    report_path = (
        Path(args.output) if args.output else llm_eval_dir / "efficiency_report.json"
    )
    save_json(report, report_path)

    if not args.no_notebook:
        notebook_path = (
            Path(args.notebook)
            if args.notebook
            else llm_eval_dir / "efficiency_report.ipynb"
        )
        build_efficiency_notebook(report_path, notebook_path)

    print(f"Efficiency report: {report_path}")
    if not args.no_notebook:
        print(f"Efficiency notebook: {notebook_path}")


if __name__ == "__main__":
    main()
