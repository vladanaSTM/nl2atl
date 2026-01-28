#!/usr/bin/env python
"""Generate a model efficiency report from seed aggregate metrics."""

import argparse
from pathlib import Path
from typing import Optional

from ..infra.io import save_json
from ..evaluation.model_efficiency import (
    EfficiencyWeights,
    build_efficiency_notebook,
    build_efficiency_report_from_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model efficiency report from seed aggregate metrics"
    )
    parser.add_argument(
        "--aggregate_file",
        default="outputs/seed_aggregate_metrics_from_judged.json",
        help="Path to seed_aggregate_metrics_from_judged.json",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for output files (default: outputs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report JSON path (default: <output_dir>/efficiency_report.json)",
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="Output notebook path (default: <output_dir>/efficiency_report.ipynb)",
    )
    parser.add_argument(
        "--no_notebook",
        action="store_true",
        help="Disable notebook output",
    )
    parser.add_argument(
        "--include_per_seed",
        action="store_true",
        help="Include efficiency calculations per seed",
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

    aggregate_path = Path(args.aggregate_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregate_path.exists():
        raise FileNotFoundError(f"Aggregate file not found: {aggregate_path}")

    report = build_efficiency_report_from_aggregate(
        aggregate_path,
        weights=EfficiencyWeights(
            accuracy=args.weight_accuracy,
            cost=args.weight_cost,
            latency=args.weight_latency,
        ),
        top_k=args.top_k,
        include_per_seed=args.include_per_seed,
    )

    report_path = (
        Path(args.output) if args.output else output_dir / "efficiency_report.json"
    )
    save_json(report, report_path)

    if not args.no_notebook:
        notebook_path = (
            Path(args.notebook)
            if args.notebook
            else output_dir / "efficiency_report.ipynb"
        )
        build_efficiency_notebook(report_path, notebook_path)

    print(f"Efficiency report: {report_path}")
    if not args.no_notebook:
        print(f"Efficiency notebook: {notebook_path}")


if __name__ == "__main__":
    main()
