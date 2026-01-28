#!/usr/bin/env python
"""Comprehensive evaluation pipeline orchestrator.

Runs all evaluation steps in sequence:
1. Summarize judge evaluations per model
2. Generate inter-rater agreement report
3. Aggregate metrics across seeds
4. Generate model efficiency report
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .summarize_judge_evaluations import summarize_judge_evaluations
from .aggregate_seeds import aggregate_predictions, _build_notebook
from ..evaluation.judge_agreement import (
    generate_agreement_report,
    print_agreement_summary,
    build_agreement_notebook,
)
from ..evaluation.model_efficiency import (
    build_efficiency_report_from_aggregate,
    build_efficiency_notebook,
    EfficiencyWeights,
)
from ..infra.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation pipeline with all reports."
    )
    parser.add_argument(
        "--eval_dir",
        default="outputs/LLM-evaluation",
        help="Main evaluation output directory (default: outputs/LLM-evaluation).",
    )
    parser.add_argument(
        "--predictions_dir",
        default="outputs/model_predictions",
        help="Directory with model predictions for efficiency analysis.",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Specific judge names to process (default: all)",
    )
    parser.add_argument(
        "--skip_judge_summaries",
        action="store_true",
        help="Skip judge summary generation.",
    )
    parser.add_argument(
        "--skip_agreement",
        action="store_true",
        help="Skip inter-rater agreement analysis.",
    )
    parser.add_argument(
        "--skip_seed_aggregation",
        action="store_true",
        help="Skip seed metric aggregation.",
    )
    parser.add_argument(
        "--skip_efficiency",
        action="store_true",
        help="Skip model efficiency analysis.",
    )
    parser.add_argument(
        "--gpu_hour_usd",
        type=float,
        default=None,
        help="GPU cost per hour in USD for efficiency calculations.",
    )
    parser.add_argument(
        "--weight_accuracy",
        type=float,
        default=0.5,
        help="Weight for accuracy in efficiency scoring (default: 0.5).",
    )
    parser.add_argument(
        "--weight_cost",
        type=float,
        default=0.25,
        help="Weight for cost in efficiency scoring (default: 0.25).",
    )
    parser.add_argument(
        "--weight_latency",
        type=float,
        default=0.25,
        help="Weight for latency in efficiency scoring (default: 0.25).",
    )
    parser.add_argument(
        "--no_notebook",
        action="store_true",
        help="Do not generate Jupyter notebooks.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    evaluated_datasets_dir = eval_dir / "evaluated_datasets"
    predictions_dir = Path(args.predictions_dir)

    try:
        # Step 1: Generate judge summaries
        if not args.skip_judge_summaries:
            print("\n" + "=" * 70)
            print("STEP 1: Summarizing judge evaluations")
            print("=" * 70)
            try:
                summarize_judge_evaluations(
                    input_dir=evaluated_datasets_dir,
                    output_dir=eval_dir,
                    judge_filter=args.judges,
                )
                print("✓ Judge summaries generated successfully")
            except Exception as e:
                print(f"✗ Error generating judge summaries: {e}")
                raise

        # Step 2: Generate agreement report
        if not args.skip_agreement:
            print("\n" + "=" * 70)
            print("STEP 2: Computing inter-rater agreement")
            print("=" * 70)
            try:
                if evaluated_datasets_dir.exists():
                    agreement_report = generate_agreement_report(
                        eval_dir=evaluated_datasets_dir,
                        output_path=eval_dir / "agreement_report.json",
                    )
                    print_agreement_summary(agreement_report)
                    if not args.no_notebook:
                        try:
                            build_agreement_notebook(
                                eval_dir / "agreement_report.json",
                                eval_dir / "agreement_report.ipynb",
                            )
                            print("✓ Agreement notebook generated")
                        except Exception as e:
                            print(f"Warning: Could not build agreement notebook: {e}")
                    print("✓ Agreement report generated successfully")
                else:
                    print(
                        f"Warning: {evaluated_datasets_dir} not found, skipping agreement analysis"
                    )
            except ValueError as e:
                print(f"Warning: Skipping agreement analysis: {e}")
            except Exception as e:
                print(f"✗ Error generating agreement report: {e}")
                raise

        # Step 3: Aggregate seed metrics
        if not args.skip_seed_aggregation:
            print("\n" + "=" * 70)
            print("STEP 3: Aggregating metrics across seeds")
            print("=" * 70)
            try:
                agreement_report_path = eval_dir / "agreement_report.json"
                aggregates = aggregate_predictions(
                    input_dir=evaluated_datasets_dir,
                    agreement_report_path=(
                        agreement_report_path
                        if agreement_report_path.exists()
                        else None
                    ),
                )

                agg_output = eval_dir / "seed_aggregate_metrics_from_judged.json"
                agg_output.write_text(
                    __import__("json").dumps(aggregates, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"✓ Aggregated {len(aggregates)} groups into {agg_output}")

                if not args.no_notebook:
                    try:
                        nb = _build_notebook(aggregates, str(agg_output))
                        nb_path = eval_dir / "seed_aggregate_metrics_from_judged.ipynb"
                        nb_path.write_text(
                            __import__("json").dumps(nb, indent=2),
                            encoding="utf-8",
                        )
                        print(f"✓ Aggregation notebook generated")
                    except Exception as e:
                        print(f"Warning: Could not build aggregation notebook: {e}")
            except Exception as e:
                print(f"✗ Error aggregating seeds: {e}")
                raise

        # Step 4: Generate efficiency report
        if not args.skip_efficiency:
            print("\n" + "=" * 70)
            print("STEP 4: Generating model efficiency report")
            print("=" * 70)
            try:
                agg_path = eval_dir / "seed_aggregate_metrics_from_judged.json"
                if not agg_path.exists():
                    print(
                        f"Warning: Aggregation file not found at {agg_path}, skipping efficiency report"
                    )
                else:
                    weights = EfficiencyWeights(
                        accuracy=args.weight_accuracy,
                        cost=args.weight_cost,
                        latency=args.weight_latency,
                    )

                    agreement_report_path = eval_dir / "agreement_report.json"
                    efficiency_report = build_efficiency_report_from_aggregate(
                        aggregate_path=agg_path,
                        weights=weights,
                        gpu_hour_usd=args.gpu_hour_usd,
                        agreement_report_path=(
                            agreement_report_path
                            if agreement_report_path.exists()
                            else None
                        ),
                    )

                    eff_output = eval_dir / "efficiency_report.json"
                    save_json(efficiency_report, eff_output)
                    print(f"✓ Efficiency report saved to {eff_output}")

                    if not args.no_notebook:
                        try:
                            build_efficiency_notebook(
                                eff_output, eval_dir / "efficiency_report.ipynb"
                            )
                            print(f"✓ Efficiency notebook generated")
                        except Exception as e:
                            print(f"Warning: Could not build efficiency notebook: {e}")
            except Exception as e:
                print(f"✗ Error generating efficiency report: {e}")
                raise

        print("\n" + "=" * 70)
        print("✓ Evaluation pipeline completed successfully!")
        print("=" * 70)
        print("\nGenerated files:")
        print(f"  • Judge summaries: {eval_dir}/summary__judge-*.json")
        print(f"  • Agreement report: {eval_dir}/agreement_report.json")
        print(
            f"  • Seed aggregation: {eval_dir}/seed_aggregate_metrics_from_judged.json"
        )
        print(f"  • Efficiency report: {eval_dir}/efficiency_report.json")
        if not args.no_notebook:
            print("\nGenerated notebooks:")
            print(f"  • Judge summaries: {eval_dir}/summary__judge-*.ipynb")
            print(f"  • Agreement report: {eval_dir}/agreement_report.ipynb")
            print(
                f"  • Seed aggregation: {eval_dir}/seed_aggregate_metrics_from_judged.ipynb"
            )
            print(f"  • Efficiency report: {eval_dir}/efficiency_report.ipynb")

    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
