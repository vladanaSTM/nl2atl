#!/usr/bin/env python
"""
Compute inter-rater agreement between LLM judges.

Usage:
    python run_judge_agreement.py
    python run_judge_agreement.py --eval_dir outputs/LLM-evaluation/evaluated_datasets
    python run_judge_agreement.py --judges llama-70b gpt-4o
"""

import argparse
from pathlib import Path

from src.judge_agreement import (
    generate_agreement_report,
    print_agreement_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cohen's Kappa and Fleiss' Kappa between LLM judges"
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("outputs/LLM-evaluation/evaluated_datasets"),
        help="Directory containing judge subdirectories with evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for agreement report JSON (default: eval_dir/../agreement_report.json)",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Specific judges to compare (default: all found in eval_dir)",
    )
    parser.add_argument(
        "--include_disagreements",
        action="store_true",
        default=True,
        help="Include sample disagreements in report",
    )
    parser.add_argument(
        "--max_disagreements",
        type=int,
        default=50,
        help="Maximum number of disagreement examples to include",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.eval_dir.parent / "agreement_report.json"

    if not args.eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {args.eval_dir}")

    report = generate_agreement_report(
        eval_dir=args.eval_dir,
        output_path=args.output,
        include_disagreements=args.include_disagreements,
        max_disagreements=args.max_disagreements,
    )

    print_agreement_summary(report)

    # Return summary for programmatic use
    return report


if __name__ == "__main__":
    main()