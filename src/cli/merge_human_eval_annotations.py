#!/usr/bin/env python
"""Merge completed blind human annotations with the private sample key."""

import argparse
from pathlib import Path

from ..evaluation.human_eval_merge import merge_human_annotations


def main():
    parser = argparse.ArgumentParser(
        description="De-anonymize completed human-evaluation annotations by audit_id."
    )
    parser.add_argument(
        "annotations",
        nargs="+",
        type=Path,
        help="Completed annotator XLSX/CSV/JSON files produced from the blind sample.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=Path(
            "outputs/LLM-evaluation/human_evaluation/human_eval_sample_key.json"
        ),
        help="Private key file mapping audit_id to hidden metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/LLM-evaluation/human_evaluation/merged"),
        help="Output directory for merged analysis files.",
    )
    parser.add_argument(
        "--output_stem",
        default="human_eval_merged",
        help="Base filename for merged JSON/JSONL/CSV outputs.",
    )
    parser.add_argument(
        "--refresh_adjudication",
        action="store_true",
        help=(
            "Overwrite an existing adjudication workbook even if annotators may "
            "have started filling it. By default the workbook is created once "
            "and left untouched on later runs."
        ),
    )
    args = parser.parse_args()

    result = merge_human_annotations(
        key_path=args.key,
        annotation_paths=args.annotations,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        refresh_adjudication=args.refresh_adjudication,
    )
    print(f"Merged annotations into {args.output_dir}")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    adjudication_workbook = result["files"].get("adjudication_workbook")
    if adjudication_workbook:
        print(
            f"adjudication_workbook: {adjudication_workbook} "
            f"({result['summary']['n_items_needing_adjudication']} items to deliberate)"
        )
    return result


if __name__ == "__main__":
    main()
