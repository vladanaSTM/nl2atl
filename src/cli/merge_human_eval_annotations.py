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
            "outputs/LLM-evaluation/human_evaluation/aaai_human_eval_sample_key.json"
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
        default="aaai_human_eval_merged",
        help="Base filename for merged JSON/JSONL/CSV outputs.",
    )
    args = parser.parse_args()

    result = merge_human_annotations(
        key_path=args.key,
        annotation_paths=args.annotations,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    print(f"Merged annotations into {args.output_dir}")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    return result


if __name__ == "__main__":
    main()
