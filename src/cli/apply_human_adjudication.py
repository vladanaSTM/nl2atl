#!/usr/bin/env python
"""Apply a completed human-adjudication workbook back into the merged outputs.

Workflow:
  1. ``human-eval-merge`` auto-writes ``<stem>_adjudication.xlsx`` listing the
     items the annotators disagreed on (with each annotator's verdict + note).
  2. The annotators deliberate and fill the blank ``correct`` (yes/no) and
     ``notes`` columns in that single shared workbook.
  3. This command re-runs the merge with that filled workbook applied, updating
     the merged CSV/JSON/JSONL and the adjudicated human-gold file.

The original annotator files are recovered from the previous merge summary, so
typically only the filled adjudication workbook needs to be supplied.
"""

import argparse
from pathlib import Path

from ..evaluation.human_eval_merge import merge_human_annotations
from ..infra.io import load_json


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the human-eval merge with a completed adjudication workbook "
            "applied as the final human gold label."
        )
    )
    parser.add_argument(
        "adjudication",
        type=Path,
        help="Filled adjudication workbook (XLSX/CSV/JSON) with correct + notes.",
    )
    parser.add_argument(
        "--merged_dir",
        type=Path,
        default=Path("outputs/LLM-evaluation/human_evaluation/merged"),
        help="Directory containing the previous merged outputs.",
    )
    parser.add_argument(
        "--output_stem",
        default="human_eval_merged",
        help="Base filename for the merged outputs.",
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
        "--annotations",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Original annotator files. When omitted they are recovered from the "
            "previous merge summary in --merged_dir."
        ),
    )
    args = parser.parse_args()

    if not args.adjudication.exists():
        parser.error(f"Adjudication workbook not found: {args.adjudication}")

    if args.annotations:
        annotation_paths = list(args.annotations)
    else:
        summary_path = args.merged_dir / f"{args.output_stem}.json"
        if not summary_path.exists():
            parser.error(
                f"No previous merge summary at {summary_path}; "
                "pass the original files with --annotations."
            )
        merged = load_json(summary_path)
        recorded = merged.get("summary", {}).get("annotation_files", [])
        annotation_paths = [Path(path) for path in recorded]
        if not annotation_paths:
            parser.error(
                "Previous merge summary lists no annotation_files; "
                "pass the original files with --annotations."
            )

    # Append the adjudication workbook, de-duplicating by resolved path.
    resolved = {path.resolve() for path in annotation_paths}
    if args.adjudication.resolve() not in resolved:
        annotation_paths.append(args.adjudication)

    missing = [path for path in annotation_paths if not path.exists()]
    if missing:
        parser.error(
            "Missing annotation files: " + ", ".join(str(path) for path in missing)
        )

    result = merge_human_annotations(
        key_path=args.key,
        annotation_paths=annotation_paths,
        output_dir=args.merged_dir,
        output_stem=args.output_stem,
        refresh_adjudication=True,
    )
    summary = result["summary"]
    print(f"Applied adjudication from {args.adjudication}")
    print(f"n_adjudicated_human_labels: {result['n_adjudicated_human_labels']}")
    print(f"n_items_with_final_label: {summary['n_items_with_final_label']}")
    remaining = summary["n_items_needing_adjudication"]
    print(f"n_items_needing_adjudication: {remaining}")
    if remaining:
        print(
            "Some items are still unresolved (blank 'correct'); the adjudication "
            "workbook was refreshed to show the remaining items."
        )
    return result


if __name__ == "__main__":
    main()
