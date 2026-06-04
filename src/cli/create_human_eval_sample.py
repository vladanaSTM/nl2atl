#!/usr/bin/env python
"""Create a stratified human-evaluation sample from judged outputs."""

import argparse
from pathlib import Path

from ..evaluation.human_eval_sample import (
    DEFAULT_ANNOTATORS,
    DEFAULT_AAAI_QUOTAS,
    DEFAULT_JUDGES,
    build_human_eval_sample,
    regenerate_annotator_workbooks_from_key,
)


def _parse_quota_override(values: list[str]) -> dict[str, int]:
    quotas = dict(DEFAULT_AAAI_QUOTAS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Quota override must be STRATUM=COUNT, got: {value}")
        stratum, raw_count = value.split("=", 1)
        quotas[stratum.strip()] = int(raw_count)
    return quotas


def main():
    parser = argparse.ArgumentParser(
        description="Create blind and keyed human-evaluation audit files."
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("outputs/LLM-evaluation/evaluated_datasets"),
        help="Directory containing judge subdirectories with evaluated JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/LLM-evaluation/human_evaluation"),
        help="Output directory for human-evaluation files.",
    )
    parser.add_argument(
        "--judges",
        nargs=2,
        default=list(DEFAULT_JUDGES),
        help="Two judge directory names to align.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=20260604,
        help="Deterministic seed used for stratified sampling and row ordering.",
    )
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Override one default quota as STRATUM=COUNT. Can be repeated.",
    )
    parser.add_argument(
        "--write_disagreement_pool",
        action="store_true",
        help="Also write the optional file containing every judge disagreement.",
    )
    parser.add_argument(
        "--legacy_formats",
        action="store_true",
        help="Also write CSV/JSON/JSONL blind annotation files. XLSX/key files are always written.",
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=list(DEFAULT_ANNOTATORS),
        help="Allowed annotator IDs for XLSX dropdowns and per-annotator workbooks.",
    )
    parser.add_argument(
        "--regenerate_annotator_workbooks",
        action="store_true",
        help="Regenerate only the blank per-annotator XLSX workbooks from the existing key file.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="Private key file to use with --regenerate_annotator_workbooks. Defaults to OUTPUT_DIR/aaai_human_eval_sample_key.json.",
    )
    args = parser.parse_args()

    if args.regenerate_annotator_workbooks:
        key_path = args.key or args.output_dir / "aaai_human_eval_sample_key.json"
        files = regenerate_annotator_workbooks_from_key(
            key_path=key_path,
            output_dir=args.output_dir,
            annotator_choices=args.annotators,
            backup_existing=True,
        )
        print(f"Regenerated annotator workbooks from {key_path}")
        for annotator_id, path in files.items():
            print(f"  {annotator_id}: {path}")
        return {"files": {"annotator_workbooks": files}}

    quotas = _parse_quota_override(args.quota)
    metadata = build_human_eval_sample(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        judges=args.judges,
        quotas=quotas,
        sampling_seed=args.sampling_seed,
        write_disagreement_pool=args.write_disagreement_pool,
        write_legacy_formats=args.legacy_formats,
        annotator_choices=args.annotators,
    )

    print(f"Created human-evaluation package in {args.output_dir}")
    print(f"Population size: {metadata['population_size']}")
    print(f"Core sample size: {metadata['sample_size']}")
    print("Core sample by primary stratum:")
    for stratum, count in metadata["sample_by_primary_stratum"].items():
        print(f"  {stratum}: {count}")

    return metadata


if __name__ == "__main__":
    main()
