#!/usr/bin/env python
"""Generate summaries from evaluated judge datasets.

This script aggregates all judge evaluation results per judge model and creates
summary JSON and notebook files following the same format as run_llm_judge.py.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from ..infra.io import save_json
from ..evaluation.llm_judge import (
    PROMPT_VERSION,
    build_summary,
    build_summary_notebook,
    compute_metrics,
)


def compute_stats_from_rows(rows: List[Dict]) -> Dict[str, int]:
    """Compute statistics from judge decision rows."""
    stats = {
        "unmatched": 0,
        "auto_exact": 0,
        "llm_calls": 0,
        "no_llm": 0,
    }

    for row in rows:
        decision_method = row.get("decision_method")
        if decision_method == "unmatched":
            stats["unmatched"] += 1
        elif decision_method == "exact":
            stats["auto_exact"] += 1
        elif decision_method == "llm":
            stats["llm_calls"] += 1
        elif decision_method == "no_llm":
            stats["no_llm"] += 1

    return stats


def extract_evaluated_rows(evaluated_data: Any) -> List[Dict]:
    """Extract detailed results from evaluated data."""
    if isinstance(evaluated_data, list):
        return evaluated_data
    if isinstance(evaluated_data, dict):
        rows = evaluated_data.get("detailed_results")
        if isinstance(rows, list):
            return rows
    return []


def summarize_judge_evaluations(
    input_dir: Path,
    output_dir: Path,
    judge_filter: List[str] = None,
    prompt_version: str = PROMPT_VERSION,
) -> None:
    """Generate summaries for all judges in the evaluated_datasets directory.

    Args:
        input_dir: Directory containing judge subdirectories with evaluated JSON files
        output_dir: Directory where summary files will be saved
        judge_filter: Optional list of judge names to process (default: all)
        prompt_version: Prompt version string for the summary metadata
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover judge subdirectories
    judge_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if judge_filter:
        judge_dirs = [d for d in judge_dirs if d.name in judge_filter]

    if not judge_dirs:
        raise ValueError(f"No judge directories found in {input_dir}")

    for judge_dir in judge_dirs:
        judge_name = judge_dir.name
        print(f"\nProcessing judge: {judge_name}")

        # Collect all evaluated JSON files for this judge
        evaluated_files = sorted(judge_dir.glob("*.json"))
        if not evaluated_files:
            print(f"  Warning: No evaluated files found for judge {judge_name}")
            continue

        results = []
        totals = {
            "evaluated": 0,
            "auto_exact": 0,
            "llm_calls": 0,
            "no_llm": 0,
        }

        for evaluated_path in evaluated_files:
            try:
                evaluated_data = json.loads(evaluated_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse {evaluated_path.name}: {e}")
                continue

            rows = extract_evaluated_rows(evaluated_data)
            if not rows:
                print(f"  Warning: No detailed_results in {evaluated_path.name}")
                continue

            metrics = compute_metrics(rows)
            stats = compute_stats_from_rows(rows)

            # Extract source filename (remove judge suffix if present)
            source_file = evaluated_path.name
            # Remove judge suffix (e.g., __judge-llama-70b) from the filename
            if "__judge-" in source_file:
                stem = source_file.split("__judge-")[0]
            else:
                stem = Path(source_file).stem

            results.append(
                {
                    "source_file": source_file,
                    "stem": stem,
                    "rows": rows,
                    "metrics": metrics,
                    "stats": stats,
                    "evaluated_path": str(evaluated_path),
                }
            )

            totals["evaluated"] += int(metrics["evaluated"])
            totals["auto_exact"] += stats.get("auto_exact", 0)
            totals["llm_calls"] += stats.get("llm_calls", 0)
            totals["no_llm"] += stats.get("no_llm", 0)

        if not results:
            print(f"  Warning: No valid results for judge {judge_name}")
            continue

        # Build summary
        summary = build_summary(
            results=results,
            totals=totals,
            judge_model=judge_name,
            prompt_version=prompt_version,
        )

        # Save summary JSON
        summary_path = output_dir / f"summary__judge-{judge_name}.json"
        save_json(summary, summary_path)
        print(f"  Wrote summary: {summary_path}")

        # Build and save summary notebook
        notebook_path = output_dir / f"summary__judge-{judge_name}.ipynb"
        build_summary_notebook(summary_path, notebook_path)
        print(f"  Wrote notebook: {notebook_path}")

        print(f"  Aggregated {len(results)} datasets for {judge_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summaries from evaluated judge datasets."
    )
    parser.add_argument(
        "--input_dir",
        default="outputs/LLM-evaluation/evaluated_datasets",
        help="Input directory containing judge subdirectories with evaluated JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/LLM-evaluation",
        help="Output directory where summary files will be saved.",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Specific judge names to process (default: all)",
    )
    parser.add_argument(
        "--prompt_version",
        default=PROMPT_VERSION,
        help="Prompt version string for summary metadata.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        summarize_judge_evaluations(
            input_dir=input_dir,
            output_dir=output_dir,
            judge_filter=args.judges,
            prompt_version=args.prompt_version,
        )
        print("\nSummary generation complete!")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
