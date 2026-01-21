#!/usr/bin/env python
"""
Run the ATL LLM-as-a-judge evaluator over prediction files.
"""

import argparse
from pathlib import Path

from src.config import load_yaml

from src.llm_judge import (
    LLMJudge,
    PROMPT_VERSION,
    build_summary,
    build_summary_notebook,
    compute_metrics,
    evaluate_prediction_file,
    write_json,
)


def resolve_prediction_files(predictions_dir: Path, datasets: list) -> list:
    if not datasets or (len(datasets) == 1 and datasets[0].lower() == "all"):
        return sorted(predictions_dir.rglob("*.json"))

    resolved = []
    for entry in datasets:
        path = Path(entry)
        if path.exists():
            resolved.append(path)
            continue

        candidate = predictions_dir / entry
        if candidate.exists():
            resolved.append(candidate)
            continue

        if not entry.endswith(".json"):
            candidate = predictions_dir / f"{entry}.json"
            if candidate.exists():
                resolved.append(candidate)
                continue

        matches = list(predictions_dir.rglob(entry))
        if matches:
            resolved.extend(matches)

    return resolved


def resolve_judge_model(judge_model: str, models_config_path: Path) -> str:
    if not models_config_path.exists():
        return judge_model

    models_cfg = load_yaml(str(models_config_path))
    models = models_cfg.get("models", {}) if isinstance(models_cfg, dict) else {}
    if not isinstance(models, dict):
        return judge_model

    def normalize(token: str) -> str:
        token = token.lower()
        for prefix in "azure-":
            if token.startswith(prefix):
                token = token[len(prefix) :]
        return token

    needle = judge_model.lower()
    normalized_needle = normalize(judge_model)
    for key, data in models.items():
        if not isinstance(data, dict):
            continue
        short_name = str(data.get("short_name", ""))
        name = str(data.get("name", ""))
        if (
            key.lower() == needle
            or short_name.lower() == needle
            or name.lower() == needle
            or normalize(key) == normalized_needle
            or normalize(short_name) == normalized_needle
            or normalize(name) == normalized_needle
        ):
            return str(data.get("api_model") or name or judge_model)

    return judge_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Prediction files to evaluate (default: all)",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-5.2",
        help="Judge model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--models_config",
        default="configs/models.yaml",
        help="Models config file (default: configs/models.yaml)",
    )
    parser.add_argument(
        "--predictions_dir",
        default="outputs/model_predictions",
        help="Directory with prediction JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/LLM-evaluation",
        help="Output directory for judge results",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Disable LLM judging; only exact-match normalization is used",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    prediction_files = resolve_prediction_files(predictions_dir, args.datasets)
    if not prediction_files:
        raise ValueError("No prediction files found to evaluate.")

    judge_cache = output_dir / "judge_cache.json"
    resolved_api_model = resolve_judge_model(args.judge_model, Path(args.models_config))
    judge = LLMJudge(
        judge_model=args.judge_model,
        api_model=resolved_api_model,
        cache_path=judge_cache,
        no_llm=args.no_llm,
        prompt_version=PROMPT_VERSION,
    )

    corrected_dir = output_dir / "corrected_datasets" / args.judge_model
    corrected_dir.mkdir(parents=True, exist_ok=True)

    results = []
    totals = {
        "evaluated": 0,
        "auto_exact": 0,
        "llm_calls": 0,
        "cache_hits": 0,
        "no_llm": 0,
    }

    for pred_path in prediction_files:
        rows, stats = evaluate_prediction_file(
            prediction_path=pred_path,
            judge=judge,
            no_llm=args.no_llm,
        )
        metrics = compute_metrics(rows)

        judge_tag = f"__judge-{args.judge_model}"
        output_name = f"{pred_path.stem}{judge_tag}.json"
        corrected_path = corrected_dir / output_name
        write_json(corrected_path, rows)

        results.append(
            {
                "source_file": pred_path.name,
                "stem": pred_path.stem,
                "rows": rows,
                "metrics": metrics,
                "stats": stats,
                "corrected_path": str(corrected_path),
            }
        )

        totals["evaluated"] += int(metrics["evaluated"])
        totals["auto_exact"] += stats.get("auto_exact", 0)
        totals["llm_calls"] += stats.get("llm_calls", 0)
        totals["cache_hits"] += stats.get("cache_hits", 0)
        totals["no_llm"] += stats.get("no_llm", 0)

    summary = build_summary(
        results=results,
        totals=totals,
        judge_model=args.judge_model,
        prompt_version=PROMPT_VERSION,
    )

    summary_path = output_dir / f"summary__judge-{args.judge_model}.json"
    write_json(summary_path, summary)

    notebook_path = output_dir / f"summary__judge-{args.judge_model}.ipynb"
    build_summary_notebook(summary_path, notebook_path)

    print(f"Wrote {len(results)} corrected datasets to {corrected_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary notebook: {notebook_path}")


if __name__ == "__main__":
    main()
