#!/usr/bin/env python
"""
Run the ATL LLM-as-a-judge evaluator over prediction files.
"""

import argparse
from pathlib import Path
from typing import Optional

from src.config import ModelConfig, load_yaml

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


def load_models_config(models_config_path: Path) -> dict:
    if not models_config_path.exists():
        return {}
    models_cfg = load_yaml(str(models_config_path))
    models = models_cfg.get("models", {}) if isinstance(models_cfg, dict) else {}
    return models if isinstance(models, dict) else {}


def resolve_model_key(model_arg: str, models: dict) -> str:
    if model_arg in models:
        return model_arg

    def normalize(token: str) -> str:
        token = token.lower()
        for prefix in "azure-":
            if token.startswith(prefix):
                token = token[len(prefix) :]
        return token

    needle = model_arg.lower()
    normalized_needle = normalize(model_arg)
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
            return key

    raise KeyError(model_arg)


def resolve_judge_models(
    models_config_path: Path,
    judge_models: Optional[list],
    judge_model: Optional[str],
) -> list:
    models = load_models_config(models_config_path)

    if not models:
        if judge_model:
            return [(judge_model, ModelConfig(name=judge_model, short_name=judge_model, provider="azure"))]
        if judge_models:
            return [
                (
                    name,
                    ModelConfig(name=name, short_name=name, provider="azure"),
                )
                for name in judge_models
            ]
        return [("gpt-5.2", ModelConfig(name="gpt-5.2", short_name="gpt-5.2", provider="azure"))]

    if judge_model:
        judge_models = [judge_model]

    if judge_models:
        selected_keys = [resolve_model_key(m, models) for m in judge_models]
    else:
        selected_keys = [
            key
            for key, data in models.items()
            if isinstance(data, dict)
            and str(data.get("provider", "huggingface")).lower() == "azure"
        ]

    resolved = []
    for key in selected_keys:
        data = models.get(key)
        if not isinstance(data, dict):
            continue
        resolved.append((key, ModelConfig(**data)))

    return resolved


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
        default=None,
        help="Judge model name (deprecated; use --judge_models)",
    )
    parser.add_argument(
        "--judge_models",
        nargs="+",
        default=None,
        help="Judge model names (default: all provider=azure)",
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

    judge_entries = resolve_judge_models(
        Path(args.models_config), args.judge_models, args.judge_model
    )
    if not judge_entries:
        raise ValueError("No judge models resolved from config.")

    for _, model_config in judge_entries:
        judge_name = model_config.short_name
        judge_cache = output_dir / "judge_cache.json"
        api_model = model_config.api_model or model_config.name
        judge = LLMJudge(
            judge_model=judge_name,
            api_model=api_model,
            cache_path=judge_cache,
            no_llm=args.no_llm,
            prompt_version=PROMPT_VERSION,
            provider=model_config.provider,
            model_config=model_config,
        )

        evaluated_dir = output_dir / "evaluated_datasets" / judge_name
        evaluated_dir.mkdir(parents=True, exist_ok=True)

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

            judge_tag = f"__judge-{judge_name}"
            output_name = f"{pred_path.stem}{judge_tag}.json"
            evaluated_path = evaluated_dir / output_name
            write_json(evaluated_path, rows)

            results.append(
                {
                    "source_file": pred_path.name,
                    "stem": pred_path.stem,
                    "rows": rows,
                    "metrics": metrics,
                    "stats": stats,
                    "evaluated_path": str(evaluated_path),
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
            judge_model=judge_name,
            prompt_version=PROMPT_VERSION,
        )

        summary_path = output_dir / f"summary__judge-{judge_name}.json"
        write_json(summary_path, summary)

        notebook_path = output_dir / f"summary__judge-{judge_name}.ipynb"
        build_summary_notebook(summary_path, notebook_path)

        print(f"Wrote {len(results)} evaluated datasets to {evaluated_dir}")
        print(f"Summary JSON: {summary_path}")
        print(f"Summary notebook: {notebook_path}")


if __name__ == "__main__":
    main()
