#!/usr/bin/env python
"""
Run the ATL LLM-as-a-judge evaluator over prediction files.
"""

import argparse
from pathlib import Path
from typing import Optional

from ..config import ModelConfig
from ..infra.io import load_yaml, save_json, load_json

from ..evaluation.llm_judge import (
    LLMJudge,
    PROMPT_VERSION,
    build_summary,
    build_summary_notebook,
    compute_metrics,
    evaluate_prediction_file,
)
from ..models.utils import resolve_model_key


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


def resolve_judge_models(
    models_config_path: Path,
    judge_models: Optional[list],
    judge_model: Optional[str],
) -> list:
    """Resolve judge models from config or fall back to a fixed allowed set.

    Allowed default judges: `llama-70b`, `gpt-5.2`, `DeepSeek-V3.2`.
    If a models config exists and contains matching keys those entries are used;
    otherwise a simple `ModelConfig` with provider="azure" is returned for
    the requested names.
    """

    models = load_models_config(models_config_path)

    if judge_model:
        judge_models = [judge_model]

    # If no config is present, construct simple ModelConfig entries for the
    # requested models or fall back to the allowed defaults.
    if not models:
        if judge_models:
            return [
                (name, ModelConfig(name=name, short_name=name, provider="azure"))
                for name in judge_models
            ]
        return [
            (
                "llama-70b",
                ModelConfig(name="llama-70b", short_name="llama-70b", provider="azure"),
            ),
            (
                "gpt-5.2",
                ModelConfig(name="gpt-5.2", short_name="gpt-5.2", provider="azure"),
            ),
            (
                "DeepSeek-V3.2",
                ModelConfig(
                    name="DeepSeek-V3.2", short_name="DeepSeek-V3.2", provider="azure"
                ),
            ),
        ]

    # If explicit judge models were provided, resolve them against the config.
    if judge_models:
        selected_keys = []
        seen = set()
        for model_arg in judge_models:
            key = resolve_model_key(
                model_arg,
                models,
                require_mapping_entries=True,
                match_key_lower=True,
            )
            if key not in seen:
                selected_keys.append(key)
                seen.add(key)
    else:
        # Default selection: only the allowed three, in this order.
        default_keys = ["llama-70b", "gpt-5.2", "DeepSeek-V3.2"]
        selected_keys = [k for k in default_keys if k in models]
        if not selected_keys:
            # If none of the allowed keys exist in the config, fall back to any
            # available azure-models or the first few entries in the config.
            azure_keys = [
                key
                for key, data in models.items()
                if isinstance(data, dict)
                and str(data.get("provider", "huggingface")).lower() == "azure"
            ]
            selected_keys = azure_keys or list(models.keys())[:3]

    resolved = []
    for key in selected_keys:
        data = models.get(key)
        if not isinstance(data, dict):
            continue
        resolved.append((key, ModelConfig(**data)))

    return resolved


def compute_stats_from_rows(rows: list) -> dict:
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


def extract_prediction_metadata(prediction_data: object) -> dict:
    if not isinstance(prediction_data, dict):
        return {}

    metadata = prediction_data.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)

    return {
        key: value
        for key, value in prediction_data.items()
        if key not in {"predictions", "detailed_results"}
    }


def extract_evaluated_rows(evaluated_data: object) -> list:
    if isinstance(evaluated_data, list):
        return evaluated_data
    if isinstance(evaluated_data, dict):
        rows = evaluated_data.get("detailed_results")
        if isinstance(rows, list):
            return rows
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Prediction files to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        "--models",
        "--judge_model",
        "--judge_models",
        nargs="+",
        dest="judge_models",
        default=None,
        help="Judge model names (aliases: --models, --judge_model, --judge_models).",
    )
    parser.add_argument(
        "--models_config",
        default="configs/models.yaml",
        help="Models config file (default: configs/models.yaml)",
    )
    # Note: HF-size filtering (--hf_min_params_b / --hf_only) removed.
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
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        help="Re-evaluate datasets even if evaluated outputs already exist.",
    )
    args = parser.parse_args()

    # Note: --models / --judge_model / --judge_models now map to `args.judge_models`.

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    prediction_files = resolve_prediction_files(predictions_dir, args.datasets)
    if not prediction_files:
        raise ValueError("No prediction files found to evaluate.")

    judge_entries = resolve_judge_models(
        Path(args.models_config),
        args.judge_models,
        None,
    )
    if not judge_entries:
        raise ValueError("No judge models resolved from config.")

    for _, model_config in judge_entries:
        judge_name = model_config.short_name
        api_model = model_config.api_model or model_config.name
        judge = LLMJudge(
            judge_model=judge_name,
            api_model=api_model,
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
            "no_llm": 0,
        }

        for pred_path in prediction_files:
            judge_tag = f"__judge-{judge_name}"
            output_name = f"{pred_path.stem}{judge_tag}.json"
            evaluated_path = evaluated_dir / output_name

            if evaluated_path.exists() and not args.overwrite:
                existing_data = load_json(evaluated_path)
                rows = extract_evaluated_rows(existing_data)
                metrics = compute_metrics(rows)
                stats = compute_stats_from_rows(rows)

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
                totals["no_llm"] += stats.get("no_llm", 0)
                continue

            prediction_data = load_json(pred_path)
            metadata = extract_prediction_metadata(prediction_data)
            rows, stats = evaluate_prediction_file(
                prediction_path=pred_path,
                judge=judge,
                no_llm=args.no_llm,
            )
            metrics = compute_metrics(rows)
            evaluated_payload = {
                **metadata,
                "judge_model": judge_name,
                "source_file": pred_path.name,
                "detailed_results": rows,
            }
            save_json(evaluated_payload, evaluated_path)

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
            totals["no_llm"] += stats.get("no_llm", 0)

        summary = build_summary(
            results=results,
            totals=totals,
            judge_model=judge_name,
            prompt_version=PROMPT_VERSION,
        )

        summary_path = output_dir / f"summary__judge-{judge_name}.json"
        save_json(summary, summary_path)

        notebook_path = output_dir / f"summary__judge-{judge_name}.ipynb"
        build_summary_notebook(summary_path, notebook_path)

        print(f"Wrote {len(results)} evaluated datasets to {evaluated_dir}")
        print(f"Summary JSON: {summary_path}")
        print(f"Summary notebook: {notebook_path}")

    # After processing all judges, compute inter-rater agreement if multiple judges
    if len(judge_entries) > 1:
        from ..evaluation.judge_agreement import (
            generate_agreement_report,
            print_agreement_summary,
            build_agreement_notebook,
        )

        eval_datasets_dir = output_dir / "evaluated_datasets"
        if not eval_datasets_dir.exists():
            # Fall back to checking if judge dirs are directly in output_dir
            eval_datasets_dir = output_dir

        try:
            agreement_report = generate_agreement_report(
                eval_dir=eval_datasets_dir,
                output_path=output_dir / "agreement_report.json",
            )
            print_agreement_summary(agreement_report)
            try:
                notebook_path = output_dir / "agreement_report.ipynb"
                build_agreement_notebook(
                    output_dir / "agreement_report.json", notebook_path
                )
                print(f"Agreement notebook: {notebook_path}")
            except Exception as e:
                print(f"Warning: could not build agreement notebook: {e}")
        except ValueError as e:
            print(f"Skipping agreement analysis: {e}")


if __name__ == "__main__":
    main()
