#!/usr/bin/env python
"""
Run the ATL LLM-as-a-judge evaluator over prediction files.
"""

import argparse
import hashlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import ModelConfig
from ..infra.io import load_yaml, save_json, load_json

from ..evaluation.llm_judge import (
    LLMJudge,
    PROMPT_VERSION,
    compute_metrics,
    evaluate_prediction_file,
)
from ..models.utils import resolve_model_key

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Resolve judge models from config or fall back to allowed defaults.

    Default judges: `gpt-5.2`, `DeepSeek-V3.2` (Azure). Explicitly selected
    judges may also use provider="huggingface" for self-hosted local judges
    (e.g. Llama/Gemma run via SLURM); other providers are rejected.
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
        # Default selection: only the allowed API judges, in this order.
        default_keys = ["gpt-5.2", "DeepSeek-V3.2"]
        selected_keys = [
            k
            for k in default_keys
            if k in models
            and isinstance(models.get(k), dict)
            and str(models[k].get("provider", "")).lower() == "azure"
        ]
        if not selected_keys:
            # If none of the allowed keys exist in the config, fall back to any
            # available Azure models.
            azure_keys = [
                key
                for key, data in models.items()
                if isinstance(data, dict)
                and str(data.get("provider", "huggingface")).lower() == "azure"
            ]
            selected_keys = azure_keys

    resolved = []
    for key in selected_keys:
        data = models.get(key)
        if not isinstance(data, dict):
            continue
        model_config = ModelConfig(**data)
        provider = model_config.provider.lower()
        if provider not in ("azure", "huggingface"):
            raise ValueError(
                f"Judge model '{key}' uses provider '{model_config.provider}'. "
                "Judge models must use provider 'azure' or 'huggingface'."
            )
        resolved.append((key, model_config))

    return resolved


def compute_stats_from_rows(rows: list) -> dict:
    stats = {
        "unmatched": 0,
        "auto_exact": 0,
        "llm_calls": 0,
        "no_llm": 0,
        "cached": 0,
    }

    for row in rows:
        decision_method = row.get("decision_method")
        if decision_method == "unmatched":
            stats["unmatched"] += 1
        elif decision_method == "exact":
            stats["auto_exact"] += 1
        elif decision_method == "llm":
            stats["llm_calls"] += 1
            if row.get("from_cache"):
                stats["cached"] += 1
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


def _render_judge_sbatch(args, repo_root, judge_key, model_config, gres):
    """Render an sbatch script that judges all prediction files with one local judge."""
    logs_dir = args.logs_dir
    job_name = f"{args.job_name}-{model_config.short_name}"
    output_path = args.output or f"{logs_dir}/%x_%j.out"
    error_path = args.error or f"{logs_dir}/%x_%j.err"

    inner = [
        "-m",
        "src.cli.run_llm_judge",
        "--judge_models",
        judge_key,
        "--datasets",
        *args.datasets,
        "--models_config",
        args.models_config,
        "--predictions_dir",
        args.predictions_dir,
        "--output_dir",
        args.output_dir,
    ]
    if args.overwrite:
        inner.append("--overwrite")
    if args.no_llm:
        inner.append("--no_llm")
    command = '"$PYTHON_BIN" ' + " ".join(shlex.quote(str(a)) for a in inner)

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
    ]
    for extra in args.sbatch_arg:
        extra = extra.strip()
        if not extra:
            continue
        lines.append(extra if extra.startswith("#SBATCH") else f"#SBATCH {extra}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(logs_dir)}",
            f"PYTHON_BIN=${{PYTHON_BIN:-{shlex.quote(args.python_bin)}}}",
            f"REPO_ROOT=${{REPO_ROOT:-{shlex.quote(str(repo_root))}}}",
            'export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"',
            "export PYTHONUNBUFFERED=1",
            # Reduce CUDA allocator fragmentation when sharding large judges
            # across multiple GPUs during the transformers parallel weight load.
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            # Load weights sequentially instead of prefetching every shard with a
            # thread pool. The async loader materializes a GPU's whole shard in
            # bf16 before compacting to 4-bit (~2*params_b/N GiB peak) and OOMs a
            # 70B mid-load; sequential load quantizes one param at a time so the
            # peak stays near the 4-bit footprint (~params_b/2/N GiB).
            "export HF_DEACTIVATE_ASYNC_LOAD=1",
            'cd "$REPO_ROOT"',
        ]
    )
    lines.extend(args.env_setup)
    lines.extend(["", command, ""])
    return "\n".join(lines)


def _judge_gpu_count(params_b: Optional[float]) -> int:
    """Number of GPUs a local judge needs to load in 4-bit on A100 40GB cards.

    The sbatch template exports ``HF_DEACTIVATE_ASYNC_LOAD=1`` so the transformers
    loader quantizes one parameter at a time instead of prefetching a GPU's whole
    shard in bf16. The transient peak per GPU is then close to the final 4-bit
    footprint (~``params_b / 2 / num_gpus`` GiB) rather than the bf16 shard, so a
    judge needs only enough GPUs to hold its 4-bit weights plus inference slack:

      * 27B -> ~13.5 GiB / 2 = ~7 GiB/GPU  (verified: loads, judges fine)
      * 70B -> ~35 GiB / 3 = ~12 GiB/GPU   (fits a clean 3-GPU standard node)

    Three GPUs keeps the 70B on the cleanly-isolated standard A100 nodes; the
    shared 8-GPU node oversubscribes cards and its co-tenants steal headroom.
    """
    p = params_b or 0
    if p >= 60:
        return 3
    if p >= 20:
        return 2
    return 1


def _submit_judge_slurm(args, judge_entries):
    """Generate and submit one GPU SLURM job per local (huggingface) judge.

    Each job loads its model once and judges every prediction file, so the work
    runs unattended. GPU count scales with judge size so the model can shard and
    load in 4-bit without OOMing mid-load: 70B-class uses three GPUs, 27B-class
    two, smaller judges one. Azure judges need no GPU and are skipped.
    """
    repo_root = Path(args.repo_root).resolve()
    local = [(k, mc) for k, mc in judge_entries if mc.provider.lower() == "huggingface"]
    azure = [mc.short_name for _, mc in judge_entries if mc.provider.lower() == "azure"]
    if azure:
        print(
            f"Note: Azure judges ({', '.join(azure)}) need no GPU; run them "
            "directly without --slurm. Skipping them from SLURM submission."
        )
    if not local:
        raise ValueError("No local (huggingface) judges to submit as SLURM jobs.")

    script_dir = Path(args.script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for judge_key, model_config in local:
        gres = args.gres or f"gpu:{_judge_gpu_count(model_config.params_b)}"
        script = _render_judge_sbatch(args, repo_root, judge_key, model_config, gres)

        if args.dry_run:
            print(script)
            print("# ---")
            continue

        script_path = script_dir / f"judge_{model_config.short_name}_{stamp}.sbatch"
        script_path.write_text(script, encoding="utf-8")

        if args.no_submit:
            print(
                f"Wrote SLURM script for judge '{model_config.short_name}' "
                f"to {script_path} (gres={gres})"
            )
            continue

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise SystemExit("sbatch not found. Are you on a SLURM cluster?") from exc

        print(result.stdout.strip() or f"Submitted judge '{model_config.short_name}'.")
        print(f"  script: {script_path} (gres={gres})")


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
    parser.add_argument(
        "--agreement_notebook",
        action="store_true",
        help="Also write the legacy agreement_report.ipynb notebook.",
    )

    # SLURM submission (mirrors `nl2atl run --slurm`): generate and submit one
    # GPU job per local judge so judging runs unattended on the cluster.
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate and submit one SLURM GPU job per local judge (unattended).",
    )
    parser.add_argument("--partition", default="A100")
    parser.add_argument(
        "--gres",
        default=None,
        help=(
            "Override --gres for all judge jobs (default scales with size: "
            "gpu:3 for >=60B, gpu:2 for >=20B, else gpu:1)."
        ),
    )
    parser.add_argument("--cpus-per-task", "--cpus_per_task", type=int, default=8)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", dest="time_limit", default="04:00:00")
    parser.add_argument("--job-name", "--job_name", default="nl2atl-judge")
    parser.add_argument("--logs-dir", "--logs_dir", default="logs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--error", default=None)
    parser.add_argument("--python-bin", "--python_bin", default=sys.executable)
    parser.add_argument("--repo-root", "--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--script-dir", "--script_dir", default="outputs/manifests")
    parser.add_argument("--sbatch-arg", "--sbatch_arg", action="append", default=[])
    parser.add_argument(
        "--env-setup",
        "--env_setup",
        action="append",
        default=[],
        help="Shell line inserted into the SLURM script before execution.",
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--no-submit", "--no_submit", action="store_true")
    args = parser.parse_args()

    # Note: --models / --judge_model / --judge_models now map to `args.judge_models`.

    judge_entries = resolve_judge_models(
        Path(args.models_config),
        args.judge_models,
        None,
    )
    if not judge_entries:
        raise ValueError("No judge models resolved from config.")

    if args.slurm:
        _submit_judge_slurm(args, judge_entries)
        return

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    prediction_files = resolve_prediction_files(predictions_dir, args.datasets)
    if not prediction_files:
        raise ValueError("No prediction files found to evaluate.")

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

        totals = {
            "evaluated": 0,
            "auto_exact": 0,
            "llm_calls": 0,
            "no_llm": 0,
            "cached": 0,
        }

        for pred_path in prediction_files:
            judge_tag = f"__judge-{judge_name}"
            output_name = f"{pred_path.stem}{judge_tag}.json"
            evaluated_path = evaluated_dir / output_name

            if evaluated_path.exists() and not args.overwrite:
                existing_data = load_json(evaluated_path)
                if existing_data.get("prompt_version") == PROMPT_VERSION:
                    rows = extract_evaluated_rows(existing_data)
                    metrics = compute_metrics(rows)
                    stats = compute_stats_from_rows(rows)

                    totals["evaluated"] += int(metrics["evaluated"])
                    totals["auto_exact"] += stats.get("auto_exact", 0)
                    totals["llm_calls"] += stats.get("llm_calls", 0)
                    totals["no_llm"] += stats.get("no_llm", 0)
                    totals["cached"] += stats.get("cached", 0)
                    continue

                print(
                    f"Re-evaluating {evaluated_path.name}: "
                    f"prompt version changed to {PROMPT_VERSION}."
                )

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
                "prompt_version": PROMPT_VERSION,
                "judge_provider": model_config.provider,
                "judge_api_model": api_model,
                "judge_decoding": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
                "source_file": pred_path.name,
                "source_sha256": _sha256_file(pred_path),
                "models_config": str(Path(args.models_config)),
                "models_config_sha256": _sha256_file(Path(args.models_config)),
                "evaluated_at": _utc_now(),
                "detailed_results": rows,
            }
            save_json(evaluated_payload, evaluated_path)

            totals["evaluated"] += int(metrics["evaluated"])
            totals["auto_exact"] += stats.get("auto_exact", 0)
            totals["llm_calls"] += stats.get("llm_calls", 0)
            totals["no_llm"] += stats.get("no_llm", 0)
            totals["cached"] += stats.get("cached", 0)

        print(f"Wrote evaluated datasets to {evaluated_dir}")
        cached = totals["cached"]
        if cached:
            unique_api_calls = totals["llm_calls"] - cached
            print(
                f"  {judge_name}: {totals['llm_calls']} judged, "
                f"{unique_api_calls} unique API calls, "
                f"{cached} served from cache "
                f"({getattr(judge, 'api_calls', unique_api_calls)} actual calls)."
            )

    # After processing all judges, compute inter-rater agreement if multiple judges
    if len(judge_entries) > 1:
        from ..evaluation.judge_agreement import (
            generate_agreement_report,
            print_agreement_summary,
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
            if args.agreement_notebook:
                from ..evaluation.judge_agreement import build_agreement_notebook

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
