#!/usr/bin/env python
"""Unified experiment runner for local and SLURM execution."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import Config, ExperimentCondition, ModelConfig
from ..constants import Provider
from ..experiment.reporter import get_git_commit
from ..models.utils import resolve_model_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_PARALLEL_GPUS: Optional[int] = None


@dataclass(frozen=True)
class TaskSpec:
    """One reproducible unit of experiment work."""

    index: int
    seed: int
    model_key: str
    condition_names: List[str]
    cv_fold: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "model_key": self.model_key,
            "condition_names": list(self.condition_names),
            "cv_fold": self.cv_fold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        cv_fold = data.get("cv_fold")
        return cls(
            index=int(data["index"]),
            seed=int(data["seed"]),
            model_key=str(data["model_key"]),
            condition_names=[str(name) for name in data["condition_names"]],
            cv_fold=None if cv_fold is None else int(cv_fold),
        )


@dataclass(frozen=True)
class SkippedSpec:
    """A requested model/condition combination that is not runnable."""

    seed: int
    model_key: str
    condition_name: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "model_key": self.model_key,
            "condition_name": self.condition_name,
            "reason": self.reason,
        }


def _wants_all(values: Optional[Sequence[str]]) -> bool:
    return not values or any(str(value).lower() == "all" for value in values)


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _model_provider(model_config: ModelConfig) -> str:
    return str(model_config.provider).lower()


def _model_generation_enabled(model_config: ModelConfig) -> bool:
    return bool(getattr(model_config, "generation_enabled", True))


def can_finetune(model_config: ModelConfig) -> Tuple[bool, str]:
    """Return whether the local LoRA runner supports fine-tuning this model."""
    if model_config.is_azure:
        return False, "Azure/API models are evaluated only without fine-tuning"
    if model_config.params_b is not None and model_config.params_b > 8:
        return False, "models above 8B parameters are disabled for local fine-tuning"
    return True, ""


def select_models(
    config: Config,
    model_args: Optional[Sequence[str]],
    model_provider: str,
) -> List[str]:
    """Resolve selected model keys and apply provider filtering."""
    if model_provider not in ("hf", "azure", "all"):
        raise ValueError(
            f"Invalid model_provider '{model_provider}'. Use 'hf', 'azure', or 'all'."
        )

    wants_all = _wants_all(model_args)

    if wants_all:
        selected = list(config.models.keys())
    else:
        selected = _dedupe(
            [
                resolve_model_key(model_arg, config.models)
                for model_arg in model_args or []
            ]
        )

    if model_provider != "all":
        provider = (
            Provider.HUGGINGFACE.value
            if model_provider == "hf"
            else Provider.AZURE.value
        )
        selected = [
            model_key
            for model_key in selected
            if _model_provider(config.models[model_key]) == provider
        ]

    disabled = [
        model_key
        for model_key in selected
        if not _model_generation_enabled(config.models[model_key])
    ]
    if wants_all:
        return [model_key for model_key in selected if model_key not in disabled]

    if disabled:
        joined = ", ".join(disabled)
        raise ValueError(
            f"Model(s) are disabled for generation and reserved for judging: {joined}"
        )

    return selected


def select_conditions(
    config: Config,
    condition_args: Optional[Sequence[str]],
) -> List[ExperimentCondition]:
    """Resolve selected conditions in config order."""
    if _wants_all(condition_args):
        return list(config.conditions)

    requested = _dedupe([str(condition) for condition in condition_args or []])
    by_name = {condition.name: condition for condition in config.conditions}
    missing = [condition for condition in requested if condition not in by_name]
    if missing:
        raise ValueError(f"No conditions matched: {missing}")
    return [by_name[condition] for condition in requested]


def resolve_requested_seeds(
    config: Config,
    seed: Optional[int],
    seeds: Optional[Sequence[int]],
) -> List[int]:
    """Resolve the seed list from CLI overrides or config defaults."""
    if seed is not None and seeds:
        raise ValueError("Use either --seed or --seeds, not both.")
    if seed is not None:
        return [seed]
    if seeds:
        return [int(value) for value in seeds]
    return config.resolve_seeds()


def build_tasks(
    config: Config,
    seeds: Sequence[int],
    models: Optional[Sequence[str]],
    conditions: Optional[Sequence[str]],
    model_provider: str,
    cv_folds: int = 0,
) -> Tuple[List[TaskSpec], List[SkippedSpec]]:
    """Build the experiment task matrix.

    The headline results use a single fixed (stratified) canonical split: each
    baseline condition runs once (deterministic under greedy decoding) while
    fine-tuned conditions run for every training seed (the seed ablation, used
    for mean +/- std). When ``cv_folds >= 2``, every model/condition also runs
    once per stratified cross-validation fold (single training seed) for the
    partition-robustness analysis.
    """
    model_keys = select_models(config, models, model_provider)
    selected_conditions = select_conditions(config, conditions)

    tasks: List[TaskSpec] = []
    skipped: List[SkippedSpec] = []
    index = 0
    base_seed = int(seeds[0]) if seeds else int(config.seed or 0)

    for model_key in model_keys:
        model_config = config.models[model_key]

        # Resolve runnable conditions once per model (records skips once).
        all_runnable: List[str] = []
        finetuned_runnable: List[str] = []
        for condition in selected_conditions:
            if condition.finetuned:
                allowed, reason = can_finetune(model_config)
                if not allowed:
                    skipped.append(
                        SkippedSpec(
                            seed=base_seed,
                            model_key=model_key,
                            condition_name=condition.name,
                            reason=reason,
                        )
                    )
                    continue
                finetuned_runnable.append(condition.name)
            all_runnable.append(condition.name)

        # Canonical split: baselines once (first seed), fine-tuned per seed.
        for seed_index, seed in enumerate(seeds):
            conditions_subset = all_runnable if seed_index == 0 else finetuned_runnable
            if conditions_subset:
                tasks.append(
                    TaskSpec(
                        index=index,
                        seed=int(seed),
                        model_key=model_key,
                        condition_names=conditions_subset,
                        cv_fold=None,
                    )
                )
                index += 1

        # Cross-validation folds: all conditions per fold, single training seed.
        if cv_folds and cv_folds >= 2 and all_runnable:
            for fold in range(cv_folds):
                tasks.append(
                    TaskSpec(
                        index=index,
                        seed=base_seed,
                        model_key=model_key,
                        condition_names=all_runnable,
                        cv_fold=fold,
                    )
                )
                index += 1

    return tasks, skipped


def format_slurm_array_range(
    task_count: int,
    max_parallel_gpus: Optional[int],
) -> str:
    """Format a SLURM array range, optionally capped by concurrent GPU tasks."""
    if task_count <= 0:
        raise ValueError("No runnable experiment tasks were selected.")
    if max_parallel_gpus is None or max_parallel_gpus == 0:
        return f"0-{task_count - 1}"
    if max_parallel_gpus < 0:
        raise ValueError("--max-parallel-gpus must be non-negative.")
    if max_parallel_gpus >= task_count:
        return f"0-{task_count - 1}"
    return f"0-{task_count - 1}%{max_parallel_gpus}"


def _manifest_path(args: argparse.Namespace, repo_root: Path) -> Path:
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = repo_root / manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return manifest_dir / f"nl2atl_experiments_{timestamp}.json"


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
    tasks: Sequence[TaskSpec],
    skipped: Sequence[SkippedSpec],
) -> Path:
    """Write the frozen task manifest used by SLURM workers."""
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "models_config": args.models_config,
        "experiments_config": args.experiments_config,
        "model_provider": args.model_provider,
        "requested_models": list(args.models or ["all"]),
        "requested_conditions": list(args.conditions or ["all"]),
        "seeds": [int(seed) for seed in seeds],
        "tasks": [task.to_dict() for task in tasks],
        "skipped": [item.to_dict() for item in skipped],
        "slurm": {
            "max_parallel_gpus": args.max_parallel_gpus,
            "gres": args.gres,
            "partition": args.partition,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time_limit,
            "env_setup": list(args.env_setup),
            "train_max_steps": args.train_max_steps,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path) -> Tuple[List[int], List[TaskSpec], Dict[str, Any]]:
    """Load a previously written task manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    tasks = [TaskSpec.from_dict(item) for item in payload.get("tasks", [])]
    return seeds, tasks, payload


def _quote_args(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _render_sbatch_script(
    args: argparse.Namespace,
    repo_root: Path,
    manifest_path: Path,
    task_count: int,
) -> str:
    logs_dir = args.logs_dir
    output_path = args.output or f"{logs_dir}/%x_%A_%a.out"
    error_path = args.error or f"{logs_dir}/%x_%A_%a.err"
    array_range = format_slurm_array_range(task_count, args.max_parallel_gpus)

    worker_args = [
        "--task-manifest",
        str(manifest_path),
        "--models_config",
        args.models_config,
        "--experiments_config",
        args.experiments_config,
    ]
    if args.overwrite:
        worker_args.append("--overwrite")
    if args.train_max_steps is not None:
        worker_args.extend(["--train-max-steps", str(args.train_max_steps)])

    command = (
        '"$PYTHON_BIN" -m src.cli.run_experiments '
        '--task-id "$SLURM_ARRAY_TASK_ID" '
        f"{_quote_args(worker_args)}"
    )

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres={args.gres}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
        f"#SBATCH --array={array_range}",
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
            'cd "$REPO_ROOT"',
        ]
    )
    lines.extend(args.env_setup)
    lines.append("PYTHON_BIN=${PYTHON_BIN:-python3}")
    if args.train_max_steps is not None:
        lines.append(f"export TRAIN_MAX_STEPS={int(args.train_max_steps)}")
    lines.extend(["", command, ""])
    return "\n".join(lines)


def _write_script(script: str, script_path: Optional[str]) -> Path:
    if script_path:
        path = Path(script_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8")
        return path

    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch")
    tmp.write(script)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def submit_slurm(
    args: argparse.Namespace,
    seeds: Sequence[int],
    tasks: Sequence[TaskSpec],
    skipped: Sequence[SkippedSpec],
) -> None:
    """Submit the selected tasks as a SLURM array."""
    repo_root = Path(args.repo_root).resolve()
    if not tasks:
        raise ValueError("No runnable experiment tasks were selected.")

    manifest_path = _manifest_path(args, repo_root)
    write_manifest(manifest_path, args, seeds, tasks, skipped)
    script = _render_sbatch_script(args, repo_root, manifest_path, len(tasks))

    if args.dry_run:
        print(script)
        print(f"Manifest: {manifest_path}")
        return

    script_path = _write_script(script, args.script_path)
    if args.no_submit:
        print(f"Wrote SLURM script to {script_path}")
        print(f"Wrote task manifest to {manifest_path}")
        return

    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("sbatch not found. Are you on a SLURM cluster?") from exc

    print(result.stdout.strip() or "Submitted SLURM job.")
    print(f"Task manifest: {manifest_path}")


def _print_tasks(tasks: Sequence[TaskSpec], skipped: Sequence[SkippedSpec]) -> None:
    for task in tasks:
        conditions = ",".join(task.condition_names)
        fold = "canonical" if task.cv_fold is None else f"fold{task.cv_fold}"
        print(
            f"{task.index}\tseed={task.seed}\t{fold}\tmodel={task.model_key}\tconditions={conditions}"
        )

    if skipped:
        print("\nSkipped incompatible combinations:")
        for item in skipped:
            print(
                f"seed={item.seed}\tmodel={item.model_key}\tcondition={item.condition_name}\t{item.reason}"
            )


def run_task(
    config: Config,
    task: TaskSpec,
    selected_seeds: Sequence[int],
    overwrite: bool,
) -> None:
    """Run one model/seed task locally or inside a SLURM array worker."""
    from ..experiment.runner import ExperimentRunner

    config.seed = task.seed
    config.seeds = [int(seed) for seed in selected_seeds]
    config.cv_fold = task.cv_fold

    fold = "canonical" if task.cv_fold is None else f"fold{task.cv_fold}"
    print(
        f"Running task {task.index}: seed={task.seed}, {fold}, "
        f"model={task.model_key}, conditions={task.condition_names}"
    )
    runner = ExperimentRunner(config)
    runner.run_model_experiments(
        task.model_key,
        conditions=task.condition_names,
        overwrite=overwrite,
    )


def resolve_task_id(explicit_task_id: Optional[int]) -> Optional[int]:
    """Resolve explicit task id or SLURM_ARRAY_TASK_ID when running in an array."""
    if explicit_task_id is not None:
        return explicit_task_id

    env_value = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_value is None:
        return None

    try:
        return int(env_value)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_value}'") from exc


def _load_plan(
    args: argparse.Namespace,
) -> Tuple[Config, List[int], List[TaskSpec], List[SkippedSpec]]:
    config = Config.from_yaml(args.models_config, args.experiments_config)
    if args.cv_folds is not None:
        config.cv_folds = args.cv_folds
    if args.max_eval_samples is not None:
        if args.max_eval_samples <= 0:
            raise ValueError("--max-eval-samples must be greater than 0")
        config.max_eval_samples = args.max_eval_samples

    if args.task_manifest:
        seeds, tasks, payload = load_manifest(Path(args.task_manifest))
        # Recover the fold count from the manifest so CV workers split data
        # into the same number of folds used to generate the task plan.
        derived_folds = (
            max((t.cv_fold for t in tasks if t.cv_fold is not None), default=-1) + 1
        )
        if derived_folds >= 2:
            config.cv_folds = derived_folds
        skipped = [
            SkippedSpec(
                seed=int(item["seed"]),
                model_key=str(item["model_key"]),
                condition_name=str(item["condition_name"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("skipped", [])
        ]
        return config, seeds, tasks, skipped

    seeds = resolve_requested_seeds(config, args.seed, args.seeds)
    tasks, skipped = build_tasks(
        config=config,
        seeds=seeds,
        models=args.models,
        conditions=args.conditions,
        model_provider=args.model_provider,
        cv_folds=config.cv_folds,
    )
    return config, seeds, tasks, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NL2ATL experiments locally or as a SLURM array."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys/names to run, or 'all' (default: all for provider).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Condition names to run, or 'all' (default: all).",
    )
    parser.add_argument(
        "--model_provider",
        "--model-provider",
        choices=["hf", "azure", "all"],
        default="hf",
        help="Which provider to run (default: hf).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--cv-folds",
        "--cv_folds",
        type=int,
        default=None,
        help=(
            "Number of stratified cross-validation folds for the robustness "
            "analysis (overrides experiments.yaml data.cv_folds). 0 disables CV."
        ),
    )
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--experiments_config", default="configs/experiments.yaml")
    parser.add_argument("--overwrite", "--force", action="store_true")
    parser.add_argument(
        "--count", action="store_true", help="Print task count and exit."
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List task plan and exit.",
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-manifest", default=None)
    parser.add_argument(
        "--train-max-steps",
        type=int,
        default=None,
        help="Limit training steps for OOM smoke tests; omitted for full training.",
    )
    parser.add_argument(
        "--max-eval-samples",
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "Smoke test only: evaluate just this many test examples, stratified "
            "to include both single- and multi-formula cases. Local runs only "
            "(not serialized into SLURM manifests). Omit for full evaluation."
        ),
    )

    parser.add_argument("--slurm", action="store_true", help="Submit as a SLURM array.")
    parser.add_argument(
        "--max-parallel-gpus",
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL_GPUS,
        help=(
            "Maximum concurrent one-GPU SLURM array tasks; omit or use 0 "
            "to run all selected tasks as soon as resources are available."
        ),
    )
    parser.add_argument("--job-name", default="nl2atl-experiments")
    parser.add_argument("--partition", default="A100")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", dest="time_limit", default="06:00:00")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--error", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest-dir", default="outputs/manifests")
    parser.add_argument("--script-path", default=None)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    parser.add_argument(
        "--env-setup",
        action="append",
        default=[],
        help="Shell line inserted into the SLURM script before execution.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    if args.slurm and args.task_id is not None:
        raise ValueError("Use either --slurm or --task-id, not both.")

    config, seeds, tasks, skipped = _load_plan(args)
    task_id = resolve_task_id(args.task_id)

    if args.train_max_steps is not None:
        if args.train_max_steps <= 0:
            raise ValueError("--train-max-steps must be greater than 0")
        os.environ["TRAIN_MAX_STEPS"] = str(args.train_max_steps)

    if args.count:
        print(len(tasks))
        return

    if args.list_tasks:
        _print_tasks(tasks, skipped)
        return

    if args.slurm:
        submit_slurm(args, seeds, tasks, skipped)
        return

    if task_id is not None:
        if task_id < 0 or task_id >= len(tasks):
            raise IndexError(f"Task index {task_id} out of range [0, {len(tasks) - 1}]")
        run_task(config, tasks[task_id], seeds, args.overwrite)
        return

    if not tasks:
        raise ValueError("No runnable experiment tasks were selected.")

    print(
        f"Running {len(tasks)} model/seed tasks locally "
        f"for seeds={seeds}. Use --slurm for SLURM parallelism."
    )
    if skipped:
        print(f"Skipping {len(skipped)} incompatible combinations.")

    for task in tasks:
        run_task(config, task, seeds, args.overwrite)


if __name__ == "__main__":
    main()
