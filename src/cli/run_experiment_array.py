#!/usr/bin/env python
"""Run a single (seed, model, condition) experiment from a SLURM array.

This command maps a task index to a concrete experiment, so you can
parallelize run_all_experiments across multiple SLURM jobs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

from ..config import Config, ExperimentCondition
from ..constants import Provider
from ..experiment import ExperimentRunner
from ..models.utils import resolve_model_key


@dataclass(frozen=True)
class TaskSpec:
    index: int
    seed: int
    model_key: str
    condition: ExperimentCondition


def _filter_models(
    config: Config, models: Optional[List[str]], model_provider: str
) -> List[str]:
    if models is None:
        models = list(config.models.keys())

    if model_provider not in ("hf", "azure", "all"):
        raise ValueError(
            f"Invalid model_provider '{model_provider}'. Use 'hf', 'azure', or 'all'."
        )

    if model_provider == "all":
        return models

    provider = Provider.HUGGINGFACE if model_provider == "hf" else Provider.AZURE
    filtered = []
    for model_key in models:
        if model_key not in config.models:
            raise KeyError(f"Model '{model_key}' not found in configuration")
        model_cfg = config.models[model_key]
        if model_cfg.provider.lower() == provider.value:
            filtered.append(model_key)
    return filtered


def _filter_conditions(
    config: Config, conditions: Optional[List[str]]
) -> List[ExperimentCondition]:
    if conditions is None:
        return list(config.conditions)

    selected = [c for c in config.conditions if c.name in conditions]
    if not selected:
        raise ValueError(f"No conditions matched: {conditions}")
    return selected


def build_tasks(
    config: Config,
    seeds: List[int],
    models: Optional[List[str]],
    conditions: Optional[List[str]],
    model_provider: str,
) -> List[TaskSpec]:
    model_keys = _filter_models(config, models, model_provider)
    run_conditions = _filter_conditions(config, conditions)

    tasks: List[TaskSpec] = []
    idx = 0
    for seed in seeds:
        for model_key in model_keys:
            for condition in run_conditions:
                tasks.append(TaskSpec(idx, seed, model_key, condition))
                idx += 1
    return tasks


def resolve_task_id(explicit_task_id: Optional[int]) -> int:
    if explicit_task_id is not None:
        return explicit_task_id

    env_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_id is None:
        raise ValueError("Missing --task-id and SLURM_ARRAY_TASK_ID is not set")

    try:
        return int(env_id)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_id}'") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment for a given SLURM array task index."
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument(
        "--model_provider",
        choices=["hf", "azure", "all"],
        default="hf",
        help="Which model provider to run (default: hf).",
    )
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--experiments_config", default="configs/experiments.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List task indices and exit.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the total number of tasks and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even if output exists for this task.",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.models_config, args.experiments_config)
    seeds = config.resolve_seeds()
    tasks = build_tasks(
        config, seeds, args.models, args.conditions, args.model_provider
    )

    if args.count:
        print(len(tasks))
        return

    if args.list_tasks:
        for task in tasks:
            print(
                f"{task.index}\tseed={task.seed}\tmodel={task.model_key}\tcondition={task.condition.name}"
            )
        return

    task_id = resolve_task_id(args.task_id)
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"Task index {task_id} out of range [0, {len(tasks) - 1}]")

    task = tasks[task_id]
    config.seed = task.seed
    config.seeds = list(seeds)

    runner = ExperimentRunner(config)

    resolved_key = resolve_model_key(task.model_key, config.models)
    run_name = runner._build_run_name(resolved_key, task.condition)
    result_path = runner.reporter.get_result_path(run_name)

    if result_path.exists() and not args.overwrite:
        print(f"Skipping {run_name}: results exist at {result_path}")
        return

    print(
        f"Running task {task.index}: seed={task.seed}, model={resolved_key}, condition={task.condition.name}"
    )
    runner.run_single_experiment(resolved_key, task.condition)


if __name__ == "__main__":
    main()
