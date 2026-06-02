import pytest
from argparse import Namespace
from pathlib import Path

from src.cli.run_experiments import (
    _render_sbatch_script,
    build_tasks,
    format_slurm_array_range,
    select_models,
)
from src.config import Config, ExperimentCondition, ModelConfig


def _config():
    config = Config(
        seed=42,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        augment_factor=2,
    )
    config.models = {
        "qwen-3b": ModelConfig(
            name="Qwen/Qwen2.5-3B-Instruct",
            short_name="qwen-3b",
            provider="huggingface",
            params_b=3,
        ),
        "gpt-5.4": ModelConfig(
            name="gpt-5.4",
            short_name="gpt-5.4",
            provider="azure",
        ),
        "gpt-5.2": ModelConfig(
            name="gpt-5.2",
            short_name="gpt-5.2",
            provider="azure",
            generation_enabled=False,
        ),
        "DeepSeek-V3.2": ModelConfig(
            name="DeepSeek-V3.2",
            short_name="ds-v3.2",
            provider="azure",
            generation_enabled=False,
        ),
    }
    config.conditions = [
        ExperimentCondition(
            name="baseline_zero_shot",
            finetuned=False,
            few_shot=False,
        ),
        ExperimentCondition(
            name="baseline_few_shot",
            finetuned=False,
            few_shot=True,
        ),
        ExperimentCondition(
            name="finetuned_zero_shot",
            finetuned=True,
            few_shot=False,
        ),
        ExperimentCondition(
            name="finetuned_few_shot",
            finetuned=True,
            few_shot=True,
        ),
    ]
    return config


def test_build_tasks_groups_conditions_by_model_and_seed():
    tasks, skipped = build_tasks(
        config=_config(),
        seeds=[42, 43],
        models=["qwen-3b"],
        conditions=["all"],
        model_provider="hf",
    )

    assert len(tasks) == 2
    assert tasks[0].seed == 42
    assert tasks[0].model_key == "qwen-3b"
    assert tasks[0].condition_names == [
        "baseline_zero_shot",
        "baseline_few_shot",
        "finetuned_zero_shot",
        "finetuned_few_shot",
    ]
    assert skipped == []


def test_build_tasks_skips_finetuning_for_azure_models():
    tasks, skipped = build_tasks(
        config=_config(),
        seeds=[42],
        models=["all"],
        conditions=["all"],
        model_provider="all",
    )

    model_keys = [task.model_key for task in tasks]

    assert "gpt-5.2" not in model_keys
    assert "DeepSeek-V3.2" not in model_keys

    azure_task = next(task for task in tasks if task.model_key == "gpt-5.4")
    assert azure_task.condition_names == ["baseline_zero_shot", "baseline_few_shot"]
    assert [item.condition_name for item in skipped] == [
        "finetuned_zero_shot",
        "finetuned_few_shot",
    ]


def test_select_models_keeps_judge_only_models_out_of_generation_all():
    assert select_models(_config(), ["all"], "azure") == ["gpt-5.4"]


def test_select_models_rejects_explicit_judge_only_generation_model():
    with pytest.raises(ValueError, match="reserved for judging"):
        select_models(_config(), ["gpt-5.2"], "azure")


def test_format_slurm_array_range_is_uncapped_by_default():
    assert format_slurm_array_range(task_count=5, max_parallel_gpus=None) == "0-4"


def test_format_slurm_array_range_allows_zero_as_uncapped():
    assert format_slurm_array_range(task_count=5, max_parallel_gpus=0) == "0-4"


def test_format_slurm_array_range_caps_parallel_tasks_when_requested():
    assert format_slurm_array_range(task_count=5, max_parallel_gpus=2) == "0-4%2"


def test_format_slurm_array_range_omits_unneeded_cap():
    assert format_slurm_array_range(task_count=5, max_parallel_gpus=5) == "0-4"


def test_format_slurm_array_range_rejects_invalid_counts():
    with pytest.raises(ValueError):
        format_slurm_array_range(task_count=0, max_parallel_gpus=2)
    with pytest.raises(ValueError):
        format_slurm_array_range(task_count=1, max_parallel_gpus=-1)


def test_render_sbatch_script_includes_tuning_controls(tmp_path):
    args = Namespace(
        logs_dir="logs",
        output=None,
        error=None,
        max_parallel_gpus=None,
        models_config="configs/models.yaml",
        experiments_config="configs/experiments.yaml",
        overwrite=False,
        train_max_steps=20,
        job_name="nl2atl-tune",
        partition="A100",
        gres="gpu:1",
        cpus_per_task=8,
        mem="32G",
        time_limit="02:00:00",
        sbatch_arg=[],
        python_bin="python3",
        env_setup=[
            "module load python/3.12.3 cuda/12.4.1",
            "source .venv/bin/activate",
        ],
    )

    script = _render_sbatch_script(
        args=args,
        repo_root=Path("/repo"),
        manifest_path=tmp_path / "manifest.json",
        task_count=3,
    )

    assert "#SBATCH --array=0-2" in script
    assert "module load python/3.12.3 cuda/12.4.1" in script
    assert "source .venv/bin/activate" in script
    assert "export TRAIN_MAX_STEPS=20" in script
