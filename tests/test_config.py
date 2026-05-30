import yaml
import pytest

from src.config import Config, ModelConfig


def test_from_yaml_and_resolve_seeds(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    experiments_yaml = tmp_path / "experiments.yaml"

    models = {"m1": {"name": "m1", "short_name": "m1", "provider": "huggingface"}}
    models_yaml.write_text(yaml.dump({"models": models}))

    experiments = {
        "experiment": {"seed": 42, "seeds": [42], "num_seeds": 1},
        "data": {
            "path": "data/dataset_gold_no_difficulty.json",
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "augment_factor": 2,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "bf16": False,
        },
        "few_shot": {"num_examples": 2},
        "conditions": [
            {"name": "baseline_zero_shot", "finetuned": False, "few_shot": False}
        ],
    }
    experiments_yaml.write_text(yaml.dump(experiments))

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))
    assert cfg.seed == 42
    assert cfg.train_size == 0.8
    assert cfg.val_size == 0.1
    assert cfg.test_size == 0.1
    assert cfg.models
    assert cfg.conditions and cfg.conditions[0].name == "baseline_zero_shot"
    assert cfg.resolve_seeds() == [42]


def test_from_yaml_loads_paths_and_training_strategy(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    experiments_yaml = tmp_path / "experiments.yaml"

    models = {"m1": {"name": "m1", "short_name": "m1", "provider": "huggingface"}}
    models_yaml.write_text(yaml.dump({"models": models}))

    experiments = {
        "paths": {"output_dir": "outputs/tuning", "models_dir": "models/tuning"},
        "experiment": {"seed": 42, "seeds": [], "num_seeds": 1},
        "data": {
            "path": "data/dataset_gold_no_difficulty.json",
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "augment_factor": 2,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "bf16": False,
            "optim": "paged_adamw_8bit",
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "gradient_checkpointing": True,
            "packing": False,
        },
        "few_shot": {"num_examples": 2},
        "conditions": [
            {"name": "baseline_zero_shot", "finetuned": False, "few_shot": False}
        ],
    }
    experiments_yaml.write_text(yaml.dump(experiments))

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))

    assert cfg.output_dir == "outputs/tuning"
    assert cfg.models_dir == "models/tuning"
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.eval_strategy == "epoch"
    assert cfg.save_strategy == "epoch"
    assert cfg.gradient_checkpointing is True
    assert cfg.packing is False


def test_resolve_seeds_with_list(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    experiments_yaml = tmp_path / "experiments.yaml"

    models = {"m1": {"name": "m1", "short_name": "m1", "provider": "huggingface"}}
    models_yaml.write_text(yaml.dump({"models": models}))

    experiments = {
        "experiment": {"seed": 7, "seeds": [7, 11], "num_seeds": 1},
        "data": {
            "path": "data/dataset_gold_no_difficulty.json",
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "augment_factor": 2,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "bf16": False,
        },
        "few_shot": {"num_examples": 2},
        "conditions": [
            {"name": "baseline_zero_shot", "finetuned": False, "few_shot": False}
        ],
    }
    experiments_yaml.write_text(yaml.dump(experiments))

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))
    assert cfg.resolve_seeds() == [7, 11]


def test_resolve_seeds_with_num_seeds(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    experiments_yaml = tmp_path / "experiments.yaml"

    models = {"m1": {"name": "m1", "short_name": "m1", "provider": "huggingface"}}
    models_yaml.write_text(yaml.dump({"models": models}))

    experiments = {
        "experiment": {"seed": 5, "seeds": [], "num_seeds": 3},
        "data": {
            "path": "data/dataset_gold_no_difficulty.json",
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "augment_factor": 2,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "bf16": False,
        },
        "few_shot": {"num_examples": 2},
        "conditions": [
            {"name": "baseline_zero_shot", "finetuned": False, "few_shot": False}
        ],
    }
    experiments_yaml.write_text(yaml.dump(experiments))

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))
    assert cfg.resolve_seeds() == [5, 6, 7]


def test_legacy_holdout_split_config_is_converted(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    experiments_yaml = tmp_path / "experiments.yaml"

    models = {"m1": {"name": "m1", "short_name": "m1", "provider": "huggingface"}}
    models_yaml.write_text(yaml.dump({"models": models}))

    experiments = {
        "experiment": {"seed": 5, "seeds": [], "num_seeds": 1},
        "data": {
            "path": "data/dataset_gold_no_difficulty.json",
            "test_size": 0.3,
            "val_size": 2 / 3,
            "augment_factor": 2,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "bf16": False,
        },
        "few_shot": {"num_examples": 2},
        "conditions": [
            {"name": "baseline_zero_shot", "finetuned": False, "few_shot": False}
        ],
    }
    experiments_yaml.write_text(yaml.dump(experiments))

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))
    assert cfg.train_size == pytest.approx(0.7)
    assert cfg.val_size == pytest.approx(0.2)
    assert cfg.test_size == pytest.approx(0.1)


def test_config_validate_missing_values(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "azure-gpt": {
                        "name": "gpt-4",
                        "short_name": "gpt-4",
                        "provider": "azure",
                    }
                }
            }
        )
    )

    experiments_yaml = tmp_path / "experiments.yaml"
    experiments_yaml.write_text(
        yaml.dump(
            {
                "experiment": {"seed": None, "seeds": [], "num_seeds": 1},
                "data": {
                    "path": "data/dataset_gold_no_difficulty.json",
                    "train_size": None,
                    "test_size": None,
                    "val_size": None,
                    "augment_factor": None,
                },
                "training": {
                    "num_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "bf16": False,
                },
                "few_shot": {"num_examples": 2},
                "conditions": [
                    {
                        "name": "baseline_zero_shot",
                        "finetuned": False,
                        "few_shot": False,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError) as exc:
        Config.from_yaml(str(models_yaml), str(experiments_yaml))
    assert "experiment.seed" in str(exc.value)


def test_get_model_and_is_azure(tmp_path):
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "azure-gpt": {
                        "name": "gpt-4",
                        "short_name": "gpt-4",
                        "provider": "azure",
                    }
                }
            }
        )
    )

    experiments_yaml = tmp_path / "experiments.yaml"
    experiments_yaml.write_text(
        yaml.dump(
            {
                "experiment": {"seed": 1, "seeds": [], "num_seeds": 1},
                "data": {
                    "path": "data/dataset_gold_no_difficulty.json",
                    "train_size": 0.8,
                    "val_size": 0.1,
                    "test_size": 0.1,
                    "augment_factor": 2,
                },
                "training": {
                    "num_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "bf16": False,
                },
                "few_shot": {"num_examples": 2},
                "conditions": [
                    {
                        "name": "baseline_zero_shot",
                        "finetuned": False,
                        "few_shot": False,
                    }
                ],
            }
        )
    )

    cfg = Config.from_yaml(str(models_yaml), str(experiments_yaml))
    model = cfg.get_model("azure-gpt")
    assert model.name == "gpt-4"
    assert model.is_azure

    local = ModelConfig(name="local", short_name="local", provider="huggingface")
    assert not local.is_azure

    with pytest.raises(KeyError):
        cfg.get_model("missing")
