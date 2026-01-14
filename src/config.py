"""
Configuration management for experiments.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    name: str
    short_name: str
    provider: str = "huggingface"  # huggingface or elysium
    api_model: Optional[str] = None  # Optional override for remote model id
    max_seq_length: int = 512
    load_in_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    target_modules: List[str] = field(default_factory=list)


@dataclass
class ExperimentCondition:
    name: str
    finetuned: bool
    few_shot: bool


@dataclass
class Config:
    # Paths
    data_path: str = "./data/dataset.json"
    output_dir: str = "./outputs"
    models_dir: str = "./models"

    # Data settings
    test_size: float = 0.20
    val_size: float = 0.50
    augment_factor: int = 10
    seed: int = 42

    # Training settings
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    bf16: bool = True

    # Few-shot settings
    num_few_shot_examples: int = 5

    # Wandb settings
    wandb_project: str = "atl-formula-generation"
    wandb_entity: Optional[str] = None

    # Model configs
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Experiment conditions
    conditions: List[ExperimentCondition] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, models_path: str, experiments_path: str) -> "Config":
        models_cfg = load_yaml(models_path)
        exp_cfg = load_yaml(experiments_path)

        config = cls(
            data_path=exp_cfg["data"]["path"],
            test_size=exp_cfg["data"]["test_size"],
            val_size=exp_cfg["data"]["val_size"],
            augment_factor=exp_cfg["data"]["augment_factor"],
            seed=exp_cfg["experiment"]["seed"],
            num_epochs=exp_cfg["training"]["num_epochs"],
            batch_size=exp_cfg["training"]["batch_size"],
            gradient_accumulation_steps=exp_cfg["training"][
                "gradient_accumulation_steps"
            ],
            learning_rate=exp_cfg["training"]["learning_rate"],
            weight_decay=exp_cfg["training"]["weight_decay"],
            warmup_ratio=exp_cfg["training"]["warmup_ratio"],
            bf16=exp_cfg["training"]["bf16"],
            num_few_shot_examples=exp_cfg["few_shot"]["num_examples"],
            wandb_project=exp_cfg["wandb"]["project"],
            wandb_entity=exp_cfg["wandb"]["entity"],
        )

        # Load models
        for model_key, model_data in models_cfg["models"].items():
            config.models[model_key] = ModelConfig(**model_data)

        # Load conditions
        for cond in exp_cfg["conditions"]:
            config.conditions.append(ExperimentCondition(**cond))

        return config
