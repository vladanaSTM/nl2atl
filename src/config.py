"""
Configuration management for experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .infra.io import load_yaml
from .constants import (
    Provider,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODELS_DIR,
)


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    short_name: str
    provider: str = Provider.HUGGINGFACE
    api_model: Optional[str] = None
    max_seq_length: int = 512
    load_in_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    train_batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    target_modules: List[str] = field(default_factory=list)
    params_b: Optional[float] = None

    @property
    def is_azure(self) -> bool:
        """Check if this model uses Azure provider."""
        return self.provider.lower() == Provider.AZURE


@dataclass
class ExperimentCondition:
    """Defines an experimental condition."""

    name: str
    finetuned: bool
    few_shot: bool


@dataclass
class Config:
    """Main configuration container."""

    # Paths
    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    models_dir: str = DEFAULT_MODELS_DIR

    # Data settings
    test_size: Optional[float] = None
    val_size: Optional[float] = None
    augment_factor: Optional[int] = None
    seed: Optional[int] = None
    seeds: List[int] = field(default_factory=list)
    num_seeds: int = 1

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

    # W&B settings
    wandb_project: str = "atl-formula-generation"
    wandb_entity: Optional[str] = None

    # Model and experiment configs
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    conditions: List[ExperimentCondition] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, models_path: str, experiments_path: str) -> "Config":
        """Load configuration from YAML files."""
        models_cfg = load_yaml(models_path)
        exp_cfg = load_yaml(experiments_path)

        # Extract experiment settings
        exp_settings = exp_cfg.get("experiment", {})
        seed = exp_settings.get("seed")
        seeds = exp_settings.get("seeds") or []
        num_seeds = exp_settings.get("num_seeds", 1)

        if seeds:
            seed = seeds[0]

        # Build config
        config = cls(
            data_path=exp_cfg["data"]["path"],
            test_size=exp_cfg["data"]["test_size"],
            val_size=exp_cfg["data"]["val_size"],
            augment_factor=exp_cfg["data"]["augment_factor"],
            seed=seed,
            seeds=seeds,
            num_seeds=num_seeds,
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
        for model_key, model_data in models_cfg.get("models", {}).items():
            config.models[model_key] = ModelConfig(**model_data)

        # Load conditions
        for cond in exp_cfg.get("conditions", []):
            config.conditions.append(ExperimentCondition(**cond))

        # Validate required fields
        config._validate()

        return config

    def _validate(self) -> None:
        """Validate that required configuration values are present."""
        missing = []
        if self.seed is None:
            missing.append("experiment.seed")
        if self.test_size is None:
            missing.append("data.test_size")
        if self.val_size is None:
            missing.append("data.val_size")
        if self.augment_factor is None:
            missing.append("data.augment_factor")

        if missing:
            raise ValueError(
                f"Missing required config values in experiments.yaml: {', '.join(missing)}"
            )

    def resolve_seeds(self) -> List[int]:
        """Get the list of seeds to run experiments with."""
        if self.seeds:
            return list(self.seeds)
        if self.num_seeds > 1:
            return [self.seed + i for i in range(self.num_seeds)]
        return [self.seed]

    def get_model(self, model_key: str) -> ModelConfig:
        """Get a model configuration by key."""
        if model_key not in self.models:
            raise KeyError(f"Model '{model_key}' not found in configuration")
        return self.models[model_key]
