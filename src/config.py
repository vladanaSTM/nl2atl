"""Configuration management for experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    revision: Optional[str] = None
    max_seq_length: int = 512
    load_in_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
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
    train_size: Optional[float] = None
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
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    tf32: bool = True
    packing: bool = False

    # Few-shot settings
    num_few_shot_examples: int = 5

    # Model and experiment configs
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    conditions: List[ExperimentCondition] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, models_path: str, experiments_path: str) -> "Config":
        """Load configuration from YAML files."""
        models_cfg = load_yaml(models_path)
        exp_cfg = load_yaml(experiments_path)

        paths = exp_cfg.get("paths", {})

        # Extract experiment settings
        exp_settings = exp_cfg.get("experiment", {})
        seed = exp_settings.get("seed")
        seeds = exp_settings.get("seeds") or []
        num_seeds = exp_settings.get("num_seeds", 1)

        if seeds:
            seed = seeds[0]

        data_settings = exp_cfg["data"]
        train_size, val_size, test_size = cls._load_split_sizes(data_settings)

        # Build config
        config = cls(
            data_path=paths.get("data_path", data_settings["path"]),
            output_dir=paths.get("output_dir", DEFAULT_OUTPUT_DIR),
            models_dir=paths.get("models_dir", DEFAULT_MODELS_DIR),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            augment_factor=data_settings["augment_factor"],
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
            optim=exp_cfg["training"].get("optim", cls.optim),
            lr_scheduler_type=exp_cfg["training"].get(
                "lr_scheduler_type", cls.lr_scheduler_type
            ),
            eval_strategy=exp_cfg["training"].get("eval_strategy", cls.eval_strategy),
            save_strategy=exp_cfg["training"].get("save_strategy", cls.save_strategy),
            save_total_limit=exp_cfg["training"].get(
                "save_total_limit", cls.save_total_limit
            ),
            max_grad_norm=exp_cfg["training"].get("max_grad_norm", cls.max_grad_norm),
            gradient_checkpointing=exp_cfg["training"].get(
                "gradient_checkpointing", cls.gradient_checkpointing
            ),
            dataloader_num_workers=exp_cfg["training"].get(
                "dataloader_num_workers", cls.dataloader_num_workers
            ),
            dataloader_pin_memory=exp_cfg["training"].get(
                "dataloader_pin_memory", cls.dataloader_pin_memory
            ),
            group_by_length=exp_cfg["training"].get(
                "group_by_length", cls.group_by_length
            ),
            tf32=exp_cfg["training"].get("tf32", cls.tf32),
            packing=exp_cfg["training"].get("packing", cls.packing),
            num_few_shot_examples=exp_cfg["few_shot"]["num_examples"],
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

    @staticmethod
    def _load_split_sizes(
        data_settings: Dict,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Load explicit split sizes, converting old holdout-style configs."""
        if "train_size" in data_settings:
            return (
                data_settings["train_size"],
                data_settings["val_size"],
                data_settings["test_size"],
            )

        holdout_size = data_settings.get("test_size")
        val_share_of_holdout = data_settings.get("val_size")
        if holdout_size is None or val_share_of_holdout is None:
            return None, None, None

        val_size = holdout_size * val_share_of_holdout
        test_size = holdout_size - val_size
        train_size = 1.0 - holdout_size
        return train_size, val_size, test_size

    def _validate(self) -> None:
        """Validate that required configuration values are present."""
        missing = []
        if self.seed is None:
            missing.append("experiment.seed")
        if self.train_size is None:
            missing.append("data.train_size")
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

        split_total = self.train_size + self.val_size + self.test_size
        if not 0.999 <= split_total <= 1.001:
            raise ValueError(
                "data.train_size, data.val_size, and data.test_size must sum to 1.0"
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
