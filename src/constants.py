"""
Shared constants used across the codebase.
"""

from enum import Enum
from typing import FrozenSet


class Provider(str, Enum):
    """Model provider types."""

    HUGGINGFACE = "huggingface"
    AZURE = "azure"


class ModelType(str, Enum):
    """Model family types for prompt formatting."""

    QWEN = "qwen"
    PHI3 = "phi3"
    MISTRAL = "mistral"
    LLAMA = "llama"
    GEMMA = "gemma"
    GENERIC = "generic"


# Temporal operators used in ATL formulas
TEMPORAL_OPERATORS: FrozenSet[str] = frozenset({"G", "F", "X", "U", "W", "R"})

# Default paths
DEFAULT_DATA_PATH = "./data/dataset.json"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_MODELS_DIR = "./models"
DEFAULT_MODELS_CONFIG = "configs/models.yaml"
DEFAULT_EXPERIMENTS_CONFIG = "configs/experiments.yaml"
DEFAULT_PREDICTIONS_DIR = "outputs/model_predictions"
DEFAULT_LLM_EVAL_DIR = "outputs/LLM-evaluation"

# Azure prefix for model name normalization
AZURE_PREFIX = "azure-"
