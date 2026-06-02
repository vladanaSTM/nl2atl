"""
Shared constants used across the codebase.
"""

from enum import Enum


class Provider(str, Enum):
    """Model provider types."""

    HUGGINGFACE = "huggingface"
    AZURE = "azure"


class ModelType(str, Enum):
    """Model family types for prompt formatting."""

    QWEN = "qwen"
    PHI3 = "phi3"
    MISTRAL = "mistral"
    GENERIC = "generic"


# Default paths
DEFAULT_DATA_PATH = "./data/dataset_gold.json"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_MODELS_DIR = "./models"
DEFAULT_LLM_EVAL_DIR = "outputs/LLM-evaluation"
