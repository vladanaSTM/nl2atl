"""Model loading and management subpackage."""

from .utils import normalize_model_token, resolve_model_key
from .few_shot import get_few_shot_examples, get_system_prompt, format_prompt

__all__ = [
    "load_model",
    "get_model_type",
    "clear_gpu_memory",
    "generate",
    "normalize_model_token",
    "resolve_model_key",
    "get_few_shot_examples",
    "get_system_prompt",
    "format_prompt",
]


def __getattr__(name: str):
    if name in {"load_model", "get_model_type", "clear_gpu_memory", "generate"}:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
