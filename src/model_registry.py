"""Deprecated: Import from src.models.registry instead."""

from src.models.registry import *  # noqa
import warnings

warnings.warn(
    "Importing from src.model_registry is deprecated. "
    "Use src.models.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)
