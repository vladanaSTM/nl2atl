"""Deprecated: Import from src.models.utils instead."""

from src.models.utils import *  # noqa
import warnings

warnings.warn(
    "Importing from src.model_utils is deprecated. " "Use src.models.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
