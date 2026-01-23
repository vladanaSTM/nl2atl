"""Deprecated: Import from src.infra.azure instead."""

from src.infra.azure import *  # noqa
import warnings

warnings.warn(
    "Importing from src.azure_utils is deprecated. " "Use src.infra.azure instead.",
    DeprecationWarning,
    stacklevel=2,
)
