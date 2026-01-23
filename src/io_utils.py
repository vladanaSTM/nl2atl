"""Deprecated: Import from src.infra.io instead."""

from src.infra.io import *  # noqa
import warnings

warnings.warn(
    "Importing from src.io_utils is deprecated. " "Use src.infra.io instead.",
    DeprecationWarning,
    stacklevel=2,
)
