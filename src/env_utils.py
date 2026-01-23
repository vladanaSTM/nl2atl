"""Deprecated: Import from src.infra.env instead."""

from src.infra.env import *  # noqa
import warnings

warnings.warn(
    "Importing from src.env_utils is deprecated. " "Use src.infra.env instead.",
    DeprecationWarning,
    stacklevel=2,
)
