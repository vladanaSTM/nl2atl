"""Deprecated: Import from src.experiment instead."""

from src.experiment import *  # noqa
import warnings

warnings.warn(
    "Importing from src.experiment_runner is deprecated. "
    "Use src.experiment instead.",
    DeprecationWarning,
    stacklevel=2,
)
