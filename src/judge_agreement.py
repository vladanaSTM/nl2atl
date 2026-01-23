"""Deprecated: Import from src.evaluation.judge_agreement instead."""

from src.evaluation.judge_agreement import *  # noqa
import warnings

warnings.warn(
    "Importing from src.judge_agreement is deprecated. "
    "Use src.evaluation.judge_agreement instead.",
    DeprecationWarning,
    stacklevel=2,
)
