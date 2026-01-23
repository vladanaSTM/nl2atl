"""Deprecated: Import from src.evaluation.exact_match instead."""

from src.evaluation.exact_match import *  # noqa
import warnings

warnings.warn(
    "Importing from src.exact_match_evaluator is deprecated. "
    "Use src.evaluation.exact_match instead.",
    DeprecationWarning,
    stacklevel=2,
)
