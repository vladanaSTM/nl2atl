"""Deprecated: Import from src.evaluation.llm_judge instead."""

from src.evaluation.llm_judge import *  # noqa
import warnings

warnings.warn(
    "Importing from src.llm_judge is deprecated. "
    "Use src.evaluation.llm_judge instead.",
    DeprecationWarning,
    stacklevel=2,
)
