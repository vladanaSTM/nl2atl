"""Deprecated: Import from src.models.few_shot instead."""

from src.models.few_shot import *  # noqa
from src.models.few_shot import _format_few_shot_section  # noqa
import warnings

warnings.warn(
    "Importing from src.few_shot is deprecated. " "Use src.models.few_shot instead.",
    DeprecationWarning,
    stacklevel=2,
)
