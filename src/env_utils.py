"""
Environment helpers.

Deprecated: Azure configuration helpers moved to src.azure_utils.AzureConfig.
"""

import os

from dotenv import load_dotenv

_ENV_LOADED = False


def load_env() -> None:
    """Load environment variables from .env once."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True
