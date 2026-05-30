"""Environment helpers."""

import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_ENV_LOADED = False


def load_env() -> None:
    """Load environment variables from .env once."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        if load_dotenv is not None:
            load_dotenv()
        _ENV_LOADED = True
