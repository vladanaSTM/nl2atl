"""
Environment helpers.
"""

from dataclasses import dataclass
from typing import Optional
import os

from dotenv import load_dotenv

_ENV_LOADED = False


def load_env() -> None:
    """Load environment variables from .env once."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


@dataclass(frozen=True)
class AzureSettings:
    api_key: Optional[str]
    endpoint: Optional[str]
    api_version: Optional[str]
    use_cache: bool
    verify_ssl: bool
    api_model: str


def get_azure_settings(
    default_model: str, *, api_model_override: Optional[str] = None
) -> AzureSettings:
    """Load Azure-related settings from the environment."""
    load_env()
    api_key = os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_INFER_ENDPOINT")
    api_version = os.getenv("AZURE_API_VERSION")
    use_cache = os.getenv("AZURE_USE_CACHE", "true").lower() == "true"
    verify_ssl_env = os.getenv("AZURE_VERIFY_SSL", "false").lower()
    verify_ssl = verify_ssl_env in ["1", "true", "yes"]
    api_model = api_model_override or os.getenv("AZURE_INFER_MODEL") or default_model

    return AzureSettings(
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        use_cache=use_cache,
        verify_ssl=verify_ssl,
        api_model=api_model,
    )
