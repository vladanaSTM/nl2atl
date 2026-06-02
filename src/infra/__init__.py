"""Infrastructure utilities subpackage."""

from .io import load_json, load_yaml, save_json

__all__ = [
    "AzureConfig",
    "AzureClient",
    "load_json",
    "load_yaml",
    "save_json",
    "load_env",
]


def __getattr__(name):
    if name in {"AzureConfig", "AzureClient"}:
        from .azure import AzureConfig, AzureClient

        return {"AzureConfig": AzureConfig, "AzureClient": AzureClient}[name]
    if name == "load_env":
        from .env import load_env

        return load_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
