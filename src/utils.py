"""
Deprecated compatibility shim.

Use src/io_utils.py and src/azure_utils.py directly.
"""

from .io_utils import load_json, load_json_safe, load_yaml, save_json
from .azure_utils import AzureConfig

__all__ = [
    "load_json",
    "load_json_safe",
    "load_yaml",
    "save_json",
    "AzureConfig",
]
