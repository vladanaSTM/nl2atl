"""Consolidated file I/O utilities for JSON and YAML operations."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save data to a JSON file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)


def load_json_safe(path: Union[str, Path], default: Any = None) -> Any:
    """Load a JSON file, returning default if file doesn't exist or is invalid."""
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    try:
        return load_json(path)
    except (json.JSONDecodeError, IOError):
        return default if default is not None else {}
