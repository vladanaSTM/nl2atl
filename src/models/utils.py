"""
Model helper utilities shared across CLI entry points and runners.
"""

from typing import Any, Mapping, Tuple

from ..constants import AZURE_PREFIX


def _get_model_names(entry: Any) -> Tuple[str, str]:
    """Extract short_name and name from a model entry (dict or object)."""
    if isinstance(entry, dict):
        short_name = entry.get("short_name", "")
        name = entry.get("name", "")
    else:
        short_name = getattr(entry, "short_name", "")
        name = getattr(entry, "name", "")
    return str(short_name), str(name)


def normalize_model_token(
    token: str, prefixes: Tuple[str, ...] = (AZURE_PREFIX,)
) -> str:
    """Normalize a model token by lowercasing and removing known prefixes."""
    token = str(token).lower()
    for prefix in prefixes:
        if token.startswith(prefix):
            token = token[len(prefix) :]
    return token


def resolve_model_key(
    model_arg: str,
    models: Mapping[str, Any],
    *,
    prefixes: Tuple[str, ...] = (AZURE_PREFIX,),
    require_mapping_entries: bool = False,
    match_key_lower: bool = True,
) -> str:
    """
    Resolve a user-provided model argument to a configuration key.

    Args:
        model_arg: User-provided model identifier
        models: Mapping of model keys to model configurations
        prefixes: Prefixes to strip during normalization
        require_mapping_entries: If True, only match dict entries
        match_key_lower: If True, allow case-insensitive key matching

    Returns:
        The resolved model key

    Raises:
        KeyError: If model cannot be resolved
    """
    # Direct match
    if model_arg in models:
        return model_arg

    needle = str(model_arg).lower()
    normalized_needle = normalize_model_token(model_arg, prefixes=prefixes)

    for key, entry in models.items():
        if require_mapping_entries and not isinstance(entry, dict):
            continue

        short_name, name = _get_model_names(entry)
        key_str = str(key)

        # Case-insensitive key match
        if match_key_lower and key_str.lower() == needle:
            return key

        # Match by short_name, name, or normalized versions
        if (
            short_name.lower() == needle
            or name.lower() == needle
            or normalize_model_token(short_name, prefixes=prefixes) == normalized_needle
            or normalize_model_token(name, prefixes=prefixes) == normalized_needle
            or normalize_model_token(key_str, prefixes=prefixes) == normalized_needle
        ):
            return key

    raise KeyError(f"Model '{model_arg}' not found in configuration")
