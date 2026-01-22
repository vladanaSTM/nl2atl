"""
Model helper utilities shared across CLI entrypoints and runners.
"""

from typing import Any, Mapping, Tuple


def _get_model_names(entry: Any) -> Tuple[str, str]:
    if isinstance(entry, dict):
        short_name = entry.get("short_name", "")
        name = entry.get("name", "")
    else:
        short_name = getattr(entry, "short_name", "")
        name = getattr(entry, "name", "")
    return str(short_name), str(name)


def normalize_model_token(token: str, prefixes=("azure-",)) -> str:
    token = str(token).lower()
    for prefix in prefixes:
        if token.startswith(prefix):
            token = token[len(prefix) :]
    return token


def resolve_model_key(
    model_arg: str,
    models: Mapping[str, Any],
    *,
    prefixes=("azure-",),
    require_mapping_entries: bool = False,
    match_key_lower: bool = True,
) -> str:
    if model_arg in models:
        return model_arg

    needle = str(model_arg).lower()
    normalized_needle = normalize_model_token(model_arg, prefixes=prefixes)

    for key, entry in models.items():
        if require_mapping_entries and not isinstance(entry, dict):
            continue

        short_name, name = _get_model_names(entry)
        key_str = str(key)

        if match_key_lower and key_str.lower() == needle:
            return key

        if (
            short_name.lower() == needle
            or name.lower() == needle
            or normalize_model_token(short_name, prefixes=prefixes) == normalized_needle
            or normalize_model_token(name, prefixes=prefixes) == normalized_needle
            or normalize_model_token(key_str, prefixes=prefixes) == normalized_needle
        ):
            return key

    raise KeyError(model_arg)
