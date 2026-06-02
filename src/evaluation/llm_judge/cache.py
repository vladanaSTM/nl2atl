"""Caching layer for LLM judge evaluations."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...infra.io import load_json_safe, save_json


def _normalize_cache_text(value: Any) -> str:
    """Normalize cache text fields without changing formula semantics."""
    return " ".join(str(value or "").split())


def _canonical_gold_options(gold: Any) -> List[str]:
    """Return accepted gold formulas as a sorted, de-duplicated list."""
    options: List[str] = []

    def append(value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                append(item)
            return

        text = _normalize_cache_text(value)
        if text and text not in options:
            options.append(text)

    append(gold)
    return sorted(options)


class JudgeCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = load_json_safe(self.cache_path, default={})

    def get_cache_key(
        self,
        input_text: str,
        gold: Any,
        prediction: str,
        judge_model: str,
        prompt_version: str,
    ) -> str:
        """Generate deterministic cache key."""
        key_payload = {
            "input_text": _normalize_cache_text(input_text),
            "gold_options": _canonical_gold_options(gold),
            "prediction": _normalize_cache_text(prediction),
            "judge_model": _normalize_cache_text(judge_model),
            "prompt_version": _normalize_cache_text(prompt_version),
        }
        key_json = json.dumps(
            key_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result."""
        value = self._cache.get(key)
        return value if isinstance(value, dict) else None

    def set(self, key: str, result: Dict[str, Any]) -> None:
        """Store result in cache."""
        self._cache[key] = result
        self.save()

    def save(self) -> None:
        save_json(self._cache, self.cache_path)
