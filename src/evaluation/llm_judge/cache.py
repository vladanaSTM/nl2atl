"""Caching layer for LLM judge evaluations."""

import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from ...io_utils import load_json_safe, save_json


class JudgeCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = load_json_safe(self.cache_path, default={})

    def get_cache_key(
        self,
        input_text: str,
        gold: str,
        prediction: str,
        judge_model: str,
        prompt_version: str,
    ) -> str:
        """Generate deterministic cache key."""
        key_payload = [input_text, gold, prediction, judge_model, prompt_version]
        return hashlib.sha256(str(key_payload).encode("utf-8")).hexdigest()

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
