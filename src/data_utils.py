"""
Data loading, splitting, and augmentation utilities.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .infra.io import load_json, save_json


def _append_unique_output(outputs: List[str], value: Any) -> None:
    """Append one or more formula strings, preserving order and uniqueness.

    Supports both legacy fields such as ``output``/``output_1`` and the
    current unified schema ``outputs: [{"formula": "..."}, ...]``.
    """
    if isinstance(value, list):
        for item in value:
            _append_unique_output(outputs, item)
        return

    if isinstance(value, dict):
        # Current dataset schema: {"formula": "..."}.
        if "formula" in value:
            _append_unique_output(outputs, value.get("formula"))
            return
        # Defensive fallback for result-like dictionaries.
        for key in ("output", "expected", "gold", "reference"):
            if key in value:
                _append_unique_output(outputs, value.get(key))
        return

    if not isinstance(value, str) or not value.strip():
        return

    cleaned = value.strip()
    if cleaned not in outputs:
        outputs.append(cleaned)


def get_output_options(item: Dict[str, Any]) -> List[str]:
    """Return all acceptable output formulas for a dataset or result item."""
    outputs: List[str] = []
    for key in ("outputs", "expected_options", "gold_options", "reference_options"):
        _append_unique_output(outputs, item.get(key))

    for key in ("output", "output_2", "output_1", "expected", "gold", "reference"):
        _append_unique_output(outputs, item.get(key))

    return outputs


def get_preferred_output(item: Dict[str, Any]) -> Optional[str]:
    """Return the preferred output formula for a dataset item."""
    outputs = get_output_options(item)
    return outputs[0] if outputs else None


def normalize_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset rows to the in-memory schema used by the project."""
    if not isinstance(item.get("input"), str) or not item["input"].strip():
        raise ValueError("Dataset item is missing a non-empty 'input' field")

    normalized = dict(item)
    output_options = get_output_options(normalized)
    if not output_options:
        raise ValueError(
            "Dataset item is missing an output, output_1, or output_2 field"
        )

    normalized["outputs"] = output_options
    normalized["output"] = output_options[0]
    return normalized


def load_data(filepath: str) -> List[Dict]:
    """Load dataset from JSON file."""
    data = load_json(filepath)
    if not isinstance(data, list):
        raise ValueError(f"Expected dataset list in {filepath}")
    normalized = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} is not an object")
        try:
            normalized.append(normalize_dataset_item(item))
        except ValueError as exc:
            raise ValueError(f"Invalid dataset item {index}: {exc}") from exc
    return normalized


def save_data(data: List[Dict], filepath: str) -> None:
    """Save dataset to JSON file."""
    save_json(data, filepath)


def split_data(
    data: List[Dict],
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: List of data items
        train_size: Fraction of data for training
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if not data:
        return [], [], []

    split_total = train_size + val_size + test_size
    if not 0.999 <= split_total <= 1.001:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)

    train_count = int(round(len(shuffled) * train_size))
    val_count = int(round(len(shuffled) * val_size))
    train_count = max(0, min(len(shuffled), train_count))
    val_count = max(0, min(len(shuffled) - train_count, val_count))

    train_data = shuffled[:train_count]
    val_data = shuffled[train_count : train_count + val_count]
    test_data = shuffled[train_count + val_count :]

    return train_data, val_data, test_data


# Paraphrase templates for data augmentation
_PARAPHRASE_TEMPLATES = [
    (
        "can guarantee that",
        ["can ensure that", "guarantees that", "is able to guarantee that"],
    ),
    ("sooner or later", ["eventually", "at some point in the future", "finally"]),
    ("at the next step", ["in the next moment", "immediately after"]),
    ("will always", ["will forever", "will continuously", "will perpetually"]),
    ("will keep", ["will continue to", "will maintain"]),
]


def _apply_paraphrase(
    text: str,
    random_generator: Optional[random.Random] = None,
) -> str:
    """Apply the first matching paraphrase to the text."""
    chooser = random_generator or random
    for phrase, replacements in _PARAPHRASE_TEMPLATES:
        if phrase in text.lower():
            return re.sub(
                re.escape(phrase),
                chooser.choice(replacements),
                text,
                flags=re.IGNORECASE,
                count=1,
            )
    return text


def augment_data(
    data_list: List[Dict],
    augment_factor: int = 5,
    seed: Optional[int] = None,
    random_generator: Optional[random.Random] = None,
) -> List[Dict]:
    """
    Augment training data by applying paraphrasing.

    Args:
        data_list: Original data items
        augment_factor: Total copies of each item (including original)
        seed: Optional seed for deterministic augmentation independent of global state
        random_generator: Optional random generator supplied by the caller

    Returns:
        Augmented list with original + paraphrased versions
    """
    if seed is not None and random_generator is not None:
        raise ValueError("Pass either seed or random_generator, not both")

    local_random = random_generator or (
        random.Random(seed) if seed is not None else None
    )
    augmented = []

    for item in data_list:
        normalized_item = normalize_dataset_item(item)
        # Always include original
        augmented.append(normalized_item)

        # Add paraphrased versions
        for _ in range(augment_factor - 1):
            new_input = _apply_paraphrase(
                normalized_item["input"],
                random_generator=local_random,
            )
            augmented.append(
                {
                    "input": new_input,
                    "output": get_preferred_output(normalized_item),
                    "outputs": get_output_options(normalized_item),
                }
            )

    return augmented
