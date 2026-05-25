"""
Data loading, splitting, and augmentation utilities.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .infra.io import load_json, save_json


def get_preferred_output(item: Dict[str, Any]) -> Optional[str]:
    """Return the preferred output formula for a dataset item."""
    for key in ("output", "output_2", "output_1"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def normalize_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset rows to the in-memory schema used by the project."""
    normalized = dict(item)
    preferred_output = get_preferred_output(normalized)
    if preferred_output is not None:
        normalized["output"] = preferred_output
    return normalized


def load_data(filepath: str) -> List[Dict]:
    """Load dataset from JSON file."""
    data = load_json(filepath)
    if not isinstance(data, list):
        raise ValueError(f"Expected dataset list in {filepath}")
    return [normalize_dataset_item(item) for item in data if isinstance(item, dict)]


def save_data(data: List[Dict], filepath: str) -> None:
    """Save dataset to JSON file."""
    save_json(data, filepath)


def split_data(
    data: List[Dict],
    test_size: float = 0.2,
    val_size: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: List of data items
        test_size: Fraction of data for test+validation combined
        val_size: Fraction of test+validation to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if not data:
        return [], [], []

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)

    temp_count = int(round(len(shuffled) * test_size))
    temp_count = max(0, min(len(shuffled), temp_count))
    train_count = len(shuffled) - temp_count

    train_data = shuffled[:train_count]
    temp_data = shuffled[train_count:]

    val_count = int(round(len(temp_data) * val_size))
    val_count = max(0, min(len(temp_data), val_count))

    val_data = temp_data[:val_count]
    test_data = temp_data[val_count:]

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


def _apply_paraphrase(text: str) -> str:
    """Apply the first matching paraphrase to the text."""
    for phrase, replacements in _PARAPHRASE_TEMPLATES:
        if phrase in text.lower():
            return re.sub(
                re.escape(phrase),
                random.choice(replacements),
                text,
                flags=re.IGNORECASE,
                count=1,
            )
    return text


def augment_data(data_list: List[Dict], augment_factor: int = 5) -> List[Dict]:
    """
    Augment training data by applying paraphrasing.

    Args:
        data_list: Original data items
        augment_factor: Total copies of each item (including original)

    Returns:
        Augmented list with original + paraphrased versions
    """
    augmented = []

    for item in data_list:
        normalized_item = normalize_dataset_item(item)
        # Always include original
        augmented.append(normalized_item)

        # Add paraphrased versions
        for _ in range(augment_factor - 1):
            new_input = _apply_paraphrase(normalized_item["input"])
            augmented.append(
                {
                    "input": new_input,
                    "output": get_preferred_output(normalized_item),
                }
            )

    return augmented
