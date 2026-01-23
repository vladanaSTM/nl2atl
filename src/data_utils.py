"""
Data loading, splitting, and augmentation utilities.
"""

import random
import re
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from .infra.io import load_json, save_json


def load_data(filepath: str) -> List[Dict]:
    """Load dataset from JSON file."""
    return load_json(filepath)


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
    Split data into train, validation, and test sets with stratification.

    Args:
        data: List of data items with 'difficulty' field for stratification
        test_size: Fraction of data for test+validation combined
        val_size: Fraction of test+validation to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    labels = [item.get("difficulty") for item in data]

    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=val_size,
        random_state=seed,
        stratify=temp_labels,
    )

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
        # Always include original
        augmented.append(item)

        # Add paraphrased versions
        for _ in range(augment_factor - 1):
            new_input = _apply_paraphrase(item["input"])
            augmented.append(
                {
                    "input": new_input,
                    "output": item["output"],
                }
            )

    return augmented
