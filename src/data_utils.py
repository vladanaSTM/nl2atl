"""
Data utilities.
"""

import json
import re
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: List[Dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def split_data(
    data: List[Dict], test_size: float = 0.2, val_size: float = 0.5, seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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


def augment_data(data_list: List[Dict], augment_factor: int = 5) -> List[Dict]:
    augmented = []

    paraphrase_templates = [
        (
            "can guarantee that",
            [
                "can ensure that",
                "guarantees that",
                "is able to guarantee that",
            ],
        ),
        ("sooner or later", ["eventually", "at some point in the future", "finally"]),
        ("at the next step", ["in the next moment", "immediately after"]),
        ("will always", ["will forever", "will continuously", "will perpetually"]),
        ("will keep", ["will continue to", "will maintain"]),
    ]

    def apply_first_paraphrase(text: str) -> str:
        for phrase, replacements in paraphrase_templates:
            if phrase in text.lower():
                return re.sub(
                    re.escape(phrase),
                    random.choice(replacements),
                    text,
                    flags=re.IGNORECASE,
                    count=1,
                )
        return text

    for item in data_list:
        augmented.append(item)

        for _ in range(augment_factor - 1):
            new_input = apply_first_paraphrase(item["input"])
            augmented.append({"input": new_input, "output": item["output"]})

    return augmented
