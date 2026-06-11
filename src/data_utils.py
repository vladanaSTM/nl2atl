"""
Data loading, splitting, and augmentation utilities.
"""

import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .infra.io import load_json, save_json


def _append_unique_output(outputs: List[str], value: Any) -> None:
    """Append one or more formula strings, preserving order and uniqueness.

    Handles the unified dataset schema ``outputs: [{"formula": "..."}, ...]``
    as well as result-file gold fields such as ``expected``/``gold``.
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
        for key in ("expected", "gold", "reference"):
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

    for key in ("expected", "gold", "reference"):
        _append_unique_output(outputs, item.get(key))

    return outputs


def normalize_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset rows to the in-memory schema used by the project."""
    if not isinstance(item.get("input"), str) or not item["input"].strip():
        raise ValueError("Dataset item is missing a non-empty 'input' field")

    normalized = dict(item)
    output_options = get_output_options(normalized)
    if not output_options:
        raise ValueError("Dataset item is missing a non-empty 'outputs' field")

    normalized["outputs"] = output_options
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


def default_stratum(item: Dict[str, Any]) -> str:
    """Return a stratification label for a dataset item.

    There is no difficulty field in the dataset, so we stratify on the number
    of jointly-required readings: multi-reading (QSA) items are the hardest and
    rarest cases, and stratifying on them keeps every split/fold balanced.
    """
    return "multi" if len(get_output_options(item)) > 1 else "single"


def _stratum_groups(
    data: List[Dict],
    stratify_key: Callable[[Dict], Any],
) -> Dict[Any, List[Dict]]:
    groups: Dict[Any, List[Dict]] = {}
    for item in data:
        groups.setdefault(stratify_key(item), []).append(item)
    return groups


def _slice_three(
    shuffled: List[Dict],
    train_size: float,
    val_size: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_count = int(round(len(shuffled) * train_size))
    val_count = int(round(len(shuffled) * val_size))
    train_count = max(0, min(len(shuffled), train_count))
    val_count = max(0, min(len(shuffled) - train_count, val_count))
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def split_data(
    data: List[Dict],
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_key: Optional[Callable[[Dict], Any]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: List of data items
        train_size: Fraction of data for training
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        seed: Random seed for reproducibility
        stratify_key: Optional function mapping an item to a stratum label. When
            provided, each stratum is split independently so the train/val/test
            sets preserve the stratum proportions.

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if not data:
        return [], [], []

    split_total = train_size + val_size + test_size
    if not 0.999 <= split_total <= 1.001:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    if stratify_key is None:
        shuffled = list(data)
        random.Random(seed).shuffle(shuffled)
        return _slice_three(shuffled, train_size, val_size)

    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []
    for key, items in sorted(
        _stratum_groups(data, stratify_key).items(), key=lambda kv: str(kv[0])
    ):
        shuffled = list(items)
        random.Random(f"{seed}:{key}").shuffle(shuffled)
        g_train, g_val, g_test = _slice_three(shuffled, train_size, val_size)
        train.extend(g_train)
        val.extend(g_val)
        test.extend(g_test)

    # Shuffle the combined splits so strata are not stored contiguously.
    random.Random(f"{seed}:train").shuffle(train)
    random.Random(f"{seed}:val").shuffle(val)
    random.Random(f"{seed}:test").shuffle(test)
    return train, val, test


def stratified_folds(
    data: List[Dict],
    n_folds: int,
    seed: int = 42,
    stratify_key: Optional[Callable[[Dict], Any]] = None,
) -> List[List[Dict]]:
    """Partition data into ``n_folds`` stratified folds.

    Within each stratum items are shuffled deterministically and dealt
    round-robin across folds, so every fold preserves the stratum proportions
    and each item appears in exactly one fold.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    key_fn = stratify_key or default_stratum
    folds: List[List[Dict]] = [[] for _ in range(n_folds)]
    for key, items in sorted(
        _stratum_groups(data, key_fn).items(), key=lambda kv: str(kv[0])
    ):
        shuffled = list(items)
        random.Random(f"{seed}:{key}").shuffle(shuffled)
        for offset, item in enumerate(shuffled):
            folds[offset % n_folds].append(item)
    return folds


def kfold_split(
    data: List[Dict],
    n_folds: int,
    fold_index: int,
    val_size: float = 0.1,
    seed: int = 42,
    stratify_key: Optional[Callable[[Dict], Any]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (train, val, test) for a single stratified cross-validation fold.

    The held-out fold is the test set; the remaining folds form train+val, from
    which a validation slice of ``val_size`` (relative to the whole dataset) is
    carved deterministically.
    """
    if not data:
        return [], [], []
    if not 0 <= fold_index < n_folds:
        raise ValueError("fold_index must satisfy 0 <= fold_index < n_folds")

    folds = stratified_folds(data, n_folds, seed=seed, stratify_key=stratify_key)
    test = folds[fold_index]
    train_val = [
        item for index, fold in enumerate(folds) if index != fold_index for item in fold
    ]

    random.Random(f"{seed}:fold{fold_index}:val").shuffle(train_val)
    val_count = int(round(len(data) * val_size))
    val_count = max(0, min(len(train_val), val_count))
    val = train_val[:val_count]
    train = train_val[val_count:]
    return train, val, test


def stratified_sample(
    data: List[Dict],
    n: Optional[int],
    seed: int = 42,
    stratify_key: Optional[Callable[[Dict], Any]] = None,
) -> List[Dict]:
    """Return up to ``n`` items chosen to cover every stratum.

    Items are shuffled deterministically within each stratum, then dealt
    round-robin across strata (sorted for determinism), so a small ``n`` still
    includes examples from each stratum when possible. Used by the
    ``--max-eval-samples`` smoke flag to guarantee both single- and
    multi-formula examples appear in a tiny evaluation set.
    """
    if n is None or n >= len(data):
        return list(data)
    if n <= 0:
        return []

    key_fn = stratify_key or default_stratum
    groups = _stratum_groups(data, key_fn)
    for key in groups:
        random.Random(f"{seed}:sample:{key}").shuffle(groups[key])

    ordered_keys = sorted(groups, key=str)
    cursors = {key: 0 for key in ordered_keys}
    sample: List[Dict] = []
    while len(sample) < n:
        advanced = False
        for key in ordered_keys:
            if cursors[key] < len(groups[key]):
                sample.append(groups[key][cursors[key]])
                cursors[key] += 1
                advanced = True
                if len(sample) >= n:
                    break
        if not advanced:
            break
    return sample


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
                    "outputs": get_output_options(normalized_item),
                }
            )

    return augmented
