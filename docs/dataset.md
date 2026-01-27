# Dataset

This document describes the NL2ATL dataset structure, schema, and how to use it in experiments.

## Overview

The dataset is stored in [data/dataset.json](../data/dataset.json) as a JSON list. Each item contains a
natural‑language requirement and its reference ATL formula.

Key properties:

- Language: English
- Logic: ATL
- Labels: `easy` or `hard` (present in the released dataset)

## Schema

Each dataset item follows this structure:

```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed",
  "difficulty": "easy"
}
```

Field meanings:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier |
| `input` | string | Natural‑language requirement |
| `output` | string | Reference ATL formula |
| `difficulty` | string | `easy` or `hard` |
| `difficulty_scores` | object | Optional score breakdown (only present if you run the classifier) |

## Difficulty labels

Difficulty labels are produced by the rule‑based classifier in
[src/evaluation/difficulty.py](../src/evaluation/difficulty.py). You can recompute labels (and optionally
add `difficulty_scores`) with:

```bash
nl2atl classify-difficulty --input data/dataset.json --verbose
```

See [difficulty_classification.md](difficulty_classification.md) for the scoring model.

## Examples

### Easy example

```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed",
  "difficulty": "easy"
}
```

### Hard example

```json
{
  "id": "ex11",
  "input": "The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.",
  "output": "<<Machine>>G (paid -> X ticket_printed)",
  "difficulty": "hard"
}
```

## Using the dataset

### Load from Python

```python
from src.infra.io import load_json

dataset = load_json("data/dataset.json")
for sample in dataset:
    print(sample["input"], sample["output"], sample.get("difficulty"))
```

### Split into train, validation, test

```python
from pathlib import Path
from src.experiment import ExperimentDataManager

manager = ExperimentDataManager(
    data_path=Path("data/dataset.json"),
    test_size=0.30,
    val_size=0.6667,
    seed=42,
    augment_factor=10,
)
train_aug, val, test, full = manager.prepare_data()
```

  The split is stratified by `difficulty` when available. Augmentation uses simple paraphrasing and
  returns training examples that only include `input` and `output` fields.
