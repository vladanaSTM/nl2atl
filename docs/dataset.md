# Dataset

This document describes the NL2ATL dataset structure, content, and usage.

## Table of Contents

- [Overview](#overview)
- [Difficulty Labels](#difficulty-labels)
- [Data Format](#data-format)
- [Examples](#examples)
- [Using the Dataset](#using-the-dataset)

---

## Overview

The dataset is stored in [data/dataset.json](../data/dataset.json) as a JSON list. Each item is a natural-language input paired with an ATL formula.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 298 |
| Difficulty Labels | easy, hard |
| Language | English |
| Logic | ATL |

---

## Difficulty Labels

Difficulty is stored as `easy` or `hard`. The rule-based classifier in `src/evaluation/difficulty.py` can overwrite or recompute these labels and add a `difficulty_scores` field.

---

## Data Format

### File Structure

```json
[
  {
    "id": "ex01",
    "input": "The user can guarantee that sooner or later the ticket will be printed.",
    "output": "<<User>>F ticket_printed",
    "difficulty": "easy"
  }
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `input` | string | Natural language requirement |
| `output` | string | Reference ATL formula |
| `difficulty` | string | Difficulty label (`easy` or `hard`) |
| `difficulty_scores` | object | Optional per-sample score breakdown (after classification) |

---

## Examples

### Easy Example

```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed",
  "difficulty": "easy"
}
```

### Hard Example

```json
{
  "id": "ex11",
  "input": "The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.",
  "output": "<<Machine>>G (paid -> X ticket_printed)",
  "difficulty": "hard"
}
```

---

## Using the Dataset

### Loading in Python

```python
from src.infra.io import load_json

dataset = load_json("data/dataset.json")
for sample in dataset:
    print(sample["input"], sample["output"], sample["difficulty"])
```

### Train/Val/Test Split

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
