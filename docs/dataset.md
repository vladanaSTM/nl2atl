# Dataset

NL2ATL uses a JSON list of natural-language requirements and reference ATL formulas.

## Default File

The default dataset is:

[../data/dataset_gold_no_difficulty.json](../data/dataset_gold_no_difficulty.json)

## Required Fields

Each item must contain:

| Field | Type | Meaning |
|---|---|---|
| `input` | string | Natural-language requirement |
| `output` | string | Reference ATL formula |

The loader also accepts `output_1` and `output_2`. When a row has more than one correct formula, the project keeps all accepted formulas in an in-memory `outputs` list and stores the first preferred formula in `output` for compatibility.

Optional fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable example identifier |
| `difficulty` | string | Optional label such as `easy` or `hard` |
| `difficulty_scores` | object | Optional score breakdown produced by the classifier |

## Example

```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed"
}
```

## Loading Data

Use `load_data` when the data will be used by experiments. It validates rows and records every accepted formula.

```python
from src.data_utils import load_data

dataset = load_data("data/dataset_gold_no_difficulty.json")
print(dataset[0]["input"])
print(dataset[0]["output"])
print(dataset[0]["outputs"])
```

Training uses every formula in `outputs`. Evaluation first checks whether the prediction exactly matches any accepted formula. If none match, the LLM judge receives the same list and can approve predictions that are semantically equivalent to any one accepted formula.

## Splitting Data

The project uses seeded shuffle splits. It does not stratify by difficulty.

```python
from src.data_utils import split_data

train, val, test = split_data(
    dataset,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    seed=42,
)
```

The three sizes must sum to `1.0`.

## Augmentation

Training data can be augmented with simple paraphrases:

```python
from src.data_utils import augment_data

train_aug = augment_data(train, augment_factor=10)
```

The original row is kept, and extra rows keep the same normalized `output` and `outputs` list.

## Difficulty Labels

Difficulty labels are optional. Add them only when you need difficulty analysis:

```bash
uv run nl2atl classify-difficulty --input data/dataset_gold_no_difficulty.json --verbose
```

See [difficulty_classification.md](difficulty_classification.md) for the scoring method.
