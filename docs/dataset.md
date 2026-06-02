# Dataset

NL2ATL uses a JSON list of natural-language requirements and accepted ATL formulas.

## Default File

[../data/dataset_gold.json](../data/dataset_gold.json)

## Row Schema

Required:

| Field | Meaning |
|---|---|
| `input` | Natural-language requirement |
| `output`, `output_1`, or `output_2` | Accepted ATL formula |

Optional:

| Field | Meaning |
|---|---|
| `id` | Stable example identifier |

Rows may have multiple correct formulas. `load_data` creates:

- `outputs`: all accepted formulas, deduplicated in preferred order
- `output`: the first preferred formula, kept for compatibility

Preferred order is `output`, then `output_2`, then `output_1`.

## Example

```json
{
  "id": "ex952",
  "input": "Every authentication server can guarantee that in the immediately succeeding state the recovery token will be issued.",
  "output_1": "<<AuthenticationServer1>>X recovery_token_issued_1 && <<AuthenticationServer2>>X recovery_token_issued_2",
  "output_2": "<<AuthenticationServer1,AuthenticationServer2>>X recovery_token_issued"
}
```

## Loading

```python
from src.data_utils import load_data

data = load_data("data/dataset_gold.json")
print(data[0]["input"])
print(data[0]["outputs"])
```

## Splits And Augmentation

Splits are seeded shuffles.

```python
from src.data_utils import split_data, augment_data

train, val, test = split_data(data, train_size=0.70, val_size=0.10, test_size=0.20, seed=42)
train_aug = augment_data(train, augment_factor=2)
```

Augmentation happens after splitting and only on the training split. Augmented rows keep the same `outputs` list.

## How Multiple Gold Answers Are Used

- Training creates one supervised target per accepted formula.
- Exact match is correct if the prediction matches any accepted formula after normalization.
- LLM judging sees all accepted formulas and can approve semantic equivalence to any one of them.