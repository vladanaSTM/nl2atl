# Dataset

NL2ATL uses a JSON list of natural-language requirements and accepted ATL formulas.

## Default File

[../data/dataset_gold.json](../data/dataset_gold.json)

## Row Schema

Required:

| Field | Meaning |
|---|---|
| `input` | Natural-language requirement |
| `outputs`, `output`, `output_1`, or `output_2` | At least one accepted ATL formula |

Optional:

| Field | Meaning |
|---|---|
| `id` | Stable example identifier |

Rows may have multiple correct formulas. `load_data` creates:

- `outputs`: all accepted formulas, deduplicated in preferred order
- `output`: the first preferred formula, kept for compatibility

Preferred order is an existing `outputs` list when present, then `output`, then `output_2`, then `output_1`. Evaluation helpers also understand result-style fields such as `expected_options`, `gold_options`, and `reference_options`.

## Example

```json
{
  "id": "ex306",
  "input": "Every agent can eventually prevent a breach",
  "outputs": [
    { "formula": "<<agent_1>>F prevent_breach_1 && <<agent_2>>F prevent_breach_2" },
    { "formula": "<<agent_1,agent_2>>F prevent_breach" }
  ]
}
```

This row has quantifier-scope ambiguity: it admits a distributive reading (each agent acts alone) and a collective reading (the agents act together). Both formulas are required, not interchangeable.

## Loading

```python
from src.data_utils import load_data

data = load_data("data/dataset_gold.json")
print(data[0]["input"])
print(data[0]["outputs"])
```

## Splits And Augmentation

Splits are seeded shuffles, not stratified splits. The splitter uses `random.Random(seed).shuffle`, rounds the train and validation counts, and leaves the remainder for test.

```python
from src.data_utils import split_data, augment_data

train, val, test = split_data(data, train_size=0.70, val_size=0.10, test_size=0.20, seed=42)
train_aug = augment_data(train, augment_factor=2)
```

Augmentation happens after splitting and only on the training split. Augmented rows keep the same `outputs` list.

`augment_factor` is the total number of copies per original training row, including the original row. With the default factor of `2`, each training item contributes one original and one paraphrased training item. Split manifests record only original train/validation/test membership; augmented rows are deterministic from the training split, seed, and config.

## How Multiple Gold Answers Are Used

Some inputs have quantifier-scope ambiguity (QSA) and admit more than one correct reading. The pipeline treats every accepted formula as jointly required, not as interchangeable alternatives:

- Training joins all accepted formulas into a single supervised target, one formula per line, so the model learns to emit every required reading instead of choosing one.
- Exact match requires all accepted formulas after normalization. Order does not matter and exact duplicates collapse, but a prediction that returns only a subset of the required formulas is counted as incorrect.
- LLM judging runs only on predictions that are not exact matches. It sees all accepted formulas and approves only when the prediction covers every required reading.
- Few-shot examples display all accepted formulas, one per line, mirroring the expected output format.