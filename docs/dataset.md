# Dataset

NL2ATL uses a JSON list of natural-language requirements and accepted ATL formulas.

## Default File

[../data/dataset_gold.json](../data/dataset_gold.json)

## Row Schema

Required:

| Field | Meaning |
|---|---|
| `input` | Natural-language requirement |
| `outputs` | List of accepted ATL formulas (at least one) |

Optional:

| Field | Meaning |
|---|---|
| `id` | Stable example identifier |

Rows may have multiple correct formulas. Each entry in `outputs` is an object with a `formula` field. `load_data` validates rows and normalizes `outputs` into a deduplicated list of formula strings, preserving order. Evaluation helpers also understand result-style fields such as `expected_options`, `gold_options`, and `reference_options` when re-scoring external prediction files.

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

Splits are deterministic and, by default, stratified on formula structure. Stratification keeps the rare multi-reading (QSA) items balanced across train, validation, and test, because there is no difficulty field in the dataset. The splitter groups rows by stratum (single-formula vs multi-formula), shuffles each stratum with a fixed `split_seed`, rounds the train and validation counts per stratum, leaves the remainder for test, then shuffles the combined splits. Set `stratify: false` to fall back to a plain seeded shuffle.

```python
from src.data_utils import split_data, augment_data, default_stratum

train, val, test = split_data(
    data,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    seed=42,
    stratify_key=default_stratum,
)
train_aug = augment_data(train, augment_factor=2)
```

The split seed is decoupled from the training seed: a single fixed `split_seed` defines one canonical split that stays comparable across models and future work, while the training `seed` only controls training stochasticity. Curated few-shot exemplars are held out of every split so demonstrations never leak into train, validation, or test.

For partition-robustness, the same data can be split into stratified cross-validation folds. `kfold_split` uses the held-out fold as the test set and carves a validation slice from the remaining folds:

```python
from src.data_utils import kfold_split, stratified_folds

# One fold (held-out fold = test, rest = train + val).
train, val, test = kfold_split(data, n_folds=5, fold_index=0, val_size=0.10, seed=42)

# Or the raw fold partition for custom loops.
folds = stratified_folds(data, n_folds=5, seed=42)
```

Augmentation happens after splitting and only on the training split. Augmented rows keep the same `outputs` list.

`augment_factor` is the total number of copies per original training row, including the original row. With the default factor of `2`, each training item contributes one original and one paraphrased training item. Split manifests record only original train/validation/test membership; augmented rows are deterministic from the training split, seed, and config.

## How Multiple Gold Answers Are Used

Some inputs have quantifier-scope ambiguity (QSA) and admit more than one correct reading. The pipeline treats every accepted formula as jointly required, not as interchangeable alternatives:

- Training joins all accepted formulas into a single supervised target, one formula per line, so the model learns to emit every required reading instead of choosing one.
- Exact match requires all accepted formulas after normalization. Order does not matter and exact duplicates collapse, but a prediction that returns only a subset of the required formulas is counted as incorrect.
- LLM judging runs only on predictions that are not exact matches. It sees all accepted formulas and approves only when the prediction covers every required reading.
- Few-shot examples display all accepted formulas, one per line, mirroring the expected output format.