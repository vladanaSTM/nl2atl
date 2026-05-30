# NL2ATL Docs

NL2ATL translates natural-language strategic requirements into ATL formulas, runs model experiments, and evaluates predictions with exact match plus optional LLM judging.

These are the only docs needed for day-to-day use and development:

1. [Quickstart](quickstart.md) - install the project, run a model, inspect outputs, and start the API.
2. [Configuration](configuration.md) - edit dataset settings, seeds, conditions, and models.
3. [Dataset](dataset.md) - understand row format, multiple gold formulas, splits, and augmentation.
4. [Evaluation](evaluation.md) - understand exact match, LLM judging, agreement, and accuracy-latency reports.
5. [Architecture](architecture.md) - see how the project is organized and how a run flows through the code.
6. [Development](development.md) - run tests and find the right module to change.

## Workflow

```text
dataset + configs
  -> seeded train/validation/test split
  -> optional training on the training split
  -> one ATL prediction per test input
  -> exact match against accepted gold formulas
  -> LLM judge only for non-exact predictions
  -> judge agreement and accuracy-latency reports
```

## Defaults

- Dataset: [../data/dataset_gold_no_difficulty.json](../data/dataset_gold_no_difficulty.json)
- Split: `train_size=0.70`, `val_size=0.10`, `test_size=0.20`
- Seeds: `num_seeds=3` starting at `seed=42` unless overridden
- Augmentation: applied after splitting, to training data only
- Dependencies: uv and [../pyproject.toml](../pyproject.toml)