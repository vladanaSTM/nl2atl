# NL2ATL Documentation

This documentation explains how to install, configure, run, evaluate, and extend NL2ATL.

## Recommended Reading Order

1. [Installation](installation.md) - install with uv and configure credentials.
2. [Quickstart](quickstart.md) - run one experiment and inspect output.
3. [Dataset](dataset.md) - understand the JSON schema and split behavior.
4. [Configuration](configuration.md) - edit models, experiments, and split sizes.
5. [Evaluation](evaluation.md) - exact match, LLM judge, agreement, and efficiency reports.

## Task Navigation

| Task | Read |
|---|---|
| Install the project | [Installation](installation.md) |
| Run a first experiment | [Quickstart](quickstart.md) |
| Understand ATL syntax | [ATL Primer](atl_primer.md) |
| Edit the dataset | [Dataset](dataset.md) |
| Add or change models | [Configuration](configuration.md) |
| Run large sweeps | [SLURM](slurm.md) |
| Evaluate predictions | [Evaluation](evaluation.md), [Usage](usage.md) |
| Use the API | [API](api.md) |
| Understand internals | [Architecture](architecture.md) |
| Contribute code | [Development](development.md) |
| Use genVITAMIN | [genVITAMIN](genvitamin.md) |

## Core Workflow

```text
dataset + configs
      |
      v
seeded train/validation/test split
      |
      v
train or load model
      |
      v
generate ATL formulas
      |
      v
exact-match and optional LLM-judge evaluation
      |
      v
reports under outputs/
```

## Current Dataset Contract

The default dataset is [../data/dataset_gold_no_difficulty.json](../data/dataset_gold_no_difficulty.json). Rows require `input` plus one formula field: `output`, `output_1`, or `output_2`. The loader keeps all accepted formulas in `outputs` and stores a preferred formula in `output` for compatibility.

Splits are simple seeded shuffles using explicit final proportions:

```yaml
train_size: 0.70
val_size: 0.10
test_size: 0.20
```

Splits are not stratified by difficulty.

## Main Commands

```bash
uv run nl2atl run-single --model qwen-3b --few_shot
uv run nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
uv run nl2atl llm-judge --datasets all
uv run nl2atl model-efficiency --predictions_dir outputs/model_predictions
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```
