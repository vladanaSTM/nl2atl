# Configuration

NL2ATL reads two YAML files by default:

- [../configs/experiments.yaml](../configs/experiments.yaml) for data, training, seeds, and conditions
- [../configs/models.yaml](../configs/models.yaml) for model definitions

Override them when needed:

```bash
uv run nl2atl run-single --model qwen-3b --models_config configs/models.yaml --experiments_config configs/experiments.yaml
```

## Experiment Config

Current defaults:

```yaml
experiment:
  seed: 42
  num_seeds: 5

data:
  path: "./data/dataset_gold_no_difficulty.json"
  train_size: 0.70
  val_size: 0.10
  test_size: 0.20
  augment_factor: 10

training:
  num_epochs: 10
  batch_size: 10
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  bf16: true
```

`train_size + val_size + test_size` must equal `1.0`.

## Conditions

Conditions combine fine-tuning and few-shot prompting:

```yaml
conditions:
  - name: "baseline_zero_shot"
    finetuned: false
    few_shot: false
  - name: "baseline_few_shot"
    finetuned: false
    few_shot: true
  - name: "finetuned_zero_shot"
    finetuned: true
    few_shot: false
  - name: "finetuned_few_shot"
    finetuned: true
    few_shot: true
```

## Seeds

Use either an explicit list:

```yaml
experiment:
  seeds: [42, 43, 44]
```

or a starting seed plus count:

```yaml
experiment:
  seed: 42
  num_seeds: 5
```

Each seed creates a different train/validation/test shuffle.

## Models

A model entry defines how to load or call a model:

```yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    params_b: 3
```

Common fields are `name`, `short_name`, `provider`, optional training batch settings, optional Azure API name, and optional parameter-count metadata.

## Environment Variables

| Variable | Purpose |
|---|---|
| `AZURE_API_KEY` | Azure-hosted models |
| `AZURE_INFER_ENDPOINT` | Azure endpoint |
| `HUGGINGFACE_TOKEN` | Gated Hugging Face models |
| `TRAIN_MAX_STEPS` | Optional short training run for debugging |
| `REUSE_MODEL_CACHE` | Set `0` to disable model reuse during sweeps |
| `NL2ATL_DEFAULT_MODEL` | Default API model |
| `NL2ATL_MODELS_CONFIG` | API config override |
| `NL2ATL_EXPERIMENTS_CONFIG` | API config override |