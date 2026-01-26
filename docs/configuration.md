# Configuration Guide

NL2ATL uses YAML configuration files for experiments and models.

## Table of Contents

- [Configuration Files](#configuration-files)
- [Experiment Configuration](#experiment-configuration)
- [Model Configuration](#model-configuration)
- [Environment Variables](#environment-variables)
- [Validation](#validation)

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/experiments.yaml` | Experiment settings, data, training, and conditions |
| `configs/models.yaml` | Model definitions (HF and Azure) |
| `.env` | Environment variables and secrets |

---

## Experiment Configuration

### `configs/experiments.yaml`

```yaml
experiment:
  name: "nl2atl_300_examples"
  seed: 42
  num_seeds: 5

data:
  path: "./data/dataset.json"
  test_size: 0.30
  val_size: 0.6667
  augment_factor: 10

training:
  num_epochs: 10
  batch_size: 10
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  bf16: true

few_shot:
  num_examples: 5

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

wandb:
  project: "nl2atl_300_examples"
  entity: "nl2atl"
```

### Notes

- If `experiment.seeds` is provided, it overrides `seed` and enables multi-seed runs.
- `num_seeds` generates seeds as `[seed, seed+1, ...]` when `seeds` is empty.
- Data splitting uses stratified sampling by `difficulty`.

---

## Model Configuration

### `configs/models.yaml`

Each model entry maps to the `ModelConfig` dataclass.

```yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    params_b: 3
    max_seq_length: 512
    load_in_4bit: false
    lora_r: 64
    lora_alpha: 128
    train_batch_size: 24
    eval_batch_size: 32
    gradient_accumulation_steps: 4
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

  gpt-5.2:
    name: "gpt-5.2"
    short_name: "gpt-5.2"
    provider: "azure"
    api_model: "gpt-5.2"
    params_b: 0
    max_seq_length: 8192
    load_in_4bit: false
    lora_r: 0
    lora_alpha: 0
    target_modules: []
```

### Model Schema Reference

```yaml
models:
  <model-key>:
    name: string
    short_name: string
    provider: huggingface|azure
    api_model: string?          # Azure deployment override
    max_seq_length: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    train_batch_size: int?
    eval_batch_size: int?
    gradient_accumulation_steps: int?
    target_modules: [string]
    params_b: float?
    price_input_per_1k: float?
    price_output_per_1k: float?
    gpu_hour_usd: float?
```

  Notes:

  - For Azure models, set `price_input_per_1k` and `price_output_per_1k` using the official Azure OpenAI pricing page.
  - For local GPU runs (e.g., A100), set `gpu_hour_usd` to your estimated per‑GPU hourly cost; the efficiency report will derive per‑1k‑token costs.

---

## Environment Variables

Create a `.env` file (see `.env.example`):

```bash
AZURE_API_KEY=...
AZURE_INFER_ENDPOINT=...
AZURE_INFER_MODEL=...
AZURE_API_VERSION=2024-08-01-preview
AZURE_USE_CACHE=true
AZURE_VERIFY_SSL=false
HUGGINGFACE_TOKEN=...
WANDB_API_KEY=...
```

---

## Validation

`Config.from_yaml()` validates that required fields in `experiments.yaml` are present and raises a `ValueError` if missing.
