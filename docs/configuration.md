# Configuration

NL2ATL reads two YAML files by default:

- [../configs/experiments.yaml](../configs/experiments.yaml) for data, training, seeds, and conditions
- [../configs/models.yaml](../configs/models.yaml) for model definitions

Override them when needed:

```bash
uv run nl2atl run --models qwen-3b --models_config configs/models.yaml --experiments_config configs/experiments.yaml
```

## Experiment Config

Current defaults:

```yaml
experiment:
  name: "nl2atl_300_examples"
  seed: 42
  num_seeds: 3

data:
  path: "./data/dataset_gold.json"
  train_size: 0.70
  val_size: 0.10
  test_size: 0.20
  augment_factor: 2

training:
  num_epochs: 8
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  bf16: true
  optim: "paged_adamw_8bit"
  lr_scheduler_type: "cosine"
  eval_strategy: "epoch"
  save_strategy: "epoch"
  save_total_limit: 1
  max_grad_norm: 0.3
  gradient_checkpointing: true
  dataloader_num_workers: 2
  dataloader_pin_memory: true
  group_by_length: true
  tf32: true
  packing: false

few_shot:
  num_examples: 5
```

`train_size + val_size + test_size` must equal `1.0`.

An optional `paths` block can override runtime locations without changing project constants:

```yaml
paths:
  data_path: "./data/dataset_gold.json"
  output_dir: "./outputs"
  models_dir: "./models"
```

Older holdout-style configs with `test_size` plus `val_size` as a validation share of the holdout are still converted by the loader, but new configs should use explicit train/validation/test fractions.

## Fine-Tuning Strategy

The default fine-tuning setup is frozen for reproducible paper runs:

- Qwen 3B and Phi-3.5 Mini use full-precision LoRA with small per-device micro-batches.
- Qwen Coder 7B and Mistral 7B use 4-bit QLoRA with smaller micro-batches.
- All trainable models use gradient checkpointing, cosine scheduling, epoch evaluation, one retained checkpoint, clipped gradients, and deterministic seeds.
- `bf16` and `tf32` stay enabled in config, but the runner only activates them when the CUDA device supports them.
- Fine-tuned zero-shot and fine-tuned few-shot conditions share one adapter per model and seed.
- Model-level batch and gradient-accumulation settings override the global training defaults.
- Set `--train-max-steps` or `TRAIN_MAX_STEPS` only for smoke tests; omit it for full paper runs.

Frozen model-level profiles:

| Model | Precision | LoRA rank | Dropout | Train batch | Grad accumulation |
|---|---:|---:|---:|---:|---:|
| Qwen 3B | BF16 LoRA | 64 | 0.05 | 8 | 4 |
| Phi-3.5 Mini | BF16 LoRA | 32 | 0.05 | 6 | 6 |
| Qwen Coder 7B | 4-bit QLoRA | 64 | 0.05 | 4 | 8 |
| Mistral 7B | 4-bit QLoRA | 32 | 0.05 | 2 | 16 |

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
  num_seeds: 3
```

Each seed creates a different train/validation/test shuffle. CLI overrides use either `--seed 42` for one seed or `--seeds 42 43 44` for a list. Passing both is an error.

## Models

A model entry defines how to load or call a model:

```yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    revision: "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    params_b: 3
```

Common fields are `name`, `short_name`, `provider`, pinned Hugging Face `revision`, optional training batch settings, optional Azure API name, optional `generation_enabled`, and optional parameter-count metadata. Set `generation_enabled: false` for judge-only API models that should remain available to `nl2atl llm-judge` but stay out of generation experiments.

Useful model fields:

| Field | Meaning |
|---|---|
| `name` | Hugging Face model ID or Azure deployment label |
| `short_name` | Stable name used in output file names and reports |
| `provider` | `huggingface` or `azure` |
| `api_model` | Azure deployment/model name when it differs from `name` |
| `generation_enabled` | Whether `nl2atl run --models all` may generate with the model |
| `revision` | Pinned Hugging Face revision for reproducibility |
| `max_seq_length` | Training sequence window |
| `load_in_4bit` | Use QLoRA-style 4-bit loading for local fine-tuning |
| `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules` | LoRA adapter profile |
| `params_b` | Parameter count metadata and fine-tuning guardrail |

## Experiment Launching

Use `nl2atl run` for local runs, task inspection, and SLURM submission:

```bash
uv run nl2atl run --list-tasks --models all --conditions all --model_provider hf
uv run nl2atl run --count --models all --conditions all --model_provider hf
uv run nl2atl run --slurm --models all --conditions all --model_provider hf
uv run nl2atl run --slurm --no-submit --script-path outputs/manifests/nl2atl.sbatch
uv run nl2atl run --list-tasks --models all --conditions baseline_zero_shot baseline_few_shot --model_provider azure
```

SLURM tasks are grouped by model and seed. If both fine-tuned conditions are selected for the same model, the runner trains one shared adapter and evaluates both prompting conditions from it. Azure/API generation baselines automatically skip fine-tuned conditions. The default Azure generation set is GPT-4.1 and GPT-5.4; GPT-5.2 and DeepSeek V3.2 are kept judge-only.

Existing result files are skipped by default. Add `--overwrite` when you want to regenerate prediction files or retrain an adapter.

SLURM arrays are uncapped by default. Add `--max-parallel-gpus N` only when you want to throttle the array to `N` concurrent one-GPU tasks.

For OOM smoke tests, cap training steps without changing the YAML files:

```bash
uv run nl2atl run --slurm \
  --models all --conditions all --model_provider hf \
  --train-max-steps 20
```

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
| `PYTHON_BIN` | Python executable used inside generated SLURM scripts |
| `REPO_ROOT` | Repository path used inside generated SLURM scripts |