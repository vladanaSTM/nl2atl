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
  seed: 42
  num_seeds: 3

data:
  path: "./data/dataset_gold_no_difficulty.json"
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
```

`train_size + val_size + test_size` must equal `1.0`.

## Fine-Tuning Strategy

The default fine-tuning setup is frozen for reproducible paper runs:

- Qwen 3B and Phi-3.5 Mini use full-precision LoRA with small per-device micro-batches.
- Qwen Coder 7B and Mistral 7B use 4-bit QLoRA with smaller micro-batches.
- All trainable models use gradient checkpointing, cosine scheduling, epoch evaluation, one retained checkpoint, clipped gradients, and deterministic seeds.
- `bf16` and `tf32` stay enabled in config, but the runner only activates them when the CUDA device supports them.
- Fine-tuned zero-shot and fine-tuned few-shot conditions share one adapter per model and seed.

Frozen model-level profiles after the bounded smoke sweep:

| Model | Precision | LoRA rank | Dropout | Train batch | Grad accumulation |
|---|---:|---:|---:|---:|---:|
| Qwen 3B | BF16 LoRA | 64 | 0.05 | 8 | 4 |
| Phi-3.5 Mini | BF16 LoRA | 32 | 0.05 | 6 | 6 |
| Qwen Coder 7B | 4-bit QLoRA | 64 | 0.05 | 4 | 8 |
| Mistral 7B | 4-bit QLoRA | 32 | 0.05 | 2 | 16 |

For parameter testing, use [../configs/models_finetune_sweep.yaml](../configs/models_finetune_sweep.yaml) with [../configs/experiments_finetune_sweep.yaml](../configs/experiments_finetune_sweep.yaml). Those files write under `outputs/tuning/` and `models/tuning/` so tuning adapters do not overwrite the frozen production adapters.

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

Each seed creates a different train/validation/test shuffle.

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

Common fields are `name`, `short_name`, `provider`, pinned Hugging Face `revision`, optional training batch settings, optional Azure API name, and optional parameter-count metadata.

## Experiment Launching

Use `nl2atl run` for local runs, task inspection, and SLURM submission:

```bash
uv run nl2atl run --list-tasks --models all --conditions all --model_provider hf
uv run nl2atl run --slurm --max-parallel-gpus 2 --models all --conditions all --model_provider hf
```

SLURM tasks are grouped by model and seed. If both fine-tuned conditions are selected for the same model, the runner trains one shared adapter and evaluates both prompting conditions from it. Azure/API models automatically skip fine-tuned conditions.

For OOM smoke tests, cap training steps without changing the YAML files:

```bash
uv run nl2atl run --slurm --max-parallel-gpus 2 \
  --models_config configs/models_finetune_sweep.yaml \
  --experiments_config configs/experiments_finetune_sweep.yaml \
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