# Configuration

NL2ATL reads two YAML files by default:

- [../configs/models.yaml](../configs/models.yaml) for model definitions.
- [../configs/experiments.yaml](../configs/experiments.yaml) for data, training, seeds, and conditions.

Override them with:

```bash
uv run nl2atl run-single \
  --model qwen-3b \
  --models_config configs/models.yaml \
  --experiments_config configs/experiments.yaml
```

## Experiment Config

Minimal structure:

```yaml
experiment:
  name: "nl2atl_300_examples"
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

few_shot:
  num_examples: 5

conditions:
  - name: "baseline_zero_shot"
    finetuned: false
    few_shot: false
```

## Data Settings

| Key | Meaning |
|---|---|
| `path` | Dataset JSON path |
| `train_size` | Final training fraction |
| `val_size` | Final validation fraction |
| `test_size` | Final test fraction |
| `augment_factor` | Number of copies per training row, including the original |

`train_size + val_size + test_size` must equal `1.0`. Splits are seeded shuffles and are not stratified.

Older configs with only `test_size` and `val_size` are converted for compatibility, but new configs should use all three explicit fields.

## Model Config

Each model entry in [../configs/models.yaml](../configs/models.yaml) defines one runnable model:

```yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    max_seq_length: 512
    load_in_4bit: true
    params_b: 3
```

Common keys:

| Key | Meaning |
|---|---|
| `name` | Hugging Face model name or Azure deployment/model name |
| `short_name` | Short label used in output filenames |
| `provider` | `huggingface` or `azure` |
| `max_seq_length` | Tokenizer/model context length used by loaders |
| `load_in_4bit` | Whether local Hugging Face loading uses 4-bit quantization |
| `params_b` | Approximate parameter count in billions |
| `price_input_per_1k` | Optional input-token price for cost reports |
| `price_output_per_1k` | Optional output-token price for cost reports |

Fine-tuning keys for Hugging Face models:

```yaml
lora_r: 64
lora_alpha: 128
target_modules: [q_proj, k_proj, v_proj, o_proj]
train_batch_size: 4
eval_batch_size: 4
gradient_accumulation_steps: 8
```

## Seeds

Use either a single seed or an explicit list:

```yaml
experiment:
  seed: 42
  seeds: [42, 43, 44]
```

If `seeds` is absent and `num_seeds` is greater than 1, the project runs consecutive seeds starting at `seed`.

## Environment Variables

Create `.env` from `.env.example` and set only what you need.

| Variable | Used For |
|---|---|
| `AZURE_API_KEY` | Azure model calls |
| `AZURE_INFER_ENDPOINT` | Azure inference endpoint |
| `HUGGINGFACE_TOKEN` | Gated Hugging Face models |
| `NL2ATL_MODELS_CONFIG` | API server model config override |
| `NL2ATL_EXPERIMENTS_CONFIG` | API server experiment config override |
| `NL2ATL_DEFAULT_MODEL` | API server default model key |
| `TRAIN_MAX_STEPS` | Optional short-run fine-tuning probe |

## Output Paths

Default output paths:

```text
outputs/model_predictions/
outputs/LLM-evaluation/
models/
```

These paths are configurable through `Config` defaults and experiment settings.
