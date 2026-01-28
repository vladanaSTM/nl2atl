# Configuration Guide

This guide explains NL2ATL's configuration system, covering YAML config files, environment variables, and validation. Learn how to configure models, experiments, and deployment settings.

## Table of Contents

1. [Configuration Files Overview](#configuration-files-overview)
2. [models.yaml Reference](#modelsyaml-reference)
3. [experiments.yaml Reference](#experimentsyaml-reference)
4. [Environment Variables](#environment-variables)
5. [Configuration Validation](#configuration-validation)
6. [Common Configurations](#common-configurations)
7. [Troubleshooting](#troubleshooting)

---

## Configuration Files Overview

NL2ATL uses three configuration sources:

| File | Purpose | When Required |
|------|---------|---------------|
| `configs/models.yaml` | Model registry and settings | Always |
| `configs/experiments.yaml` | Experiment conditions and hyperparameters | Always |
| `.env` | API keys and secrets | When using Azure, HuggingFace, or W&B |

### Configuration Priority

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **YAML config files**
4. **Default values** (lowest priority)

---

## models.yaml Reference

Defines available models, their providers, and training/inference settings.

### File Location

```
configs/models.yaml
```

### Structure

```yaml
models:
  model-key:
    # Model identification
    name: string              # Full model name or HuggingFace ID
    short_name: string        # Short identifier
    provider: string          # huggingface or azure
    
    # Model parameters
    params_b: float           # Billion parameters (for documentation)
    max_seq_length: int       # Maximum sequence length
    
    # Quantization
    load_in_4bit: bool        # Enable 4-bit quantization
    
    # LoRA configuration
    lora_r: int               # LoRA rank
    lora_alpha: int           # LoRA alpha scaling
    target_modules: [string]  # Modules to apply LoRA
    
    # Batch sizes
    train_batch_size: int     # Training batch size
    eval_batch_size: int      # Evaluation batch size
    gradient_accumulation_steps: int  # Gradient accumulation
    
    # Pricing (optional)
    price_input_per_1k: float   # Input token cost (USD per 1K)
    price_output_per_1k: float  # Output token cost (USD per 1K)
    gpu_hour_usd: float         # GPU cost per hour (for local models)
    
    # Azure-specific
    api_model: string         # Azure deployment name (overrides 'name')
```

### Complete Examples

#### HuggingFace Model (Small)

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
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    train_batch_size: 24
    eval_batch_size: 32
    gradient_accumulation_steps: 4
    gpu_hour_usd: 0.50  # Estimated for RTX 3090
```

#### HuggingFace Model (Large)

```yaml
models:
  llama-70b:
    name: "meta-llama/Llama-3.1-70B-Instruct"
    short_name: "llama-70b"
    provider: "huggingface"
    params_b: 70
    max_seq_length: 2048
    load_in_4bit: true  # Required for single GPU
    lora_r: 32
    lora_alpha: 64
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
    train_batch_size: 2
    eval_batch_size: 4
    gradient_accumulation_steps: 16
    gpu_hour_usd: 3.00  # Estimated for A100
```

#### Azure OpenAI Model

```yaml
models:
  gpt-5.2:
    name: "gpt-5.2"
    short_name: "gpt-5.2"
    provider: "azure"
    api_model: "gpt-5.2"  # Your Azure deployment name
    params_b: 0  # Not applicable
    max_seq_length: 8192
    load_in_4bit: false
    lora_r: 0
    lora_alpha: 0
    target_modules: []
    price_input_per_1k: 0.01   # From Azure pricing
    price_output_per_1k: 0.03  # From Azure pricing
```

### Field Descriptions

#### Model Identification

**`name`** (string, required)
- For HuggingFace: Model ID (e.g., `"Qwen/Qwen2.5-3B-Instruct"`)
- For Azure: Deployment name (or use `api_model` to override)

**`short_name`** (string, required)
- Short identifier used in output filenames
- Convention: `{model}-{size}` (e.g., `qwen-3b`, `llama-8b`)

**`provider`** (string, required)
- `"huggingface"` — HuggingFace Transformers
- `"azure"` — Azure OpenAI

#### Model Parameters

**`params_b`** (float, optional, default: 0)
- Number of parameters in billions
- Used for documentation and reports
- Set to 0 for API models

**`max_seq_length`** (int, required)
- Maximum sequence length in tokens
- Truncates longer inputs

#### Quantization

**`load_in_4bit`** (bool, optional, default: false)
- Enable 4-bit quantization (bitsandbytes)
- Reduces memory usage ~4x
- Slight accuracy loss
- Recommended for large models on limited GPUs

#### LoRA Configuration

**`lora_r`** (int, optional, default: 64)
- LoRA rank (dimensionality of low-rank matrices)
- Higher = more parameters, better fit, slower training
- Typical range: 8-128

**`lora_alpha`** (int, optional, default: 128)
- LoRA scaling factor
- Rule of thumb: `alpha = 2 * r`

**`target_modules`** (list of strings, required)
- Which model modules to apply LoRA
- Common for Llama/Qwen: `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
- Common for BERT-style: `[query, key, value]`

#### Batch Sizes

**`train_batch_size`** (int, optional, default: 8)
- Per-device training batch size
- Adjust based on GPU memory

**`eval_batch_size`** (int, optional, default: 16)
- Per-device evaluation batch size
- Can be larger than train batch size

**`gradient_accumulation_steps`** (int, optional, default: 1)
- Number of steps to accumulate gradients
- Effective batch size = `batch_size * accumulation_steps * num_gpus`

#### Pricing

**`price_input_per_1k`** (float, optional)
- Cost in USD per 1,000 input tokens
- Check Azure/OpenAI pricing page

**`price_output_per_1k`** (float, optional)
- Cost in USD per 1,000 output tokens
- Usually higher than input cost

**`gpu_hour_usd`** (float, optional)
- GPU cost per hour (for local models)
- Used to estimate cost from runtime
- Can estimate as: `(GPU_MSRP / lifespan_hours) + power_cost`

**Note**: If neither per-token prices nor `gpu_hour_usd` is set, cost rankings in efficiency report will be skipped for that model.

---

## experiments.yaml Reference

Defines experiment conditions, data splits, and hyperparameters.

### File Location

```
configs/experiments.yaml
```

### Structure

```yaml
experiment:
  name: string              # Experiment identifier
  seed: int                 # Base random seed
  num_seeds: int            # Number of seeds for reproducibility
  seeds: [int]              # Explicit seed list (overrides seed + num_seeds)

data:
  path: string              # Path to dataset JSON
  test_size: float          # Test set proportion (0-1)
  val_size: float           # Validation proportion of non-test (0-1)
  augment_factor: int       # Data augmentation multiplier

training:
  num_epochs: int           # Training epochs
  batch_size: int           # Global training batch size
  gradient_accumulation_steps: int  # Gradient accumulation
  learning_rate: float      # Learning rate
  weight_decay: float       # Weight decay
  warmup_ratio: float       # Warmup proportion
  bf16: bool                # Use bfloat16 training
  save_strategy: string     # epoch or steps
  evaluation_strategy: string  # epoch or steps
  load_best_model_at_end: bool  # Load best checkpoint

few_shot:
  num_examples: int         # Default few-shot examples

conditions:
  - name: string            # Condition identifier
    finetuned: bool         # Use fine-tuned model
    few_shot: bool          # Enable few-shot prompting

wandb:
  project: string           # W&B project name
  entity: string            # W&B entity (username/team)
  enabled: bool             # Enable W&B logging
```

### Complete Example

```yaml
experiment:
  name: "nl2atl_300_examples"
  seed: 42
  num_seeds: 5

data:
  path: "./data/dataset.json"
  test_size: 0.30
  val_size: 0.6667  # 2/3 of remaining after test split
  augment_factor: 10

training:
  num_epochs: 10
  batch_size: 10
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  bf16: true
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  load_best_model_at_end: true

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
  entity: "your-wandb-username"
  enabled: true
```

### Field Descriptions

#### Experiment Settings

**`experiment.name`** (string, required)
- Identifier for this experiment set
- Used in W&B project names and logs

**`experiment.seed`** (int, required)
- Base random seed for reproducibility
- Used if `seeds` list not provided

**`experiment.num_seeds`** (int, optional, default: 1)
- Number of random seeds to generate
- Seeds will be `[seed, seed+1, ..., seed+num_seeds-1]`
- Ignored if `seeds` is provided

**`experiment.seeds`** (list of int, optional)
- Explicit list of seeds to use
- Overrides `seed` and `num_seeds`
- Example: `[42, 123, 456, 789, 1337]`

#### Data Configuration

**`data.path`** (string, required)
- Path to dataset JSON file
- Can be relative to repo root

**`data.test_size`** (float, required)
- Proportion of data for test set
- Range: 0.0-1.0
- Example: 0.30 = 30% test, 70% train+val

**`data.val_size`** (float, optional, default: 0.0)
- Proportion of *remaining* data for validation
- After splitting test set, val_size is applied to remainder
- Example: `test_size=0.3, val_size=0.6667` → 30% test, 47% train, 23% val
- Set to 0.0 to disable validation

**`data.augment_factor`** (int, optional, default: 1)
- Data augmentation multiplier
- Training set is augmented by this factor
- Example: 210 samples × 10 = 2100 training samples

**Note**: Splits are stratified by `difficulty` when available in dataset.

#### Training Hyperparameters

**`training.num_epochs`** (int, required)
- Number of training epochs

**`training.batch_size`** (int, required)
- Global batch size (across all devices)
- Actual per-device batch size: `batch_size / num_devices`

**`training.gradient_accumulation_steps`** (int, optional, default: 1)
- Gradient accumulation steps
- Effective batch size: `batch_size * gradient_accumulation_steps`

**`training.learning_rate`** (float, required)
- Peak learning rate
- Typical range for fine-tuning: 1e-5 to 3e-4

**`training.weight_decay`** (float, optional, default: 0.0)
- L2 regularization weight

**`training.warmup_ratio`** (float, optional, default: 0.0)
- Proportion of steps for warmup
- Example: 0.1 = 10% of steps are warmup

**`training.bf16`** (bool, optional, default: false)
- Enable bfloat16 mixed precision
- Reduces memory, faster on modern GPUs

**`training.save_strategy`** (string, optional, default: "epoch")
- When to save checkpoints
- Options: `"epoch"`, `"steps"`, `"no"`

**`training.evaluation_strategy`** (string, optional, default: "epoch")
- When to evaluate on validation set
- Options: `"epoch"`, `"steps"`, `"no"`

**`training.load_best_model_at_end`** (bool, optional, default: true)
- Load best checkpoint at end of training

#### Few-Shot Configuration

**`few_shot.num_examples`** (int, optional, default: 5)
- Default number of few-shot examples
- Can be overridden per command

#### Conditions

Conditions define experimental configurations to test.

**`conditions[].name`** (string, required)
- Condition identifier (used in output filenames)

**`conditions[].finetuned`** (bool, required)
- Whether to use fine-tuned model

**`conditions[].few_shot`** (bool, required)
- Whether to use few-shot prompting

**Common conditions:**
- `baseline_zero_shot`: Pretrained, no examples
- `baseline_few_shot`: Pretrained, with examples
- `finetuned_zero_shot`: Fine-tuned, no examples
- `finetuned_few_shot`: Fine-tuned, with examples

#### Weights & Biases

**`wandb.project`** (string, optional)
- W&B project name

**`wandb.entity`** (string, optional)
- W&B username or team name

**`wandb.enabled`** (bool, optional, default: false)
- Enable W&B logging

---

## Environment Variables

### Core Variables

```bash
# Python path (if imports fail)
PYTHONPATH=.

# Random seed
SEED=42
```

### Azure OpenAI

```bash
AZURE_API_KEY=your_api_key_here
AZURE_INFER_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_INFER_MODEL=gpt-5.2
AZURE_API_VERSION=2024-08-01-preview
AZURE_USE_CACHE=true
AZURE_VERIFY_SSL=false  # Set true in production
```

### HuggingFace

```bash
HUGGINGFACE_TOKEN=your_hf_token_here
```

### Weights & Biases

```bash
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=nl2atl
WANDB_ENTITY=your_username
```

### API Service

```bash
NL2ATL_DEFAULT_MODEL=qwen-3b
NL2ATL_MODELS_CONFIG=configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=configs/experiments.yaml
```

### Loading Environment

```bash
# From .env file (automatic)
python -c "from dotenv import load_env; load_env()"

# From .env file (CLI)
export $(cat .env | xargs)

# Manually
export AZURE_API_KEY=...
```

---

## Configuration Validation

### Validation Rules

NL2ATL validates configuration at load time:

**Required fields:**
- `models` (at least one model)
- `experiment.name`
- `experiment.seed` or `experiment.seeds`
- `data.path`
- `data.test_size`

**Type checking:**
- Strings are strings
- Numbers are floats/ints
- Booleans are booleans
- Lists are lists

**Value constraints:**
- `0 < test_size < 1`
- `0 <= val_size < 1`
- `seed >= 0`
- `num_epochs > 0`
- `batch_size > 0`

### Validation Errors

```python
from src.config import Config

try:
    config = Config.from_yaml(
        models_yaml="configs/models.yaml",
        experiments_yaml="configs/experiments.yaml"
    )
except ValueError as e:
    print(f"Invalid config: {e}")
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
```

---

## Common Configurations

### Small Model (RTX 3090, 24GB)

```yaml
# configs/models.yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    max_seq_length: 512
    load_in_4bit: false
    lora_r: 64
    lora_alpha: 128
    train_batch_size: 24
    eval_batch_size: 32
    gradient_accumulation_steps: 4
```

```yaml
# configs/experiments.yaml
training:
  batch_size: 24
  gradient_accumulation_steps: 4
  # Effective batch size: 24 * 4 = 96
```

### Large Model (A100, 80GB)

```yaml
# configs/models.yaml
models:
  llama-70b:
    name: "meta-llama/Llama-3.1-70B-Instruct"
    short_name: "llama-70b"
    provider: "huggingface"
    max_seq_length: 2048
    load_in_4bit: true  # Still needed even on A100
    lora_r: 32
    lora_alpha: 64
    train_batch_size: 2
    eval_batch_size: 4
    gradient_accumulation_steps: 16
```

```yaml
# configs/experiments.yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16
  bf16: true
  # Effective batch size: 2 * 16 = 32
```

### CPU-Only (No GPU)

```yaml
# configs/models.yaml
models:
  qwen-1.5b:
    name: "Qwen/Qwen2.5-1.5B-Instruct"
    short_name: "qwen-1.5b"
    provider: "huggingface"
    max_seq_length: 256  # Shorter for speed
    load_in_4bit: false  # Quantization requires GPU
    train_batch_size: 4
    eval_batch_size: 8
```

```yaml
# configs/experiments.yaml
training:
  batch_size: 4
  bf16: false  # bf16 requires GPU
```

### Azure-Only (No Local Models)

```yaml
# configs/models.yaml
models:
  gpt-5.2:
    name: "gpt-5.2"
    short_name: "gpt-5.2"
    provider: "azure"
    max_seq_length: 8192
    price_input_per_1k: 0.01
    price_output_per_1k: 0.03
```

```bash
# .env
AZURE_API_KEY=your_key
AZURE_INFER_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_INFER_MODEL=gpt-5.2
```

### Multi-Seed Reproducibility

```yaml
# configs/experiments.yaml
experiment:
  name: "reproducibility_study"
  seeds: [42, 123, 456, 789, 1337]  # Explicit seeds

# Alternative using num_seeds
experiment:
  name: "reproducibility_study"
  seed: 42
  num_seeds: 5  # Generates [42, 43, 44, 45, 46]
```

---

## Troubleshooting

### "Config validation failed"

**Problem**: Invalid YAML structure or values

**Solution**:
1. Validate YAML syntax: https://www.yamllint.com/
2. Check required fields are present
3. Verify value types (string vs int vs float)
4. Check constraints (e.g., `0 < test_size < 1`)

### "Model not found in config"

**Problem**: Model key doesn't exist

**Solution**:
```bash
# List available models
cat configs/models.yaml | grep "^  [a-z]"

# Check exact spelling
nl2atl run-single --model qwen-3b  # Correct
nl2atl run-single --model Qwen-3B  # Wrong (case sensitive)
```

### "CUDA out of memory" during training

**Problem**: GPU memory exhausted

**Solutions**:
1. Enable 4-bit quantization:
   ```yaml
   load_in_4bit: true
   ```
2. Reduce batch size:
   ```yaml
   train_batch_size: 4
   gradient_accumulation_steps: 16
   ```
3. Reduce sequence length:
   ```yaml
   max_seq_length: 256
   ```
4. Use smaller model

### "File not found: configs/models.yaml"

**Problem**: Running from wrong directory

**Solution**:
```bash
# Ensure you're in repo root
cd /path/to/nl2atl

# Or use absolute paths
NL2ATL_MODELS_CONFIG=/absolute/path/to/configs/models.yaml \
nl2atl run-single --model qwen-3b
```

### "Azure authentication failed"

**Problem**: Invalid or missing credentials

**Solution**:
```bash
# Check .env file
cat .env | grep AZURE

# Test credentials
python -c "
from src.infra.azure import AzureConfig, AzureClient
config = AzureConfig.from_env()
client = AzureClient(config)
print('✓ Azure configured')
"
```

### "Validation size must be between 0 and 1"

**Problem**: `val_size` incorrectly specified

**Solution**:
```yaml
# Correct: val_size is proportion of non-test data
data:
  test_size: 0.30    # 30% for test
  val_size: 0.6667   # 2/3 of remaining 70% = 47% of total

# Wrong: val_size > 1
data:
  test_size: 0.30
  val_size: 2.0  # Error!
```

---

## Best Practices

### 1. Version Control

```bash
# Track config files
git add configs/models.yaml configs/experiments.yaml

# Don't track .env (secrets)
echo ".env" >> .gitignore
```

### 2. Backup Configs

```bash
# Before major changes
cp configs/models.yaml configs/models.yaml.backup
cp configs/experiments.yaml configs/experiments.yaml.backup
```

### 3. Use Comments

```yaml
models:
  qwen-3b:
    # Optimized for RTX 3090 (24GB VRAM)
    name: "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: false  # 4-bit not needed with 24GB
```

### 4. Separate Configs for Different Setups

```bash
configs/
├── models.yaml              # Development (local)
├── models_production.yaml   # Production (Azure)
├── experiments.yaml         # Standard experiments
└── experiments_ablation.yaml  # Ablation studies
```

Load specific config:
```bash
NL2ATL_MODELS_CONFIG=configs/models_production.yaml nl2atl run-all
```

---

## Next Steps

- **Usage Guide**: [usage.md](usage.md) — CLI commands with config examples
- **Installation**: [installation.md](installation.md) — Environment setup
- **Development**: [development.md](development.md) — Extending config system

---

**Questions?** Check [full documentation](index.md) or [open an issue](https://github.com/vladanaSTM/nl2atl/issues).
