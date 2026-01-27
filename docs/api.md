# API Reference

This document provides a comprehensive reference for NL2ATL's programmatic interfaces, including the REST API service, public Python modules, and key classes.

## Table of Contents

1. [REST API Service](#rest-api-service)
2. [Python API](#python-api)
3. [Configuration Classes](#configuration-classes)
4. [Model Registry](#model-registry)
5. [Evaluation Modules](#evaluation-modules)
6. [Infrastructure Utilities](#infrastructure-utilities)

---

## REST API Service

NL2ATL provides a FastAPI-based REST API for real-time NL→ATL translation. This is the primary interface for integrating NL2ATL into external applications.

### Starting the Server

```bash
# From repository root
uvicorn src.api_server:app --host 0.0.0.0 --port 8081

# With custom config paths (if running from elsewhere)
NL2ATL_MODELS_CONFIG=/absolute/path/to/configs/models.yaml \
NL2ATL_EXPERIMENTS_CONFIG=/absolute/path/to/configs/experiments.yaml \
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:8081/health
```

---

#### `POST /generate`

Generate ATL formula from natural language description.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `description` | string | ✓ | - | Natural language requirement |
| `model` | string | ✗ | env default | Model key from `models.yaml` |
| `few_shot` | boolean | ✗ | `false` | Enable few-shot prompting |
| `num_few_shot` | integer | ✗ | 5 | Number of few-shot examples |
| `adapter` | string | ✗ | `null` | LoRA adapter path (HuggingFace only) |
| `max_new_tokens` | integer | ✗ | 128 | Maximum tokens to generate |
| `temperature` | float | ✗ | 0.0 | Sampling temperature |
| `return_raw` | boolean | ✗ | `false` | Include raw model output |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `formula` | string | Generated ATL formula |
| `model_key` | string | Model key used |
| `model_name` | string | Full model name |
| `provider` | string | `huggingface` or `azure` |
| `latency_ms` | float | Generation latency in milliseconds |
| `raw_output` | string | Raw model output (if `return_raw=true`) |

**Example Request:**
```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "The user can guarantee that after payment, the ticket is eventually printed",
    "model": "qwen-3b",
    "few_shot": true,
    "max_new_tokens": 128
  }'
```

**Example Response:**
```json
{
  "formula": "<<User>>G (paid -> F ticket_printed)",
  "model_key": "qwen-3b",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "provider": "huggingface",
  "latency_ms": 425.3
}
```

---

#### `GET /models`

List available models from configuration.

**Response:**
```json
{
  "models": [
    {
      "key": "qwen-3b",
      "name": "Qwen/Qwen2.5-3B-Instruct",
      "provider": "huggingface",
      "params_b": 3
    },
    {
      "key": "gpt-5.2",
      "name": "gpt-5.2",
      "provider": "azure",
      "params_b": 0
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8081/models
```

---

### Error Responses

All endpoints return structured errors:

```json
{
  "detail": "Error message here"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (invalid model key)
- `500` - Internal Server Error (generation failure)

---

### Environment Variables

Required for API service:

```bash
# Default model (optional)
NL2ATL_DEFAULT_MODEL=qwen-3b

# Config paths (required if not running from repo root)
NL2ATL_MODELS_CONFIG=configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=configs/experiments.yaml

# Azure (if using Azure models)
AZURE_API_KEY=your_key
AZURE_INFER_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_INFER_MODEL=gpt-5.2

# HuggingFace (if using gated models)
HUGGINGFACE_TOKEN=your_token
```

---

## Python API

For programmatic use of NL2ATL components in Python scripts.

### Experiment Module

#### `ExperimentRunner`

Main class for running experiments.

```python
from pathlib import Path
from src.experiment import ExperimentRunner
from src.config import Config

# Load configuration
config = Config.from_yaml(
    models_yaml=Path("configs/models.yaml"),
    experiments_yaml=Path("configs/experiments.yaml")
)

# Create runner
runner = ExperimentRunner(config)

# Run experiment
result = runner.run_experiment(
    model_key="qwen-3b",
    condition="baseline_few_shot",
    seed=42
)

print(f"Accuracy: {result.metrics['exact_match']}")
```

**Key Methods:**
- `run_experiment(model_key, condition, seed)` - Run single experiment
- `run_all_experiments(model_keys, conditions, seeds)` - Run multiple experiments

---

#### `ExperimentDataManager`

Handles data loading, splitting, and augmentation.

```python
from pathlib import Path
from src.experiment import ExperimentDataManager

manager = ExperimentDataManager(
    data_path=Path("data/dataset.json"),
    test_size=0.30,
    val_size=0.6667,  # 2/3 of non-test as validation
    seed=42,
    augment_factor=10
)

# Prepare data splits
train_aug, val, test, full = manager.prepare_data()

print(f"Train: {len(train_aug)} (augmented)")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")
```

**Key Methods:**
- `prepare_data()` - Returns (train_augmented, validation, test, full_dataset)
- `load_dataset()` - Load raw dataset
- `split_dataset()` - Stratified split by difficulty

---

### Model Module

#### Model Registry

```python
from src.models import load_model, ModelConfig

# Load model from registry
model, tokenizer = load_model(
    model_key="qwen-3b",
    models_config=model_config,
    load_in_4bit=False
)

# Generate prediction
prompt = "Generate ATL: User can eventually print ticket"
output = model.generate(
    tokenizer(prompt, return_tensors="pt").input_ids,
    max_new_tokens=128
)
formula = tokenizer.decode(output[0], skip_special_tokens=True)
```

**Key Functions:**
- `load_model(model_key, models_config, **kwargs)` - Load model and tokenizer
- `format_prompt(example, few_shot_examples=None)` - Format input prompt
- `extract_formula(raw_output)` - Extract formula from model output

---

### Evaluation Module

#### Exact Match Evaluator

```python
from src.evaluation import ExactMatchEvaluator

evaluator = ExactMatchEvaluator()

# Evaluate single prediction
result = evaluator.evaluate(
    prediction="<<User>>F ticket_printed",
    reference="<<User>>F ticket_printed"
)
print(result.exact_match)  # True

# Evaluate dataset
predictions = [
    {"generated": "<<A>>F p", "expected": "<<A>>F p"},
    {"generated": "<<B>>G q", "expected": "<<B>>G !q"}
]
metrics = evaluator.evaluate_dataset(predictions)
print(f"Accuracy: {metrics['exact_match']}")
```

**Key Methods:**
- `evaluate(prediction, reference)` - Single comparison
- `evaluate_dataset(predictions)` - Batch evaluation
- `normalize_formula(formula)` - Normalize for comparison

---

#### LLM Judge

```python
from src.evaluation.llm_judge import LLMJudge, evaluate_prediction_file

# Create judge
judge = LLMJudge(
    model="gpt-5.2",
    temperature=0.0
)

# Evaluate single prediction
verdict = judge.evaluate(
    input_text="User can eventually print ticket",
    prediction="<<User>>F ticket_printed",
    reference="<<User>>F ticket_printed"
)
print(verdict.correct)  # "yes"
print(verdict.reasoning)

# Evaluate entire prediction file
evaluate_prediction_file(
    predictions_file=Path("outputs/model_predictions/qwen-3b.json"),
    output_file=Path("outputs/LLM-evaluation/qwen-3b_judged.json"),
    judge_model="gpt-5.2"
)
```

**Key Classes:**
- `LLMJudge` - Judge implementation
- `JudgeVerdict` - Verdict data class (contains `decision: str` and `reasoning: str`)

**Key Functions:**
- `evaluate_prediction_file(...)` - Evaluate entire file
- `compute_metrics(results)` - Aggregate metrics
- `build_summary(eval_files)` - Create summary report

---

#### Agreement Analysis

```python
from src.evaluation.judge_agreement import generate_agreement_report

# Compute inter-rater agreement
report = generate_agreement_report(
    eval_dir=Path("outputs/LLM-evaluation/evaluated_datasets"),
    output_path=Path("outputs/LLM-evaluation/agreement_report.json"),
    judges=["gpt-4", "gpt-5.2", "claude-3.5"],
    include_disagreements=True,
    max_disagreements=10
)

print(f"Cohen's κ: {report['pairwise_agreements']}")
print(f"Fleiss' κ: {report['fleiss_kappa']}")
```

**Key Functions:**
- `generate_agreement_report(...)` - Compute all agreement metrics
- `generate_agreement_report_with_human(...)` - Include human annotations
- `compute_cohens_kappa(rater1, rater2)` - Pairwise agreement
- `compute_fleiss_kappa(ratings)` - Multi-rater agreement
- `compute_krippendorff_alpha(ratings)` - Krippendorff's alpha

---

#### Efficiency Analysis

```python
from src.evaluation.model_efficiency import build_efficiency_report

# Generate efficiency report
report = build_efficiency_report(
    predictions_dir=Path("outputs/model_predictions"),
    output_path=Path("outputs/LLM-evaluation/efficiency_report.json"),
    weight_accuracy=0.5,
    weight_cost=0.3,
    weight_latency=0.2
)

# Access rankings
print("Most accurate:", report['rankings']['by_accuracy'][0])
print("Cheapest:", report['rankings']['by_cost'][0])
print("Fastest:", report['rankings']['by_latency'][0])
print("Best composite:", report['rankings']['by_composite'][0])
```

**Key Functions:**
- `build_efficiency_report(...)` - Complete efficiency analysis
- `build_efficiency_notebook(...)` - Generate Jupyter notebook
- `calculate_composite_score(...)` - Weighted efficiency score

---

#### Difficulty Classification

```python
from src.evaluation.difficulty import (
    classify_difficulty,
    process_dataset,
    formula_complexity_score,
    nl_ambiguity_score
)

# Classify single example
label, scores = classify_difficulty(
    nl_input="User can guarantee ticket is eventually printed after payment",
    formula="<<User>>G (paid -> F ticket_printed)",
    formula_weight=0.4,
    nl_weight=0.6,
    threshold=5.0
)

print(f"Difficulty: {label}")
print(f"Formula complexity: {scores['formula_complexity']}")
print(f"NL ambiguity: {scores['nl_ambiguity']}")

# Process entire dataset
updated_dataset = process_dataset(
    input_path=Path("data/dataset.json"),
    output_path=Path("data/dataset_labeled.json"),
    verbose=True
)
```

**Key Functions:**
- `classify_difficulty(nl_input, formula, ...)` - Single classification
- `process_dataset(input_path, output_path, ...)` - Batch processing
- `formula_complexity_score(formula)` - Formula complexity
- `nl_ambiguity_score(nl_text, formula)` - NL ambiguity

---

## Configuration Classes

### Config

Main configuration class.

```python
from pathlib import Path
from src.config import Config

# Load from YAML files
config = Config.from_yaml(
    models_yaml=Path("configs/models.yaml"),
    experiments_yaml=Path("configs/experiments.yaml")
)

# Access configuration
print(config.experiment_name)
print(config.data_path)
print(config.models)
print(config.conditions)
```

**Attributes:**
- `experiment_name` - Experiment identifier
- `seed` - Random seed
- `num_seeds` - Number of seeds for sweep
- `data_path` - Path to dataset
- `test_size` - Test split ratio
- `val_size` - Validation split ratio
- `augment_factor` - Data augmentation factor
- `models` - Dictionary of model configurations
- `conditions` - List of experiment conditions
- `training` - Training hyperparameters
- `wandb` - Weights & Biases config

---

### ModelConfig

Model-specific configuration.

```python
from src.models import ModelConfig

model_config = ModelConfig(
    name="Qwen/Qwen2.5-3B-Instruct",
    short_name="qwen-3b",
    provider="huggingface",
    max_seq_length=512,
    load_in_4bit=False,
    lora_r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

**Attributes:**
- `name` - Full model name (HuggingFace ID or Azure deployment)
- `short_name` - Short identifier
- `provider` - `huggingface` or `azure`
- `max_seq_length` - Maximum sequence length
- `load_in_4bit` - Enable 4-bit quantization
- `lora_r` - LoRA rank
- `lora_alpha` - LoRA alpha parameter
- `target_modules` - Modules to apply LoRA
- `price_input_per_1k` - Input token price (USD per 1K)
- `price_output_per_1k` - Output token price (USD per 1K)
- `gpu_hour_usd` - GPU cost per hour (for local models)

---

## Infrastructure Utilities

### I/O Functions

```python
from pathlib import Path
from src.infra.io import load_json, save_json, load_yaml, save_yaml

# JSON operations
data = load_json(Path("data/dataset.json"))
save_json(data, Path("output.json"), indent=2)

# YAML operations
config = load_yaml(Path("configs/models.yaml"))
save_yaml(config, Path("configs/models_backup.yaml"))

# Safe loading (returns default on error)
from src.infra.io import load_json_safe
data = load_json_safe(Path("may_not_exist.json"), default=[])
```

**Key Functions:**
- `load_json(path)` - Load JSON file
- `save_json(data, path, indent=2)` - Save JSON file
- `load_yaml(path)` - Load YAML file
- `save_yaml(data, path)` - Save YAML file
- `load_json_safe(path, default=None)` - Safe JSON loading

---

### Azure Client

```python
from src.infra.azure import AzureClient, AzureConfig

# Create client from environment
config = AzureConfig.from_env()
client = AzureClient(config)

# Generate text
result = client.generate(
    prompt="Generate ATL for: User can eventually print ticket",
    max_new_tokens=128,
    return_usage=True
)

print(result.text)
print(f"Input tokens: {result.usage.input_tokens}")
print(f"Output tokens: {result.usage.output_tokens}")
```

**Classes:**
- `AzureConfig` - Azure configuration
- `AzureClient` - Azure API client
- `GenerationResult` - Generation result with usage stats

**Key Methods:**
- `AzureConfig.from_env()` - Load config from environment
- `client.generate(prompt, ...)` - Generate text
- `client.generate_batch(prompts, ...)` - Batch generation

---

## Constants and Enums

```python
from src.constants import (
    Provider,
    ModelType,
    TEMPORAL_OPERATORS,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_LLM_EVAL_DIR
)

# Provider enum
print(Provider.HUGGINGFACE)  # Provider.HUGGINGFACE
print(Provider.AZURE)  # Provider.AZURE

# ModelType enum
print(ModelType.QWEN)  # ModelType.QWEN
print(ModelType.LLAMA)  # ModelType.LLAMA
print(ModelType.MISTRAL)  # ModelType.MISTRAL
print(ModelType.PHI3)  # ModelType.PHI3
print(ModelType.GEMMA)  # ModelType.GEMMA
print(ModelType.GENERIC)  # ModelType.GENERIC

# Temporal operators used in ATL formulas
print(TEMPORAL_OPERATORS)  # frozenset({"G", "F", "X", "U", "W", "R"})

# Default paths
print(DEFAULT_DATA_PATH)  # "./data/dataset.json"
print(DEFAULT_OUTPUT_DIR)  # "./outputs"
print(DEFAULT_PREDICTIONS_DIR)  # "outputs/model_predictions"
print(DEFAULT_LLM_EVAL_DIR)  # "outputs/LLM-evaluation"
```

### Judge Responses

LLM judge responses use simple string values (not enums):
- **Correct**: `"yes"`
- **Incorrect**: `"no"`

Example from `JudgeVerdict`:
```python
@dataclass
class JudgeVerdict:
    decision: str  # "yes" or "no"
    reasoning: Optional[str] = None
```

### Difficulty Labels

Difficulty classification uses string values (not enums):
- **Easy**: `"easy"`
- **Hard**: `"hard"`

These are assigned directly in the dataset:
```json
{
  "id": "ex01",
  "difficulty": "easy"
}
```

---

## Type Hints

NL2ATL uses Python type hints throughout. Key types:

```python
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Common type aliases
DatasetItem = Dict[str, Any]
Dataset = List[DatasetItem]
Metrics = Dict[str, float]
PredictionResult = Dict[str, Any]
```

---

## Complete Example: Custom Evaluation Pipeline

```python
from pathlib import Path
from src.experiment import ExperimentRunner, ExperimentDataManager
from src.config import Config
from src.evaluation import ExactMatchEvaluator
from src.evaluation.llm_judge import LLMJudge, evaluate_prediction_file
from src.evaluation.judge_agreement import generate_agreement_report
from src.evaluation.model_efficiency import build_efficiency_report

# 1. Setup
config = Config.from_yaml(
    models_yaml=Path("configs/models.yaml"),
    experiments_yaml=Path("configs/experiments.yaml")
)

# 2. Run experiments
runner = ExperimentRunner(config)
result = runner.run_experiment(
    model_key="qwen-3b",
    condition="baseline_few_shot",
    seed=42
)

# 3. LLM judge evaluation
evaluate_prediction_file(
    predictions_file=Path(f"outputs/model_predictions/{result.run_id}.json"),
    output_file=Path(f"outputs/LLM-evaluation/{result.run_id}_judged.json"),
    judge_model="gpt-5.2"
)

# 4. Compute agreement (if multiple judges)
agreement_report = generate_agreement_report(
    eval_dir=Path("outputs/LLM-evaluation/evaluated_datasets"),
    output_path=Path("outputs/LLM-evaluation/agreement_report.json")
)

# 5. Efficiency analysis
efficiency_report = build_efficiency_report(
    predictions_dir=Path("outputs/model_predictions"),
    output_path=Path("outputs/LLM-evaluation/efficiency_report.json")
)

print("Pipeline complete!")
print(f"Accuracy: {result.metrics['exact_match']}")
print(f"Agreement: {agreement_report['fleiss_kappa']}")
print(f"Best model: {efficiency_report['rankings']['by_composite'][0]}")
```

---

## Error Handling

All public functions raise appropriate exceptions:

```python
from src.config import Config

try:
    config = Config.from_yaml(
        models_yaml=Path("invalid.yaml"),
        experiments_yaml=Path("configs/experiments.yaml")
    )
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
except ValueError as e:
    print(f"Invalid config: {e}")
```

**Common Exceptions:**
- `FileNotFoundError` - Missing config or data files
- `ValueError` - Invalid configuration values
- `KeyError` - Missing required config keys
- `RuntimeError` - Model loading or inference errors

---

## Further Reading

- **Usage Guide**: [usage.md](usage.md) - CLI command reference
- **Configuration**: [configuration.md](configuration.md) - Config file format
- **Architecture**: [architecture.md](architecture.md) - System design
- **Development**: [development.md](development.md) - Extending NL2ATL

---

## API Versioning

Current version: `0.1.0`

The API follows semantic versioning. Breaking changes will increment the major version.

---

## Support

- **Documentation**: See [index.md](index.md) for full documentation
- **Issues**: [GitHub Issues](https://github.com/vladanaSTM/nl2atl/issues)
- **Examples**: Check `tests/` directory for usage examples
