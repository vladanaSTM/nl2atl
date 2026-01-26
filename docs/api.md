# API Reference

This document summarizes the public API surface based on the current source code.

## Table of Contents

- [Configuration](#configuration)
- [Constants](#constants)
- [Experiment Module](#experiment-module)
- [Models Module](#models-module)
- [Evaluation Module](#evaluation-module)
- [Infrastructure Module](#infrastructure-module)

---

## Configuration

### `src.config.ModelConfig`

Dataclass describing a model entry from `configs/models.yaml`.

Key fields: `name`, `short_name`, `provider`, `api_model`, `max_seq_length`, `load_in_4bit`, `lora_r`, `lora_alpha`, `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `target_modules`, `params_b`.

Property: `is_azure` returns true when `provider` is `azure`.

### `src.config.ExperimentCondition`

Dataclass for experiment conditions:

- `name`
- `finetuned`
- `few_shot`

### `src.config.Config`

Main configuration container. Load with:

```python
Config.from_yaml(models_path: str, experiments_path: str) -> Config
```

Utility methods:

- `resolve_seeds()`
- `get_model(model_key: str)`

---

## Constants

`src.constants` includes:

- `Provider` and `ModelType` enums
- `TEMPORAL_OPERATORS`
- `DEFAULT_*` paths
- `AZURE_PREFIX`

---

## Experiment Module

### `src.experiment.ExperimentRunner`

Orchestrates model training (optional), inference, and evaluation.

Key methods:

- `run_single_experiment(model_key: str, condition: ExperimentCondition, save_model: bool = True)`
- `run_all_experiments(models: Optional[List[str]], conditions: Optional[List[str]], model_provider: str, overwrite: bool)`
- `run()`

### `src.experiment.ExperimentDataManager`

Handles data loading/splitting/augmentation.

```python
ExperimentDataManager(
    data_path: Path,
    test_size: float,
    val_size: float,
    seed: int,
    augment_factor: int,
)
```

Key methods: `load_dataset()`, `split_dataset()`, `augment_training_data()`, `prepare_data()`.

### `src.experiment.ExperimentReporter`

Responsible for saving results and W&B logging.

Key methods: `init_wandb_run()`, `build_run_metadata()`, `save_result()`, `log_metrics()`, `log_predictions_table()`, `finalize()`.

---

## Models Module

### `src.models.registry`

Public functions (also re-exported by `src.models`):

- `load_model(model_config, for_training=False, load_adapter=None)`
- `get_model_type(model_name: str)`
- `clear_gpu_memory()`
- `generate(model, tokenizer, prompt, max_new_tokens=256, return_usage=False)`

### `src.models.utils`

- `normalize_model_token(token, prefixes=(AZURE_PREFIX,))`
- `resolve_model_key(model_arg, models, require_mapping_entries=False, match_key_lower=True)`

### `src.models.few_shot`

- `get_few_shot_examples(n=5, seed=None, exclude_inputs=None)`
- `get_system_prompt(few_shot=False, num_examples=5, seed=42, exclude_inputs=None)`
- `format_prompt(input_text, output_text=None, few_shot=False, num_examples=5, model_type=ModelType.QWEN, exclude_inputs=None, tokenizer=None)`

---

## Evaluation Module

### `src.evaluation.BaseEvaluator`

Abstract base class with:

- `evaluate(predictions, references)`
- `evaluate_single(prediction, reference)`

### `src.evaluation.ExactMatchEvaluator`

Evaluates and normalizes ATL outputs.

Key methods:

- `evaluate(...)` (model-based or predictions-based)
- `evaluate_predictions(predictions, references)`

### `src.evaluation.DifficultyClassifier`

Wraps `classify_difficulty()` and `process_dataset()`.

### `src.evaluation.generate_agreement_report`

Generates Cohen’s κ, Fleiss’ κ, and Krippendorff’s α reports from evaluated datasets.

### `src.evaluation.model_efficiency`

Key functions:

- `build_efficiency_report(...)`
- `build_efficiency_notebook(...)`

### `src.evaluation.judge_agreement.generate_agreement_report_with_human`

Same as `generate_agreement_report`, but merges a human annotation JSON as an additional judge.

### `src.evaluation.llm_judge`

Key classes and functions:

- `LLMJudge`
- `LLMJudgeEvaluator`
- `evaluate_prediction_file(...)`
- `compute_metrics(...)`
- `build_summary(...)`
- `build_summary_notebook(...)`
- `run_llm_judge(...)`

Data classes:

- `JudgeDecision`
- `JudgePromptConfig`
- `JudgeVerdict`
- `JudgeMetrics`

---

## Infrastructure Module

### `src.infra.azure`

- `AzureConfig.from_env()` and `AzureConfig.from_env_optional()`
- `AzureClient.generate(prompt, max_new_tokens=256, return_usage=False)`
- `GenerationResult` dataclass (text + usage token counts)

### `src.infra.io`

- `load_yaml(path)`
- `load_json(path)`
- `save_json(data, path)`
- `load_json_safe(path, default=None)`

### `src.infra.env`

- `load_env()`
```python
def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with UTF-8 encoding."""
    
def save_json(
    data: Dict[str, Any], 
    path: Path, 
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with safe loading."""
    
def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save data to YAML file."""
```

---


```python
from enum import Enum

class Provider(Enum):
    """Model provider types."""
    AZURE = "azure"
    LOCAL = "local"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

class Verdict(Enum):
    """Judge verdict types."""
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"

class Difficulty(Enum):
    """Sample difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ATL* Operators
ATL_OPERATORS = {"G", "F", "X", "U", "R", "W"}

# Default paths
DEFAULT_DATA_PATH = Path("data/dataset.json")
DEFAULT_OUTPUT_DIR = Path("outputs/model_predictions")
DEFAULT_EVAL_DIR = Path("outputs/LLM-evaluation")

# Default parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = 42
```
