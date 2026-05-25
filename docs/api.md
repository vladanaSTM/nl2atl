# API Reference

This page lists the project APIs most useful for scripts and integrations. Prefer these entry points over importing deep helper functions.

## Data Utilities

```python
from src.data_utils import load_data, split_data, augment_data, get_output_options, get_preferred_output
```

```python
data = load_data("data/dataset_gold_no_difficulty.json")
train, val, test = split_data(data, train_size=0.7, val_size=0.1, test_size=0.2)
train_aug = augment_data(train, augment_factor=10)
formula = get_preferred_output(data[0])
accepted = get_output_options(data[0])
```

`load_data` validates rows, preserves all accepted formulas in `outputs`, and keeps the first preferred formula in `output` for compatibility.

## Configuration

```python
from src.config import Config

config = Config.from_yaml("configs/models.yaml", "configs/experiments.yaml")
print(config.data_path)
print(config.train_size, config.val_size, config.test_size)
```

## Experiments

```python
from src.config import Config, ExperimentCondition
from src.experiment import ExperimentRunner

config = Config.from_yaml("configs/models.yaml", "configs/experiments.yaml")
runner = ExperimentRunner(config)
condition = ExperimentCondition(
    name="baseline_few_shot",
    finetuned=False,
    few_shot=True,
)
runner.run_single_experiment("qwen-3b", condition)
```

## Prompt Formatting

```python
from src.models.few_shot import format_prompt

prompt = format_prompt(
    input_text="The user can eventually print a ticket.",
    few_shot=True,
    model_type="qwen",
)
```

## Exact-Match Evaluation

```python
from src.evaluation.exact_match import ExactMatchEvaluator

evaluator = ExactMatchEvaluator()
result = evaluator.evaluate_single(
    {"input": "x", "generated": "<<A>>F p"},
    {"input": "x", "output": "<<A>>F p"},
)
print(result["exact_match"])
```

## Difficulty Classification

```python
from src.evaluation.difficulty import classify_difficulty, process_dataset

label, scores = classify_difficulty(
    "The user can eventually print a ticket.",
    "<<User>>F ticket_printed",
)

process_dataset(
    input_path="data/dataset_gold_no_difficulty.json",
    output_path="data/dataset_with_difficulty.json",
)
```

Difficulty labels are optional and are not used for splitting.

## FastAPI Service

Start the server:

```bash
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

Health check:

```bash
curl http://localhost:8081/health
```

Generate a formula:

```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "The user can eventually print a ticket.",
    "model": "qwen-3b",
    "few_shot": true,
    "max_new_tokens": 128
  }'
```

Response shape:

```json
{
  "formula": "<<User>>F ticket_printed",
  "model_key": "qwen-3b",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "provider": "huggingface",
  "latency_ms": 412.7,
  "raw_output": null
}
```

Useful API environment variables:

| Variable | Meaning |
|---|---|
| `NL2ATL_MODELS_CONFIG` | Override model config path |
| `NL2ATL_EXPERIMENTS_CONFIG` | Override experiment config path |
| `NL2ATL_DEFAULT_MODEL` | Default model key when request omits `model` |
