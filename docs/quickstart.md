# Quickstart

This page runs one experiment and shows where the output goes.

## 1. Verify the CLI

```bash
uv run nl2atl --help
```

You should see commands such as `run-single`, `run-all`, `llm-judge`, `judge-agreement`, `model-efficiency`, `classify-difficulty`, and `slurm`.

## 2. Inspect the Dataset

```bash
uv run python -c "from src.data_utils import load_data; data = load_data('data/dataset_gold_no_difficulty.json'); print(len(data)); print(data[0]['input']); print(data[0]['output'])"
```

Rows are loaded from JSON and normalized to `input` and `output` in memory.

## 3. Run One Experiment

```bash
uv run nl2atl run-single --model qwen-3b --few_shot
```

This will:

1. Load config from [../configs/experiments.yaml](../configs/experiments.yaml) and [../configs/models.yaml](../configs/models.yaml).
2. Load and validate the dataset.
3. Create a seeded train/validation/test split.
4. Load the selected model.
5. Generate formulas for the test split.
6. Save predictions and metadata under `outputs/model_predictions/`.

Default splits are not stratified. They use the explicit proportions in config.

## 4. Inspect Predictions

```bash
uv run python -c "from src.infra.io import load_json; result = load_json('outputs/model_predictions/qwen-3b_baseline_few_shot.json'); row = result['predictions'][0]; print(row['input']); print(row['expected']); print(row['generated']); print(row['exact_match'])"
```

Prediction files look like this:

```json
{
  "metadata": {
    "run_id": "qwen-3b_baseline_few_shot",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "total_samples": 30,
    "metrics": {
      "exact_match": 0.82
    }
  },
  "predictions": [
    {
      "input": "...",
      "expected": "<<User>>F ticket_printed",
      "generated": "<<User>>F ticket_printed",
      "exact_match": 1,
      "latency_ms": 412.7
    }
  ]
}
```

## 5. Optional LLM Judge

Exact match is strict, but it accepts any listed gold formula for rows with multiple correct answers. Use the LLM judge when you want semantic evaluation for non-exact predictions:

```bash
uv run nl2atl llm-judge --datasets all
```

Results are written to `outputs/LLM-evaluation/`.

## 6. Optional Efficiency Report

```bash
uv run nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

The report combines accuracy, cost, latency, and throughput information when those fields are available.

## Next Pages

- [Dataset](dataset.md)
- [Configuration](configuration.md)
- [Evaluation](evaluation.md)
- [Architecture](architecture.md)
