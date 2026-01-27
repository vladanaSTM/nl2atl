# Quick Start

This guide runs a minimal end‑to‑end workflow: generate predictions, evaluate them, and inspect outputs.

## Prerequisite

Complete [installation.md](installation.md) first.

## Step 1 — Run experiments (SLURM recommended)

For multi‑model/seed sweeps, use SLURM arrays. Benefits include parallel GPU usage, scheduler‑managed
resources, and fault isolation.

```bash
nl2atl run-array --count
nl2atl run-array --list-tasks
```

Then submit the array job using the helper script:

```bash
sbatch scripts/slurm/submit_array.sh
```

This runs one $(seed, model, condition)$ per array task and writes predictions to
`outputs/model_predictions/`.

## Step 2 — Local fallback (single node)

Use this when SLURM is unavailable or for quick checks.

### Run a single experiment

```bash
nl2atl run-single --model qwen-3b --few_shot
```

Output:

- Predictions file in `outputs/model_predictions/<run_name>.json`

### Run a sweep of experiments

```bash
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
```

This uses `configs/experiments.yaml` to expand model/condition combinations.

## Step 3 — Evaluate with the LLM judge

```bash
nl2atl llm-judge --datasets all
```

To re‑evaluate existing outputs:

```bash
nl2atl llm-judge --datasets all --overwrite
```

Output:

- `outputs/LLM-evaluation/evaluated_datasets/<judge>/`

## Step 4 — Compute judge agreement

```bash
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
```

Output:

- `outputs/LLM-evaluation/agreement_report.json`

## Step 5 — Compare model efficiency

```bash
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

Outputs:

- `outputs/LLM-evaluation/efficiency_report.json`
- `outputs/LLM-evaluation/efficiency_report.ipynb`

These reports summarize accuracy–cost–latency trade‑offs.

## Step 6 — Inspect the dataset

```python
from src.infra.io import load_json

dataset = load_json("data/dataset.json")
sample = dataset[0]
print(sample["input"])
print(sample["output"])
```

## Step 7 — Optional: run the API service

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If running from another directory, set absolute config paths:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

## Next steps

- CLI details: [usage.md](usage.md)
- Experiment configuration: [configuration.md](configuration.md)
- Dataset schema: [dataset.md](dataset.md)
- Evaluation methods: [evaluation.md](evaluation.md)