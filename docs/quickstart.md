# Quick Start

Get up and running with NL2ATL in a few minutes.

## Prerequisites

Ensure you have completed the [Installation](installation.md) steps.

---

## 1. Run a Single Experiment

```bash
nl2atl run-single --model qwen-3b --few_shot
```

This runs the default dataset split and writes predictions to:

```
outputs/model_predictions/<run_name>.json
```

---

## 2. Run All Experiments

```bash
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
```

This iterates over the selected models and conditions from `configs/experiments.yaml`.

---

## 3. Evaluate Predictions with LLM Judge

```bash
nl2atl llm-judge --datasets all
```

Re-run existing evaluations with:

```bash
nl2atl llm-judge --datasets all --overwrite
```

Results are written under:

```
outputs/LLM-evaluation/evaluated_datasets/<judge>/
```

---

## 4. Compute Judge Agreement

```bash
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
```

This produces `outputs/LLM-evaluation/agreement_report.json`.

---

## 5. Compare Model Efficiency

```bash
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

This produces:

- `outputs/LLM-evaluation/efficiency_report.json`
- `outputs/LLM-evaluation/efficiency_report.ipynb`

The report is useful because it summarizes accuracy–cost–latency
trade-offs in a single, comparable view.

---

## 6. Explore the Dataset

```python
from src.infra.io import load_json

dataset = load_json("data/dataset.json")
sample = dataset[0]
print(sample["input"])
print(sample["output"])
```

---

## Next Steps

- [Usage Guide](usage.md) — Full CLI documentation
- [Dataset](dataset.md) — Dataset structure and format
- [Evaluation](evaluation.md) — Evaluation methods and metrics
- [Configuration](configuration.md) — Customizing experiments