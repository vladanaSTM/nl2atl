# Quickstart

Use this page to install NL2ATL, run one model, and find the output files.

## Install

NL2ATL uses Python 3.10+ and uv.

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
python -m pip install uv
uv sync --group dev
uv run nl2atl --help
```

Create `.env` from `.env.example` when you need Azure-hosted models or gated Hugging Face models. Common variables are `AZURE_API_KEY`, `AZURE_INFER_ENDPOINT`, and `HUGGINGFACE_TOKEN`.

## Inspect The Dataset

```bash
uv run python -c "from src.data_utils import load_data; d=load_data('data/dataset_gold_no_difficulty.json'); print(len(d)); print(d[0]['input']); print(d[0]['outputs'])"
```

The loader validates rows and records every accepted gold formula in `outputs`.

## Run One Experiment

```bash
uv run nl2atl run-single --model qwen-3b --few_shot
```

This loads the dataset, creates a seeded split, loads the model, generates formulas for the test split, and writes a prediction file under `outputs/model_predictions/`.

## Inspect Predictions

```bash
uv run python -c "from src.infra.io import load_json; r=load_json('outputs/model_predictions/qwen-3b_baseline_few_shot.json'); p=r['predictions'][0]; print(p['input']); print(p['expected_options']); print(p['generated']); print(p['exact_match'])"
```

Each prediction row includes the input, accepted gold formulas, generated formula, exact-match flag, and latency.

## Evaluate And Report

```bash
uv run nl2atl llm-judge --datasets all
uv run nl2atl judge-agreement
uv run nl2atl generate-eval-reports
```

Exact matches are accepted automatically. The LLM judge is called only for non-exact predictions and receives all accepted gold formulas. Reports are written under `outputs/LLM-evaluation/`.

## Common Commands

| Command | Purpose |
|---|---|
| `uv run nl2atl run-single --model qwen-3b --few_shot` | Run one model and condition |
| `uv run nl2atl run-all --models qwen-3b --conditions baseline_zero_shot` | Run selected models, conditions, and seeds |
| `uv run nl2atl llm-judge --datasets all` | Judge non-exact predictions semantically |
| `uv run nl2atl generate-eval-reports` | Build summaries, agreement, aggregates, and accuracy-latency reports |
| `uv run nl2atl model-efficiency --aggregate_file outputs/LLM-evaluation/seed_aggregate_metrics_from_judged.json --output_dir outputs/LLM-evaluation` | Rebuild only the accuracy-latency report |

Use `uv run nl2atl COMMAND --help` for the full option list.

## API Service

```bash
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
curl http://localhost:8081/health
```

`POST /generate` accepts `description`, optional `model`, `few_shot`, `num_few_shot`, `max_new_tokens`, `adapter`, and `return_raw`.

## Outputs

| Path | Contents |
|---|---|
| `outputs/model_predictions/` | Prediction rows and run metadata |
| `outputs/LLM-evaluation/evaluated_datasets/` | LLM judge decisions |
| `outputs/LLM-evaluation/agreement_report.json` | Judge agreement report |
| `outputs/LLM-evaluation/seed_aggregate_metrics_from_judged.json` | Metrics aggregated across seeds |
| `outputs/LLM-evaluation/efficiency_report.json` | Accuracy, latency, rankings, and Pareto frontier |