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
uv run python -c "from src.data_utils import load_data; d=load_data('data/dataset_gold.json'); print(len(d)); print(d[0]['input']); print(d[0]['outputs'])"
```

The loader validates rows and records every accepted gold formula in `outputs`.

## Run One Experiment

```bash
uv run nl2atl run --models qwen-3b --conditions baseline_few_shot --seed 42
```

This loads the dataset, creates a seeded split, loads the model, generates formulas for the test split, and writes a prediction file under `outputs/model_predictions/`.

`nl2atl run` uses Hugging Face models by default. Use `--model_provider azure` for Azure generation baselines, or `--model_provider all` when you intentionally want both runnable providers. Existing prediction files are skipped unless you add `--overwrite`.

Preview the task plan before running a sweep:

```bash
uv run nl2atl run --list-tasks --models all --conditions all --model_provider hf
uv run nl2atl run --count --models all --conditions all --model_provider hf
```

To submit the selected sweep to SLURM without an array concurrency cap:

```bash
uv run nl2atl run --slurm --models all --conditions all
```

Add `--max-parallel-gpus N` only when you want to throttle the array to `N` concurrent one-GPU tasks.

The SLURM runner writes a frozen task manifest under `outputs/manifests/` so array tasks use the same model, condition, and seed plan that was submitted.

Use `--dry-run` to print the generated batch script, or `--no-submit --script-path path/to/job.sbatch` to write the script and manifest without calling `sbatch`.

For fine-tuning smoke tests, use a short step cap:

```bash
uv run nl2atl run --slurm \
	--models all --conditions all --model_provider hf \
	--train-max-steps 20
```

Fine-tuned zero-shot and fine-tuned few-shot runs share one adapter per model and seed. Multi-seed runs write adapters under `models/<model>_finetuned_seed<seed>/final`; single-seed runs omit the seed suffix. Existing adapters are reused unless `--overwrite` is set.

For a quick check that predictions come out in the expected format, evaluate just a couple of test examples. Azure/API models run locally:

```bash
uv run nl2atl run \
	--models gpt-5.4 --model_provider azure \
	--conditions baseline_zero_shot \
	--max-eval-samples 2
```

Hugging Face models need a GPU, so smoke-test them through SLURM; the cap is forwarded to each array worker:

```bash
uv run nl2atl run --slurm \
	--models all --model_provider hf \
	--conditions baseline_zero_shot \
	--max-eval-samples 2
```

`--max-eval-samples 2` runs a stratified sample of the test set, giving one single-formula and one multi-formula (QSA) example so you can confirm the output shape for both. Predictions land in `outputs/model_predictions/smoke_test/` so they stay separate from real runs. Use it for smoke checks only, never for reported numbers.

## Inspect Predictions

```bash
uv run python -c "from src.infra.io import load_json; r=load_json('outputs/model_predictions/qwen-3b_baseline_few_shot_seed42.json'); p=r['predictions'][0]; print(p['input']); print(p['expected_options']); print(p['generated']); print(p['exact_match'])"
```

Each prediction row includes the input, accepted gold formulas, minimally cleaned model output, the raw generation when cleaning changed it, prompt hash, deterministic decoding settings, few-shot example IDs when used, token usage when available, exact-match flag, and latency. The top-level metadata also records dataset/config hashes, command arguments, and a split manifest path/hash.

## Evaluate And Report

```bash
uv run nl2atl llm-judge --datasets all
uv run nl2atl judge-agreement
uv run nl2atl generate-eval-reports
```

Exact matches are accepted automatically. The LLM judge is called only for non-exact predictions and receives all accepted gold formulas. Judged rows preserve the prompt hash, raw response, parse status, prompt version, and judge latency. Reports are written under `outputs/LLM-evaluation/`.

By default, `llm-judge` evaluates with GPT-5.2 and DeepSeek V3.2. Override that with `--models gpt-5.2` or another Azure judge model from [../configs/models.yaml](../configs/models.yaml). Add `--no_llm` for an offline exact-match-only pass.

## Common Commands

| Command | Purpose |
|---|---|
| `uv run nl2atl run --models qwen-3b --conditions baseline_few_shot --seed 42` | Run one model, condition, and seed |
| `uv run nl2atl run --models all --conditions all --model_provider hf` | Run selected local models, conditions, and seeds |
| `uv run nl2atl run --slurm --models all --conditions all` | Submit an uncapped SLURM array |
| `uv run nl2atl run --slurm --no-submit --script-path outputs/manifests/job.sbatch` | Write the SLURM script and task manifest without submitting |
| `uv run nl2atl run --list-tasks --models all --conditions all` | Inspect the planned model/seed tasks |
| `uv run nl2atl llm-judge --datasets all` | Judge non-exact predictions semantically |
| `uv run nl2atl llm-judge --datasets all --no_llm` | Recompute exact-match-only judged artifacts without API calls |
| `uv run nl2atl generate-eval-reports` | Build summaries, agreement, aggregates, and accuracy-latency reports |
| `uv run nl2atl model-efficiency` | Rebuild only the accuracy-latency report |

Use `uv run nl2atl COMMAND --help` for the full option list.

## API Service

```bash
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
curl http://localhost:8081/health
```

`POST /generate` accepts `description`, optional `model`, `few_shot`, `num_few_shot`, `max_new_tokens`, `adapter`, and `return_raw`.

Set `NL2ATL_DEFAULT_MODEL`, `NL2ATL_MODELS_CONFIG`, or `NL2ATL_EXPERIMENTS_CONFIG` to change API defaults without editing YAML. Adapters are supported only for Hugging Face models.

## Outputs

| Path | Contents |
|---|---|
| `outputs/model_predictions/` | Prediction rows and run metadata |
| `outputs/split_manifests/` | Train/validation/test membership and dataset hash per run |
| `outputs/manifests/` | Frozen SLURM task manifests and optional generated scripts |
| `outputs/LLM-evaluation/evaluated_datasets/` | LLM judge decisions |
| `outputs/LLM-evaluation/agreement_report.json` | Judge agreement report |
| `outputs/LLM-evaluation/seed_aggregate_metrics_from_judged.json` | Metrics aggregated across seeds |
| `outputs/LLM-evaluation/efficiency_report.json` | Accuracy, latency, rankings, and Pareto frontier |
| `outputs/LLM-evaluation/publication_analysis.ipynb` | Single notebook for publication analysis |
| `models/` | Fine-tuned LoRA/QLoRA adapters |