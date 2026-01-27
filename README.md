# NL2ATL

Natural language to ATL (Alternating-Time Temporal Logic) formula generation, evaluation, and difficulty classification for the NL2ATL research project.

## Quick Start

- Create an environment, install dependencies, and enable the `nl2atl` CLI.
- Configure `.env` for Azure inference (optional if you only use local HuggingFace models).
- Run an experiment and (optionally) evaluate outputs with the LLM judge.

## Features

- **Experiments** — Run baseline and fine-tuned generation experiments.
- **Evaluation** — Exact-match scoring and LLM-as-a-judge evaluation.
- **Analysis** — Inter-rater agreement across judge models.
- **Efficiency** — Cost/latency/accuracy trade-off reporting for paper-ready comparisons.
- **Classification** — Rule-based dataset difficulty scoring.

## Project Structure

```
src/           Core library
src/cli/       CLI entry points
configs/       Experiment and model configurations
data/          Datasets
outputs/       Predictions and evaluation results
tests/         Unit tests
docs/          Documentation
```

## Installation

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development (enables `nl2atl` command):

```bash
pip install -e .
```

## Configuration

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_API_KEY` | No | Azure API key for judge/inference models |
| `AZURE_INFER_ENDPOINT` | No | Azure inference endpoint |
| `AZURE_INFER_MODEL` | No | Default Azure deployment/model name |
| `AZURE_API_VERSION` | No | API version (defaults to `2024-08-01-preview`) |
| `AZURE_USE_CACHE` | No | Enable Azure response caching (default: true) |
| `AZURE_VERIFY_SSL` | No | SSL verification (default: false) |
| `HUGGINGFACE_TOKEN` | No | Token for gated/private HF models |
| `WANDB_API_KEY` | No | W&B API key for experiment logging |
| `NL2ATL_DEFAULT_MODEL` | No | Default model key/name for the API service |
| `NL2ATL_MODELS_CONFIG` | No | Models config path (API service) |
| `NL2ATL_EXPERIMENTS_CONFIG` | No | Experiments config path (API service) |

## Usage

### CLI Commands

```bash
nl2atl <command> [options]
# or: python nl2atl.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `run-all` | Run experiments across multiple models/conditions |
| `run-single` | Run a single model/condition experiment |
| `llm-judge` | Evaluate prediction files with LLM judge |
| `judge-agreement` | Compute inter-rater agreement metrics |
| `model-efficiency` | Compare accuracy, cost, and latency across models |
| `classify-difficulty` | Score dataset difficulty |

### Examples

```bash
# Run experiments
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
nl2atl run-single --model qwen-3b --few_shot

# Evaluate predictions
nl2atl llm-judge --datasets all
nl2atl llm-judge --datasets all --overwrite  # re-evaluate existing outputs
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets

# Compare model efficiency
nl2atl model-efficiency --predictions_dir outputs/model_predictions

# Classify difficulty
nl2atl classify-difficulty --input data/dataset.json --verbose
```

### Subcommand Modules

Subcommand handlers live under `src/cli/` and can be invoked directly:

```bash
python -m src.cli.run_all_experiments
python -m src.cli.run_single_experiment
python -m src.cli.run_llm_judge
python -m src.cli.run_judge_agreement
python -m src.cli.run_model_efficiency
python -m src.cli.classify_difficulty
```

### API Service (for UI integration)

Run the lightweight FastAPI server for NL→ATL generation:

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

Example request:

```bash
curl -X POST http://localhost:8081/generate \
	-H "Content-Type: application/json" \
	-d '{
		"description": "Agent A can eventually reach goal",
		"model": "gpt-5.2",
		"few_shot": true,
		"max_new_tokens": 128
	}'
```

For genVITAMIN GUI wiring instructions, see [docs/integrations/genvitamin.md](docs/integrations/genvitamin.md).

### genVITAMIN Integration (one-click patch)

The integration is designed so genVITAMIN users only apply a single backend patch and set an env var.
**Only one file in genVITAMIN is modified**: `api/routes/ai/generate.py` (a timestamped backup is created).

Steps (Windows/Linux/macOS):

1) Clone genVITAMIN next to this repo.
2) Start NL2ATL API (from repo root so configs resolve):

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If you start NL2ATL from another working directory, set:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

3) Apply the patch from this repo:

```bash
python integrations/genvitamin/apply_genvitamin_patch.py \
	--genvitamin-path "/path/to/genVITAMIN"
```

4) Set env in genVITAMIN backend (persisted in genVITAMIN/api/.env or shell):

```bash
NL2ATL_URL=http://localhost:8081
```

5) Start genVITAMIN backend and frontend as usual.

Validation: from the genVITAMIN UI, choose logic `ATL` and generate a formula. The backend log should include:
“Generated formula using NL2ATL”. If `NL2ATL_URL` is not set or logic is not `ATL`, genVITAMIN falls back to its
original generator.

## Outputs

- Predictions: `outputs/model_predictions/<run_name>.json`
- LLM judge results: `outputs/LLM-evaluation/evaluated_datasets/<judge>/<file>.json`
- LLM judge summary: `outputs/LLM-evaluation/summary__judge-<judge>.json`
- Model efficiency report: `outputs/LLM-evaluation/efficiency_report.json`
- Model efficiency notebook: `outputs/LLM-evaluation/efficiency_report.ipynb`
- Agreement report: `outputs/LLM-evaluation/agreement_report.json`

## Testing

```bash
pytest -q
```

## Documentation

For detailed documentation including architecture, design decisions, and API reference, see the **[full documentation](docs/index.md)**.

## Notes

- **GPU dependencies**: The CLI lazy-loads subcommands, so non-GPU tasks (difficulty classification, judge agreement) do not require CUDA. If you encounter GPU issues, run only the subcommand you need.
- **Model caching**: Set `REUSE_MODEL_CACHE=0` to disable HF model reuse between runs.
- **Training probes**: Set `TRAIN_MAX_STEPS` to run a short training loop.
- **LLM judge outputs**: Existing evaluated datasets are skipped unless `--overwrite/--force` is used.