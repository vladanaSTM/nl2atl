# NL2ATL

Natural language → ATL (Alternating-Time Temporal Logic) formula generation, evaluation, and difficulty classification for the NL2ATL research project.

For full documentation (architecture, configuration, API, datasets, and integration details), see [docs/index.md](docs/index.md).

## Highlights

- Experiment runner for baseline and fine‑tuned models
- LLM‑as‑judge evaluation and agreement analysis
- Efficiency reporting (cost/latency/accuracy)
- Dataset difficulty classification
- Lightweight FastAPI service for UI integration

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configure

```bash
cp .env.example .env
```

Common environment variables (see [docs/configuration.md](docs/configuration.md) for full list):

- `AZURE_API_KEY`, `AZURE_INFER_ENDPOINT`, `AZURE_INFER_MODEL` (optional, for Azure inference/judge)
- `HUGGINGFACE_TOKEN` (optional, for gated HF models)
- `WANDB_API_KEY` (optional, for experiment logging)
- `NL2ATL_DEFAULT_MODEL`, `NL2ATL_MODELS_CONFIG`, `NL2ATL_EXPERIMENTS_CONFIG` (API service)

## Run experiments (SLURM recommended)

For multi‑model/seed sweeps, use SLURM arrays for parallel execution and scheduler‑managed resources.

```bash
nl2atl run-array --count
nl2atl run-array --list-tasks
sbatch scripts/slurm/submit_array.sh
```

Local fallback (single node):

```bash
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
nl2atl run-single --model qwen-3b --few_shot
```

Other common commands (see [docs/usage.md](docs/usage.md)):

- `nl2atl llm-judge --datasets all`
- `nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets`
- `nl2atl model-efficiency --predictions_dir outputs/model_predictions`
- `nl2atl classify-difficulty --input data/dataset.json --verbose`

## API service (NL → ATL)

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

Example request:

```bash
curl -X POST http://localhost:8081/generate \
	-H "Content-Type: application/json" \
	-d '{
		"description": "Agent A can eventually reach goal",
		"model": "qwen-3b",
		"few_shot": true,
		"max_new_tokens": 128
	}'
```

## genVITAMIN integration (patch workflow)

genVITAMIN is an open‑source model checker for multi‑agent systems, supporting ATL (and other logics) with a
web UI for building models and verifying formulas. Integrating NL2ATL is useful because it lets users generate
ATL formulas from natural language directly inside the genVITAMIN workflow.

Use the one‑click patch to wire the genVITAMIN backend to NL2ATL. Full setup and troubleshooting are in
[docs/integrations/genvitamin.md](docs/integrations/genvitamin.md).

```bash
python integrations/genvitamin/apply_genvitamin_patch.py \
	--genvitamin-path "/path/to/nl2atl/genVITAMIN"
```

## Outputs

- Predictions: `outputs/model_predictions/<run_name>.json`
- LLM judge results: `outputs/LLM-evaluation/evaluated_datasets/<judge>/<file>.json`
- LLM judge summary: `outputs/LLM-evaluation/summary__judge-<judge>.json`
- Model efficiency report: `outputs/LLM-evaluation/efficiency_report.json`
- Agreement report: `outputs/LLM-evaluation/agreement_report.json`

## Testing

```bash
pytest -q
```

## Documentation

- Quickstart: [docs/quickstart.md](docs/quickstart.md)
- Installation: [docs/installation.md](docs/installation.md)
- Usage & CLI: [docs/usage.md](docs/usage.md)
- SLURM guide: [docs/slurm.md](docs/slurm.md)
- Configuration: [docs/configuration.md](docs/configuration.md)
- Evaluation: [docs/evaluation.md](docs/evaluation.md)
- Integrations: [docs/integrations/genvitamin.md](docs/integrations/genvitamin.md)