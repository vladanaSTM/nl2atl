# Installation

This guide sets up NL2ATL for research use and development. It assumes a clean Python environment.

## Prerequisites

- Python 3.10+
- Git
- Optional accounts:
  - Azure OpenAI (for Azure‑hosted inference and judge models)
  - Weights & Biases (for experiment tracking)
  - Hugging Face token (for gated models)

## Step 1 — Clone the repository

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
```

## Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

For development (enables the `nl2atl` CLI as an editable install):

```bash
pip install -e .
```

## Step 4 — Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Set only what you need. Common variables:

- Azure (only if you use Azure models):
  - `AZURE_API_KEY`
  - `AZURE_INFER_ENDPOINT`
  - `AZURE_INFER_MODEL`
- Optional:
  - `HUGGINGFACE_TOKEN` (gated HF models)
  - `WANDB_API_KEY` (experiment tracking)
  - `NL2ATL_DEFAULT_MODEL` (default model key)
  - `NL2ATL_MODELS_CONFIG`, `NL2ATL_EXPERIMENTS_CONFIG` (API server config paths)

See [configuration.md](configuration.md) for the complete list and examples.

## Step 5 — Verify the installation

```bash
python nl2atl.py --help
```

You should see the consolidated CLI with commands like `run-all`, `run-single`, and `classify-difficulty`.

## Step 6 — Optional: run the NL2ATL API service

Use this if you integrate NL2ATL into external UIs (e.g., genVITAMIN):

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If you run from a different working directory, set absolute config paths:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

## Tests

```bash
pytest -q
```

## Troubleshooting

| Issue | Fix |
|------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run from repo root or set `PYTHONPATH=.` |
| Azure auth errors | Check `.env` credentials and endpoint URL |
| CUDA out of memory | Reduce batch size in `configs/experiments.yaml` or use 4‑bit loading |
| Tokenizer load errors | Ensure `HUGGINGFACE_TOKEN` is set for gated models |

If issues persist, see [development.md](development.md) for contribution and debugging guidance.
