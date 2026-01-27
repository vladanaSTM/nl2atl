# Installation

This guide covers setting up NL2ATL for development and research use.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Azure OpenAI access for Azure-hosted models
- (Optional) Weights & Biases account for experiment tracking

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/[your-org]/nl2atl.git
cd nl2atl
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows (PowerShell): .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Required variables only if you use Azure models:

```bash
AZURE_API_KEY=your-api-key
AZURE_INFER_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_INFER_MODEL=your-deployment-name
```

Optional variables:

```bash
AZURE_API_VERSION=2024-08-01-preview
AZURE_USE_CACHE=true
AZURE_VERIFY_SSL=false
HUGGINGFACE_TOKEN=your-hf-token
WANDB_API_KEY=your-wandb-key
NL2ATL_DEFAULT_MODEL=gpt-5.2
NL2ATL_MODELS_CONFIG=configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=configs/experiments.yaml
```

### 5. Verify Installation

```bash
python nl2atl.py --help
```

### 6. (Optional) Run the NL2ATL API Service

If you want to integrate NL2ATL with external UIs (e.g., VITAMIN), run the API server:

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If you start NL2ATL from another working directory, set absolute paths so configs resolve:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

Expected output (command list may vary by version):

```
usage: nl2atl [-h] {classify-difficulty,judge-agreement,llm-judge,run-all,run-single} ...

NL2ATL consolidated CLI

positional arguments:
  command
  args

optional arguments:
  -h, --help  show this help message and exit
```

---

## Configuration

See [Configuration Guide](configuration.md) for detailed configuration options.

---

## Development Setup

NL2ATL uses the same `requirements.txt` for both development and research usage. For local development, install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest -q
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root or set `PYTHONPATH=.` |
| Azure authentication errors | Verify `.env` credentials and endpoint URL |
| CUDA out of memory | Reduce batch size in `configs/experiments.yaml` or use 4-bit loading |
| Tokenizer loading errors | Ensure `HUGGINGFACE_TOKEN` is set for gated models |

### Getting Help

- Open an issue on GitHub
- Check existing issues for similar problems
- See [Development Guide](development.md) for contributing
