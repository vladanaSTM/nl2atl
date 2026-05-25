# Installation

NL2ATL uses Python 3.10+ and uv for dependency management.

## 1. Clone

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
```

## 2. Install uv

```bash
python -m pip install uv
```

Check it:

```bash
uv --version
```

## 3. Install Dependencies

For normal usage:

```bash
uv sync
```

For development and tests:

```bash
uv sync --group dev
```

You can run commands through uv without manually activating the environment:

```bash
uv run nl2atl --help
uv run pytest -q
```

If you prefer activation:

```bash
uv venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 4. Configure Secrets

Copy the template:

```bash
cp .env.example .env
```

Set only the values you need:

```text
AZURE_API_KEY=...
AZURE_INFER_ENDPOINT=...
HUGGINGFACE_TOKEN=...
```

Do not commit `.env`.

## 5. Verify

```bash
uv run nl2atl --help
uv run pytest tests/test_data_utils.py -q
```

## GPU Notes

Local Hugging Face inference and fine-tuning are much faster on a CUDA GPU. Azure models do not require local GPU resources.

## Git LFS

Some generated outputs may be stored with Git LFS. If you need them:

```bash
git lfs install
git lfs pull
```

The source code and default dataset do not require LFS to run basic tests.
