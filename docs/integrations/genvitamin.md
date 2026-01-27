# genVITAMIN Integration (NL2ATL as Generator)

This guide shows how to wire genVITAMIN’s GUI to use NL2ATL as the backend generator. All changes live in the NL2ATL repo; genVITAMIN only needs a small patch to its backend route.

## Overview

- The genVITAMIN GUI calls its own backend at `/api/ai/generate`.
- You keep genVITAMIN’s UI and backend, but patch that single route to forward to NL2ATL.
- NL2ATL runs as a standalone FastAPI service and returns ATL formulas.

## 1) Start the NL2ATL API

Run the NL2ATL API service (see [Usage Guide](../usage.md)) **from the NL2ATL repo root** so config files resolve correctly:

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If you start NL2ATL from another working directory, set absolute paths:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

Verify health:

```bash
curl http://localhost:8081/health
```

### Port already in use

If port `8081` is busy, pick another one (e.g., `8082`):

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8082
```

## 2) Configure genVITAMIN Backend

In your genVITAMIN backend environment, set the NL2ATL variables **for the backend process**.

Recommended (persisted): add to genVITAMIN’s backend env file at genVITAMIN/api/.env.

Or set them in the shell **before** starting the backend:

```bash
NL2ATL_URL=http://localhost:8081
# Optional overrides
NL2ATL_MODEL=gpt-5.2
NL2ATL_FEW_SHOT=true
NL2ATL_ADAPTER=my-finetuned-adapter
NL2ATL_MAX_NEW_TOKENS=128
NL2ATL_TIMEOUT=30
NL2ATL_NUM_FEW_SHOT=4
```

If you changed the NL2ATL port, update `NL2ATL_URL` accordingly (e.g., `http://localhost:8082`).

`NL2ATL_ADAPTER` can be a name or path. If it is a relative name, NL2ATL resolves it against its models directory (default `./models`).
`NL2ATL_NUM_FEW_SHOT` is optional and overrides NL2ATL’s default few‑shot count.

## 3) Apply the Patch (Backend Only)

One-click patch (recommended):

```bash
python integrations/genvitamin/apply_genvitamin_patch.py \
	--genvitamin-path "C:\\path\\to\\genVITAMIN"
```

This script backs up the original file and replaces it with the NL2ATL-wired version.

Manual patch (advanced):

Apply the patch from this repo:

- Patch file: [integrations/genvitamin/patches/use-nl2atl-generator.patch](../../integrations/genvitamin/patches/use-nl2atl-generator.patch)
- Target file in genVITAMIN: `api/routes/ai/generate.py`

The patch makes genVITAMIN call NL2ATL when `NL2ATL_URL` is set. If the URL is not set or the logic is not ATL, it falls back to its original generator.

## 4) Run genVITAMIN Normally

Start genVITAMIN backend and frontend as usual (see its README). The GUI will use NL2ATL automatically via the patched backend.

If the genVITAMIN backend fails with `ModuleNotFoundError: langchain_ollama`, install its backend deps:

```bash
pip install -r api/requirements-api.txt
```

### Recommended: separate virtual environments

Use a dedicated venv for NL2ATL and another for genVITAMIN to avoid dependency conflicts.

**NL2ATL (repo root):**

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

**genVITAMIN backend (repo root):**

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If genVITAMIN has a backend-specific requirements file, use that instead
# pip install -r api/requirements.txt

# Point backend to NL2ATL (PowerShell)
$env:NL2ATL_URL="http://localhost:8081"

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Windows note (npm)

If PowerShell is in constrained language mode, run npm commands from **cmd** and from the **genVITAMIN/frontend** folder:

```bat
cd C:\path\to\genVITAMIN\frontend
cmd /c "npm install"
cmd /c "npm run dev"
```

### Quick run checklist

1) Start NL2ATL API (`uvicorn src.api_server:app --host 0.0.0.0 --port 8081`).
2) Apply the patch with the one‑click script.
3) Start genVITAMIN backend (FastAPI).
4) Start genVITAMIN frontend (React/Vite) and open the GUI URL.

### Troubleshooting
- **NL2ATL not used even though it’s running**
	- Ensure `NL2ATL_URL` is visible to the genVITAMIN backend process.
	- If you set it in `api/.env`, restart the backend.
	- If you set it in the shell, make sure you start the backend from the same shell session.

- **“Failed to load examples / Please ensure the API is running”**
	- The genVITAMIN frontend can’t reach the backend. Start the backend and ensure it is on the URL expected by the frontend.
- **`ModuleNotFoundError: pydantic_settings`**
	- You are likely using the NL2ATL venv to run genVITAMIN. Activate the genVITAMIN venv and install its requirements.
- **Port already in use**
	- Either stop the existing process or change the port and update `NL2ATL_URL`.
- **`ModuleNotFoundError: langchain_ollama` when starting genVITAMIN backend**
	- Install genVITAMIN backend deps: `pip install -r api/requirements-api.txt` (or follow its README).
- **NL2ATL returns 500 with `configs/models.yaml` not found**
	- Start NL2ATL from its repo root or set `NL2ATL_MODELS_CONFIG` and `NL2ATL_EXPERIMENTS_CONFIG` to absolute paths.

## Notes

- NL2ATL currently produces ATL only. The patch routes non‑ATL requests to genVITAMIN’s original generator.
- No changes are required to the genVITAMIN frontend.
