# genVITAMIN Integration (NL2ATL as Generator)

This guide shows how to wire genVITAMIN’s GUI to use NL2ATL as the backend generator. All changes live in the NL2ATL repo; genVITAMIN only needs a small backend patch.
If you don’t control genVITAMIN, do not commit changes there—use the patch script or manual patch steps below.

Patch compatibility tested with genVITAMIN commit: c5231bcbdaf43c01b2b6dd4c3ad5945224e0c68f
Repo: https://github.com/MarcoAruta/genVITAMIN (default branch: quyen@optimize_model_checker)

## Overview

- The genVITAMIN GUI calls its own backend at `/api/ai/generate`.
- You keep genVITAMIN’s UI and backend, but patch two backend files to forward to NL2ATL.
- NL2ATL runs as a standalone FastAPI service and returns ATL formulas.

## 1) Prepare environments (one per repo)

Create two separate virtual environments and keep them isolated:

**NL2ATL venv (repo root):**

```bash
cd /path/to/nl2atl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**genVITAMIN venv (repo root):**

```bash
cd /path/to/nl2atl/genVITAMIN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If missing modules, use: pip install -r api/requirements-api.txt
```

**Faster backend-only install (recommended):**

If you only need the genVITAMIN backend (and not its Streamlit UI), install the minimal backend requirements instead of the full stack:

```bash
pip install -r api/requirements-api.txt
```

This is faster than `requirements.txt`, which pulls Streamlit/frontend extras.

## 2) Start the NL2ATL API

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

## 3) Configure genVITAMIN Backend

In your genVITAMIN backend environment, set the NL2ATL variables **for the backend process**.

Recommended (persisted): use **genVITAMIN/.env** (repo root). The patch script now creates/updates this file for you.

Or set them in the shell **before** starting the backend:

```bash
NL2ATL_URL=http://localhost:8081
# Optional overrides
NL2ATL_MODEL=qwen-3b
NL2ATL_FEW_SHOT=true
NL2ATL_ADAPTER=qwen-3b_finetuned_few_shot/final
NL2ATL_MAX_NEW_TOKENS=128
NL2ATL_TIMEOUT=300
NL2ATL_NUM_FEW_SHOT=4
```

If you changed the NL2ATL port, update `NL2ATL_URL` accordingly (e.g., `http://localhost:8082`).

**Changing the model (quick):**

- `NL2ATL_MODEL` is the base model key from NL2ATL’s `configs/models.yaml`.
- `NL2ATL_ADAPTER` is an optional fine‑tuned adapter path under `./models` (relative paths are resolved against NL2ATL’s `./models`).

Example:

```bash
NL2ATL_MODEL=qwen-3b
NL2ATL_ADAPTER=qwen-3b_finetuned_few_shot/final
```

`NL2ATL_NUM_FEW_SHOT` is optional and overrides NL2ATL’s default few‑shot count.

## 4) Apply the Patch (Backend Only)

One-click patch (recommended):

```bash
python integrations/genvitamin/apply_genvitamin_patch.py \
	--genvitamin-path "C:\\path\\to\\genVITAMIN"
```

This script backs up the original files and replaces them with the NL2ATL-wired versions. It also creates/updates genVITAMIN/.env with sensible defaults.

Manual patch (advanced):

Apply the patch from this repo:

- Patch file: [integrations/genvitamin/patches/use-nl2atl-generator.patch](../../integrations/genvitamin/patches/use-nl2atl-generator.patch)
- Target files in genVITAMIN: `api/routes/ai/generate.py` and `api/core/config.py`

The patch makes genVITAMIN call NL2ATL when `NL2ATL_URL` is set. If the URL is not set or the logic is not ATL, it falls back to its original generator.

## 5) Start genVITAMIN Backend

From the genVITAMIN repo root (using the genVITAMIN venv):

```bash
cd /path/to/nl2atl/genVITAMIN
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If port 8000 is busy, use 8001 and update the frontend env (next step).

## 6) Start genVITAMIN Frontend

Set [genVITAMIN/frontend/.env.local](genVITAMIN/frontend/.env.local) to match the backend port:

```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Then run:

```bash
cd /path/to/nl2atl/genVITAMIN/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open http://localhost:5173

The GUI will use NL2ATL automatically via the patched backend.

**Reproducible run order (every time):**

- Activate NL2ATL venv → start NL2ATL API.
- Apply the patch (updates genVITAMIN/.env).
- Activate genVITAMIN venv → start genVITAMIN backend.
- Start genVITAMIN frontend (points to genVITAMIN backend).

If the genVITAMIN backend fails with `ModuleNotFoundError: langchain_ollama`, install its backend deps:

```bash
pip install -r api/requirements-api.txt
```

If you still see missing modules, install the two commonly-missed extras:

```bash
pip install langchain-ollama langchain-chroma
```

### Frontend without admin rights (no apt)

If you cannot install npm via apt, use a user-level Node 20 binary (required by Vite).

1) Download and extract Node 20:

```bash
mkdir -p ~/.local
cd ~/.local
curl -L -o node20.tar.xz https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz
tar -xf node20.tar.xz
```

2) Install frontend deps and start the dev server:

```bash
cd /path/to/genVITAMIN/frontend
PATH=~/.local/node-v20.19.0-linux-x64/bin:$PATH npm install
PATH=~/.local/node-v20.19.0-linux-x64/bin:$PATH npm run dev -- --host 0.0.0.0 --port 5173
```

Open http://localhost:5173
```

### Windows (PowerShell) equivalents

Activate the correct venv before each command:

- NL2ATL venv: `\.venv\Scripts\Activate.ps1`
- genVITAMIN venv: `\.venv\Scripts\Activate.ps1`

### Windows note (npm)

If PowerShell is in constrained language mode, run npm commands from **cmd** and from the **genVITAMIN/frontend** folder:

```bat
cd C:\path\to\genVITAMIN\frontend
cmd /c "npm install"
cmd /c "npm run dev"
```

### Quick run checklist

1) Start NL2ATL API (NL2ATL venv).
2) Apply the patch (NL2ATL venv; updates genVITAMIN/.env).
3) Start genVITAMIN backend (genVITAMIN venv).
4) Start genVITAMIN frontend (points to genVITAMIN backend).

### Troubleshooting
- **NL2ATL not used even though it’s running**
	- Ensure `NL2ATL_URL` is visible to the genVITAMIN backend process.
	- If you set it in `.env` at the genVITAMIN repo root, restart the backend.
	- If you set it in the shell, make sure you start the backend from the same shell session.

- **“Failed to load examples / Please ensure the API is running”**
	- The genVITAMIN frontend can’t reach the backend.
	- Ensure the backend is running and reachable at the URL configured in the frontend.
	- If the frontend is opened via a network host (not localhost), set `VITE_API_URL` and `VITE_WS_URL` to the network host and restart the frontend.
- **`ModuleNotFoundError: pydantic_settings`**
	- You are likely using the NL2ATL venv to run genVITAMIN. Activate the genVITAMIN venv and install its requirements.
- **Port already in use**
	- Stop the existing process on the port and restart, or change the port and update `NL2ATL_URL` accordingly.
	- If you change the genVITAMIN backend port, also update `VITE_API_URL` and `VITE_WS_URL` for the frontend.
- **`ModuleNotFoundError: langchain_ollama` when starting genVITAMIN backend**
	- Install genVITAMIN backend deps: `pip install -r api/requirements-api.txt` (or follow its README).
	- If still missing: `pip install langchain-ollama langchain-chroma`
- **Vite error: “requires Node.js 20.19+”**
	- Use the user-level Node 20 install above.
- **NL2ATL times out on first request**
	- Set `NL2ATL_TIMEOUT=300` in genVITAMIN/.env and restart the genVITAMIN backend.
- **Want to use a local adapter**
	- Set `NL2ATL_ADAPTER` to a path under ./models (e.g., `qwen-3b_finetuned_few_shot/final`).
- **NL2ATL returns 500 with `configs/models.yaml` not found**
	- Start NL2ATL from its repo root or set `NL2ATL_MODELS_CONFIG` and `NL2ATL_EXPERIMENTS_CONFIG` to absolute paths.

### Ports used

- NL2ATL API: default 8081 (use 8082 if busy).
- genVITAMIN backend: default 8000 (often 8001 if 8000 is busy).
- genVITAMIN frontend: default 5173.

### Closing ports

- Stop servers with Ctrl+C in the terminal where they run.
- If a port is stuck, stop the process and restart on the same port.

## Notes

- NL2ATL currently produces ATL only. The patch routes non‑ATL requests to genVITAMIN’s original generator.
- The patch also normalizes coalitions from `<<Agent>>` to `<Agent>` for genVITAMIN compatibility.
- Default model is `qwen-3b` with adapter `qwen-3b_finetuned_few_shot/final`.
- No changes are required to the genVITAMIN frontend.
