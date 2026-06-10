# genVITAMIN Integration

This integration lets genVITAMIN use the NL2ATL FastAPI service for ATL formula generation. NL2ATL is tried first for ATL requests; if the NL2ATL service is not configured, unavailable, or returns an invalid response, genVITAMIN falls back to its native AI generation path.

## 1. Start NL2ATL

From this repository:

```bash
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
curl http://localhost:8081/health
```

For best quality, point `NL2ATL_ADAPTER` at a fine-tuned adapter that exists under this repository's `models/` directory. If no adapter is configured, NL2ATL uses the selected base model.

## 2. Patch genVITAMIN

From this repository, pass the path to your genVITAMIN checkout:

```bash
uv run nl2atl genvitamin install --genvitamin-path ../genVITAMIN
```

You can also run the standalone wrapper:

```bash
uv run python integrations/genvitamin/apply_genvitamin_patch.py --genvitamin-path ../genVITAMIN
```

The installer updates these files in genVITAMIN:

| File | Purpose |
|---|---|
| `api/services/nl2atl_client.py` | Small standard-library client for the NL2ATL API |
| `api/routes/ai/generate.py` | Minimal hook that tries NL2ATL before genVITAMIN fallback |
| `api/core/config.py` | Optional `NL2ATL_*` settings |
| `api/.env` | Runtime configuration used by `cd api && make dev` |
| `api/env.example` | Commented example settings for future setup |

Backups are created next to changed files unless you pass `--no-backup`. Use `--dry-run` to preview the file list without writing anything.

## Configuration

The installer does not overwrite existing `NL2ATL_*` values in `api/.env` unless `--force-env` is provided.

Useful options:

```bash
uv run nl2atl genvitamin install \
  --genvitamin-path ../genVITAMIN \
  --nl2atl-url http://localhost:8081 \
  --model qwen-3b \
  --adapter qwen-3b_finetuned_seed42/final \
  --max-new-tokens 128 \
  --timeout 300
```

Environment variables added to `api/.env`:

| Variable | Default | Meaning |
|---|---|---|
| `NL2ATL_URL` | `http://localhost:8081` | NL2ATL API base URL. Empty disables the integration. |
| `NL2ATL_MODEL` | `qwen-3b` | NL2ATL model key from `configs/models.yaml`. |
| `NL2ATL_FEW_SHOT` | `true` | Whether NL2ATL should use few-shot prompting. |
| `NL2ATL_NUM_FEW_SHOT` | unset | Optional few-shot example count. |
| `NL2ATL_ADAPTER` | unset | Optional adapter path relative to NL2ATL `models/`. |
| `NL2ATL_MAX_NEW_TOKENS` | `128` | Generation token cap. |
| `NL2ATL_TIMEOUT` | `300` | genVITAMIN-to-NL2ATL request timeout in seconds. |

## Check Status

```bash
uv run nl2atl genvitamin status --genvitamin-path ../genVITAMIN
```

## Smoke Test

With both services running:

```bash
curl -X POST http://localhost:8000/api/ai/generate/ \
  -H 'Content-Type: application/json' \
  -d '{"description":"the controller can eventually reach a safe state","logic_type":"ATL"}'
```

The response should contain a formula. NL2ATL returns coalitions as `<<agent>>`; the bridge normalizes them to genVITAMIN's `<agent>` ATL syntax before returning the result.
