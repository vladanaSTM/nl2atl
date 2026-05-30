# Development

This page is for changing NL2ATL code.

## Setup

```bash
uv sync --group dev
uv run pytest -q
```

On the ENST SLURM cluster where `uv` is unavailable, this project has also been validated with a local venv created from the `python/3.12.3` module:

```bash
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load python/3.12.3 cuda/12.4.1; fi
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest==9.0.2
ln -sfn /usr/lib64/libbz2.so.1.0.8 .venv/lib/libbz2.so.1.0
export LD_LIBRARY_PATH="$PWD/.venv/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
python -m pytest -q
```

Root pytest collects only `tests/`; the nested `genVITAMIN/` directory is a separate project.

## Useful Tests

```bash
uv run pytest tests/test_data_utils.py -q
uv run pytest tests/test_exact_match.py tests/test_llm_judge.py -q
uv run pytest tests/test_config.py tests/test_experiment_data_manager.py -q
```

On the Windows workstation used for this repo, this command is reliable:

```powershell
cmd /c C:\Users\perlicv1\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q
```

## Where To Make Changes

| Change | Start here |
|---|---|
| Dataset schema or splitting | `src/data_utils.py`, `src/experiment/data_manager.py` |
| Experiment orchestration | `src/experiment/runner.py` |
| Config fields | `src/config.py`, `configs/experiments.yaml`, `configs/models.yaml` |
| Prompt format | `src/models/few_shot.py` |
| Exact match | `src/evaluation/exact_match.py` |
| LLM judging | `src/evaluation/llm_judge/` |
| Judge agreement | `src/evaluation/judge_agreement.py` |
| Experiment CLI | `src/cli/run_experiments.py` |
| Other CLI commands | `src/cli/` |
| API service | `src/api_server.py` |

## Code Guidelines

- Keep data normalization at the dataset boundary.
- Do not read `output_1` or `output_2` directly outside data utilities; use `get_output_options`.
- Keep augmentation train-only.
- Keep generated artifacts under `outputs/` or `models/`.
- Add focused tests for behavior changes.
- Update docs when command behavior or data contracts change.

## Before Finishing

```bash
uv run pytest -q
uv run nl2atl --help
```

For cluster-only validation, use `.venv/bin/python -m pytest -q` with the module and library path shown above.

For doc-only changes, also search for old dependency commands, old dataset paths, and old split values.