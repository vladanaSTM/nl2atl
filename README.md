# NL2ATL

NL2ATL translates natural-language strategic requirements into ATL formulas. It provides a compact experiment framework for running Hugging Face or Azure models, evaluating predictions, and serving generation through FastAPI.

## What This Repository Contains

- A CLI named `nl2atl` for experiments and evaluation.
- Model loading for Hugging Face and Azure-hosted models.
- Prompt formatting for zero-shot and few-shot generation.
- Exact-match, LLM-judge, agreement, and efficiency evaluation tools.
- A JSON dataset of natural-language requirements paired with ATL formulas.
- Optional integration helpers for genVITAMIN.

## Quick Start

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl

python -m pip install uv
uv sync --group dev

uv run nl2atl --help
```

Create a local `.env` from `.env.example` when you need remote or gated models:

```bash
cp .env.example .env
```

Common variables:

- `AZURE_API_KEY` and `AZURE_INFER_ENDPOINT` for Azure models.
- `HUGGINGFACE_TOKEN` for gated Hugging Face models.

## Common Commands

```bash
# Run one model/condition
uv run nl2atl run-single --model qwen-3b --few_shot

# Run a local sweep
uv run nl2atl run-all --models qwen-3b --conditions baseline_zero_shot

# Evaluate generated predictions with an LLM judge
uv run nl2atl llm-judge --datasets all

# Summarize cost, latency, and accuracy
uv run nl2atl model-efficiency --predictions_dir outputs/model_predictions

# Start the API service
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

## Dataset

The checked-in dataset is [data/dataset_gold_no_difficulty.json](data/dataset_gold_no_difficulty.json). Each item must have:

- `input`: natural-language requirement.
- `output`: reference ATL formula.

The loader also accepts `output_1` and `output_2`. When a row has multiple correct formulas, they are preserved in an in-memory `outputs` list and the first preferred formula is kept in `output` for backward compatibility. Training uses every formula in `outputs`, exact match accepts any of them, and the LLM judge receives the full list when semantic judging is needed. Difficulty labels are optional and are not required for splitting.

Example:

```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed"
}
```

## Data Splits

Splits are seeded shuffles, not stratified splits. Configure final proportions directly in [configs/experiments.yaml](configs/experiments.yaml):

```yaml
data:
  path: "./data/dataset_gold_no_difficulty.json"
  train_size: 0.70
  val_size: 0.10
  test_size: 0.20
  augment_factor: 10
```

The three split sizes must sum to `1.0`.

## Project Layout

```text
src/
  cli/          command-line entry points
  experiment/   data preparation, training, evaluation orchestration, reporting
  models/       prompt formatting, model loading, generation
  evaluation/   exact match, LLM judge, agreement, efficiency, difficulty tools
  infra/        JSON/YAML/env helpers and Azure client
  api_server.py FastAPI generation service
configs/        model and experiment configuration
data/           dataset files
docs/           user and developer documentation
tests/          unit tests
```

## Outputs

Experiment outputs are written under `outputs/`:

```text
outputs/model_predictions/      generated formulas and per-run metadata
outputs/LLM-evaluation/         LLM-judge results, summaries, and reports
```

Large output artifacts may be tracked with Git LFS. Install Git LFS and run `git lfs pull` if you need those files after cloning.

## Documentation

Start with [docs/index.md](docs/index.md). The most useful pages are:

- [docs/installation.md](docs/installation.md)
- [docs/quickstart.md](docs/quickstart.md)
- [docs/dataset.md](docs/dataset.md)
- [docs/configuration.md](docs/configuration.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/api.md](docs/api.md)

## Development

```bash
uv sync --group dev
uv run pytest -q
```

Keep changes small and run focused tests for the modules you touch before running the full suite.

## License

NL2ATL is released under the MIT License.
