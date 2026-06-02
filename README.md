# NL2ATL

NL2ATL translates natural-language strategic requirements into ATL formulas. The repository contains a small experiment framework for running Hugging Face or Azure models, evaluating predictions, and serving generation through FastAPI.

## Workflow

```text
dataset + configs
  -> seeded train/validation/test split
  -> optional training on the training split
  -> one ATL prediction per test input
  -> normalized exact match against accepted gold formulas
  -> LLM judge only for non-exact predictions
  -> judge agreement and accuracy-latency reports
```

## Install

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
python -m pip install uv
uv sync --group dev
uv run nl2atl --help
```

Create `.env` from `.env.example` when you need Azure or gated Hugging Face models.

## Main Commands

```bash
# One model and condition
uv run nl2atl run --models qwen-3b --conditions baseline_few_shot --seed 42

# Local sweep across selected models, conditions, and seeds
uv run nl2atl run --models qwen-3b --conditions baseline_zero_shot

# SLURM sweep, uncapped by default so all selected tasks can start when resources allow
uv run nl2atl run --slurm --models all --conditions all

# Semantic evaluation for non-exact predictions
uv run nl2atl llm-judge --datasets all

# Agreement and accuracy-latency reports
uv run nl2atl judge-agreement
uv run nl2atl generate-eval-reports
uv run nl2atl model-efficiency

# API service
uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

## Dataset

The default dataset is [data/dataset_gold_no_difficulty.json](data/dataset_gold_no_difficulty.json). Each row needs `input` and at least one gold formula field: `output`, `output_1`, or `output_2`.

Rows can have more than one correct formula. `load_data` normalizes those formulas into `outputs` and keeps the preferred formula in `output` for compatibility. Training uses every formula in `outputs`; exact match accepts any of them; the LLM judge sees all of them when needed.

Splits are seeded shuffles, not stratified splits:

```yaml
data:
  path: "./data/dataset_gold_no_difficulty.json"
  train_size: 0.70
  val_size: 0.10
  test_size: 0.20
  augment_factor: 2
```

Augmentation is applied only after splitting, and only to the training split.

## Code Layout

```text
src/cli/          command-line entry points
src/experiment/   data preparation, training, evaluation orchestration, reporting
src/models/       prompt formatting, model loading, generation
src/evaluation/   exact match, LLM judge, agreement, accuracy-latency
src/infra/        JSON/YAML/env helpers and Azure client
src/api_server.py FastAPI generation service
configs/          model and experiment configuration
data/             dataset files
docs/             documentation
tests/            unit tests
```

## Outputs

```text
outputs/model_predictions/      model predictions and run metadata
outputs/LLM-evaluation/         judge outputs, agreement reports, aggregate metrics
models/                         fine-tuned adapters
```

## Fine-Tuning Defaults

The default fine-tuning configuration is tuned for reproducible, memory-aware LoRA/QLoRA runs on one GPU per task: pinned Hugging Face model revisions, 8 epochs, 1024-token SFT windows, learning rate `1e-4`, cosine schedule, gradient checkpointing, paged 8-bit AdamW, epoch validation, one retained checkpoint, and uncapped SLURM arrays unless `--max-parallel-gpus` is explicitly provided. Fine-tuned zero-shot and fine-tuned few-shot conditions share one adapter for each model and seed. The frozen profiles are Qwen 3B r64/b8, Phi-3.5 r32/b6, Qwen Coder 7B QLoRA r64/b4, and Mistral 7B QLoRA r32/b2.

## Documentation

Start with [docs/index.md](docs/index.md). The docs are intentionally small: [quickstart](docs/quickstart.md), [configuration](docs/configuration.md), [dataset](docs/dataset.md), [evaluation](docs/evaluation.md), [architecture](docs/architecture.md), and [development](docs/development.md).

## Development

```bash
uv sync --group dev
uv run pytest -q
```

NL2ATL is released under the MIT License.