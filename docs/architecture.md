# Architecture

NL2ATL is organized around one flow: load data, prepare prompts, run a model, evaluate predictions, and save results.

## High-Level Flow

```text
configs + dataset
      |
      v
Config + ExperimentDataManager
      |
      v
ExperimentRunner
      |
      +--> models.registry loads/generates
      +--> models.few_shot formats prompts
      +--> evaluation.exact_match scores predictions
      +--> experiment.reporter saves metadata/results
```

## Main Packages

| Path | Responsibility |
|---|---|
| `src/cli/` | Command-line entry points |
| `src/config.py` | YAML config loading and validation |
| `src/data_utils.py` | Dataset validation, normalization, splitting, augmentation |
| `src/experiment/` | Experiment data preparation, orchestration, and reporting |
| `src/models/` | Prompt formatting, model loading, generation |
| `src/evaluation/` | Exact match, LLM judge, agreement, efficiency, difficulty tooling |
| `src/infra/` | JSON/YAML/env helpers and Azure client |
| `src/api_server.py` | FastAPI generation endpoint |

## Dataset Boundary

`src.data_utils.load_data` is the dataset boundary. It validates raw JSON rows, stores every accepted formula in `outputs`, and keeps a preferred `output` for compatibility. Downstream code should use `outputs` or `get_output_options` instead of reading `output_1` or `output_2` directly.

## Experiment Boundary

`ExperimentRunner` coordinates a run but delegates narrow work:

- `ExperimentDataManager` loads, splits, and augments data.
- `models.registry` loads models and generates text.
- `models.few_shot` builds prompts.
- `ExactMatchEvaluator` cleans model output and scores exact matches against every accepted formula.
- `ExperimentReporter` builds metadata and writes JSON files.

## Evaluation Boundary

Prediction files use:

```json
{
  "metadata": {},
  "predictions": []
}
```

Evaluation tools read those files and write derived outputs under `outputs/LLM-evaluation/`.

## API Boundary

The FastAPI app in `src/api_server.py` loads the same configs as the CLI. Environment variables can override config paths and the default model.

## Design Principles

- Keep dataset normalization in one place: `src/data_utils.py`.
- Keep config semantics explicit: train/validation/test sizes are final fractions.
- Keep model-provider differences behind `src/models/registry.py` and `src/infra/azure.py`.
- Keep generated artifacts under `outputs/`.
