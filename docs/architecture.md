# Architecture

NL2ATL is organized around a simple experiment pipeline.

```text
configs + dataset
  -> ExperimentDataManager
  -> ExperimentRunner
  -> model loading / training / generation
  -> ExactMatchEvaluator
  -> LLM judge and reports
```

## Main Packages

| Path | Role |
|---|---|
| `src/cli/` | Command-line entry points |
| `src/config.py` | YAML config loading and validation |
| `src/data_utils.py` | Dataset validation, normalization, splitting, augmentation |
| `src/experiment/` | Experiment orchestration and result writing |
| `src/models/` | Prompt formatting, model loading, generation utilities |
| `src/evaluation/` | Exact match, LLM judge, judge agreement, accuracy-latency tools |
| `src/infra/` | JSON/YAML/env helpers and Azure client wrapper |
| `src/api_server.py` | FastAPI generation service |

## CLI Boundary

The consolidated `nl2atl` command dispatches to focused modules:

| Command | Module |
|---|---|
| `nl2atl run` | `src/cli/run_experiments.py` |
| `nl2atl llm-judge` | `src/cli/run_llm_judge.py` |
| `nl2atl judge-agreement` | `src/cli/run_judge_agreement.py` |
| `nl2atl aggregate-seeds` | `src/cli/aggregate_seeds.py` |
| `nl2atl model-efficiency` | `src/cli/run_model_efficiency.py` |
| `nl2atl generate-eval-reports` | `src/cli/generate_eval_reports.py` |
| `nl2atl genvitamin` | `src/cli/genvitamin.py` |

## Dataset Boundary

Use `load_data` for experiment data. It validates raw rows, builds `outputs` for all accepted gold formulas, and keeps preferred `output` for compatibility. Downstream code should use `get_output_options` instead of reading `output_1` or `output_2` directly.

## Experiment Boundary

`ExperimentDataManager` loads originals, performs the seeded train/validation/test split, and augments only the training split.

`ExperimentRunner` loads or trains models, evaluates on the test split, and writes prediction files under `outputs/model_predictions/`.

For fine-tuned sweeps, `run_experiments` groups work by model and seed. When both fine-tuned conditions are selected, the runner trains one shared adapter, then evaluates zero-shot and few-shot prompts from that adapter. Baseline model loads can be cached within a process unless `REUSE_MODEL_CACHE=0`.

## Evaluation Boundary

`ExactMatchEvaluator` cleans model output and checks it against every accepted gold formula. LLM judging consumes prediction files and only judges non-exact predictions.

Report generation is layered: judged datasets feed judge summaries and agreement, those feed seed aggregation, and seed aggregation feeds the accuracy-latency report and publication notebook. Aggregates remain judge-specific unless `aggregate-seeds --combine_judges` is used intentionally.

## API Boundary

The FastAPI service exposes `/health` and `/generate`. It loads models from the same config files as the CLI.

## Design Rules

- Keep raw dataset normalization in `src/data_utils.py`.
- Keep generated artifacts under `outputs/` or `models/`.
- Treat `genVITAMIN/` as a separate nested project.
- Prefer focused modules over large cross-cutting helpers.
- Keep pricing, GPU-hour, and token-derived efficiency estimates out of computed reports unless the research question changes explicitly.