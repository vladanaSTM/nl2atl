# Architecture

This document summarizes NL2ATL’s current module layout and data flow as implemented in the codebase.

## High‑level flow

```mermaid
flowchart TD
  CLI[nl2atl CLI] --> Runner[ExperimentRunner]
  Runner --> Data[ExperimentDataManager]
  Runner --> Models[Model Registry]
  Runner --> Eval[Exact Match Evaluator]
  Runner --> Report[ExperimentReporter]
  Eval --> Outputs[outputs/model_predictions]

  Judge[LLM Judge] --> Outputs
  Judge --> EvalOut[outputs/LLM-evaluation]
  Agreement[Judge Agreement] --> EvalOut
  Efficiency[Model Efficiency] --> EvalOut
```

## Package structure

```
src/
  cli/              CLI entry points (run-all, run-single, run-array, judge, etc.)
  experiment/       experiment orchestration and reporting
  models/           model loading, prompt formatting, generation
  evaluation/       exact-match, LLM judge, agreement, efficiency, difficulty
  infra/            I/O helpers and Azure utilities
  data_utils.py     dataset split + augmentation helpers
  api_server.py     FastAPI service
```

## Key responsibilities

- `src/cli/` — command parsing and task dispatch
- `src/experiment/` — data splits, training/inference runs, output persistence
- `src/models/` — model loading (HF/Azure), caching, few‑shot prompt formatting
- `src/evaluation/` — exact‑match evaluation, LLM‑as‑judge pipeline, agreement, efficiency, difficulty
- `src/infra/` — I/O helpers, Azure config/client, environment utilities
- `src/data_utils.py` — stratified split and augmentation utilities

## Execution workflows

### Experiment workflow (local or single node)

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Runner
    participant DataMgr
    participant Registry
    participant Evaluator
    participant Reporter

    User->>CLI: nl2atl run-all / run-single
    CLI->>Runner: Config.from_yaml(models.yaml, experiments.yaml)
    Runner->>DataMgr: prepare_data()
    DataMgr-->>Runner: train_aug, val, test
    Runner->>Registry: load_model(...)
    Runner->>Evaluator: evaluate(model, test_data)
    Evaluator-->>Runner: metrics + per-sample results
    Runner->>Reporter: save_result(run_name, result)
```

### SLURM array workflow (recommended for sweeps)

```mermaid
sequenceDiagram
    participant Scheduler
    participant CLI
    participant Runner

    Scheduler->>CLI: nl2atl run-array (task index)
    CLI->>Runner: Run single (seed, model, condition)
    Runner-->>CLI: outputs/model_predictions/<run>.json
```

`run-array` maps each SLURM array index to exactly one $(seed, model, condition)$ task.

## Output artifacts

```
outputs/
  model_predictions/<run_name>.json
  LLM-evaluation/
    evaluated_datasets/<judge>/...
    summary__judge-<judge>.json
    agreement_report.json
    efficiency_report.json
```

## Where to extend

- Add models: update `configs/models.yaml` and provider logic in `src/models/registry.py`.
- Add CLI tasks: create `src/cli/run_*.py` and register in `src/cli/main.py`.
- Add evaluators: implement `BaseEvaluator` in `src/evaluation/base.py`.
