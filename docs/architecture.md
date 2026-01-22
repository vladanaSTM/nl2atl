# Architecture

NL2ATL is structured as a lightweight Python package plus CLI entry points. The project separates experiment logic, data utilities, and evaluation logic into focused modules under src/.

## Core Modules

- src/experiment_runner.py: orchestrates experiment execution (training, eval, logging).
- src/model_registry.py: model loading and GPU memory management.
- src/llm_judge.py: LLM-as-a-judge evaluation pipeline and summaries.
- src/judge_agreement.py: inter-rater agreement metrics.
- src/classify_difficulty.py: rule-based difficulty scoring.

## CLI Layout

- src/cli/*: thin wrappers that parse CLI arguments and call the core modules.
- nl2atl.py: repository-root entrypoint for the consolidated CLI.

## Data Flow

1. configs/ define models and experiment settings.
2. experiment_runner produces predictions in outputs/model_predictions/.
3. llm_judge evaluates predictions and writes outputs/LLM-evaluation/.
4. judge_agreement computes agreement metrics across judges.
5. classify_difficulty updates difficulty labels in data/.
