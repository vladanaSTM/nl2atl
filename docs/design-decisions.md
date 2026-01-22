# Design Decisions

This document captures the main design choices and rationale.

## CLI consolidation

A single `nl2atl` command with subcommands simplifies usage and scripting. Subcommands are lazy-loaded to avoid importing heavy GPU dependencies when they are not needed.

## Module separation

- Experiment execution is isolated in `experiment_runner`.
- Model management is isolated in `model_registry`.
- Evaluation logic is isolated in `llm_judge` and `judge_agreement`.
- Data utilities and I/O live in `data_utils` and `io_utils`.

This keeps each module small and testable.

## Environment handling

Dotenv loading is centralized in `env_utils` to avoid redundant or out-of-order environment initialization.

## Determinism and seeds

Configuration supports a single seed, a list of seeds, or a number of seeds. The `Config.resolve_seeds()` method standardizes this for experiment runners.

## Backwards compatibility

Top-level scripts are kept as wrappers for compatibility with existing usage, but the canonical entry points live in `src/cli/`.
