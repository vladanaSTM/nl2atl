# NL2ATL

NL2ATL is a research codebase for natural language to ATL (Alternating-time Temporal Logic) formula generation, evaluation, and difficulty classification.

## Features

- Run baseline and fine-tuned experiments
- Evaluate predictions with exact match and LLM-as-a-judge
- Measure inter-rater agreement across judges
- Classify dataset difficulty with a rule-based scorer

## Project Structure

- src/ — core library
- src/cli/ — CLI entry points
- configs/ — experiment and model configuration files
- data/ — datasets
- outputs/ — predictions and evaluation outputs
- tests/ — unit tests

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables (for Azure judging):

- AZURE_API_KEY
- AZURE_INFER_ENDPOINT
- AZURE_API_VERSION (optional)
- AZURE_USE_CACHE (optional)
- AZURE_VERIFY_SSL (optional)

## Consolidated CLI

Use the consolidated CLI entrypoint:

```bash
python nl2atl.py <command> [args]
```

Or install the console script:

```bash
pip install -e .
nl2atl <command> [args]
```

Available commands:

- run-all
- run-single
- llm-judge
- judge-agreement
- classify-difficulty

Examples:

```bash
python nl2atl.py run-all --models qwen-3b --conditions baseline_zero_shot
python nl2atl.py run-single --model qwen-3b --few_shot
python nl2atl.py llm-judge --datasets all
python nl2atl.py judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
python nl2atl.py classify-difficulty --input data/dataset.json --verbose
```

## Legacy Scripts

The original top-level scripts are kept as thin wrappers and still work:

- run_all_experiments.py
- run_single_experiment.py
- run_llm_judge.py
- run_judge_agreement.py
- classify_difficulty.py

## Notes on GPU Dependencies

Some commands import GPU libraries (e.g., torch). The consolidated CLI now lazy-loads subcommands, so non-GPU tasks like classification and judge agreement do not require torch to be importable. If you hit CUDA/NCCL errors, run only the specific subcommand you need.

## Tests

```bash
pytest -q
```

## Documentation

Start with [docs/index.md](docs/index.md).
