# CLI Usage

NL2ATL provides a consolidated CLI with subcommands.

## Installation (console script)

```bash
pip install -e .
```

After installation, run:

```bash
nl2atl <command> [args]
```

## Commands

- run-all
- run-single
- llm-judge
- judge-agreement
- classify-difficulty

## Examples

```bash
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot
nl2atl run-single --model qwen-3b --few_shot
nl2atl llm-judge --datasets all
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
nl2atl classify-difficulty --input data/dataset.json --verbose
```

## Direct Python invocation

```bash
python nl2atl.py run-all --models qwen-3b --conditions baseline_zero_shot
```
