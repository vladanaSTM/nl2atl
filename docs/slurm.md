# SLURM Guide

Use SLURM arrays to scale experiments across GPUs. Each array task runs exactly one
$(\text{seed},\text{model},\text{condition})$ combination via `nl2atl run-array`.

## Why SLURM is recommended

- **Parallel execution** across multiple GPUs/nodes
- **Reliable scheduling** with explicit resource requests
- **Isolation of failures** (one task failing doesn’t stop the sweep)
- **Reproducible mapping** from array index to experiment tuple
- **Better utilization** for large model grids and multiple seeds

## Step‑by‑step

### 1) Inspect the task map

```bash
nl2atl run-array --count
nl2atl run-array --list-tasks
```

Add filters as needed:

```bash
nl2atl run-array --models qwen-3b --conditions baseline_zero_shot --count
```

### 2) Submit the array from the CLI (recommended)

The CLI can generate and submit the SLURM script for you, while still letting you pass any
`run-array` filters.

```bash
nl2atl slurm array \
  --partition A100 \
  --time 06:00:00 \
  --mem 32G \
  --cpus-per-task 8 \
  --gres gpu:1 \
  --seed 123 \
  --models qwen-3b \
  --conditions baseline_zero_shot
```

This automatically computes the task count and submits the array with the correct `0-(count-1)` range.
Use `--dry-run` to print the generated script without submitting.

### 3) (Optional) Use the legacy scripts

If you prefer static scripts, you can still use [slurm_scripts/submit_array.sh](../slurm_scripts/submit_array.sh)
and adapt it to your cluster.

### 4) Monitor jobs

```bash
squeue --me
```

Or use the CLI helper:

```bash
nl2atl slurm status
```

## LLM-judge on SLURM

Submit the evaluator as a single SLURM job:

```bash
nl2atl slurm llm-judge --datasets all --models gpt-5.2
```

## Notes

- `run-array` skips outputs unless `--overwrite` is set.
- LLM‑judge and agreement are not executed inside `run-array`.
- Use [evaluation.md](evaluation.md) for post‑processing commands.
- Outputs are written to `outputs/model_predictions/`.

## Troubleshooting

- Skipped tasks: check for existing files in `outputs/model_predictions/`.
- OOM errors: reduce `training.batch_size` or increase `gradient_accumulation_steps`.
- Scheduling delays: reduce array size or request longer time limits.
