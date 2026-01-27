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

### 2) Create or edit the array job script

Use the helper in [scripts/slurm/submit_array.sh](../scripts/slurm/submit_array.sh), or adapt it to your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=nl2atl-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-99

cd /path/to/nl2atl
source .venv/bin/activate

nl2atl run-array \
  --model_provider hf \
  --models qwen-3b \
  --conditions baseline_zero_shot
```

The helper computes the task count and submits the array with the correct `0-(count-1)` range.

### 3) Submit and monitor

```bash
sbatch scripts/slurm/submit_array.sh
squeue --me
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
