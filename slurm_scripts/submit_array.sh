#!/usr/bin/env bash
#SBATCH --job-name=nl2atl-array
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python3}
REPO_ROOT=/home/infres/vperlic/projects/nl2atl
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

TASKS=$($PYTHON_BIN -m src.cli.run_experiment_array \
  --model_provider hf \
  --count)
ARRAY_MAX=$((TASKS - 1))

if [[ "${SLURM_ARRAY_TASK_ID:-}" == "" ]]; then
  echo "Submitting array with 0-${ARRAY_MAX} tasks"
  sbatch --array=0-${ARRAY_MAX} "$0"
  exit 0
fi

$PYTHON_BIN -m src.cli.run_experiment_array \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --model_provider hf
