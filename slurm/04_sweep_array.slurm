#!/bin/bash
#SBATCH --job-name=sql_sweep
#SBATCH --output=slurm/logs/sweep_%A_%a.log
#SBATCH --error=slurm/logs/sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=1-8%1        # Sequential execution: 8 tasks, 1 at a time

# ---------------------------------------------------------------------------
# UNIVERSAL PREAMBLE
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"

source ~/.conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ask-the-database 2>/dev/null || true

export PYTHONNOUSERSITE=1
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID is required. Usage: sbatch --export=ALL,SWEEP_ID='...' slurm/04_sweep_array.slurm"
    exit 1
fi

COUNT=${COUNT:-5}

echo "Starting W&B Sweep Agent for $SWEEP_ID (Array Task ${SLURM_ARRAY_TASK_ID:-1}) with count $COUNT..."
wandb agent "$SWEEP_ID" --count "$COUNT"
echo "Sweep Agent Task finished."
