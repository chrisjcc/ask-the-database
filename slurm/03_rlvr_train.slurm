#!/bin/bash
#SBATCH --job-name=sql_rlvr
#SBATCH --output=slurm/logs/rlvr_%j.log
#SBATCH --error=slurm/logs/rlvr_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1           # 1x A100 / H100 80GB
#SBATCH --time=48:00:00

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

echo "Starting Multi-Turn RLVR Training Pipeline..."
python scripts/train_rlvr.py

echo "RLVR Training Complete. Checkpoints saved to models/rlvr/final"
