#!/bin/bash
#SBATCH --job-name=sql_sft
#SBATCH --output=slurm/logs/sft_%j.log
#SBATCH --error=slurm/logs/sft_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

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

echo "Starting Supervised Fine-Tuning (SFT) Warmup..."
python scripts/train_sft.py
echo "SFT Warmup complete. Model saved to models/sft/final"
