#!/bin/bash
#SBATCH --job-name=sql_prep
#SBATCH --output=slurm/logs/prep_%j.log
#SBATCH --error=slurm/logs/prep_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

echo "Starting Data Ingestion & Preprocessing..."
python scripts/preprocess_data.py
echo "Preprocessing complete."
