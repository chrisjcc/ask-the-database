#!/bin/bash
#SBATCH --job-name=sql_demo
#SBATCH --output=slurm/logs/demo_%j.log
#SBATCH --error=slurm/logs/demo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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

COMPUTE_NODE=$(hostname -s)
PORT=8501

echo "=========================================================================="
echo "🚀 STREAMLIT APP RUNNING ON COMPUTE NODE: $COMPUTE_NODE"
echo "=========================================================================="
echo "To view the app, run this command in your LOCAL Macbook terminal:"
echo ""
echo "    ssh -N -f -L ${PORT}:${COMPUTE_NODE}:${PORT} \${USER}@YOUR_LOGIN_NODE.edu"
echo ""
echo "Then open your browser at: http://localhost:${PORT}"
echo "=========================================================================="

streamlit run app/app.py --server.port "$PORT" --server.address=0.0.0.0 --browser.gatherUsageStats=false
