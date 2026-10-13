# AskTheDatabase — HPC Slurm Operations Manual

This directory contains hardened production Slurm scripts for training, hyperparameter sweeps, evaluation, and demonstration on high-performance compute clusters (e.g. SLURM clusters equipped with NVIDIA A100 / H100 80GB GPUs).

---

## 1. Job Pipeline Architecture

Jobs must be submitted in sequential progression:

```text
01_preprocess.slurm   ──> Ingest Spider & download SQLite databases (CPU Queue)
        │
        ▼
02_sft_warmup.slurm   ──> Supervised Fine-Tuning warm-up on Qwen2.5-Coder-7B (1x GPU)
        │
        ▼
03_rlvr_train.slurm   ──> Multi-turn RLVR SkyRL-Agent training (1x A100/H100 80GB)
        │
        ├────────────────────────────────┐
        ▼                                ▼
04_sweep_array.slurm              05_evaluate.slurm
(Bayesian Sweep %1 Array)          (Full Benchmark Evaluation)
                                         │
                                         ▼
                                  06_demo_tunnel.slurm
                                  (Streamlit SSH Port Forwarding)
```

---

## 2. Universal Cluster Preamble Standards

All scripts include the universal defensive preamble:

1. **Submit Directory Snapping**: Ensures paths resolve to project root via `cd "${SLURM_SUBMIT_DIR:-.}"`.
2. **Conda Isolation**: Dynamically activates conda environment with fallback paths.
3. **User Site Isolation**: Sets `export PYTHONNOUSERSITE=1` preventing conflicts with user-level packages in `~/.local`.
4. **Dynamic Library Linking**: Resolves SQLite and CUDA dynamic libraries via `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`.
5. **Environment Loading**: Automatically sources `.env` for W&B, Hugging Face, and model tokens.

---

## 3. Operational Command Guide

### Step 1: Preprocess Data (CPU Queue)
```bash
sbatch slurm/01_preprocess.slurm
```
*Logs*: `slurm/logs/prep_<job_id>.log`

### Step 2: SFT Warmup Baseline (GPU Queue)
```bash
sbatch slurm/02_sft_warmup.slurm
```
*Output*: `models/sft/final`

### Step 3: Multi-Turn RLVR Agent Training
```bash
sbatch slurm/03_rlvr_train.slurm
```
*Output*: `models/rlvr/final`

### Step 4: Bayesian Hyperparameter Sweep (Throttled Array)
Initialize the sweep **once** on the login node:
```bash
wandb sweep configs/sweeps/rlvr_sweep.yaml
```
Copy the returned `SWEEP_ID` and launch the array:
```bash
sbatch --export=ALL,SWEEP_ID='<entity>/ask-the-database/<sweep_id>' slurm/04_sweep_array.slurm
```
*Note*: Uses `#SBATCH --array=1-8%1` to guarantee sequential Bayesian trials, updating the Gaussian Process prior between runs and bypassing 24h walltime limits.

### Step 5: Benchmark Evaluation & Report Synthesis
```bash
sbatch slurm/05_evaluate.slurm
```
*Output*: `results/evaluation_summary.json` and `docs/ablation_report.md`

### Step 6: Interactive Streamlit Demonstration
```bash
sbatch slurm/06_demo_tunnel.slurm
```
Check log for SSH port forwarding instructions:
```bash
cat slurm/logs/demo_<job_id>.log
```

---

## 4. Log Inspection & Monitoring

Check queue status:
```bash
squeue -u $USER
```

Tail real-time logs:
```bash
tail -f slurm/logs/<job_name>_<job_id>.log
```

Cancel jobs:
```bash
scancel <job_id>
```
