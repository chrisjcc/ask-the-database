"""Unit tests for Bayesian sweep configurations and Slurm array scripts."""

import subprocess
from pathlib import Path

import yaml


def test_rlvr_sweep_config_structure() -> None:
    """Validate Bayesian sweep YAML structure and hyperparameter ranges."""
    sweep_path = Path("configs/sweeps/rlvr_sweep.yaml")
    assert sweep_path.exists(), "Sweep config file not found."

    with open(sweep_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["method"] == "bayes"
    assert cfg["metric"]["name"] == "train/test_suite_reward"
    assert cfg["metric"]["goal"] == "maximize"

    params = cfg["parameters"]
    assert "learning_rate" in params
    assert "beta" in params
    assert "step_penalty" in params
    assert params["learning_rate"]["min"] < params["learning_rate"]["max"]


def test_sweep_array_slurm_syntax_and_throttling() -> None:
    """Validate slurm array script syntax and %1 sequential throttling flag."""
    script_path = Path("slurm/04_sweep_array.slurm")
    assert script_path.exists(), "Slurm sweep array script not found."

    content = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --array=1-8%1" in content, "Must use %1 throttling for Bayesian GP integrity."
    assert "SWEEP_ID is required" in content

    # Test syntax with bash -n
    proc = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"Bash syntax check failed: {proc.stderr}"
