"""Unit tests for HPC Slurm script verification and defensive engineering standards."""

import subprocess
from pathlib import Path


def test_slurm_scripts_exist_and_pass_syntax() -> None:
    """Verify all production Slurm scripts exist and pass bash syntax parsing."""
    slurm_dir = Path("slurm")
    expected_scripts = [
        "01_preprocess.slurm",
        "02_sft_warmup.slurm",
        "03_rlvr_train.slurm",
        "04_sweep_array.slurm",
        "05_evaluate.slurm",
        "06_demo_tunnel.slurm",
    ]

    for script_name in expected_scripts:
        script_path = slurm_dir / script_name
        assert script_path.exists(), f"Slurm script {script_name} does not exist."

        proc = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"Syntax error in {script_name}: {proc.stderr}"


def test_slurm_scripts_adhere_to_preamble_standards() -> None:
    """Verify each Slurm script contains the defensive cluster preamble."""
    slurm_dir = Path("slurm")
    for script_path in slurm_dir.glob("*.slurm"):
        content = script_path.read_text(encoding="utf-8")
        assert "#!/bin/bash" in content, f"{script_path.name} missing shebang"
        assert "#SBATCH" in content, f"{script_path.name} missing SBATCH directives"
        assert "export PYTHONNOUSERSITE=1" in content, f"{script_path.name} missing site isolation"
        assert "SLURM_SUBMIT_DIR" in content, f"{script_path.name} missing submit dir snapping"
