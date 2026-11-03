"""Unit tests for interactive Streamlit application and tunneling deployment."""

import ast
import subprocess
from pathlib import Path

import pytest

pytest.importorskip(
    "streamlit",
    reason="Streamlit is required for app UI tests. Install via 'make install' or 'pip install -e .[all]'.",
)

from ask_the_database.inference.pipeline import NL2SQLPipeline


def test_app_script_syntax_and_imports() -> None:
    """Verify that app/app.py is syntactically valid Python and defines main()."""
    app_path = Path("app/app.py")
    assert app_path.exists(), "app/app.py does not exist."

    content = app_path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename="app/app.py")

    # Verify functions are defined
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "main" in func_names
    assert "init_demo_db" in func_names
    assert "get_pipeline" in func_names


def test_init_demo_db(tmp_path: Path) -> None:
    """Test demo database initialization helper in app.py."""
    from app.app import init_demo_db

    db_path = tmp_path / "test_demo" / "test_demo.sqlite"
    init_demo_db(db_path)

    assert db_path.exists()
    from ask_the_database.environment.schema_manager import SchemaManager

    sm = SchemaManager(str(db_path))
    tables = sm.get_tables()
    assert "singer" in tables

    ddl = sm.get_all_tables_ddl()
    assert "CREATE TABLE" in ddl
    assert "singer" in ddl


def test_get_pipeline_caching() -> None:
    """Test that get_pipeline returns an NL2SQLPipeline instance."""
    from app.app import get_pipeline

    pipe = get_pipeline(model_path="dummy_path", max_turns=3)
    assert isinstance(pipe, NL2SQLPipeline)
    assert pipe.max_turns == 3


def test_demo_tunnel_slurm_script() -> None:
    """Verify slurm/06_demo_tunnel.slurm configuration and tunneling directives."""
    tunnel_script = Path("slurm/06_demo_tunnel.slurm")
    assert tunnel_script.exists()

    content = tunnel_script.read_text(encoding="utf-8")
    assert "ssh -N -f -L" in content
    assert "streamlit run app/app.py" in content
    assert "--server.port" in content
    assert "COMPUTE_NODE" in content

    # Check bash syntax
    proc = subprocess.run(["bash", "-n", str(tunnel_script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"Syntax error in 06_demo_tunnel.slurm: {proc.stderr}"


def test_deployment_artifacts_exist() -> None:
    """Verify Dockerfile, HF Space readme, and workflow configuration files."""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists()
    dockerfile_content = dockerfile.read_text(encoding="utf-8")
    assert "streamlit" in dockerfile_content
    assert "app/app.py" in dockerfile_content

    app_req = Path("app/requirements.txt")
    assert app_req.exists()
    assert "streamlit" in app_req.read_text(encoding="utf-8")
    assert "torch" in app_req.read_text(encoding="utf-8")
    assert "transformers" in app_req.read_text(encoding="utf-8")

    space_readme = Path(".github/hf_space_readme.md")
    assert space_readme.exists()
    content = space_readme.read_text(encoding="utf-8")
    assert "sdk: docker" in content
    assert "app_port: 8501" in content

    hf_deploy_wf = Path(".github/workflows/deploy-hf-demo.yml")
    assert hf_deploy_wf.exists()
    hf_wf_content = hf_deploy_wf.read_text(encoding="utf-8")
    assert "Validate Hugging Face credentials" in hf_wf_content
    assert "HF_TOKEN" in hf_wf_content

    docker_wf = Path(".github/workflows/docker-publish.yml")
    assert docker_wf.exists()


def test_pipeline_model_loading_status() -> None:
    """Verify is_model_loaded reflects weights loading and handles missing checkpoints."""
    # Offline dummy path
    p_offline = NL2SQLPipeline(model_path="nonexistent/local/path", max_turns=3)
    assert not p_offline.is_model_loaded

    # Dummy remote hub path (should fail gracefully and record load_error)
    p_remote = NL2SQLPipeline(model_path="dummy-org/nonexistent-model-repo", max_turns=3)
    assert not p_remote.is_model_loaded
    assert p_remote.load_error is not None

    # Fallback heuristic generation remains functional
    step_output = p_remote.generate_step(
        messages=[{"role": "user", "content": "list singers"}],
        available_tables=["singer"],
    )
    assert "inspect_schema" in step_output or "SELECT" in step_output


def test_pipeline_hardware_memory_guardrail(monkeypatch) -> None:
    """Verify memory guardrail prevents OOM on low-RAM CPU and warns on 32GB CPU."""
    import torch

    # Case 1: Low RAM CPU (< 20 GB) triggers OOM guardrail
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "os.sysconf", lambda name: 4096 if "PAGE_SIZE" in name else 4194304
    )  # 16 GB

    p_low_ram = NL2SQLPipeline(model_path="dummy-org/nonexistent-model-repo", max_turns=3)
    assert not p_low_ram.is_model_loaded
    assert p_low_ram.load_error is not None
    assert "insufficient for 7B parameter model" in p_low_ram.load_error
    assert "16.0 GB" in p_low_ram.load_error

    # Case 2: 32 GB RAM CPU plan allows loading and sets hardware_warning
    monkeypatch.setattr(
        "os.sysconf", lambda name: 4096 if "PAGE_SIZE" in name else 8388608
    )  # 32 GB
    monkeypatch.setattr(
        "peft.PeftConfig.from_pretrained",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("mock error")),
    )
    p_32gb = NL2SQLPipeline(model_path="dummy-org/nonexistent-model-repo", max_turns=3)
    assert p_32gb.hardware_warning is not None
    assert "32 GB RAM detected" in p_32gb.hardware_warning
    assert "T4 GPU" in p_32gb.hardware_warning
