"""Unit tests for interactive Streamlit application and tunneling deployment."""

import ast
import subprocess
from pathlib import Path

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
