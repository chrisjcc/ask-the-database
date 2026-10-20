"""Unit tests for SkyRLSQLEnvironment adapter."""

from pathlib import Path

from omegaconf import OmegaConf

from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment


def test_skyrl_env_reset(sample_sqlite_db: Path) -> None:
    """Test environment reset initializes turns, registry, and system prompt."""
    cfg = OmegaConf.create({"max_turns": 4})
    env = SkyRLSQLEnvironment(cfg)

    instance = {
        "question": "What is the name of the oldest user?",
        "gold_sql": "SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
    }

    obs = env.reset(instance)
    assert env.current_turn == 0
    assert "Database: company" in obs
    assert "What is the name of the oldest user?" in obs
    assert "system:" in obs


def test_skyrl_env_multiturn_interaction_and_final_answer(
    sample_sqlite_db: Path,
) -> None:
    """Test sequence of tool calls followed by a correct final answer."""
    cfg = OmegaConf.create({"max_turns": 5, "step_penalty": -0.05})
    env = SkyRLSQLEnvironment(cfg)

    instance = {
        "question": "What is the name of the oldest user?",
        "gold_sql": "SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
    }
    env.reset(instance)

    # Step 1: inspect schema
    obs1, r1, done1, info1 = env.step('<tool name="inspect_schema">["users"]</tool>')
    assert not done1
    assert env.current_turn == 1
    assert r1 == -0.05
    assert "CREATE TABLE users" in obs1
    assert info1["action"] == "inspect_schema"

    # Step 2: sample data
    obs2, r2, done2, info2 = env.step('<tool name="sample_data">users</tool>')
    assert not done2
    assert env.current_turn == 2
    assert r2 == -0.05
    assert "Alice" in obs2
    assert info2["action"] == "sample_data"

    # Step 3: trial query execution
    obs3, r3, done3, info3 = env.step(
        '<tool name="execute_sql">SELECT name, age FROM users ORDER BY age DESC LIMIT 1;</tool>'
    )
    assert not done3
    assert env.current_turn == 3
    assert "Success: 1 rows returned" in obs3
    assert info3["action"] == "execute_sql"

    # Step 4: final answer
    obs4, r4, done4, info4 = env.step(
        "<final_answer>SELECT name FROM users ORDER BY age DESC LIMIT 1;</final_answer>"
    )
    assert done4
    assert env.current_turn == 4
    assert info4["is_terminal"] is True
    # Gold match should yield positive reward
    assert r4 > 0.5


def test_skyrl_env_max_turns_truncation(sample_sqlite_db: Path) -> None:
    """Test that environment terminates when exceeding max_turns without final answer."""
    cfg = OmegaConf.create({"max_turns": 2, "step_penalty": -0.05})
    env = SkyRLSQLEnvironment(cfg)

    instance = {
        "question": "Show all departments.",
        "gold_sql": "SELECT department_name FROM departments;",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
    }
    env.reset(instance)

    # Turn 1
    _, _, done1, _ = env.step('<tool name="sample_data">departments</tool>')
    assert not done1

    # Turn 2: reaches max_turns
    _, r2, done2, info2 = env.step('<tool name="sample_data">departments</tool>')
    assert done2
    assert info2["max_turns_exceeded"] is True
    # Terminal step without final answer results in negative penalty
    assert r2 <= -0.5


def test_skyrl_env_action_parsing_variants(sample_sqlite_db: Path) -> None:
    """Test action parser handles tag spacing and fallback formatting."""
    cfg = OmegaConf.create({"max_turns": 3})
    env = SkyRLSQLEnvironment(cfg)

    # Single quotes in attribute
    act1, pay1 = env._parse_action("<tool name='execute_sql'>SELECT 1;</tool>")
    assert act1 == "execute_sql"
    assert pay1 == "SELECT 1;"

    # Extra whitespaces inside tags
    act2, pay2 = env._parse_action('<tool    name="sample_data"   >   orders   </tool>')
    assert act2 == "sample_data"
    assert pay2 == "orders"

    # Raw fallback
    act3, pay3 = env._parse_action("SELECT * FROM departments;")
    assert act3 == "final_answer"
    assert pay3 == "SELECT * FROM departments;"
