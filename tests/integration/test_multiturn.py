"""Integration tests for end-to-end multi-turn agentic trajectories in SkyRLSQLEnvironment."""

import sqlite3
from pathlib import Path

from omegaconf import OmegaConf

from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment


def _get_db_snapshot(db_path: Path) -> dict[str, list[tuple]]:
    """Capture a snapshot of all table contents in the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [r[0] for r in cursor.fetchall()]
    snapshot = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table} ORDER BY 1;")
        snapshot[table] = cursor.fetchall()
    conn.close()
    return snapshot


def test_full_multiturn_agentic_trajectory(sample_sqlite_db: Path) -> None:
    """Verify full agent trajectory from user query -> inspect -> sample -> trial -> final answer.

    Guarantees that database state remains completely unmutated.
    """
    initial_snapshot = _get_db_snapshot(sample_sqlite_db)

    config = OmegaConf.create(
        {
            "max_turns": 5,
            "step_penalty": -0.05,
            "recovery_bonus": 0.3,
            "reward": {
                "float_tolerance": 0.0001,
                "use_test_suite": False,
            },
        }
    )
    env = SkyRLSQLEnvironment(config)

    gold_sql = (
        "SELECT u.name, d.department_name "
        "FROM users u "
        "JOIN departments d ON u.department_id = d.department_id "
        "WHERE u.age > 25 "
        "ORDER BY u.age ASC;"
    )

    obs = env.reset(
        {
            "question": "Which users older than 25 have a department? List names and department names.",
            "gold_sql": gold_sql,
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
        }
    )

    assert "Database: company" in obs
    assert env.current_turn == 0

    # Turn 1: Inspect schemas
    obs, r1, done, info = env.step('<tool name="inspect_schema">["users", "departments"]</tool>')
    assert not done
    assert env.current_turn == 1
    assert r1 == -0.05
    assert "CREATE TABLE users" in obs
    assert "CREATE TABLE departments" in obs
    assert info["action"] == "inspect_schema"

    # Turn 2: Sample data
    obs, r2, done, info = env.step('<tool name="sample_data">departments</tool>')
    assert not done
    assert env.current_turn == 2
    assert r2 == -0.05
    assert "Engineering" in obs
    assert info["action"] == "sample_data"

    # Turn 3: Run trial query
    obs, r3, done, info = env.step(f'<tool name="execute_sql">{gold_sql}</tool>')
    assert not done
    assert env.current_turn == 3
    assert "Success: 2 rows returned" in obs
    assert info["action"] == "execute_sql"

    # Turn 4: Final Answer
    obs, r4, done, info = env.step(f"<final_answer>{gold_sql}</final_answer>")
    assert done
    assert env.current_turn == 4
    assert info["is_terminal"] is True
    # Base reward 1.0 discounted for 3 additional steps: 1.0 - (3 * 0.05) = 0.85
    assert r4 >= 0.80

    # Assert database state remains completely unmutated
    final_snapshot = _get_db_snapshot(sample_sqlite_db)
    assert final_snapshot == initial_snapshot


def test_multiturn_self_correction_trajectory(sample_sqlite_db: Path) -> None:
    """Verify recovery bonus is granted when agent self-corrects after runtime error."""
    config = OmegaConf.create(
        {
            "max_turns": 5,
            "step_penalty": -0.05,
            "recovery_bonus": 0.3,
            "reward": {
                "float_tolerance": 0.0001,
                "use_test_suite": False,
            },
        }
    )
    env = SkyRLSQLEnvironment(config)

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"

    env.reset(
        {
            "question": "Who is the oldest user?",
            "gold_sql": gold_sql,
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
        }
    )

    # Turn 1: agent hallucinated column name 'user_full_name'
    obs, r1, done, info = env.step(
        '<tool name="execute_sql">SELECT user_full_name FROM users ORDER BY age DESC LIMIT 1;</tool>'
    )
    assert not done
    assert env.current_turn == 1
    assert "Database Error" in obs
    assert info["had_intermediate_error"] is True

    # Turn 2: agent inspects schema to check valid column names
    obs, r2, done, info = env.step('<tool name="inspect_schema">["users"]</tool>')
    assert not done
    assert env.current_turn == 2
    assert "name" in obs

    # Turn 3: agent submits corrected final answer
    obs, r3, done, info = env.step(f"<final_answer>{gold_sql}</final_answer>")
    assert done
    assert env.current_turn == 3
    # Turn discount for turn 3 is 2 * 0.05 = 0.10.
    # Base reward 1.0 - 0.10 + 0.30 recovery bonus = 1.20 -> clamped to 1.0.
    assert r3 == 1.0


def test_multiturn_security_mutation_defense(sample_sqlite_db: Path) -> None:
    """Verify sandbox blocks destructive queries during multi-turn exploration."""
    initial_snapshot = _get_db_snapshot(sample_sqlite_db)

    config = OmegaConf.create({"max_turns": 3})
    env = SkyRLSQLEnvironment(config)

    env.reset(
        {
            "question": "Clean old records.",
            "gold_sql": "SELECT 1;",
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
        }
    )

    # Attempt mutation
    obs, r1, done, info = env.step(
        '<tool name="execute_sql">DELETE FROM users WHERE age < 30;</tool>'
    )
    assert not done
    assert "Database Error" in obs
    assert "Prohibited SQL" in obs or "Security" in obs or "readonly" in obs or "DELETE" in obs

    # Verify no records deleted
    post_snapshot = _get_db_snapshot(sample_sqlite_db)
    assert post_snapshot == initial_snapshot
