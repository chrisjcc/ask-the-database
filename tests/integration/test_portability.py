"""Integration test harness verifying framework portability and reward determinism.

Verifies that 50 diverse text-to-SQL rollouts produce 100% reproducible,
deterministic rewards and group advantages across multiple execution runs,
protecting against framework churn and non-deterministic environment behavior.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip(
    "torch",
    reason="PyTorch is required for training and adapter tests. Install via 'make install' or 'pip install -e .[training]'.",
)

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import RolloutBatch
from ask_the_database.training.adapters.trl_grpo import TRLGRPOAdapter


def build_50_portability_benchmark(db_path: Path) -> list[dict[str, str]]:
    """Build a fixed, deterministic suite of 50 query evaluation samples.

    Includes exact matches, semantically equivalent reformulations, syntax errors,
    schema linking errors, and subtle boundary cases across departments, users, and orders.
    """
    samples: list[dict[str, str]] = []

    # 1-10: Exact SELECT and Filter Queries
    exact_queries = [
        ("SELECT * FROM departments", "SELECT * FROM departments"),
        (
            "SELECT department_id, department_name FROM departments",
            "SELECT department_name, department_id FROM departments",
        ),
        (
            "SELECT name, age FROM users WHERE age > 25",
            "SELECT name, age FROM users WHERE age > 25",
        ),
        ("SELECT COUNT(*) FROM users", "SELECT count(*) FROM users"),
        (
            "SELECT AVG(age) FROM users WHERE department_id = 1",
            "SELECT avg(age) FROM users WHERE department_id = 1",
        ),
        ("SELECT MAX(budget) FROM departments", "SELECT max(budget) FROM departments"),
        ("SELECT MIN(amount) FROM orders", "SELECT min(amount) FROM orders"),
        (
            "SELECT SUM(amount) FROM orders WHERE user_id = 1",
            "SELECT sum(amount) FROM orders WHERE user_id = 1",
        ),
        (
            "SELECT department_id FROM departments WHERE budget >= 300000.0",
            "SELECT department_id FROM departments WHERE budget >= 300000.0",
        ),
        (
            "SELECT name FROM users WHERE department_id IS NULL",
            "SELECT name FROM users WHERE department_id is null",
        ),
    ]
    for cand, gold in exact_queries:
        samples.append({"candidate": cand, "gold": gold, "db_path": str(db_path)})

    # 11-20: Join and Aggregation Queries
    join_queries = [
        (
            "SELECT u.name, d.department_name FROM users u JOIN departments d ON u.department_id = d.department_id",
            "SELECT u.name, d.department_name FROM users AS u JOIN departments AS d ON d.department_id = u.department_id",
        ),
        (
            "SELECT d.department_name, COUNT(u.user_id) FROM departments d LEFT JOIN users u ON d.department_id = u.department_id GROUP BY d.department_name",
            "SELECT d.department_name, count(u.user_id) FROM departments d LEFT JOIN users u ON d.department_id = u.department_id GROUP BY d.department_name",
        ),
        (
            "SELECT u.name, SUM(o.amount) FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.name",
            "SELECT u.name, sum(o.amount) FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.name",
        ),
        (
            "SELECT department_name FROM departments WHERE budget > 250000.0 ORDER BY budget DESC",
            "SELECT department_name FROM departments WHERE budget > 250000.0 ORDER BY budget DESC",
        ),
        (
            "SELECT user_id, count(order_id) FROM orders GROUP BY user_id HAVING count(order_id) > 1",
            "SELECT user_id, COUNT(order_id) FROM orders GROUP BY user_id HAVING count(order_id) > 1",
        ),
        (
            "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.user_id = o.user_id WHERE o.amount >= 200.0",
            "SELECT u.name, o.amount FROM users u JOIN orders o ON u.user_id = o.user_id WHERE o.amount >= 200.0",
        ),
        (
            "SELECT d.department_id, AVG(u.age) FROM departments d JOIN users u ON d.department_id = u.department_id GROUP BY d.department_id",
            "SELECT d.department_id, avg(u.age) FROM departments d JOIN users u ON d.department_id = u.department_id GROUP BY d.department_id",
        ),
        (
            "SELECT name FROM users WHERE age BETWEEN 25 AND 35",
            "SELECT name FROM users WHERE age >= 25 AND age <= 35",
        ),
        (
            "SELECT order_id FROM orders WHERE user_id IN (SELECT user_id FROM users WHERE age > 25)",
            "SELECT order_id FROM orders JOIN users ON orders.user_id = users.user_id WHERE users.age > 25",
        ),
        (
            "SELECT DISTINCT department_id FROM users WHERE department_id IS NOT NULL",
            "SELECT department_id FROM users WHERE department_id IS NOT NULL GROUP BY department_id",
        ),
    ]
    for cand, gold in join_queries:
        samples.append({"candidate": cand, "gold": gold, "db_path": str(db_path)})

    # 21-30: Semantic Equivalence with Formatting & Markdown Variations
    semantic_queries = [
        ("```sql\nSELECT name FROM users;\n```", "SELECT name FROM users"),
        ("```SQL\nSELECT count(*) FROM orders\n```", "SELECT count(*) FROM orders"),
        ("SELECT  u.name   FROM   users   u", "SELECT name FROM users"),
        (
            "SELECT name AS user_name, age AS user_age FROM users",
            "SELECT name, age FROM users",
        ),
        (
            "SELECT * FROM users WHERE age > 20 AND age < 35",
            "SELECT * FROM users WHERE age < 35 AND age > 20",
        ),
        (
            "SELECT user_id, amount FROM orders WHERE amount != 150.0",
            "SELECT user_id, amount FROM orders WHERE amount <> 150.0",
        ),
        (
            "SELECT department_name FROM departments WHERE budget = 500000.0",
            "SELECT department_name FROM departments WHERE budget = 500000",
        ),
        (
            "SELECT name FROM users ORDER BY age ASC",
            "SELECT name FROM users ORDER BY age",
        ),
        ("SELECT 100.0 * 2", "SELECT 200.0"),
        (
            "SELECT u.name FROM users u WHERE u.age >= (SELECT 25)",
            "SELECT name FROM users WHERE age >= 25",
        ),
    ]
    for cand, gold in semantic_queries:
        samples.append({"candidate": cand, "gold": gold, "db_path": str(db_path)})

    # 31-40: Syntax Errors and Malformed Queries (should be deterministically penalized)
    syntax_error_queries = [
        ("SELEC * FROM users", "SELECT * FROM users"),
        ("SELECT FROM users", "SELECT name FROM users"),
        ("SELECT * FROM WHERE", "SELECT * FROM users"),
        ("SELECT 'unclosed string FROM users", "SELECT name FROM users"),
        ("NOT SQL QUERY AT ALL", "SELECT count(*) FROM users"),
        ("SELECT (1 + 2 FROM users", "SELECT 3"),
        ("DROP TABLE users", "SELECT * FROM users"),
        ("DELETE FROM orders", "SELECT * FROM orders"),
        ("UPDATE users SET age = 99", "SELECT * FROM users"),
        ("", "SELECT count(*) FROM users"),
    ]
    for cand, gold in syntax_error_queries:
        samples.append({"candidate": cand, "gold": gold, "db_path": str(db_path)})

    # 41-50: Schema linking errors and semantic mismatches
    mismatch_queries = [
        ("SELECT nonexistent_col FROM users", "SELECT name FROM users"),
        ("SELECT * FROM nonexistent_table", "SELECT * FROM users"),
        (
            "SELECT name FROM users WHERE age = 999",
            "SELECT name FROM users WHERE age = 25",
        ),
        ("SELECT department_id FROM users", "SELECT department_name FROM departments"),
        (
            "SELECT name FROM users ORDER BY age DESC",
            "SELECT name FROM users ORDER BY age ASC",
        ),
        (
            "SELECT amount FROM orders WHERE amount > 1000",
            "SELECT amount FROM orders WHERE amount > 100",
        ),
        (
            "SELECT user_id FROM orders WHERE amount = 0",
            "SELECT user_id FROM orders WHERE amount = 150",
        ),
        (
            "SELECT count(*) FROM users WHERE age < 10",
            "SELECT count(*) FROM users WHERE age > 10",
        ),
        (
            "SELECT name FROM users WHERE name = 'Zoe'",
            "SELECT name FROM users WHERE name = 'Alice'",
        ),
        (
            "SELECT budget FROM departments WHERE department_id = 99",
            "SELECT budget FROM departments WHERE department_id = 1",
        ),
    ]
    for cand, gold in mismatch_queries:
        samples.append({"candidate": cand, "gold": gold, "db_path": str(db_path)})

    return samples


def test_portability_rollout_invariance(sample_sqlite_db: Path) -> None:
    """Verify that candidate SQL queries receive 100% deterministic rewards across runs."""
    benchmark_samples = build_50_portability_benchmark(sample_sqlite_db)
    assert len(benchmark_samples) == 50

    rubric = SQLRewardRubric(
        {
            "float_tolerance": 1e-4,
            "use_test_suite": True,
            "num_mutations": 3,
            "weights": {
                "syntax_penalty": -1.0,
                "exec_penalty": -0.5,
                "schema_bonus": 0.2,
                "base_execution_reward": 0.5,
                "test_suite_reward": 0.5,
            },
        }
    )

    # Pass 1: Run all 50 samples
    rewards_pass1: list[float] = []
    syntax_pass1: list[bool] = []
    exec_pass1: list[bool] = []
    equiv_pass1: list[bool] = []

    for item in benchmark_samples:
        out = rubric.evaluate(
            candidate_sql=item["candidate"],
            gold_sql=item["gold"],
            db_path=item["db_path"],
        )
        rewards_pass1.append(out.total_reward)
        syntax_pass1.append(out.is_valid_syntax)
        exec_pass1.append(out.is_valid_execution)
        equiv_pass1.append(out.is_execution_equivalent)

    # Pass 2: Re-run all 50 samples to verify zero drift / 100% determinism
    rewards_pass2: list[float] = []
    syntax_pass2: list[bool] = []
    exec_pass2: list[bool] = []
    equiv_pass2: list[bool] = []

    for item in benchmark_samples:
        out = rubric.evaluate(
            candidate_sql=item["candidate"],
            gold_sql=item["gold"],
            db_path=item["db_path"],
        )
        rewards_pass2.append(out.total_reward)
        syntax_pass2.append(out.is_valid_syntax)
        exec_pass2.append(out.is_valid_execution)
        equiv_pass2.append(out.is_execution_equivalent)

    assert len(rewards_pass1) == 50
    assert len(rewards_pass2) == 50
    assert rewards_pass1 == rewards_pass2, "Rewards diverged across runs!"
    assert syntax_pass1 == syntax_pass2, "Syntax checks diverged across runs!"
    assert exec_pass1 == exec_pass2, "Execution results diverged across runs!"
    assert equiv_pass1 == equiv_pass2, "Equivalence evaluation diverged across runs!"


def test_portability_adapter_reward_alignment(sample_sqlite_db: Path) -> None:
    """Verify that TRLGRPOAdapter computes identical rewards and advantages across multiple passes."""
    benchmark_samples = build_50_portability_benchmark(sample_sqlite_db)
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": "/tmp/portability_out",
                "learning_rate": 1e-5,
                "num_generations": 5,
                "max_completion_length": 256,
                "beta": 0.04,
            },
            "reward": {
                "use_test_suite": False,
            },
        }
    )
    rubric = SQLRewardRubric(getattr(cfg, "reward", None))
    adapter = TRLGRPOAdapter(cfg, rubric)

    batch = RolloutBatch(
        prompts=[f"Sample prompt {i}" for i in range(50)],
        completions=[item["candidate"] for item in benchmark_samples],
        gold_sqls=[item["gold"] for item in benchmark_samples],
        db_paths=[item["db_path"] for item in benchmark_samples],
    )

    # First pass
    rewards_1 = adapter.compute_rewards(batch)
    advantages_1 = adapter.compute_advantages(batch, group_size=5)

    # Second pass
    rewards_2 = adapter.compute_rewards(batch)
    advantages_2 = adapter.compute_advantages(batch, group_size=5)

    assert len(rewards_1) == 50
    assert rewards_1 == rewards_2
    assert len(advantages_1) == 50
    assert advantages_1 == advantages_2

    # Verify group normalization invariance
    for g_idx in range(0, 50, 5):
        chunk = advantages_1[g_idx : g_idx + 5]
        mean_chunk = sum(chunk) / len(chunk)
        assert abs(mean_chunk) < 1e-5, (
            f"Group {g_idx // 5} mean advantage is non-zero: {mean_chunk}"
        )
