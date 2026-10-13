"""Unit tests for multi-turn step discounting and recovery bonus reward evaluation."""

from pathlib import Path

from omegaconf import OmegaConf

from ask_the_database.rewards.multi_turn_reward import (
    MultiTurnRewardEvaluator,
    MultiTurnRewardOutput,
)


def test_single_turn_perfect_answer(sample_sqlite_db: Path) -> None:
    """1-turn perfect answer receives maximum reward (1.0)."""
    evaluator = MultiTurnRewardEvaluator(OmegaConf.create({"step_penalty": -0.05, "recovery_bonus": 0.3}))

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
    pred_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"

    reward = evaluator.evaluate_step(
        action_name="final_answer",
        tool_output="Submitted final query.",
        final_query=pred_sql,
        gold_sql=gold_sql,
        db_path=str(sample_sqlite_db),
        turn=1,
        is_terminal=True,
    )

    assert reward == 1.0


def test_three_turn_answer_with_recovery_bonus(sample_sqlite_db: Path) -> None:
    """3-turn trajectory recovering from a syntax error receives discount + recovery bonus."""
    evaluator = MultiTurnRewardEvaluator(OmegaConf.create({"step_penalty": -0.05, "recovery_bonus": 0.3}))

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
    db_path = str(sample_sqlite_db)

    # Turn 1: tool call with database syntax error
    r1 = evaluator.evaluate_step(
        action_name="execute_sql",
        tool_output="Database Error: near 'SELEKT': syntax error",
        final_query=None,
        gold_sql=gold_sql,
        db_path=db_path,
        turn=1,
        is_terminal=False,
    )
    assert r1 == -0.05
    assert evaluator.had_intermediate_error is True

    # Turn 2: tool call with successful schema check
    r2 = evaluator.evaluate_step(
        action_name="inspect_schema",
        tool_output="CREATE TABLE users (name TEXT, age INT);",
        final_query=None,
        gold_sql=gold_sql,
        db_path=db_path,
        turn=2,
        is_terminal=False,
    )
    assert r2 == -0.05
    assert evaluator.had_intermediate_error is True

    # Turn 3: final answer successfully resolving the question
    detailed = evaluator.evaluate_step_detailed(
        action_name="final_answer",
        tool_output="Submitted final query.",
        final_query="SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        gold_sql=gold_sql,
        db_path=db_path,
        turn=3,
        is_terminal=True,
    )

    assert isinstance(detailed, MultiTurnRewardOutput)
    assert detailed.is_execution_equivalent is True
    assert detailed.had_intermediate_error is True
    assert detailed.recovery_bonus == 0.3
    # Turn discount for turn 3 is 2 * 0.05 = 0.10
    assert abs(detailed.turn_discount - 0.10) < 1e-5
    # Base reward 1.0 - 0.10 + 0.30 = 1.20 -> clamped to 1.0
    assert detailed.total_reward == 1.0


def test_three_turn_answer_without_error_has_no_recovery_bonus(sample_sqlite_db: Path) -> None:
    """3-turn trajectory with clean tool calls receives discount but NO recovery bonus."""
    evaluator = MultiTurnRewardEvaluator(OmegaConf.create({"step_penalty": -0.05, "recovery_bonus": 0.3}))

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
    db_path = str(sample_sqlite_db)

    # Turn 1: clean inspect
    evaluator.evaluate_step(
        action_name="inspect_schema",
        tool_output="CREATE TABLE users ...",
        final_query=None,
        gold_sql=gold_sql,
        db_path=db_path,
        turn=1,
        is_terminal=False,
    )
    assert evaluator.had_intermediate_error is False

    # Turn 2: clean sample
    evaluator.evaluate_step(
        action_name="sample_data",
        tool_output="Alice | 25",
        final_query=None,
        gold_sql=gold_sql,
        db_path=db_path,
        turn=2,
        is_terminal=False,
    )
    assert evaluator.had_intermediate_error is False

    # Turn 3: final answer
    detailed = evaluator.evaluate_step_detailed(
        action_name="final_answer",
        tool_output="Submitted final query.",
        final_query="SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        gold_sql=gold_sql,
        db_path=db_path,
        turn=3,
        is_terminal=True,
    )

    assert detailed.recovery_bonus == 0.0
    assert abs(detailed.turn_discount - 0.10) < 1e-5
    # Base 1.0 - 0.10 = 0.90
    assert abs(detailed.total_reward - 0.90) < 1e-5


def test_terminal_without_final_query(sample_sqlite_db: Path) -> None:
    """Exhausting turn budget without final answer receives -1.0 penalty."""
    evaluator = MultiTurnRewardEvaluator()

    r = evaluator.evaluate_step(
        action_name="sample_data",
        tool_output="data",
        final_query=None,
        gold_sql="SELECT 1;",
        db_path=str(sample_sqlite_db),
        turn=5,
        is_terminal=True,
    )
    assert r == -1.0


def test_evaluator_reset(sample_sqlite_db: Path) -> None:
    """Test reset clears error memory across episodes."""
    evaluator = MultiTurnRewardEvaluator()

    evaluator.evaluate_step(
        action_name="execute_sql",
        tool_output="Database Error: syntax error",
        final_query=None,
        gold_sql="SELECT 1;",
        db_path=str(sample_sqlite_db),
        turn=1,
        is_terminal=False,
    )
    assert evaluator.had_intermediate_error is True

    evaluator.reset()
    assert evaluator.had_intermediate_error is False
