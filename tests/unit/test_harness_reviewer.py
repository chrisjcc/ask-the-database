"""Unit tests for HarnessReviewer and K_SQL Reference Harness (Sub-Issue #61)."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ask_the_database.distillation.harness_reviewer import (
    HarnessReviewer,
    ReviewDecision,
    ReviewedSkyRLEnvironment,
)
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment


def test_reviewer_initialization_defaults() -> None:
    """Verify reviewer initializes with default K_SQL configuration."""
    reviewer = HarnessReviewer()
    assert reviewer.rule_schema_inspection_enabled is True
    assert reviewer.rule_row_sampling_enabled is True
    assert reviewer.rule_dry_run_enabled is True
    assert reviewer.rule_recovery_enabled is True
    assert reviewer.max_interventions >= 1
    assert reviewer.history.interventions_count == 0


def test_review_decision_dataclass() -> None:
    """Verify ReviewDecision formatting and properties."""
    pass_decision = ReviewDecision(decision="PASS", rationale="All good", turn_index=1)
    assert pass_decision.is_pass is True
    assert pass_decision.is_replacement is False
    assert pass_decision.formatted_replacement == ""

    replace_inspect = ReviewDecision(
        decision="REPLACE",
        rule_triggered="require_schema_inspection",
        replacement_action="inspect_schema",
        replacement_payload=["users", "departments"],
        turn_index=1,
    )
    assert replace_inspect.is_pass is False
    assert replace_inspect.is_replacement is True
    assert (
        '<tool name="inspect_schema">["users", "departments"]</tool>'
        in replace_inspect.formatted_replacement
    )

    replace_final = ReviewDecision(
        decision="REPLACE",
        rule_triggered="enforce_dry_run_before_finish",
        replacement_action="final_answer",
        replacement_payload="SELECT * FROM users;",
        turn_index=3,
    )
    assert (
        replace_final.formatted_replacement == "<final_answer>SELECT * FROM users;</final_answer>"
    )

    # Serialization
    d_dict = replace_final.to_dict()
    assert d_dict["decision"] == "REPLACE"
    assert d_dict["replacement_action"] == "final_answer"
    assert d_dict["is_replacement"] is True


def test_rule_require_schema_inspection_trigger(sample_sqlite_db: Path) -> None:
    """Rule 1: Intercept execute_sql and final_answer on uninspected tables."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    # 1. Propose final_answer without prior inspect_schema
    proposal = "<final_answer>SELECT name FROM users WHERE age > 25;</final_answer>"
    decision = reviewer.review(proposal, db_path=db_path, turn_index=1)

    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "require_schema_inspection"
    assert decision.replacement_action == "inspect_schema"
    assert "users" in [t.lower() for t in decision.replacement_payload]
    assert reviewer.history.interventions_count == 1

    # 2. Propose execute_sql without prior inspect_schema
    reviewer.reset()
    proposal_sql = '<tool name="execute_sql">SELECT * FROM departments;</tool>'
    decision2 = reviewer.review(proposal_sql, db_path=db_path, turn_index=1)

    assert decision2.decision == "REPLACE"
    assert decision2.rule_triggered == "require_schema_inspection"
    assert decision2.replacement_action == "inspect_schema"
    assert "departments" in [t.lower() for t in decision2.replacement_payload]


def test_rule_require_schema_inspection_pass_after_inspection(
    sample_sqlite_db: Path,
) -> None:
    """Rule 1: Pass proposal when referenced tables have already been inspected."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    # Record that users table was inspected
    reviewer.record_observation(
        "inspect_schema", ["users"], "CREATE TABLE users (id INT, age INT);"
    )
    assert "users" in reviewer.history.inspected_tables

    # Propose query on users (without categorical string filter)
    proposal = '<tool name="execute_sql">SELECT count(*) FROM users WHERE age > 25;</tool>'
    decision = reviewer.review(proposal, db_path=db_path, turn_index=2)

    assert decision.decision == "PASS"
    assert decision.is_pass is True


def test_rule_require_row_sampling_trigger_and_pass(sample_sqlite_db: Path) -> None:
    """Rule 2: Intercept queries filtering on string literals when table not yet sampled."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    # Step 1: Record schema inspection for users
    reviewer.record_observation(
        "inspect_schema", ["users"], "CREATE TABLE users (name TEXT, age INT);"
    )

    # Step 2: Propose query filtering on string literal 'Alice'
    proposal = "<tool name=\"execute_sql\">SELECT age FROM users WHERE name = 'Alice';</tool>"
    decision = reviewer.review(proposal, db_path=db_path, turn_index=2)

    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "require_row_sampling"
    assert decision.replacement_action == "sample_data"
    assert decision.replacement_payload == "users"
    assert '<tool name="sample_data">users</tool>' in decision.formatted_replacement

    # Step 3: Record sample_data execution
    reviewer.record_observation("sample_data", "users", "Alice | 25\nBob | 30")
    assert "users" in reviewer.history.sampled_tables

    # Step 4: Propose the same query again -> Now passes!
    decision2 = reviewer.review(proposal, db_path=db_path, turn_index=3)
    assert decision2.decision == "PASS"


def test_rule_enforce_dry_run_before_finish(sample_sqlite_db: Path) -> None:
    """Rule 3: Enforce trial execution via execute_sql before final_answer submission."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    # Record schema inspection and sampling
    reviewer.record_observation("inspect_schema", ["users"], "CREATE TABLE users (name TEXT);")
    reviewer.record_observation("sample_data", "users", "Alice | 25")

    query = "SELECT name FROM users WHERE name = 'Alice';"
    proposal = f"<final_answer>{query}</final_answer>"

    # Attempt final_answer before execute_sql
    decision = reviewer.review(proposal, db_path=db_path, turn_index=3)
    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "enforce_dry_run_before_finish"
    assert decision.replacement_action == "execute_sql"
    assert decision.replacement_payload == query
    assert f'<tool name="execute_sql">{query}</tool>' in decision.formatted_replacement

    # Record successful trial execution
    reviewer.record_observation(
        "execute_sql", query, "Success: 1 rows returned. Preview: [('Alice',)]"
    )

    # Now final_answer passes
    decision2 = reviewer.review(proposal, db_path=db_path, turn_index=4)
    assert decision2.decision == "PASS"
    assert decision2.is_pass is True


def test_rule_diagnostic_recovery_repeated_query(sample_sqlite_db: Path) -> None:
    """Rule 4: Intercept identical repeated query following execution error."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    reviewer.record_observation("inspect_schema", ["users"], "DDL")

    bad_query = "SELECT non_existent_col FROM users;"
    reviewer.record_observation(
        "execute_sql", bad_query, "Database Error: no such column: non_existent_col"
    )

    # Propose the exact same bad query again
    proposal = f'<tool name="execute_sql">{bad_query}</tool>'
    decision = reviewer.review(proposal, db_path=db_path, turn_index=3)

    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "diagnostic_recovery"
    assert decision.replacement_action == "inspect_schema"
    assert "users" in [t.lower() for t in decision.replacement_payload]


def test_rule_diagnostic_recovery_missing_table(sample_sqlite_db: Path) -> None:
    """Rule 4: Intercept when previous query threw no such table."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    reviewer.record_observation(
        "execute_sql",
        "SELECT * FROM ghost_table;",
        "Database Error: no such table: ghost_table",
    )

    proposal = '<tool name="execute_sql">SELECT id FROM ghost_table WHERE id = 1;</tool>'
    decision = reviewer.review(proposal, db_path=db_path, turn_index=2)

    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "diagnostic_recovery"
    assert decision.replacement_action == "inspect_schema"


def test_intervention_budget_exhaustion(sample_sqlite_db: Path) -> None:
    """Reviewer must PASS candidate proposals once intervention budget is exhausted."""
    cfg = OmegaConf.create(
        {
            "rules": {
                "require_schema_inspection": {"enabled": True},
                "require_row_sampling": {"enabled": True},
                "enforce_dry_run_before_finish": {"enabled": True},
                "diagnostic_recovery": {"enabled": True},
            },
            "teacher": {
                "max_interventions_per_episode": 2,
                "allow_rule_based_fallback": True,
            },
        }
    )
    reviewer = HarnessReviewer(config=cfg)
    db_path = str(sample_sqlite_db)

    # Intervention 1
    d1 = reviewer.review(
        "<final_answer>SELECT * FROM users;</final_answer>",
        db_path=db_path,
        turn_index=1,
    )
    assert d1.is_replacement is True
    assert reviewer.history.interventions_count == 1

    # Intervention 2
    d2 = reviewer.review(
        "<final_answer>SELECT * FROM departments;</final_answer>",
        db_path=db_path,
        turn_index=2,
    )
    assert d2.is_replacement is True
    assert reviewer.history.interventions_count == 2

    # Intervention 3 -> Budget reached (2/2), proposal must pass
    d3 = reviewer.review(
        "<final_answer>SELECT * FROM orders;</final_answer>",
        db_path=db_path,
        turn_index=3,
    )
    assert d3.decision == "PASS"
    assert d3.is_pass is True
    assert "budget" in d3.rationale.lower()


def test_teacher_llm_interception_and_graceful_fallback(sample_sqlite_db: Path) -> None:
    """Verify optional teacher LLM callable interception and fallback on failure."""

    # 1. Custom Teacher LLM that returns a valid decision
    def mock_teacher_ok(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": "REPLACE",
            "rule_triggered": "require_schema_inspection",
            "replacement_action": "inspect_schema",
            "replacement_payload": ["users"],
            "rationale": "Mock teacher recommends inspecting users schema first.",
        }

    reviewer = HarnessReviewer(teacher_llm_callable=mock_teacher_ok)
    decision = reviewer.review("<final_answer>SELECT * FROM users;</final_answer>", turn_index=1)
    assert decision.is_replacement is True
    assert decision.replacement_action == "inspect_schema"
    assert "Mock teacher" in decision.rationale

    # 2. Teacher LLM raises exception -> falls back to deterministic rule engine
    def mock_teacher_error(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Teacher API connection error")

    reviewer_fb = HarnessReviewer(teacher_llm_callable=mock_teacher_error)
    db_path = str(sample_sqlite_db)
    decision_fb = reviewer_fb.review(
        "<final_answer>SELECT * FROM users;</final_answer>",
        db_path=db_path,
        turn_index=1,
    )
    assert decision_fb.is_replacement is True
    assert decision_fb.rule_triggered == "require_schema_inspection"


def test_strip_reviewer_metadata_from_chat() -> None:
    """Verify trajectory masking strips reviewer rationale and keeps clean agent ChatML."""
    chat = [
        {"role": "system", "content": "You are a database agent."},
        {"role": "user", "content": "Database: company\nQuestion: Find users"},
        {
            "role": "assistant",
            "content": '<tool name="inspect_schema">["users"]</tool>',
        },
        {"role": "tool", "content": "CREATE TABLE users (id INT);"},
        {
            "role": "reviewer_thought",
            "content": "Teacher private rationale that must not leak",
        },
    ]

    clean = HarnessReviewer.strip_reviewer_metadata_from_chat(chat)
    assert len(clean) == 4
    assert not any(m["role"] == "reviewer_thought" for m in clean)
    assert clean[2]["role"] == "assistant"
    assert clean[2]["content"] == '<tool name="inspect_schema">["users"]</tool>'


def test_skyrl_env_integration_with_harness_reviewer(sample_sqlite_db: Path) -> None:
    """Verify full multi-turn step_env interception loop with SkyRLSQLEnvironment."""
    env_cfg = OmegaConf.create({"max_turns": 5, "step_penalty": -0.05})
    env = SkyRLSQLEnvironment(env_cfg)
    reviewer = HarnessReviewer()

    instance = {
        "question": "What is the name of user Alice?",
        "gold_sql": "SELECT name FROM users WHERE name = 'Alice';",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
    }

    env.reset(instance)
    reviewer.reset()

    # Turn 1: Student prematurely proposes final_answer
    obs1, r1, done1, info1, d1 = reviewer.step_env(
        env,
        "<final_answer>SELECT name FROM users WHERE name = 'Alice';</final_answer>",
    )
    assert not done1
    assert d1.is_replacement is True
    assert d1.rule_triggered == "require_schema_inspection"
    assert info1["was_replaced"] is True
    assert info1["action"] == "inspect_schema"
    assert "CREATE TABLE users" in obs1

    # Turn 2: Student proposes execute_sql with string literal 'Alice' without sampling
    obs2, r2, done2, info2, d2 = reviewer.step_env(
        env,
        "<tool name=\"execute_sql\">SELECT name FROM users WHERE name = 'Alice';</tool>",
    )
    assert not done2
    assert d2.is_replacement is True
    assert d2.rule_triggered == "require_row_sampling"
    assert info2["action"] == "sample_data"
    assert "Alice" in obs2

    # Turn 3: Student proposes final_answer before dry-run execution
    obs3, r3, done3, info3, d3 = reviewer.step_env(
        env,
        "<final_answer>SELECT name FROM users WHERE name = 'Alice';</final_answer>",
    )
    assert not done3
    assert d3.is_replacement is True
    assert d3.rule_triggered == "enforce_dry_run_before_finish"
    assert info3["action"] == "execute_sql"
    assert "Success: 1 rows returned" in obs3

    # Turn 4: Student proposes verified final_answer -> PASSED and finishes episode
    obs4, r4, done4, info4, d4 = reviewer.step_env(
        env,
        "<final_answer>SELECT name FROM users WHERE name = 'Alice';</final_answer>",
    )
    assert done4 is True
    assert d4.is_pass is True
    assert info4["was_replaced"] is False
    assert info4["action"] == "final_answer"
    assert info4["is_terminal"] is True


def test_reviewed_skyrl_environment_wrapper(sample_sqlite_db: Path) -> None:
    """Verify Gymnasium-compatible ReviewedSkyRLEnvironment class wrapper."""
    env = SkyRLSQLEnvironment()
    reviewer = HarnessReviewer()
    reviewed_env = ReviewedSkyRLEnvironment(env, reviewer)

    instance = {
        "question": "How many users are there?",
        "gold_sql": "SELECT count(*) FROM users;",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
    }

    obs = reviewed_env.reset(instance)
    assert "company" in obs

    # Step with uninspected query
    obs1, r1, done1, info1 = reviewed_env.step(
        "<final_answer>SELECT count(*) FROM users;</final_answer>"
    )
    assert not done1
    assert info1["was_replaced"] is True
    assert info1["action"] == "inspect_schema"

    clean_history = reviewed_env.clean_chat_history
    assert len(clean_history) >= 2
    assert all(msg["role"] in ("system", "user", "assistant", "tool") for msg in clean_history)
    assert len(reviewed_env.history) >= 2


def test_config_loading_variants(tmp_path: Path) -> None:
    """Verify loading configuration from DictConfig, dict, and YAML file path."""
    cfg_dict = {
        "rules": {
            "require_schema_inspection": {"enabled": True},
            "require_row_sampling": {"enabled": False},
        },
        "teacher": {"max_interventions_per_episode": 5},
    }
    reviewer_dict = HarnessReviewer(config=cfg_dict)
    assert reviewer_dict.max_interventions == 5
    assert reviewer_dict.rule_row_sampling_enabled is False

    yaml_file = tmp_path / "custom_rules.yaml"
    yaml_file.write_text("rules:\n  diagnostic_recovery:\n    enabled: false\n")
    reviewer_path = HarnessReviewer(config=yaml_file)
    assert reviewer_path.rule_recovery_enabled is False


def test_teacher_llm_pass_and_invalid_tool_rejection() -> None:
    """Verify teacher LLM returning PASS, and teacher returning illegal action falling back."""

    # Teacher approves
    def teacher_pass(payload: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "PASS", "rationale": "Looks great"}

    reviewer_pass = HarnessReviewer(teacher_llm_callable=teacher_pass)
    d_pass = reviewer_pass.review("<final_answer>SELECT 1;</final_answer>", turn_index=1)
    assert d_pass.is_pass is True

    # Teacher returns illegal tool outside target space h -> falls back to rule engine
    def teacher_illegal(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": "REPLACE",
            "replacement_action": "unauthorized_bash_exec",
            "replacement_payload": "rm -rf /",
        }

    reviewer_illegal = HarnessReviewer(teacher_llm_callable=teacher_illegal)
    d_fallback = reviewer_illegal.review(
        "<final_answer>SELECT * FROM users;</final_answer>",
        available_tables=["users"],
        turn_index=1,
    )
    assert d_fallback.is_replacement is True
    assert d_fallback.replacement_action in ("inspect_schema", "execute_sql")
    assert d_fallback.replacement_action != "unauthorized_bash_exec"


def test_rule_diagnostic_recovery_missing_column(sample_sqlite_db: Path) -> None:
    """Rule 4: Intercept when previous query threw no such column."""
    reviewer = HarnessReviewer()
    db_path = str(sample_sqlite_db)

    reviewer.record_observation(
        "execute_sql",
        "SELECT invalid_column FROM users;",
        "Database Error: no such column: invalid_column",
    )

    proposal = '<tool name="execute_sql">SELECT invalid_column FROM users WHERE id = 1;</tool>'
    decision = reviewer.review(proposal, db_path=db_path, turn_index=2)

    assert decision.decision == "REPLACE"
    assert decision.rule_triggered == "diagnostic_recovery"
    assert decision.replacement_action == "inspect_schema"
    assert "users" in [t.lower() for t in decision.replacement_payload]
