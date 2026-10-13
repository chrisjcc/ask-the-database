"""Unit tests for DatabaseToolRegistry, InteractionState, and prompt formatting."""

from pathlib import Path

from ask_the_database.agent.interaction_state import InteractionState
from ask_the_database.agent.prompt_formatter import (
    format_chatml_prompt,
    parse_agent_action,
)
from ask_the_database.agent.tool_registry import DatabaseToolRegistry


def test_inspect_schema_valid_tables(sample_sqlite_db: Path) -> None:
    """Test inspect_schema returns DDL for valid tables."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.inspect_schema(["departments", "users"])

    assert not res.is_terminal
    assert res.action == "inspect_schema"
    assert "CREATE TABLE departments" in res.output
    assert "CREATE TABLE users" in res.output
    assert res.final_query is None


def test_inspect_schema_missing_table(sample_sqlite_db: Path) -> None:
    """Test inspect_schema gracefully handles non-existent tables."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.inspect_schema(["non_existent_table"])

    assert not res.is_terminal
    assert "Error: Table 'non_existent_table' not found" in res.output


def test_sample_data_valid_table(sample_sqlite_db: Path) -> None:
    """Test sample_data returns formatted table rows."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.sample_data("users", limit=2)

    assert not res.is_terminal
    assert res.action == "sample_data"
    assert "name" in res.output
    assert "Alice" in res.output
    assert "Bob" in res.output


def test_sample_data_invalid_table(sample_sqlite_db: Path) -> None:
    """Test sample_data gracefully returns error on invalid table."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.sample_data("ghost_table")

    assert not res.is_terminal
    assert "Execution error" in res.output


def test_execute_sql_success(sample_sqlite_db: Path) -> None:
    """Test execute_sql runs trial query and returns result preview."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.execute_sql("SELECT name, age FROM users WHERE age > 25 ORDER BY age ASC;")

    assert not res.is_terminal
    assert res.action == "execute_sql"
    assert "Success: 3 rows returned" in res.output
    assert "Bob" in res.output


def test_execute_sql_syntax_error(sample_sqlite_db: Path) -> None:
    """Test execute_sql handles syntax error."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.execute_sql("SELEKT * FROM users;")

    assert not res.is_terminal
    assert "Database Error" in res.output


def test_execute_sql_security_mutation_rejection(sample_sqlite_db: Path) -> None:
    """Test execute_sql blocks destructive mutations in sandbox."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.execute_sql("DROP TABLE users;")

    assert not res.is_terminal
    assert "Database Error" in res.output


def test_final_answer(sample_sqlite_db: Path) -> None:
    """Test final_answer terminates episode and captures query."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))
    res = registry.final_answer("SELECT COUNT(*) FROM users;")

    assert res.is_terminal
    assert res.action == "final_answer"
    assert res.final_query == "SELECT COUNT(*) FROM users;"


def test_execute_action_dispatch(sample_sqlite_db: Path) -> None:
    """Test dynamic action dispatch helper."""
    registry = DatabaseToolRegistry(str(sample_sqlite_db))

    # Inspect schema with string payload
    res1 = registry.execute_action("inspect_schema", '["departments"]')
    assert "CREATE TABLE departments" in res1.output

    # Sample data
    res2 = registry.execute_action("sample_data", "departments")
    assert "Engineering" in res2.output

    # Execute sql
    res3 = registry.execute_action("execute_sql", "SELECT department_name FROM departments;")
    assert "Success: 3 rows returned" in res3.output

    # Final answer
    res4 = registry.execute_action("final_answer", "SELECT 1;")
    assert res4.is_terminal
    assert res4.final_query == "SELECT 1;"

    # Unknown tool
    res5 = registry.execute_action("hack_database", "foo")
    assert not res5.is_terminal
    assert "Error: Unknown tool" in res5.output


def test_interaction_state(sample_sqlite_db: Path) -> None:
    """Test InteractionState turn recording and lifecycle properties."""
    state = InteractionState(
        question="How many users are in engineering?",
        db_id="company",
        db_path=str(sample_sqlite_db),
        max_turns=3,
    )

    assert not state.is_finished
    assert state.current_turn == 0
    assert not state.has_intermediate_error

    # Turn 1: intermediate tool call with error
    state.add_turn(
        action="execute_sql",
        action_input="SELECT count FROM users_typo;",
        observation="Database Error: no such table: users_typo",
        is_terminal=False,
    )
    assert state.current_turn == 1
    assert state.has_intermediate_error
    assert not state.is_finished

    # Turn 2: final answer
    state.add_turn(
        action="final_answer",
        action_input="SELECT COUNT(*) FROM users WHERE department_id = 1;",
        observation="Submitted final query.",
        is_terminal=True,
    )
    assert state.current_turn == 2
    assert state.is_finished
    assert state.final_query == "SELECT COUNT(*) FROM users WHERE department_id = 1;"

    chat_history = state.to_chat_history()
    assert len(chat_history) >= 4
    assert any("users_typo" in msg["content"] for msg in chat_history)

    dump = state.to_dict()
    assert dump["current_turn"] == 2
    assert dump["has_intermediate_error"] is True
    assert dump["final_query"] is not None


def test_prompt_formatting_and_action_parsing() -> None:
    """Test ChatML prompt formatting and action parsing regex."""
    messages = [
        {"role": "system", "content": "You are a database agent."},
        {"role": "user", "content": "Find all users."},
    ]
    formatted = format_chatml_prompt(messages)
    assert "<|im_start|>system\nYou are a database agent.<|im_end|>" in formatted
    assert "<|im_start|>user\nFind all users.<|im_end|>" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")

    # Parse tool call
    act, payload = parse_agent_action(
        'Let me inspect the schema.\n<tool name="inspect_schema">["users"]</tool>'
    )
    assert act == "inspect_schema"
    assert payload == '["users"]'

    # Parse final answer
    act2, payload2 = parse_agent_action(
        "Here is the result:\n<final_answer>\nSELECT * FROM users;\n</final_answer>"
    )
    assert act2 == "final_answer"
    assert payload2 == "SELECT * FROM users;"

    # Fallback
    act3, payload3 = parse_agent_action("SELECT * FROM users;")
    assert act3 == "final_answer"
    assert payload3 == "SELECT * FROM users;"
