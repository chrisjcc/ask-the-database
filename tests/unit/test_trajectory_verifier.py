"""Unit tests for Multi-Turn Trajectory Verification Engine (Issue #82)."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ask_the_database.inference.pipeline import AgentStep, AgentTrajectory, NL2SQLPipeline
from ask_the_database.inference.trajectory_verifier import (
    TrajectoryCandidate,
    TrajectoryCandidateSampler,
    TrajectoryScore,
    TrajectorySelectionOutcome,
    TrajectoryVerifier,
    TrajectoryVerifierSelector,
)
from scripts.evaluate_benchmark import run_benchmark_evaluation


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Create a temporary SQLite database for testing."""
    db_file = tmp_path / "test_traj.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, salary REAL, dept TEXT);"
    )
    cursor.execute("INSERT INTO employees VALUES (1, 'Alice', 95000, 'Engineering');")
    cursor.execute("INSERT INTO employees VALUES (2, 'Bob', 80000, 'Sales');")
    cursor.execute("INSERT INTO employees VALUES (3, 'Charlie', 110000, 'Engineering');")
    conn.commit()
    conn.close()
    return str(db_file)


def test_trajectory_candidate_serialization() -> None:
    """Verify serialization of TrajectoryCandidate, TrajectoryScore, and Outcome."""
    step1 = AgentStep(
        turn=1,
        action="inspect_schema",
        action_input=["employees"],
        output="DDL...",
        is_terminal=False,
    )
    step2 = AgentStep(
        turn=2,
        action="final_answer",
        action_input="SELECT * FROM employees;",
        output="Done",
        is_terminal=True,
    )
    traj = AgentTrajectory(
        question="Show all employees",
        db_id="test_db",
        db_path="/tmp/test.db",
        steps=[step1, step2],
        final_query="SELECT * FROM employees;",
        final_execution_success=True,
        had_error_recovery=False,
    )

    cand = TrajectoryCandidate(
        candidate_id="cand_1",
        trajectory=traj,
        final_sql="SELECT * FROM employees;",
        is_executable=True,
        inspected_tables=["employees"],
        trial_queries=[],
    )
    d = cand.to_dict()
    assert d["candidate_id"] == "cand_1"
    assert d["is_executable"] is True
    assert "employees" in d["inspected_tables"]

    score = TrajectoryScore(
        candidate_id="cand_1",
        final_sql="SELECT * FROM employees;",
        overall_score=0.88,
        criteria_scores={"schema_grounding": 0.9, "intent_alignment": 0.85},
        execution_valid=True,
        had_error_recovery=False,
    )
    s_dict = score.to_dict()
    assert s_dict["overall_score"] == 0.88
    assert s_dict["had_error_recovery"] is False

    outcome = TrajectorySelectionOutcome(
        selected_query="SELECT * FROM employees;",
        selected_candidate_id="cand_1",
        selected_trajectory=traj,
        selected_score=score,
        all_ranked_scores=[score],
        surviving_candidates_count=1,
        discarded_candidates_count=0,
        majority_vote=None,
        agrees_with_majority_vote=False,
        verification_latency_ms=12.5,
    )
    o_dict = outcome.to_dict()
    assert o_dict["selected_query"] == "SELECT * FROM employees;"
    assert o_dict["surviving_candidates_count"] == 1
    assert o_dict["selection_mechanism"] == "trajectory_verifier_tournament"


def test_trajectory_candidate_sampler(temp_db: str) -> None:
    """Verify TrajectoryCandidateSampler sampling and table/trial extraction."""
    pipeline = NL2SQLPipeline(model_path=None, max_turns=3)
    sampler = TrajectoryCandidateSampler(pipeline=pipeline, num_candidates=3, max_turns=3)

    candidates = sampler.sample_candidates(
        question="How many employees are in engineering?",
        db_path=temp_db,
        db_id="test_db",
        n=3,
        gold_sql="SELECT COUNT(*) FROM employees WHERE dept = 'Engineering';",
    )

    assert len(candidates) == 3
    for cand in candidates:
        assert isinstance(cand, TrajectoryCandidate)
        assert cand.final_sql != ""
        # The mock pipeline inspects table in turn 1
        assert "employees" in cand.inspected_tables or "demo_table" in cand.inspected_tables


def test_trajectory_sampler_fallback(temp_db: str) -> None:
    """Verify that if all trajectories fail, candidate 0 is preserved as fallback."""
    pipeline = NL2SQLPipeline(model_path=None, max_turns=2)
    sampler = TrajectoryCandidateSampler(pipeline=pipeline, num_candidates=2, max_turns=2)

    # Provide an intentionally broken gold query in mock mode
    candidates = sampler.sample_candidates(
        question="Test query",
        db_path=temp_db,
        n=2,
        gold_sql="SELECT * FROM non_existent_table_xyz;",
    )
    assert len(candidates) == 2
    # Candidate 0 must be preserved as fallback
    assert candidates[0].is_executable is True


def test_trajectory_verifier_heuristics() -> None:
    """Verify TrajectoryVerifier decomposed heuristic scoring across criteria."""
    verifier = TrajectoryVerifier()

    # Grounded trajectory with recovery
    step1 = AgentStep(turn=1, action="inspect_schema", action_input=["employees"], output="ok")
    step2 = AgentStep(
        turn=2, action="execute_sql", action_input="SELECT count(*) FROM employees", output="3"
    )
    traj_good = AgentTrajectory(
        question="How many employees are there?",
        db_id="db1",
        db_path="/tmp/db1",
        steps=[step1, step2],
        final_query="SELECT COUNT(*) FROM employees;",
        final_execution_success=True,
        had_error_recovery=True,
    )
    cand_good = TrajectoryCandidate(
        candidate_id="c_good",
        trajectory=traj_good,
        final_sql="SELECT COUNT(*) FROM employees;",
        is_executable=True,
        inspected_tables=["employees"],
        trial_queries=["SELECT count(*) FROM employees"],
        had_error_recovery=True,
    )

    score_good = verifier.score_trajectory(cand_good, question="How many employees are there?")
    assert score_good.criteria_scores["schema_grounding"] >= 0.85
    assert score_good.criteria_scores["trial_execution"] >= 0.90
    assert score_good.criteria_scores["error_recovery"] == 1.00
    assert score_good.criteria_scores["intent_alignment"] >= 0.70
    assert score_good.overall_score >= 0.75

    # Ungrounded, failed trajectory
    traj_bad = AgentTrajectory(
        question="How many employees?",
        db_id="db1",
        db_path="/tmp/db1",
        steps=[],
        final_query="INVALID SQL QUERY",
        final_execution_success=False,
    )
    cand_bad = TrajectoryCandidate(
        candidate_id="c_bad",
        trajectory=traj_bad,
        final_sql="INVALID SQL QUERY",
        is_executable=False,
        inspected_tables=[],
        trial_queries=[],
    )
    score_bad = verifier.score_trajectory(cand_bad, question="How many employees?")
    assert score_bad.overall_score < score_good.overall_score
    assert score_bad.criteria_scores["schema_grounding"] <= 0.40
    assert score_bad.criteria_scores["intent_alignment"] <= 0.20


def test_trajectory_verifier_continuous_expectation() -> None:
    """Verify continuous expectation scoring when client logit callback is provided."""

    # Mock logit client returning high probability for token '5' (1.00)
    def mock_client(prompt: str) -> dict[str, float]:
        return {"1": -5.0, "2": -3.0, "3": 0.0, "4": 2.0, "5": 8.0}

    verifier = TrajectoryVerifier(client=mock_client)
    traj = AgentTrajectory(
        question="Find top salary",
        db_id="db",
        db_path="/tmp/db",
        final_query="SELECT MAX(salary) FROM employees;",
        final_execution_success=True,
    )
    cand = TrajectoryCandidate(
        candidate_id="c_1",
        trajectory=traj,
        final_sql="SELECT MAX(salary) FROM employees;",
        is_executable=True,
    )
    score = verifier.score_trajectory(cand, question="Find top salary")
    assert score.overall_score > 0.80
    assert any("5" in k for k in score.scoring_logprobs)


def test_trajectory_verifier_selector_end_to_end(temp_db: str) -> None:
    """Verify TrajectoryVerifierSelector tournament and majority voting on test database."""
    pipeline = NL2SQLPipeline(model_path=None, max_turns=3)
    sampler = TrajectoryCandidateSampler(pipeline=pipeline, num_candidates=3, max_turns=3)
    selector = TrajectoryVerifierSelector(sampler=sampler, pivot_k=2)

    outcome = selector.run_pipeline(
        question="How many employees work in Engineering?",
        db_path=temp_db,
        db_id="test_db",
        num_candidates=3,
        gold_sql="SELECT COUNT(*) FROM employees WHERE dept = 'Engineering';",
    )

    assert isinstance(outcome, TrajectorySelectionOutcome)
    assert outcome.selected_query != ""
    assert outcome.surviving_candidates_count >= 1
    assert len(outcome.all_ranked_scores) >= 1
    assert outcome.verification_latency_ms > 0
    assert outcome.selection_mechanism == "trajectory_verifier_tournament"


def test_evaluate_benchmark_trajectory_mode(temp_db: str) -> None:
    """Verify that evaluate_benchmark script correctly supports verification_mode='trajectory'."""
    custom_sample = {
        "question": "Show all employees ordered by salary desc",
        "db_path": temp_db,
        "db_id": "test_db",
        "gold_sql": "SELECT name, salary FROM employees ORDER BY salary DESC;",
    }

    eval_config: dict[str, Any] = {
        "evaluation": {
            "model_path": None,  # Mock mode
            "benchmarks": ["custom_bench"],
            "custom_samples": {"custom_bench": [custom_sample]},
            "verification_mode": "trajectory",
            "num_candidates": 2,
            "max_turns": 2,
            "temperature": 0.7,
            "summary_file": "/tmp/test_eval_summary.json",
        }
    }

    results = run_benchmark_evaluation(eval_config)
    assert "custom_bench" in results
    bench_res = results["custom_bench"]
    assert bench_res["total_samples"] == 1
    assert bench_res["valid_execution_rate_va"] >= 0.0
