"""Unit tests for Reviewed Trajectory Collection & Data Pipeline (Sub-Issue #62)."""

import json
from pathlib import Path

from ask_the_database.distillation.collector import (
    MockStudentPolicy,
    ReviewedTrajectoryCollector,
    classify_sql_difficulty,
    sample_stratified_spider_tasks,
)
from ask_the_database.distillation.harness_reviewer import HarnessReviewer
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment
from scripts.collect_reviewed_trajectories import (
    build_student_policy,
    build_teacher_callable,
    parse_args,
    run_collection,
)


def test_classify_sql_difficulty() -> None:
    """Verify AST-based SQL difficulty classification."""
    # Easy
    easy_sql = "SELECT name FROM users WHERE age > 20;"
    assert classify_sql_difficulty(easy_sql) == "easy"

    # Medium (two-table join or group by)
    med_sql_1 = "SELECT users.name, departments.department_name FROM users JOIN departments ON users.department_id = departments.department_id;"
    assert classify_sql_difficulty(med_sql_1) in ("medium", "hard")

    med_sql_2 = "SELECT department_id, count(*) FROM users GROUP BY department_id;"
    assert classify_sql_difficulty(med_sql_2) == "medium"

    # Hard (having or subquery)
    hard_sql_1 = "SELECT department_id FROM users GROUP BY department_id HAVING count(*) > 2;"
    assert classify_sql_difficulty(hard_sql_1) == "hard"

    hard_sql_2 = "SELECT name FROM users WHERE age > (SELECT avg(age) FROM users);"
    assert classify_sql_difficulty(hard_sql_2) == "hard"

    # Extra Hard (set operations: UNION, INTERSECT, EXCEPT)
    extra_sql_1 = "SELECT name FROM users UNION SELECT department_name FROM departments;"
    assert classify_sql_difficulty(extra_sql_1) == "extra"

    extra_sql_2 = "SELECT user_id FROM users EXCEPT SELECT user_id FROM orders;"
    assert classify_sql_difficulty(extra_sql_2) == "extra"


def test_sample_stratified_spider_tasks(sample_sqlite_db: Path, tmp_path: Path) -> None:
    """Verify task stratification and sampling with real SQLite fixture."""
    # Create mock train JSON
    raw_tasks = [
        {
            "db_id": "company",
            "question": "What is the name of user 1?",
            "query": "SELECT name FROM users WHERE user_id = 1;",
        },
        {
            "db_id": "company",
            "question": "How many users per department?",
            "query": "SELECT department_id, count(*) FROM users GROUP BY department_id;",
        },
        {
            "db_id": "company",
            "question": "Departments with more than 1 user?",
            "query": "SELECT department_id FROM users GROUP BY department_id HAVING count(*) > 1;",
        },
        {
            "db_id": "company",
            "question": "Users without orders?",
            "query": "SELECT user_id FROM users EXCEPT SELECT user_id FROM orders;",
        },
    ]

    mock_json = tmp_path / "mock_train.json"
    mock_json.write_text(json.dumps(raw_tasks))

    # databases dir contains company.sqlite in tmp_path
    db_dir = sample_sqlite_db.parent

    stratified = sample_stratified_spider_tasks(
        spider_path=mock_json,
        databases_dir=db_dir,
        n_medium=1,
        n_hard=1,
        n_extra=1,
        seed=42,
    )

    assert len(stratified) >= 3
    for task in stratified:
        assert "instance_id" in task
        assert "db_id" in task
        assert "db_path" in task
        assert Path(task["db_path"]).exists()
        assert task["difficulty"] in ("easy", "medium", "hard", "extra")


def test_mock_student_policy() -> None:
    """Verify MockStudentPolicy emits expected action sequence."""
    policy = MockStudentPolicy()
    instance = {"gold_sql": "SELECT name FROM users;"}

    t1 = policy.generate("", [], instance, turn=1)
    assert t1 == "<final_answer>SELECT name FROM users;</final_answer>"

    t2 = policy.generate("", [], instance, turn=2)
    assert t2 == '<tool name="execute_sql">SELECT name FROM users;</tool>'

    t3 = policy.generate("", [], instance, turn=3)
    assert t3 == "<final_answer>SELECT name FROM users;</final_answer>"


def test_reviewed_trajectory_collector_single_episode(sample_sqlite_db: Path) -> None:
    """Verify single-episode interactive rollout, interception, and trajectory creation."""
    reviewer = HarnessReviewer()
    env = SkyRLSQLEnvironment({"max_turns": 5})
    collector = ReviewedTrajectoryCollector(
        reviewer=reviewer,
        env=env,
        student=MockStudentPolicy(),
        require_success=False,
    )

    instance = {
        "instance_id": "test_001",
        "db_id": "company",
        "db_path": str(sample_sqlite_db),
        "question": "What is the name of user 1?",
        "gold_sql": "SELECT name FROM users WHERE user_id = 1;",
        "difficulty": "medium",
    }

    record = collector.collect_episode(instance)
    assert record is not None
    assert record.instance_id == "test_001"
    assert record.turns_count >= 1
    assert record.interventions_count >= 1
    assert "require_schema_inspection" in record.rules_triggered
    assert len(record.messages) >= 3

    # Ensure no reviewer thoughts leak into trajectory messages
    for msg in record.messages:
        assert msg["role"] in ("system", "user", "assistant", "tool")
        assert "reviewer" not in msg["role"]
        assert "Intervention" not in msg["content"]


def test_reviewed_trajectory_collector_batch_and_jsonl(
    sample_sqlite_db: Path, tmp_path: Path
) -> None:
    """Verify batch collection, telemetry computation, and streaming to JSONL file."""
    reviewer = HarnessReviewer()
    collector = ReviewedTrajectoryCollector(
        reviewer=reviewer,
        student=MockStudentPolicy(),
        require_success=False,
    )

    tasks = [
        {
            "instance_id": "task_1",
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
            "question": "How many users?",
            "gold_sql": "SELECT count(*) FROM users;",
            "difficulty": "easy",
        },
        {
            "instance_id": "task_2",
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
            "question": "Names of users?",
            "gold_sql": "SELECT name FROM users;",
            "difficulty": "medium",
        },
    ]

    out_file = tmp_path / "trajectories.jsonl"
    records, metrics = collector.collect_batch(tasks, output_path=out_file, progress_bar=False)

    assert len(records) == 2
    assert out_file.exists()

    # Verify JSONL lines
    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        assert "instance_id" in data
        assert "messages" in data
        assert "turns_count" in data

    # Verify metrics
    assert metrics["total_tasks_processed"] == 2
    assert metrics["collected_trajectories"] == 2
    assert metrics["total_interventions"] >= 1
    assert "require_schema_inspection" in metrics["interventions_by_rule"]


def test_cli_dry_run_workflow(sample_sqlite_db: Path, tmp_path: Path) -> None:
    """Verify CLI entrypoint with dry-run mode and synthetic tasks."""
    raw_tasks = [
        {
            "db_id": "company",
            "question": "User count?",
            "query": "SELECT count(*) FROM users;",
        },
        {
            "db_id": "company",
            "question": "Department budget?",
            "query": "SELECT budget FROM departments WHERE department_id = 1;",
        },
    ]
    raw_file = tmp_path / "train_spider.json"
    raw_file.write_text(json.dumps(raw_tasks))

    out_jsonl = tmp_path / "dry_run_reviewed.jsonl"
    out_metrics = tmp_path / "dry_run_metrics.json"

    args = parse_args(
        [
            "--raw-spider",
            str(raw_file),
            "--databases-dir",
            str(sample_sqlite_db.parent),
            "--output",
            str(out_jsonl),
            "--metrics-output",
            str(out_metrics),
            "--dry-run",
            "--no-progress",
        ]
    )

    metrics = run_collection(args)
    assert metrics["collected_trajectories"] >= 1
    assert out_jsonl.exists()
    assert out_metrics.exists()


def test_policy_and_teacher_builders() -> None:
    """Verify builder functions instantiate appropriate instances."""
    mock_student = build_student_policy("mock", "test-model")
    assert isinstance(mock_student, MockStudentPolicy)

    unknown_student = build_student_policy("unknown_type", "test-model")
    assert isinstance(unknown_student, MockStudentPolicy)

    rule_teacher = build_teacher_callable("rule", "test-teacher")
    assert rule_teacher is None
