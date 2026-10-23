"""Reviewed trajectory collection harness and stratified Spider dataset pipeline.

Implements the Agent-as-Harness trajectory generation loop:
1. Samples stratified Spider tasks across difficulty levels (Medium, Hard, Extra Hard).
2. Intercepts student model proposals using HarnessReviewer guided by K_SQL.
3. Masks reviewer rationale from recorded trajectory messages.
4. Validates execution outcomes and exports clean ChatML datasets for SFT.

Reference:
    Ye et al., September 2026. 'Harness Distillation via Agent-as-Harness'
    arXiv:2609.24974
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import sqlglot
from sqlglot import exp
from tqdm import tqdm

from ask_the_database.distillation.harness_reviewer import HarnessReviewer
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment

logger = logging.getLogger(__name__)


def classify_sql_difficulty(sql: str) -> str:
    """Classify SQL complexity into easy, medium, hard, extra based on AST features."""
    clean = sql.strip().rstrip(";")
    if not clean:
        return "easy"

    try:
        ast = sqlglot.parse_one(clean, read="sqlite")
    except Exception:
        return "medium"

    if isinstance(ast, exp.Union | exp.Intersect | exp.Except):
        return "extra"

    tables = list(ast.find_all(exp.Table))
    subqueries = list(ast.find_all(exp.Select))
    joins = list(ast.find_all(exp.Join))
    havings = list(ast.find_all(exp.Having))
    group_bys = list(ast.find_all(exp.Group))

    # Multiple subqueries or set operations
    if len(subqueries) > 2 or (len(subqueries) > 1 and (joins or havings)):
        return "extra"

    # Hard: subqueries, group by + having, 3+ tables joined
    if len(subqueries) > 1 or havings or len(tables) >= 3 or len(joins) >= 2:
        return "hard"

    # Medium: 2 tables joined or group by or multiple conditions
    if len(tables) >= 2 or joins or group_bys:
        return "medium"

    return "easy"


def sample_stratified_spider_tasks(
    spider_path: str | Path,
    databases_dir: str | Path,
    n_medium: int = 150,
    n_hard: int = 200,
    n_extra: int = 150,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample tasks from Spider training split stratified by hardness.

    Args:
        spider_path: Path to train_spider.json or processed train.jsonl.
        databases_dir: Path to database directory containing .sqlite files.
        n_medium: Target count for Medium tasks (multi-table joins).
        n_hard: Target count for Hard tasks (subqueries, group-by having).
        n_extra: Target count for Extra Hard tasks (nested sets, complex agg).
        seed: Random seed for reproducible sampling.

    Returns:
        list[dict[str, Any]]: Stratified list of valid database tasks.
    """
    s_path = Path(spider_path)
    db_dir = Path(databases_dir)

    raw_items: list[dict[str, Any]] = []
    if s_path.suffix == ".jsonl":
        with open(s_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    raw_items.append(json.loads(line_str))
    else:
        with open(s_path, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                raw_items = loaded

    # Group by difficulty while validating database file existence
    by_difficulty: dict[str, list[dict[str, Any]]] = {
        "medium": [],
        "hard": [],
        "extra": [],
        "easy": [],
    }

    for idx, item in enumerate(raw_items):
        db_id = str(item.get("db_id", "")).strip()
        if not db_id:
            continue

        # Locate physical database path
        candidate_path = db_dir / db_id / f"{db_id}.sqlite"
        if not candidate_path.exists():
            matches = list(db_dir.glob(f"**/{db_id}.sqlite"))
            if matches:
                candidate_path = matches[0]
            else:
                continue

        sql = str(item.get("gold_sql") or item.get("query") or item.get("sql", "")).strip()
        question = str(item.get("question", "")).strip()
        if not sql or not question:
            continue

        difficulty = str(item.get("difficulty", "")).lower()
        if difficulty not in by_difficulty:
            difficulty = classify_sql_difficulty(sql)

        task_record = {
            "instance_id": str(item.get("instance_id", f"spider_strat_{idx:05d}")),
            "db_id": db_id,
            "db_path": str(candidate_path),
            "question": question,
            "gold_sql": sql,
            "difficulty": difficulty,
        }
        by_difficulty[difficulty].append(task_record)

    # Deterministic sampling
    rng = random.Random(seed)
    rng.shuffle(by_difficulty["medium"])
    rng.shuffle(by_difficulty["hard"])
    rng.shuffle(by_difficulty["extra"])

    sampled: list[dict[str, Any]] = []
    sampled.extend(by_difficulty["medium"][:n_medium])
    sampled.extend(by_difficulty["hard"][:n_hard])
    sampled.extend(by_difficulty["extra"][:n_extra])

    # If the file had fewer than requested (e.g. smoke test subset), supplement with remaining available
    if len(sampled) < (n_medium + n_hard + n_extra):
        logger.info(
            "Stratified sampling gathered %d tasks (requested %d). Including remaining tasks.",
            len(sampled),
            n_medium + n_hard + n_extra,
        )
        remaining_needed = (n_medium + n_hard + n_extra) - len(sampled)
        easy_tasks = by_difficulty.get("easy", [])
        rng.shuffle(easy_tasks)
        sampled.extend(easy_tasks[:remaining_needed])

    rng.shuffle(sampled)
    return sampled


class StudentPolicy(Protocol):
    """Protocol defining student response generation."""

    def generate(
        self,
        observation: str,
        history: list[dict[str, str]],
        instance: dict[str, Any],
        turn: int,
    ) -> str:
        """Produce candidate proposal string (e.g. '<tool ...>' or '<final_answer>...')."""
        ...


class MockStudentPolicy:
    """Deterministic simulated student for offline collection testing and dry runs.

    Emulates realistic student failure modes:
    - Turn 1: Premature final_answer without inspecting schema (exercises Rule 1).
    - Turn 2: Trial execution without row sampling (exercises Rule 2).
    - Turn 3: Premature final_answer without prior dry-run (exercises Rule 3).
    - Turn 4+: Submits verified final_answer.
    """

    def generate(
        self,
        observation: str,
        history: list[dict[str, str]],
        instance: dict[str, Any],
        turn: int,
    ) -> str:
        """Generate simulated student proposals across turns."""
        gold_sql = str(instance.get("gold_sql", "SELECT 1;")).strip()

        turn_proposals = {
            1: f"<final_answer>{gold_sql}</final_answer>",
            2: f'<tool name="execute_sql">{gold_sql}</tool>',
            3: f"<final_answer>{gold_sql}</final_answer>",
        }
        return turn_proposals.get(turn, f"<final_answer>{gold_sql}</final_answer>")


@dataclass
class TrajectoryRecord:
    """Structured record of a single reviewed rollout episode."""

    instance_id: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str
    turns_count: int
    interventions_count: int
    rules_triggered: list[str]
    messages: list[dict[str, str]]
    final_query: str | None
    reward: float
    is_success: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for JSONL export."""
        return {
            "instance_id": self.instance_id,
            "db_id": self.db_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "difficulty": self.difficulty,
            "turns_count": self.turns_count,
            "interventions_count": self.interventions_count,
            "rules_triggered": self.rules_triggered,
            "messages": self.messages,
            "final_query": self.final_query,
            "reward": round(self.reward, 4),
            "is_success": self.is_success,
        }


class ReviewedTrajectoryCollector:
    """Orchestrates interactive Agent-as-Harness rollouts and exports reviewed SFT datasets."""

    def __init__(
        self,
        reviewer: HarnessReviewer,
        env: SkyRLSQLEnvironment | None = None,
        student: StudentPolicy | None = None,
        max_turns: int = 5,
        require_success: bool = True,
    ) -> None:
        self.reviewer = reviewer
        self.env = env if env is not None else SkyRLSQLEnvironment({"max_turns": max_turns})
        self.student = student if student is not None else MockStudentPolicy()
        self.max_turns = max_turns
        self.require_success = require_success
        self.equivalence_evaluator = ExecutionEquivalenceEvaluator()

    def _verify_execution(self, gen_sql: str, gold_sql: str, db_path: str) -> bool:
        """Verify semantic execution equivalence between candidate and gold SQL."""
        if not gen_sql or not os.path.exists(db_path):
            return False

        try:
            env = SQLiteEnvironment(db_path, timeout_seconds=5.0)
            gold_res = env.execute(gold_sql)
            gen_res = env.execute(gen_sql)

            if not gold_res.success or not gen_res.success:
                return False

            res = self.equivalence_evaluator.compare(gen_res, gold_res)
            return bool(res.is_equivalent)
        except Exception:
            return False

    def collect_episode(self, instance: dict[str, Any]) -> TrajectoryRecord | None:
        """Run single interactive rollout with HarnessReviewer interception.

        Args:
            instance: Dictionary with keys 'instance_id', 'db_id', 'db_path', 'question', 'gold_sql'.

        Returns:
            TrajectoryRecord if episode meets acceptance criteria, else None.
        """
        obs = self.env.reset(instance)
        self.reviewer.reset()

        reward = 0.0
        is_done = False
        step_info: dict[str, Any] = {}

        for _ in range(self.max_turns):
            turn_number = self.env.current_turn + 1
            proposal = self.student.generate(
                observation=obs,
                history=self.env.history,
                instance=instance,
                turn=turn_number,
            )

            obs, reward, is_done, step_info, _ = self.reviewer.step_env(self.env, proposal)

            if is_done:
                break

        final_query = self.env.state.final_query if self.env.state else None
        gold_sql = str(instance.get("gold_sql", "")).strip()
        db_path = str(instance.get("db_path", ""))

        is_success = False
        if final_query and gold_sql and db_path:
            is_success = self._verify_execution(final_query, gold_sql, db_path)

        if self.require_success and not is_success:
            logger.debug(
                "Episode %s rejected: final query did not match gold execution.",
                instance.get("instance_id"),
            )
            return None

        # Mask reviewer reasoning and sanitize messages for SFT
        clean_messages = self.reviewer.strip_reviewer_metadata_from_chat(self.env.history)

        rules_triggered = [
            str(d.rule_triggered) for d in self.reviewer.history.decisions if d.rule_triggered
        ]

        return TrajectoryRecord(
            instance_id=str(instance.get("instance_id", "unknown")),
            db_id=str(instance.get("db_id", "")),
            question=str(instance.get("question", "")),
            gold_sql=gold_sql,
            difficulty=str(instance.get("difficulty", "medium")),
            turns_count=self.env.current_turn,
            interventions_count=self.reviewer.history.interventions_count,
            rules_triggered=rules_triggered,
            messages=clean_messages,
            final_query=final_query,
            reward=reward,
            is_success=is_success,
        )

    def collect_batch(
        self,
        instances: list[dict[str, Any]],
        output_path: str | Path | None = None,
        progress_bar: bool = True,
    ) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
        """Collect reviewed rollouts across a batch of tasks and optionally stream to JSONL.

        Args:
            instances: List of task instances.
            output_path: Optional path to output JSONL file.
            progress_bar: Display tqdm progress bar.

        Returns:
            tuple[list[TrajectoryRecord], dict[str, Any]]: Accepted records and collection telemetry.
        """
        collected: list[TrajectoryRecord] = []
        out_file = Path(output_path) if output_path else None
        if out_file:
            out_file.parent.mkdir(parents=True, exist_ok=True)

        iterator = (
            tqdm(instances, desc="Collecting reviewed rollouts") if progress_bar else instances
        )

        rule_counts: dict[str, int] = {}
        diff_counts: dict[str, int] = {}

        if out_file:
            with open(out_file, "w", encoding="utf-8") as file_handle:
                for inst in iterator:
                    record = self.collect_episode(inst)
                    if record is not None:
                        collected.append(record)
                        diff_counts[record.difficulty] = diff_counts.get(record.difficulty, 0) + 1
                        for rule in record.rules_triggered:
                            rule_counts[rule] = rule_counts.get(rule, 0) + 1
                        file_handle.write(json.dumps(record.to_dict()) + "\n")
                        file_handle.flush()
        else:
            for inst in iterator:
                record = self.collect_episode(inst)
                if record is not None:
                    collected.append(record)
                    diff_counts[record.difficulty] = diff_counts.get(record.difficulty, 0) + 1
                    for rule in record.rules_triggered:
                        rule_counts[rule] = rule_counts.get(rule, 0) + 1

        total_tasks = len(instances)
        collected_count = len(collected)
        success_rate = (collected_count / total_tasks) if total_tasks > 0 else 0.0
        avg_turns = sum(r.turns_count for r in collected) / max(collected_count, 1)
        total_interventions = sum(r.interventions_count for r in collected)
        avg_interventions = total_interventions / max(collected_count, 1)

        metrics = {
            "total_tasks_processed": total_tasks,
            "collected_trajectories": collected_count,
            "collection_success_rate": round(success_rate, 4),
            "average_trajectory_turns": round(avg_turns, 2),
            "total_interventions": total_interventions,
            "average_interventions_per_trajectory": round(avg_interventions, 2),
            "interventions_by_rule": rule_counts,
            "difficulty_breakdown": diff_counts,
            "output_path": str(out_file) if out_file else None,
        }

        logger.info(
            "Collection finished: %d / %d accepted (%.1f%%). Total interventions: %d.",
            collected_count,
            total_tasks,
            success_rate * 100.0,
            total_interventions,
        )
        return collected, metrics
