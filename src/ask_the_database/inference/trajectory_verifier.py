"""Multi-turn trajectory verification and tournament selection for autonomous database agents.

Extends LLM-as-a-Verifier from single-turn static SQL ranking to complete interactive
agent trajectories, eliminating the "Tool-Calling Tax" by evaluating catalog exploration,
intermediate sandbox dry-runs, runtime error recovery, and terminal SQL correctness.
"""

import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ask_the_database.environment.base import ExecutionResult
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.inference.pipeline import AgentTrajectory, NL2SQLPipeline
from ask_the_database.inference.verifier_selector import MajorityVoteOutcome, hash_execution_data
from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator
from ask_the_database.verification.sql_verifier import (
    DEFAULT_SCORING_TOKENS,
    compute_continuous_expectation,
)

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryCandidate:
    """Represents a sampled multi-turn agent trajectory and its execution diagnostics."""

    candidate_id: str
    trajectory: AgentTrajectory
    final_sql: str
    is_executable: bool = True
    execution_result: ExecutionResult | None = None
    had_error_recovery: bool = False
    inspected_tables: list[str] = field(default_factory=list)
    trial_queries: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "final_sql": self.final_sql,
            "is_executable": self.is_executable,
            "execution_success": (
                self.execution_result.success if self.execution_result is not None else None
            ),
            "had_error_recovery": self.had_error_recovery,
            "inspected_tables": self.inspected_tables,
            "trial_queries": self.trial_queries,
            "execution_time_ms": self.execution_time_ms,
            "trajectory": self.trajectory.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class TrajectoryScore:
    """Detailed score telemetry for a candidate agent trajectory."""

    candidate_id: str
    final_sql: str
    overall_score: float
    criteria_scores: dict[str, float] = field(default_factory=dict)
    scoring_logprobs: dict[str, float] = field(default_factory=dict)
    execution_valid: bool = True
    had_error_recovery: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory score to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "final_sql": self.final_sql,
            "overall_score": self.overall_score,
            "criteria_scores": self.criteria_scores,
            "execution_valid": self.execution_valid,
            "had_error_recovery": self.had_error_recovery,
            "metadata": self.metadata,
        }


@dataclass
class TrajectorySelectionOutcome:
    """Complete outcome of test-time multi-turn trajectory verification and selection."""

    selected_query: str
    selected_candidate_id: str
    selected_trajectory: AgentTrajectory
    selected_score: TrajectoryScore
    all_ranked_scores: list[TrajectoryScore]
    surviving_candidates_count: int
    discarded_candidates_count: int
    majority_vote: MajorityVoteOutcome | None
    agrees_with_majority_vote: bool
    verification_latency_ms: float
    selection_mechanism: str = "trajectory_verifier_tournament"

    def to_dict(self) -> dict[str, Any]:
        """Convert selection outcome to serializable dictionary."""
        return {
            "selected_query": self.selected_query,
            "selected_candidate_id": self.selected_candidate_id,
            "overall_score": self.selected_score.overall_score,
            "criteria_scores": self.selected_score.criteria_scores,
            "surviving_candidates_count": self.surviving_candidates_count,
            "discarded_candidates_count": self.discarded_candidates_count,
            "majority_vote": (
                self.majority_vote.to_dict() if self.majority_vote is not None else None
            ),
            "agrees_with_majority_vote": self.agrees_with_majority_vote,
            "verification_latency_ms": self.verification_latency_ms,
            "selection_mechanism": self.selection_mechanism,
            "selected_trajectory": self.selected_trajectory.to_dict(),
        }


class TrajectoryCandidateSampler:
    """Samples parallel/sequential multi-turn agent trajectories in isolated sandboxes."""

    def __init__(
        self,
        pipeline: NL2SQLPipeline | None = None,
        model_path: str | Path | None = None,
        num_candidates: int = 5,
        temperature: float = 0.7,
        max_turns: int = 5,
        device: str = "auto",
        torch_dtype: Any = None,
    ) -> None:
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = NL2SQLPipeline(
                model_path=model_path,
                device=device,
                torch_dtype=torch_dtype,
                max_turns=max_turns,
            )
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_turns = max_turns

    def sample_candidates(
        self,
        question: str,
        db_path: str,
        db_id: str = "default_db",
        n: int | None = None,
        temperature: float | None = None,
        gold_sql: str = "SELECT 1;",
    ) -> list[TrajectoryCandidate]:
        """Sample N independent multi-turn trajectories and inspect sandbox execution."""
        num_cands = n if n is not None and n > 0 else self.num_candidates
        temp = temperature if temperature is not None else self.temperature

        raw_candidates: list[TrajectoryCandidate] = []
        db_env = SQLiteEnvironment(db_path)

        for idx in range(num_cands):
            t_start = time.perf_counter()
            # In unit-test mock mode (no weights), temperature variations fall back to deterministic mocks
            do_sample = self.pipeline.model is not None or temp > 0.0

            traj = self.pipeline.run_interactive(
                question=question,
                db_path=db_path,
                db_id=db_id,
                max_turns=self.max_turns,
                gold_sql=gold_sql,
                do_sample=do_sample,
                temperature=temp,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            inspected_tables: list[str] = []
            trial_queries: list[str] = []
            for step in traj.steps:
                if step.action == "inspect_schema":
                    if isinstance(step.action_input, list):
                        inspected_tables.extend([str(tbl) for tbl in step.action_input])
                    elif isinstance(step.action_input, str):
                        inspected_tables.append(step.action_input)
                elif step.action == "sample_data":
                    inspected_tables.append(str(step.action_input))
                elif step.action == "execute_sql":
                    trial_queries.append(str(step.action_input))

            final_sql = traj.final_query.strip()
            exec_res = traj.final_execution_result
            if exec_res is None and final_sql:
                exec_res = db_env.execute(final_sql)

            is_executable = exec_res.success if exec_res is not None else False

            cand = TrajectoryCandidate(
                candidate_id=f"traj_{idx}",
                trajectory=traj,
                final_sql=final_sql,
                is_executable=is_executable,
                execution_result=exec_res,
                had_error_recovery=traj.had_error_recovery,
                inspected_tables=inspected_tables,
                trial_queries=trial_queries,
                execution_time_ms=elapsed_ms,
                metadata={"index": idx, "temperature": temp},
            )
            raw_candidates.append(cand)

        # Safety Fallback: If all trajectories failed sandbox execution, preserve candidate 0
        executable_cands = [c for c in raw_candidates if c.is_executable]
        if not executable_cands and raw_candidates:
            logger.warning(
                "All %d candidate trajectories failed sandbox execution. "
                "Preserving trajectory 0 as fallback.",
                len(raw_candidates),
            )
            raw_candidates[0].is_executable = True

        return raw_candidates


class TrajectoryVerifier:
    """Evaluates multi-turn agent trajectories across decomposed relational criteria."""

    def __init__(
        self,
        client: Callable[[str], dict[str, float]] | None = None,
        temperature: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.scoring_tokens = DEFAULT_SCORING_TOKENS
        self.syntax_validator = SQLSyntaxValidator()
        self.weights = weights or {
            "schema_grounding": 0.25,
            "trial_execution": 0.20,
            "error_recovery": 0.15,
            "intent_alignment": 0.40,
        }

    def build_verification_prompt(
        self,
        question: str,
        candidate: TrajectoryCandidate,
        criterion_name: str,
    ) -> str:
        """Construct prompt for external LLM verifier evaluating trajectory."""
        steps_summary = []
        for s in candidate.trajectory.steps:
            steps_summary.append(f"Turn {s.turn} [{s.action}]: {s.action_input}")

        history_str = "\n".join(steps_summary) if steps_summary else "No intermediate actions."
        return (
            "<|im_start|>system\n"
            "You are an expert evaluator assessing autonomous relational database agent trajectories. "
            "Rate how well the trajectory satisfies the specified criterion on a scale from 1 (poor) to 5 (excellent).\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"[User Question]:\n{question}\n\n"
            f"[Interaction History]:\n{history_str}\n\n"
            f"[Final SQL Query]:\n{candidate.final_sql}\n\n"
            f"[Criterion]: {criterion_name}\n"
            "Respond with only a score integer (1-5).\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\nScore: "
        )

    def fallback_heuristic_score(
        self,
        candidate: TrajectoryCandidate,
        question: str,
        criterion_name: str,
    ) -> float:
        """Deterministic heuristic scoring of agent trajectory."""
        if criterion_name == "schema_grounding":
            # Did the agent inspect any schemas?
            if candidate.inspected_tables:
                return 0.90
            return 0.30

        if criterion_name == "trial_execution":
            # Did the agent run trial queries before committing?
            if candidate.trial_queries:
                return 0.95
            return 0.40

        if criterion_name == "error_recovery":
            # Did the agent actively recover from a runtime error?
            if candidate.had_error_recovery:
                return 1.00
            return 0.50

        if criterion_name == "intent_alignment":
            # Evaluate final SQL syntax and basic keyword presence
            sql = candidate.final_sql.upper()
            q_lower = question.lower()
            val_res = self.syntax_validator.validate(candidate.final_sql)
            if not val_res.is_valid:
                return 0.10

            score = 0.60
            if ("how many" in q_lower or "count" in q_lower) and "COUNT(" in sql:
                score += 0.20
            if ("maximum" in q_lower or "highest" in q_lower or "top" in q_lower) and (
                "MAX(" in sql or "ORDER BY" in sql
            ):
                score += 0.20
            if ("average" in q_lower or "mean" in q_lower) and "AVG(" in sql:
                score += 0.20
            return min(1.0, score)

        return 0.50

    def score_trajectory(
        self,
        candidate: TrajectoryCandidate,
        question: str,
    ) -> TrajectoryScore:
        """Compute continuous expectation score across all decomposed trajectory criteria."""
        criteria_scores: dict[str, float] = {}
        all_logprobs: dict[str, float] = {}
        total_score = 0.0

        for crit_name, weight in self.weights.items():
            if self.client is not None:
                prompt = self.build_verification_prompt(question, candidate, crit_name)
                try:
                    logits = self.client(prompt)
                    score, probs = compute_continuous_expectation(
                        logits,
                        scoring_tokens=self.scoring_tokens,
                        temperature=self.temperature,
                    )
                except Exception as e:
                    logger.warning("Trajectory verifier client failed for %s: %s", crit_name, e)
                    score = self.fallback_heuristic_score(candidate, question, crit_name)
                    probs = {}
            else:
                score = self.fallback_heuristic_score(candidate, question, crit_name)
                probs = {}

            criteria_scores[crit_name] = score
            all_logprobs.update({f"{crit_name}_{k}": v for k, v in probs.items()})
            total_score += score * weight

        # Heavy discount if final query failed execution
        if not candidate.is_executable:
            total_score *= 0.15

        overall = round(max(0.0, min(1.0, total_score)), 4)
        return TrajectoryScore(
            candidate_id=candidate.candidate_id,
            final_sql=candidate.final_sql,
            overall_score=overall,
            criteria_scores=criteria_scores,
            scoring_logprobs=all_logprobs,
            execution_valid=candidate.is_executable,
            had_error_recovery=candidate.had_error_recovery,
            metadata={"inspected_tables_count": len(candidate.inspected_tables)},
        )


class TrajectoryVerifierSelector:
    """Probabilistic Pivot Tournament selection pipeline for multi-turn agent trajectories."""

    def __init__(
        self,
        sampler: TrajectoryCandidateSampler,
        verifier: TrajectoryVerifier | None = None,
        pivot_k: int = 2,
    ) -> None:
        self.sampler = sampler
        self.verifier = verifier if verifier is not None else TrajectoryVerifier()
        self.pivot_k = max(1, pivot_k)

    def run_pipeline(
        self,
        question: str,
        db_path: str,
        db_id: str = "default_db",
        num_candidates: int | None = None,
        temperature: float | None = None,
        gold_sql: str = "SELECT 1;",
    ) -> TrajectorySelectionOutcome:
        """Sample trajectories, score criteria, run tournament, and select winning query."""
        t_start = time.perf_counter()

        # 1. Sample N trajectories
        candidates = self.sampler.sample_candidates(
            question=question,
            db_path=db_path,
            db_id=db_id,
            n=num_candidates,
            temperature=temperature,
            gold_sql=gold_sql,
        )

        surviving = [c for c in candidates if c.is_executable]
        discarded_count = len(candidates) - len(surviving)

        # 2. Score trajectories
        scored_candidates: list[tuple[TrajectoryCandidate, TrajectoryScore]] = []
        for cand in surviving:
            score = self.verifier.score_trajectory(cand, question)
            scored_candidates.append((cand, score))

        # 3. Sort by initial expectation score
        scored_candidates.sort(key=lambda x: x[1].overall_score, reverse=True)

        # 4. Probabilistic Pivot Tournament Selection
        winner_cand, winner_score = self._run_tournament(scored_candidates)

        # 5. Execution-based Majority Voting baseline
        majority_vote = self._compute_majority_vote(surviving)
        agrees_with_majority = False
        if majority_vote is not None:
            # Check if winner execution matches majority hash
            w_res = winner_cand.execution_result
            if w_res is not None and w_res.success:
                w_hash = hash_execution_data(w_res.data)
                agrees_with_majority = w_hash == majority_vote.consensus_result_hash

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return TrajectorySelectionOutcome(
            selected_query=winner_cand.final_sql,
            selected_candidate_id=winner_cand.candidate_id,
            selected_trajectory=winner_cand.trajectory,
            selected_score=winner_score,
            all_ranked_scores=[s for _, s in scored_candidates],
            surviving_candidates_count=len(surviving),
            discarded_candidates_count=discarded_count,
            majority_vote=majority_vote,
            agrees_with_majority_vote=agrees_with_majority,
            verification_latency_ms=total_latency_ms,
        )

    def _run_tournament(
        self,
        scored: list[tuple[TrajectoryCandidate, TrajectoryScore]],
    ) -> tuple[TrajectoryCandidate, TrajectoryScore]:
        """Rank candidates in O(K * N) comparisons against top K pivots."""
        if not scored:
            raise ValueError("No candidate trajectories survived pre-filtering.")
        if len(scored) == 1:
            return scored[0]

        # Top K pivots
        pivots = scored[: min(self.pivot_k, len(scored))]
        tournament_points: dict[str, float] = {c.candidate_id: 0.0 for c, _ in scored}

        for cand, score in scored:
            for pivot_cand, pivot_score in pivots:
                if cand.candidate_id == pivot_cand.candidate_id:
                    continue
                # Margin-based continuous preference
                margin = score.overall_score - pivot_score.overall_score
                prob_win = 1.0 / (1.0 + 10.0 ** (-margin * 2.0))
                tournament_points[cand.candidate_id] += prob_win

        best_cand, best_score = max(
            scored,
            key=lambda x: (
                tournament_points.get(x[0].candidate_id, 0.0),
                x[1].overall_score,
            ),
        )
        return best_cand, best_score

    def _compute_majority_vote(
        self,
        candidates: Sequence[TrajectoryCandidate],
    ) -> MajorityVoteOutcome | None:
        """Compute execution consensus hash among surviving candidate trajectories."""
        valid_executions: list[tuple[str, str]] = []
        for c in candidates:
            if c.execution_result is not None and c.execution_result.success:
                h = hash_execution_data(c.execution_result.data)
                valid_executions.append((c.final_sql, h))

        if not valid_executions:
            return None

        hashes = [h for _, h in valid_executions]
        counts = Counter(hashes)
        consensus_hash, top_count = counts.most_common(1)[0]

        consensus_query = next(q for q, h in valid_executions if h == consensus_hash)
        total_exec = len(valid_executions)
        ratio = top_count / total_exec if total_exec > 0 else 0.0

        return MajorityVoteOutcome(
            consensus_query=consensus_query,
            consensus_result_hash=consensus_hash,
            agreement_count=top_count,
            total_executable_candidates=total_exec,
            agreement_ratio=round(ratio, 4),
        )
