"""Tournament re-ranking and candidate selection pipeline using LLM-as-a-Verifier.

Combines candidate generation, SQLite sandbox pre-filtering, and SQLVerifier
Probabilistic Pivot Tournament selection with majority-voting baseline comparator.
"""

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ask_the_database.inference.candidate_sampler import CandidateQuery, CandidateSampler
from ask_the_database.verification.sql_verifier import CandidateScore, SQLVerifier

logger = logging.getLogger(__name__)


@dataclass
class MajorityVoteOutcome:
    """Execution-based majority voting consensus outcome."""

    consensus_query: str
    consensus_result_hash: str
    agreement_count: int
    total_executable_candidates: int
    agreement_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert majority vote outcome to dictionary representation."""
        return {
            "consensus_query": self.consensus_query,
            "consensus_result_hash": self.consensus_result_hash,
            "agreement_count": self.agreement_count,
            "total_executable_candidates": self.total_executable_candidates,
            "agreement_ratio": self.agreement_ratio,
        }


@dataclass
class VerifierSelectionOutcome:
    """Complete diagnostic outcome of test-time candidate verification and selection."""

    selected_query: str
    selected_candidate_id: str
    selected_score: CandidateScore
    all_ranked_scores: list[CandidateScore]
    surviving_candidates_count: int
    discarded_candidates_count: int
    majority_vote: MajorityVoteOutcome | None
    agrees_with_majority_vote: bool
    verification_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    selection_mechanism: str = "verifier_tournament"

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
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "selection_mechanism": self.selection_mechanism,
        }


def hash_execution_data(data: list[tuple[Any, ...]] | None) -> str:
    """Generate deterministic hash of database query result tuples."""
    if data is None:
        return "NONE"
    if not data:
        return "EMPTY_SET"

    # Sort string representations to achieve order invariance if rows are unordered
    normalized_rows = tuple(sorted(str(row) for row in data))
    return hashlib.sha256(repr(normalized_rows).encode("utf-8")).hexdigest()[:16]


class VerifierSelector:
    """Test-time verification and tournament re-ranking selector for candidate SQL queries."""

    def __init__(
        self,
        verifier: SQLVerifier | None = None,
        sampler: CandidateSampler | None = None,
        verifier_config: str | Path | dict[str, Any] | None = None,
    ) -> None:
        """Initialize VerifierSelector.

        Args:
            verifier: Optional pre-configured SQLVerifier instance.
            sampler: Optional pre-configured CandidateSampler instance.
            verifier_config: Optional path or dict config for SQLVerifier if verifier is None.
        """
        self.verifier = verifier if verifier is not None else SQLVerifier(config=verifier_config)
        self.sampler = sampler if sampler is not None else CandidateSampler()

    def compute_majority_vote(
        self,
        candidates: Sequence[CandidateQuery],
    ) -> MajorityVoteOutcome | None:
        """Compute execution-based majority voting consensus across candidate queries.

        Args:
            candidates: Sequence of CandidateQuery objects with execution diagnostics.

        Returns:
            MajorityVoteOutcome or None if no candidates are executable.
        """
        executable_cands = [
            c for c in candidates if c.is_executable and c.execution_result is not None
        ]
        if not executable_cands:
            return None

        # Map each candidate to result hash
        hash_to_candidates: dict[str, list[CandidateQuery]] = {}
        for c in executable_cands:
            h = hash_execution_data(c.execution_result.data if c.execution_result else None)
            hash_to_candidates.setdefault(h, []).append(c)

        # Count frequencies
        counts = Counter({h: len(c_list) for h, c_list in hash_to_candidates.items()})
        most_common_hash, agreement_count = counts.most_common(1)[0]
        consensus_cand = hash_to_candidates[most_common_hash][0]

        total_exec = len(executable_cands)
        ratio = round(agreement_count / total_exec, 4) if total_exec > 0 else 0.0

        return MajorityVoteOutcome(
            consensus_query=consensus_cand.sql,
            consensus_result_hash=most_common_hash,
            agreement_count=agreement_count,
            total_executable_candidates=total_exec,
            agreement_ratio=ratio,
        )

    def select_from_candidates(
        self,
        candidates: Sequence[CandidateQuery],
        question: str,
        schema_ddl: str,
        discarded_count: int = 0,
    ) -> VerifierSelectionOutcome:
        """Score and re-rank candidates via Probabilistic Pivot Tournament.

        Args:
            candidates: Surviving CandidateQuery objects to evaluate.
            question: User question string.
            schema_ddl: Schema DDL string.
            discarded_count: Number of candidates discarded during sandbox pre-filtering.

        Returns:
            VerifierSelectionOutcome with selected query and telemetric data.
        """
        if not candidates:
            raise ValueError("Cannot select from an empty candidates sequence.")

        start_time = time.perf_counter()

        # Format candidates for tournament
        cand_payloads: list[dict[str, Any]] = [
            {
                "id": c.candidate_id,
                "sql": c.sql,
                "execution_valid": c.is_executable,
                "execution_result": c.execution_result,
            }
            for c in candidates
        ]

        ranked_scores = self.verifier.probabilistic_pivot_tournament(
            cand_payloads,
            question=question,
            schema_ddl=schema_ddl,
        )
        best_score = ranked_scores[0]

        # Majority vote comparator
        majority_vote = self.compute_majority_vote(candidates)
        agrees_with_majority = False
        if majority_vote is not None:
            # Check query string match or execution result hash match
            selected_cand = next(
                (c for c in candidates if c.candidate_id == best_score.candidate_id), None
            )
            if selected_cand and selected_cand.execution_result is not None:
                selected_hash = hash_execution_data(selected_cand.execution_result.data)
                agrees_with_majority = selected_hash == majority_vote.consensus_result_hash
            else:
                agrees_with_majority = best_score.sql_query.strip().rstrip(
                    ";"
                ) == majority_vote.consensus_query.strip().rstrip(";")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Estimate token usage: ~1 token per 4 chars prompt, 1 token completion per criterion call
        num_criteria = len(self.verifier.criteria)
        est_prompt_tokens = sum(
            len(self.verifier.build_verification_prompt(question, schema_ddl, c.sql, crit)) // 4
            for c in candidates
            for crit in self.verifier.criteria.values()
        )
        est_completion_tokens = len(candidates) * num_criteria

        return VerifierSelectionOutcome(
            selected_query=best_score.sql_query,
            selected_candidate_id=best_score.candidate_id,
            selected_score=best_score,
            all_ranked_scores=ranked_scores,
            surviving_candidates_count=len(candidates),
            discarded_candidates_count=discarded_count,
            majority_vote=majority_vote,
            agrees_with_majority_vote=agrees_with_majority,
            verification_latency_ms=round(elapsed_ms, 2),
            total_prompt_tokens=est_prompt_tokens,
            total_completion_tokens=est_completion_tokens,
            selection_mechanism="verifier_tournament",
        )

    def run_pipeline(
        self,
        question: str,
        schema_ddl: str,
        db_path: str | None = None,
        num_candidates: int = 5,
        temperature: float = 0.7,
    ) -> VerifierSelectionOutcome:
        """Run full end-to-end pipeline: generation -> sandbox filter -> tournament selection.

        Args:
            question: Natural language question.
            schema_ddl: Database schema DDL string.
            db_path: Optional path to SQLite database.
            num_candidates: Number of candidates N to generate.
            temperature: Sampling temperature T.

        Returns:
            VerifierSelectionOutcome.
        """
        # 1. Sample N candidates
        candidates = self.sampler.generate_candidates(
            question=question,
            schema_ddl=schema_ddl,
            n=num_candidates,
            temperature=temperature,
        )

        # 2. Filter via sandbox if db_path is provided
        if db_path and Path(db_path).exists():
            surviving, discarded = self.sampler.filter_candidates_in_sandbox(candidates, db_path)
            discarded_count = len(discarded)
        else:
            surviving = list(candidates)
            discarded_count = 0

        # 3. Select best candidate via verifier tournament
        return self.select_from_candidates(
            candidates=surviving,
            question=question,
            schema_ddl=schema_ddl,
            discarded_count=discarded_count,
        )
