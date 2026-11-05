"""Inference package for NL2SQL agent rollout and benchmark evaluation."""

from ask_the_database.inference.candidate_sampler import (
    CandidateQuery,
    CandidateSampler,
    clean_candidate_sql,
)
from ask_the_database.inference.pipeline import (
    AgentStep,
    AgentTrajectory,
    NL2SQLPipeline,
)
from ask_the_database.inference.trajectory_verifier import (
    TrajectoryCandidate,
    TrajectoryCandidateSampler,
    TrajectoryScore,
    TrajectorySelectionOutcome,
    TrajectoryVerifier,
    TrajectoryVerifierSelector,
)
from ask_the_database.inference.verifier_selector import (
    MajorityVoteOutcome,
    VerifierSelectionOutcome,
    VerifierSelector,
)

__all__ = [
    "AgentStep",
    "AgentTrajectory",
    "CandidateQuery",
    "CandidateSampler",
    "MajorityVoteOutcome",
    "NL2SQLPipeline",
    "TrajectoryCandidate",
    "TrajectoryCandidateSampler",
    "TrajectoryScore",
    "TrajectorySelectionOutcome",
    "TrajectoryVerifier",
    "TrajectoryVerifierSelector",
    "VerifierSelectionOutcome",
    "VerifierSelector",
    "clean_candidate_sql",
]
