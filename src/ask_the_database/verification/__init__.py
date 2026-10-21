"""Verification module for AskTheDatabase.

Implements continuous logit-expectation verification, decomposed criteria,
and test-time scaling re-ranking for relational SQL agents.
"""

from ask_the_database.verification.mve_adjudicator import (
    VerifierConditionMetrics,
    VerifierGateEvaluation,
    VerifierMVEReport,
    adjudicate_verifier_gates,
    get_empirical_verifier_metrics,
    load_verifier_mve_config,
)
from ask_the_database.verification.sql_verifier import (
    DEFAULT_SCORING_TOKENS,
    CandidateScore,
    SQLVerifier,
    VerifierCriterion,
    compute_continuous_expectation,
)

__all__ = [
    "DEFAULT_SCORING_TOKENS",
    "CandidateScore",
    "SQLVerifier",
    "VerifierConditionMetrics",
    "VerifierCriterion",
    "VerifierGateEvaluation",
    "VerifierMVEReport",
    "adjudicate_verifier_gates",
    "compute_continuous_expectation",
    "get_empirical_verifier_metrics",
    "load_verifier_mve_config",
]
