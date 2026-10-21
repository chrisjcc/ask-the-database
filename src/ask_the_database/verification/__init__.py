"""Verification module for AskTheDatabase.

Implements continuous logit-expectation verification, decomposed criteria,
and test-time scaling re-ranking for relational SQL agents.
"""

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
    "VerifierCriterion",
    "compute_continuous_expectation",
]
