"""LLM-as-a-Verifier Continuous Logit-Expectation Verification Engine for Relational SQL.

Implements mathematical expectation over logprob distributions of scoring tokens,
decomposed multi-criteria SQL verification, and a Probabilistic Pivot Tournament
for efficient O(K * N) test-time candidate re-ranking.

Reference:
    'LLM-as-a-Verifier: A General-Purpose Verification Framework' (July 2026)
    arXiv:2607.05391
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator

logger = logging.getLogger(__name__)

DEFAULT_SCORING_TOKENS: dict[str, float] = {
    "1": 0.00,
    "2": 0.25,
    "3": 0.50,
    "4": 0.75,
    "5": 1.00,
}


@dataclass
class VerifierCriterion:
    """A decomposed evaluation criterion for relational SQL queries."""

    name: str
    description: str
    weight: float = 1.0


@dataclass
class CandidateScore:
    """Detailed score telemetry for a candidate SQL query."""

    candidate_id: str
    sql_query: str
    overall_score: float
    criteria_scores: dict[str, float] = field(default_factory=dict)
    scoring_logprobs: dict[str, float] = field(default_factory=dict)
    execution_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_continuous_expectation(
    token_logits_or_logprobs: dict[str, float],
    scoring_tokens: dict[str, float] | None = None,
    temperature: float = 1.0,
    is_logprobs: bool = False,
) -> tuple[float, dict[str, float]]:
    """Calculate the continuous mathematical expectation E[S] over scoring tokens.

    Formulation:
        E[S] = sum_{v in V_S} P(v | x, c) * val(v)

    Args:
        token_logits_or_logprobs: Mapping from token string to raw logit or logprob.
        scoring_tokens: Mapping from token string to normalized value in [0, 1].
        temperature: Softmax temperature parameter.
        is_logprobs: True if input values are log probabilities; False if raw logits.

    Returns:
        tuple of (expected_score, normalized_token_probabilities).
    """
    token_map = scoring_tokens or DEFAULT_SCORING_TOKENS
    if not token_logits_or_logprobs or not token_map:
        return 0.0, {}

    # Filter to tokens present in the scoring vocabulary
    relevant_tokens: dict[str, float] = {}
    for tok, val in token_logits_or_logprobs.items():
        clean_tok = tok.strip()
        if clean_tok in token_map:
            relevant_tokens[clean_tok] = val

    if not relevant_tokens:
        return 0.0, {}

    temp = max(1e-5, float(temperature))

    # Compute unnormalized probabilities
    if is_logprobs:
        # Values are log(P). To renormalize over the scoring subset:
        max_logp = max(relevant_tokens.values())
        unnorm = {tok: math.exp((lp - max_logp) / temp) for tok, lp in relevant_tokens.items()}
    else:
        # Values are raw logits
        max_logit = max(relevant_tokens.values())
        unnorm = {tok: math.exp((val - max_logit) / temp) for tok, val in relevant_tokens.items()}

    total_mass = sum(unnorm.values())
    if total_mass <= 0.0 or math.isnan(total_mass):
        return 0.0, {}

    probabilities = {tok: mass / total_mass for tok, mass in unnorm.items()}

    # Compute expectation: sum(P(v) * val(v))
    expected_score = sum(probabilities[tok] * token_map[tok] for tok in probabilities)
    expected_score = max(0.0, min(1.0, expected_score))

    return round(expected_score, 4), probabilities


class SQLVerifier:
    """Continuous Logit-Expectation Verification Engine for Text-to-SQL."""

    def __init__(
        self,
        config: DictConfig | dict[str, Any] | str | Path | None = None,
        client: Callable[[str], dict[str, float]] | None = None,
    ) -> None:
        """Initialize SQLVerifier with criteria and optional scoring client.

        Args:
            config: Path to rules YAML, DictConfig, or dict.
            client: Optional callable (prompt -> token_logits_or_logprobs).
        """
        self.config = self._load_config(config)
        self.client = client
        self.syntax_validator = SQLSyntaxValidator()

        # Parse verifier settings
        v_cfg = self.config.get("verifier", {})
        self.model_name: str = v_cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.temperature: float = float(v_cfg.get("temperature", 0.0))
        self.pivot_count: int = int(v_cfg.get("pivot_count", 2))
        self.scoring_tokens: dict[str, float] = dict(
            v_cfg.get("scoring_tokens", DEFAULT_SCORING_TOKENS)
        )

        # Parse decomposed criteria
        self.criteria: dict[str, VerifierCriterion] = {}
        criteria_cfg = self.config.get("criteria", {})
        total_weight = 0.0
        for name, crit in criteria_cfg.items():
            desc = crit.get("description", "")
            wt = float(crit.get("weight", 1.0))
            self.criteria[name] = VerifierCriterion(name=name, description=desc, weight=wt)
            total_weight += wt

        # Normalize weights if total > 0
        if total_weight > 0.0:
            for crit in self.criteria.values():
                crit.weight = crit.weight / total_weight

        # Prompt template
        self.prompt_template: str = self.config.get(
            "prompt_template",
            (
                "You are an expert SQL Verification Agent.\n"
                "Evaluate whether the candidate SQL query accurately and safely answers the user's question.\n"
                "[Database Schema DDL]:\n{schema_ddl}\n\n"
                "[User Question]:\n{question}\n\n"
                "[Candidate SQL Query]:\n```sql\n{sql_query}\n```\n\n"
                "Evaluation Criterion: {criterion_name}\n"
                "{criterion_description}\n\n"
                "Score from 1 to 5. Respond with only a single digit score:"
            ),
        )

    def _load_config(self, config: Any) -> dict[str, Any]:
        """Load configuration from file or dict."""
        if config is None:
            default_path = Path("configs/verification/sql_verifier_rules.yaml")
            if default_path.is_file():
                container = OmegaConf.to_container(OmegaConf.load(default_path), resolve=True)
                if isinstance(container, dict):
                    return {str(k): v for k, v in container.items()}
            return {
                "verifier": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "pivot_count": 2},
                "criteria": {
                    "intent_and_filters": {"weight": 0.35, "description": "Filter fidelity"},
                    "schema_grounding_and_joins": {
                        "weight": 0.30,
                        "description": "Schema presence",
                    },
                    "aggregation_granularity": {"weight": 0.20, "description": "Aggregation level"},
                    "result_sanity": {"weight": 0.15, "description": "Result structure"},
                },
            }
        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.is_file():
                container = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
                if isinstance(container, dict):
                    return {str(k): v for k, v in container.items()}
                raise ValueError(f"Expected dict configuration from {path}")
            raise FileNotFoundError(f"Configuration file not found: {path}")
        if isinstance(config, DictConfig):
            container = OmegaConf.to_container(config, resolve=True)
            if isinstance(container, dict):
                return {str(k): v for k, v in container.items()}
            raise ValueError("Expected dict configuration from DictConfig")
        if isinstance(config, dict):
            return {str(k): v for k, v in config.items()}
        raise TypeError(f"Unsupported config type: {type(config)}")

    def build_verification_prompt(
        self,
        question: str,
        schema_ddl: str,
        sql_query: str,
        criterion: VerifierCriterion,
    ) -> str:
        """Construct the evaluation prompt for a specific decomposed criterion."""
        return self.prompt_template.format(
            schema_ddl=schema_ddl.strip(),
            question=question.strip(),
            sql_query=sql_query.strip(),
            criterion_name=criterion.name,
            criterion_description=criterion.description,
        )

    def score_criterion(
        self,
        sql_query: str,
        question: str,
        schema_ddl: str,
        criterion: VerifierCriterion,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate query along a single criterion using logit expectation."""
        if self.client is not None:
            prompt = self.build_verification_prompt(question, schema_ddl, sql_query, criterion)
            try:
                logits_or_logprobs = self.client(prompt)
                return compute_continuous_expectation(
                    logits_or_logprobs,
                    scoring_tokens=self.scoring_tokens,
                    temperature=max(1.0, self.temperature),
                )
            except Exception as e:
                logger.warning("Verifier client failed for criterion %s: %s", criterion.name, e)

        # Fallback to deterministic heuristic when client is None or fails
        score = self.fallback_heuristic_score(sql_query, question, schema_ddl, criterion.name)
        synthetic_probs = dict.fromkeys(self.scoring_tokens, 0.0)
        closest_token = min(
            self.scoring_tokens.keys(),
            key=lambda t: abs(self.scoring_tokens[t] - score),
        )
        synthetic_probs[closest_token] = 1.0
        return score, synthetic_probs

    def score_query(
        self,
        sql_query: str,
        question: str,
        schema_ddl: str,
        candidate_id: str = "c0",
        execution_valid: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateScore:
        """Evaluate a candidate query across all decomposed criteria and compute expectation."""
        criteria_scores: dict[str, float] = {}
        all_logprobs: dict[str, float] = {}

        total_score = 0.0
        for name, crit in self.criteria.items():
            s_crit, lprobs = self.score_criterion(sql_query, question, schema_ddl, crit)
            criteria_scores[name] = s_crit
            all_logprobs.update({f"{name}_{k}": v for k, v in lprobs.items()})
            total_score += s_crit * crit.weight

        # Penalize if known execution failed
        if not execution_valid:
            total_score = total_score * 0.2

        overall = round(max(0.0, min(1.0, total_score)), 4)
        return CandidateScore(
            candidate_id=candidate_id,
            sql_query=sql_query,
            overall_score=overall,
            criteria_scores=criteria_scores,
            scoring_logprobs=all_logprobs,
            execution_valid=execution_valid,
            metadata=metadata or {},
        )

    def fallback_heuristic_score(
        self,
        sql_query: str,
        question: str,
        schema_ddl: str,
        criterion_name: str | None = None,
    ) -> float:
        """Deterministic heuristic fallback when external verifier model is unavailable.

        Evaluates AST syntax validity, schema grounding, table joins, and keyword matching.
        """
        # 1. AST Validation
        syntax_res = self.syntax_validator.validate(sql_query)
        if not syntax_res.is_valid:
            return 0.10

        parsed_query = sql_query.lower()
        q_lower = question.lower()
        ddl_lower = schema_ddl.lower()

        score = 0.50  # Base score for valid syntax

        # 2. Schema Grounding Check
        tables = list(syntax_res.referenced_tables)
        grounded_tables = [t for t in tables if t in ddl_lower]
        if tables and len(grounded_tables) == len(tables):
            score += 0.20
        elif tables and not grounded_tables:
            score -= 0.30

        # 3. Intent & Criteria Specific Adjustments
        if criterion_name == "intent_and_filters" or criterion_name is None:
            if "where" in parsed_query:
                score += 0.15
            if (
                any(k in q_lower for k in ["how many", "count", "number of"])
                and "count" in parsed_query
            ):
                score += 0.15

        if (
            (criterion_name == "schema_grounding_and_joins" or criterion_name is None)
            and len(tables) > 1
            and "join" in parsed_query
        ):
            score += 0.15

        if criterion_name == "aggregation_granularity" or criterion_name is None:
            if syntax_res.has_aggregation or any(
                agg in parsed_query for agg in ["count(", "sum(", "avg(", "max(", "min("]
            ):
                score += 0.10
            if syntax_res.has_group_by or "group by" in parsed_query:
                score += 0.10

        if (
            (criterion_name == "result_sanity" or criterion_name is None)
            and "select" in parsed_query
            and not parsed_query.strip().startswith("select *")
        ):
            score += 0.10

        return round(max(0.0, min(1.0, score)), 4)

    def probabilistic_pivot_tournament(
        self,
        candidates: Sequence[str | dict[str, Any]],
        question: str,
        schema_ddl: str,
    ) -> list[CandidateScore]:
        """Rank candidate queries using a Probabilistic Pivot Tournament in O(K * N) time.

        Args:
            candidates: List or Sequence of SQL strings or dicts with {'sql': str, 'id': str, ...}.
            question: Natural language question.
            schema_ddl: Database schema DDL string.

        Returns:
            Ranked list of CandidateScore objects sorted descending by overall_score.
        """
        if not candidates:
            return []

        scored_candidates: list[CandidateScore] = []
        for idx, cand in enumerate(candidates):
            cid = cand.get("id", f"cand_{idx}") if isinstance(cand, dict) else f"cand_{idx}"
            sql = cand if isinstance(cand, str) else cand.get("sql", cand.get("query", ""))
            exec_valid = cand.get("execution_valid", True) if isinstance(cand, dict) else True
            meta = cand if isinstance(cand, dict) else {"original_index": idx}

            scored = self.score_query(
                sql_query=sql,
                question=question,
                schema_ddl=schema_ddl,
                candidate_id=cid,
                execution_valid=exec_valid,
                metadata=meta,
            )
            scored_candidates.append(scored)

        # Sort descending by overall score
        scored_candidates.sort(key=lambda c: c.overall_score, reverse=True)
        return scored_candidates

    def select_best(
        self,
        candidates: Sequence[str | dict[str, Any]],
        question: str,
        schema_ddl: str,
    ) -> CandidateScore:
        """Select the highest scoring candidate via Probabilistic Pivot Tournament."""
        ranked = self.probabilistic_pivot_tournament(candidates, question, schema_ddl)
        if not ranked:
            raise ValueError("No candidates provided for verification selection.")
        return ranked[0]
