"""Harness-Zero Agent-as-Harness Reviewer and K_SQL Reference Harness.

Implements step-level proposal interception at the response boundary guided by
procedural domain rules (K_SQL). Transfers multi-turn exploration and recovery
capabilities into model weights via supervised fine-tuning on reviewed trajectories.

Reference:
    Ye et al., September 2026. 'Harness Distillation via Agent-as-Harness'
    arXiv:2609.24974
"""

import contextlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import sqlglot
from omegaconf import DictConfig, OmegaConf
from sqlglot import exp

from ask_the_database.agent.prompt_formatter import parse_agent_action
from ask_the_database.environment.schema_manager import SchemaManager
from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment

logger = logging.getLogger(__name__)

ReviewActionType = Literal["PASS", "REPLACE"]
ValidToolName = Literal["inspect_schema", "sample_data", "execute_sql", "final_answer"]

VALID_TARGET_TOOLS: frozenset[str] = frozenset(
    {"inspect_schema", "sample_data", "execute_sql", "final_answer"}
)


def _normalize_query(query: str) -> str:
    """Normalize query text for exact comparison."""
    return re.sub(r"\s+", " ", query.strip().rstrip(";")).strip().lower()


@dataclass
class ReviewDecision:
    """Outcome of a step-level proposal review by the harnessing agent."""

    decision: ReviewActionType
    rule_triggered: str | None = None
    replacement_action: str | None = None
    replacement_payload: Any = None
    rationale: str = ""
    turn_index: int = 0

    @property
    def is_pass(self) -> bool:
        """True if the student's proposal was approved without modification."""
        return self.decision == "PASS"

    @property
    def is_replacement(self) -> bool:
        """True if the harnessing agent intercepted and replaced the proposal."""
        return self.decision == "REPLACE"

    @property
    def formatted_replacement(self) -> str:
        """Render replacement into valid ChatML action string within target action space h."""
        if not self.is_replacement or not self.replacement_action:
            return ""

        if self.replacement_action == "final_answer":
            return f"<final_answer>{str(self.replacement_payload).strip()}</final_answer>"

        if self.replacement_action == "inspect_schema":
            if isinstance(self.replacement_payload, (list, set, tuple)):
                payload_str = json.dumps([str(t).strip() for t in self.replacement_payload])
            elif isinstance(self.replacement_payload, str):
                cleaned = self.replacement_payload.strip()
                if cleaned.startswith("[") and cleaned.endswith("]"):
                    payload_str = cleaned
                else:
                    payload_str = json.dumps(
                        [t.strip().strip("'\"") for t in cleaned.split(",") if t.strip()]
                    )
            else:
                payload_str = json.dumps([])
            return f'<tool name="inspect_schema">{payload_str}</tool>'

        return (
            f'<tool name="{self.replacement_action}">{str(self.replacement_payload).strip()}</tool>'
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize review decision to dictionary."""
        return {
            "decision": self.decision,
            "rule_triggered": self.rule_triggered,
            "replacement_action": self.replacement_action,
            "replacement_payload": self.replacement_payload,
            "rationale": self.rationale,
            "turn_index": self.turn_index,
            "is_replacement": self.is_replacement,
            "formatted_replacement": self.formatted_replacement if self.is_replacement else "",
        }


@dataclass
class EpisodeReviewHistory:
    """Private review state tracking turn interventions, schema discoveries, and errors."""

    decisions: list[ReviewDecision] = field(default_factory=list)
    inspected_tables: set[str] = field(default_factory=set)
    sampled_tables: set[str] = field(default_factory=set)
    executed_queries: list[str] = field(default_factory=list)
    successful_queries: set[str] = field(default_factory=set)
    last_error: str | None = None
    last_action: str | None = None
    last_payload: Any = None

    @property
    def interventions_count(self) -> int:
        """Count of interventions (REPLACE decisions) in the current episode."""
        return sum(1 for d in self.decisions if d.is_replacement)

    def record_decision(self, decision: ReviewDecision) -> None:
        """Record a review decision in the episode trajectory."""
        self.decisions.append(decision)

    def record_step_result(self, action_name: str, action_payload: Any, observation: str) -> None:
        """Update environmental knowledge and error state after tool execution."""
        self.last_action = action_name
        self.last_payload = action_payload

        # Update error state
        if (
            "Database Error" in observation
            or "Error:" in observation
            or "Execution error:" in observation
        ):
            self.last_error = observation
        else:
            self.last_error = None

        # Update action-specific tracking
        if action_name == "inspect_schema":
            if isinstance(action_payload, list):
                for tbl in action_payload:
                    self.inspected_tables.add(str(tbl).strip().lower())
            elif isinstance(action_payload, str):
                cleaned = action_payload.strip("[]")
                for tbl in cleaned.split(","):
                    t_str = tbl.strip().strip("'\"").lower()
                    if t_str:
                        self.inspected_tables.add(t_str)
        elif action_name == "sample_data":
            tbl_clean = str(action_payload).strip().strip("'\"").lower()
            if tbl_clean:
                self.sampled_tables.add(tbl_clean)
        elif action_name == "execute_sql":
            query_str = str(action_payload).strip()
            self.executed_queries.append(query_str)
            if self.last_error is None:
                self.successful_queries.add(_normalize_query(query_str))

    def reset(self) -> None:
        """Clear all episode tracking state for a new task."""
        self.decisions.clear()
        self.inspected_tables.clear()
        self.sampled_tables.clear()
        self.executed_queries.clear()
        self.successful_queries.clear()
        self.last_error = None
        self.last_action = None
        self.last_payload = None


class HarnessReviewer:
    """Agent-as-Harness Reviewer implementing K_SQL procedural rules.

    Reviews student proposals before sandbox execution, enforcing schema inspection,
    row sampling, pre-finish dry-runs, and diagnostic recovery.
    """

    def __init__(
        self,
        config: DictConfig | dict[str, Any] | str | Path | None = None,
        teacher_llm_callable: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = self._load_config(config)
        self.teacher_llm = teacher_llm_callable
        self.syntax_validator = SQLSyntaxValidator(dialect="sqlite")
        self.history = EpisodeReviewHistory()

        rules_cfg = self.cfg.get("rules", {})
        self.rule_schema_inspection_enabled = bool(
            rules_cfg.get("require_schema_inspection", {}).get("enabled", True)
        )
        self.rule_row_sampling_enabled = bool(
            rules_cfg.get("require_row_sampling", {}).get("enabled", True)
        )
        self.rule_dry_run_enabled = bool(
            rules_cfg.get("enforce_dry_run_before_finish", {}).get("enabled", True)
        )
        self.rule_recovery_enabled = bool(
            rules_cfg.get("diagnostic_recovery", {}).get("enabled", True)
        )

        teacher_cfg = self.cfg.get("teacher", {})
        self.max_interventions = int(teacher_cfg.get("max_interventions_per_episode", 3))
        self.privacy_enforced = bool(teacher_cfg.get("privacy_enforcement", True))
        self.allow_fallback = bool(teacher_cfg.get("allow_rule_based_fallback", True))

    def _load_config(self, config: DictConfig | dict[str, Any] | str | Path | None) -> DictConfig:
        """Load configuration from DictConfig, dict, or YAML file path."""
        if config is None:
            default_path = Path("configs/distillation/k_sql_rules.yaml")
            if default_path.exists():
                loaded = OmegaConf.load(str(default_path))
                if isinstance(loaded, DictConfig):
                    return loaded
            res = OmegaConf.create(
                {
                    "rules": {
                        "require_schema_inspection": {"enabled": True},
                        "require_row_sampling": {"enabled": True},
                        "enforce_dry_run_before_finish": {"enabled": True},
                        "diagnostic_recovery": {"enabled": True},
                    },
                    "teacher": {
                        "max_interventions_per_episode": 3,
                        "privacy_enforcement": True,
                        "allow_rule_based_fallback": True,
                    },
                }
            )
            return res if isinstance(res, DictConfig) else OmegaConf.create({})
        elif isinstance(config, (str, Path)):
            loaded = OmegaConf.load(str(config))
            return loaded if isinstance(loaded, DictConfig) else OmegaConf.create({})
        elif isinstance(config, dict):
            res = OmegaConf.create(config)
            return res if isinstance(res, DictConfig) else OmegaConf.create({})
        return config

    def reset(self) -> None:
        """Reset reviewer state for a new episode."""
        self.history.reset()

    def record_observation(self, action_name: str, action_payload: Any, observation: str) -> None:
        """Record an executed action and its resulting environment observation."""
        self.history.record_step_result(action_name, action_payload, observation)

    def _extract_tables_from_query(self, query: str) -> list[str]:
        """Extract table names referenced in candidate SQL query using AST or regex fallback."""
        tables: set[str] = set()
        clean = query.strip().rstrip(";")
        if not clean:
            return []

        with contextlib.suppress(Exception):
            val_res = self.syntax_validator.validate(clean)
            if val_res.is_valid and val_res.referenced_tables:
                tables.update(val_res.referenced_tables)

        # Regex fallback for imperfect or dialect-specific syntax
        regex_matches = re.findall(
            r"(?:FROM|JOIN)\s+([`'\"\[]?[a-zA-Z0-9_]+[`'\"\]]?)",
            clean,
            re.IGNORECASE,
        )
        for match in regex_matches:
            tbl = match.strip("`'\"[]").lower()
            if tbl and tbl not in ("select", "where", "group", "order", "limit"):
                tables.add(tbl)

        return sorted(tables)

    def _extract_string_literals_in_where(self, query: str) -> list[str]:
        """Detect string literals in WHERE clause to identify categorical filter targets."""
        clean = query.strip().rstrip(";")
        if not clean:
            return []

        literals: list[str] = []
        with contextlib.suppress(Exception):
            ast = sqlglot.parse_one(clean, read="sqlite")
            where = ast.find(exp.Where)
            if where:
                for lit in where.find_all(exp.Literal):
                    if lit.is_string:
                        literals.append(str(lit.this))

        if not literals:
            # Fallback regex for WHERE col = 'val' or LIKE '%val%'
            where_match = re.search(
                r"WHERE\s+(.*?)(?:ORDER|GROUP|LIMIT|$)", clean, re.IGNORECASE | re.DOTALL
            )
            if where_match:
                where_clause = where_match.group(1)
                matches = re.findall(r"['\"]([^'\"]+)['\"]", where_clause)
                literals.extend(matches)

        return literals

    def _get_database_tables(self, db_path: str | None) -> list[str]:
        """Retrieve user tables from database sandbox."""
        if not db_path or not os.path.exists(db_path):
            return []
        with contextlib.suppress(Exception):
            schema_mgr = SchemaManager(db_path)
            return [t.lower() for t in schema_mgr.get_tables()]
        return []

    def review(
        self,
        proposal: str,
        db_path: str | None = None,
        available_tables: list[str] | None = None,
        turn_index: int = 1,
    ) -> ReviewDecision:
        """Review candidate action proposal against K_SQL rules.

        Args:
            proposal: Raw agent completion string (e.g., '<tool ...>' or '<final_answer>...').
            db_path: Optional path to SQLite database sandbox.
            available_tables: Optional explicit list of known tables in the database.
            turn_index: 1-indexed turn number.

        Returns:
            ReviewDecision: Outcome containing decision (PASS or REPLACE) and replacement action.
        """
        # 1. Parse proposed action
        action_name, action_payload = parse_agent_action(proposal)

        # 2. Intervention budget gate: if budget exhausted, pass proposal to avoid infinite loops
        if self.history.interventions_count >= self.max_interventions:
            decision = ReviewDecision(
                decision="PASS",
                rule_triggered=None,
                rationale="Intervention budget reached; passing candidate proposal.",
                turn_index=turn_index,
            )
            self.history.record_decision(decision)
            return decision

        # 3. Known database tables resolution
        known_tables: list[str] = []
        if available_tables:
            known_tables = [t.lower() for t in available_tables]
        elif db_path:
            known_tables = self._get_database_tables(db_path)

        # 4. Optional Teacher LLM interception
        if self.teacher_llm is not None:
            try:
                llm_decision = self._call_teacher_llm(
                    proposal=proposal,
                    action_name=action_name,
                    action_payload=action_payload,
                    known_tables=known_tables,
                    turn_index=turn_index,
                )
                if llm_decision is not None:
                    self.history.record_decision(llm_decision)
                    return llm_decision
            except Exception as e:
                logger.warning(f"Teacher LLM failed: {e}. Falling back to rule engine.")
                if not self.allow_fallback:
                    raise

        # 5. Deterministic K_SQL Rule Evaluation
        rule_decision = self._evaluate_rules(
            action_name=action_name,
            action_payload=action_payload,
            known_tables=known_tables,
            turn_index=turn_index,
        )
        self.history.record_decision(rule_decision)
        return rule_decision

    def _evaluate_rules(
        self,
        action_name: str,
        action_payload: Any,
        known_tables: list[str],
        turn_index: int,
    ) -> ReviewDecision:
        """Evaluate proposal against procedural rules in priority order."""
        # -------------------------------------------------------------
        # Rule 4: DiagnosticRecovery (Priority 110)
        # -------------------------------------------------------------
        if self.rule_recovery_enabled and self.history.last_error is not None:
            last_err = self.history.last_error

            # A. Check for blindly repeated query
            if action_name == "execute_sql":
                norm_cand = _normalize_query(str(action_payload))
                if self.history.executed_queries:
                    norm_prev = _normalize_query(self.history.executed_queries[-1])
                    if norm_cand == norm_prev:
                        # Repeated query after error! Intercept with schema inspection
                        cand_tables = self._extract_tables_from_query(str(action_payload))
                        tables_to_inspect = cand_tables or known_tables[:3] or ["tables"]
                        return ReviewDecision(
                            decision="REPLACE",
                            rule_triggered="diagnostic_recovery",
                            replacement_action="inspect_schema",
                            replacement_payload=tables_to_inspect,
                            rationale=(
                                f"Repeated identical failing query after error: {last_err[:100]}. "
                                "Replacing with inspect_schema to diagnose valid columns and table structures."
                            ),
                            turn_index=turn_index,
                        )

            # B. Check for "no such table" error
            table_err_match = re.search(
                r"no such table:\s*([a-zA-Z0-9_]+)", last_err, re.IGNORECASE
            )
            if table_err_match:
                missing_table = table_err_match.group(1).lower()
                cand_tables = self._extract_tables_from_query(str(action_payload))
                if missing_table in cand_tables or missing_table not in known_tables:
                    replace_tables = known_tables[:4] if known_tables else [missing_table]
                    return ReviewDecision(
                        decision="REPLACE",
                        rule_triggered="diagnostic_recovery",
                        replacement_action="inspect_schema",
                        replacement_payload=replace_tables,
                        rationale=(
                            f"Previous query failed with nonexistent table '{missing_table}'. "
                            "Replacing with inspect_schema on verified database tables."
                        ),
                        turn_index=turn_index,
                    )

            # C. Check for "no such column" error
            col_err_match = re.search(
                r"no such column:\s*([a-zA-Z0-9_.]+)", last_err, re.IGNORECASE
            )
            if col_err_match:
                missing_col = col_err_match.group(1)
                cand_tables = self._extract_tables_from_query(str(action_payload))
                target_tables = cand_tables or known_tables[:2] or ["tables"]
                return ReviewDecision(
                    decision="REPLACE",
                    rule_triggered="diagnostic_recovery",
                    replacement_action="inspect_schema",
                    replacement_payload=target_tables,
                    rationale=(
                        f"Previous query failed with invalid column '{missing_col}'. "
                        "Replacing with inspect_schema to inspect actual schema column definitions."
                    ),
                    turn_index=turn_index,
                )

        # -------------------------------------------------------------
        # Rule 1: RequireSchemaInspection (Priority 100)
        # -------------------------------------------------------------
        if self.rule_schema_inspection_enabled and action_name in ("execute_sql", "final_answer"):
            query_str = str(action_payload).strip()
            referenced_tables = self._extract_tables_from_query(query_str)

            # Identify tables referenced that have not been inspected
            uninspected_tables = [
                tbl for tbl in referenced_tables if tbl.lower() not in self.history.inspected_tables
            ]

            # If student has not inspected tables at all
            if not self.history.inspected_tables:
                tables_to_inspect = (
                    uninspected_tables
                    or referenced_tables
                    or (known_tables[:3] if known_tables else [])
                )
                if tables_to_inspect:
                    return ReviewDecision(
                        decision="REPLACE",
                        rule_triggered="require_schema_inspection",
                        replacement_action="inspect_schema",
                        replacement_payload=tables_to_inspect,
                        rationale=(
                            f"Candidate query references uninspected tables {tables_to_inspect}. "
                            "K_SQL procedural rules require schema inspection before query execution."
                        ),
                        turn_index=turn_index,
                    )
            elif uninspected_tables:
                # Some referenced tables were not inspected
                return ReviewDecision(
                    decision="REPLACE",
                    rule_triggered="require_schema_inspection",
                    replacement_action="inspect_schema",
                    replacement_payload=uninspected_tables,
                    rationale=(
                        f"Referenced tables {uninspected_tables} have not been inspected. "
                        "Enforcing inspect_schema before query execution."
                    ),
                    turn_index=turn_index,
                )

        # -------------------------------------------------------------
        # Rule 2: RequireRowSampling (Priority 80)
        # -------------------------------------------------------------
        if self.rule_row_sampling_enabled and action_name in ("execute_sql", "final_answer"):
            query_str = str(action_payload).strip()
            literals = self._extract_string_literals_in_where(query_str)
            if literals:
                cand_tables = self._extract_tables_from_query(query_str)
                # Find referenced table that has not been sampled
                unsampled_tables = [
                    t for t in cand_tables if t.lower() not in self.history.sampled_tables
                ]
                if unsampled_tables:
                    target_table = unsampled_tables[0]
                    return ReviewDecision(
                        decision="REPLACE",
                        rule_triggered="require_row_sampling",
                        replacement_action="sample_data",
                        replacement_payload=target_table,
                        rationale=(
                            f"Query filters on categorical string literal(s) {literals[:2]} in table '{target_table}' "
                            "without prior row sampling. Enforcing sample_data to verify formatting and casing."
                        ),
                        turn_index=turn_index,
                    )

        # -------------------------------------------------------------
        # Rule 3: EnforceDryRunBeforeFinish (Priority 90)
        # -------------------------------------------------------------
        if self.rule_dry_run_enabled and action_name == "final_answer":
            query_str = str(action_payload).strip()
            norm_q = _normalize_query(query_str)

            if norm_q not in self.history.successful_queries:
                return ReviewDecision(
                    decision="REPLACE",
                    rule_triggered="enforce_dry_run_before_finish",
                    replacement_action="execute_sql",
                    replacement_payload=query_str,
                    rationale=(
                        "Proposed final_answer query has not been tested in the sandbox. "
                        "Enforcing dry-run execution via execute_sql to verify syntax and execution."
                    ),
                    turn_index=turn_index,
                )

        # -------------------------------------------------------------
        # Default: Proposal is sound, PASS
        # -------------------------------------------------------------
        return ReviewDecision(
            decision="PASS",
            rule_triggered=None,
            rationale="Candidate action adheres to K_SQL procedural guidelines.",
            turn_index=turn_index,
        )

    def _call_teacher_llm(
        self,
        proposal: str,
        action_name: str,
        action_payload: Any,
        known_tables: list[str],
        turn_index: int,
    ) -> ReviewDecision | None:
        """Invoke teacher model for Agent-as-Harness review."""
        if self.teacher_llm is None:
            return None

        # Build grounded context for teacher (strictly excluding gold_sql)
        teacher_input = {
            "proposal": proposal,
            "parsed_action": action_name,
            "parsed_payload": str(action_payload),
            "known_tables": known_tables,
            "inspected_tables": list(self.history.inspected_tables),
            "sampled_tables": list(self.history.sampled_tables),
            "successful_queries": list(self.history.successful_queries),
            "last_error": self.history.last_error,
            "turn_index": turn_index,
        }

        response = self.teacher_llm(teacher_input)
        if not isinstance(response, dict):
            return None

        decision_str = str(response.get("decision", "PASS")).upper()
        if decision_str not in ("PASS", "REPLACE"):
            return None

        if decision_str == "PASS":
            return ReviewDecision(
                decision="PASS",
                rule_triggered=None,
                rationale=str(response.get("rationale", "Teacher approved proposal.")),
                turn_index=turn_index,
            )

        replacement_action = str(response.get("replacement_action", "")).strip()
        if replacement_action not in VALID_TARGET_TOOLS:
            logger.warning(
                f"Teacher proposed invalid tool '{replacement_action}' outside target action space h."
            )
            return None

        return ReviewDecision(
            decision="REPLACE",
            rule_triggered=response.get("rule_triggered"),
            replacement_action=replacement_action,
            replacement_payload=response.get("replacement_payload"),
            rationale=str(response.get("rationale", "Teacher replaced proposal.")),
            turn_index=turn_index,
        )

    def step_env(
        self,
        env: SkyRLSQLEnvironment,
        model_completion: str,
    ) -> tuple[str, float, bool, dict[str, Any], ReviewDecision]:
        """Intercept proposal, apply review decision, and advance environment.

        Args:
            env: SkyRLSQLEnvironment instance.
            model_completion: Raw completion emitted by the student agent.

        Returns:
            tuple: (observation, reward, is_done, info, review_decision)
        """
        turn_idx = env.current_turn + 1
        known_tables: list[str] = []
        if env.registry and env.registry.schema_manager:
            with contextlib.suppress(Exception):
                known_tables = [t.lower() for t in env.registry.schema_manager.get_tables()]

        # 1. Review proposal
        decision = self.review(
            proposal=model_completion,
            db_path=env.db_path,
            available_tables=known_tables,
            turn_index=turn_idx,
        )

        # 2. Determine actual action executed in the environment
        if decision.is_replacement:
            executed_completion = decision.formatted_replacement
        else:
            executed_completion = model_completion

        # 3. Execute in sandbox environment
        obs, reward, is_done, info = env.step(executed_completion)

        # 4. Record executed action outcome in reviewer history
        executed_action, executed_payload = parse_agent_action(executed_completion)
        tool_output = str(info.get("tool_output", obs))
        self.record_observation(executed_action, executed_payload, tool_output)

        # 5. Enrich info dict with review metadata (without leaking rationale to student chat)
        info["review_decision"] = decision.to_dict()
        info["was_replaced"] = decision.is_replacement
        info["rule_triggered"] = decision.rule_triggered

        return obs, reward, is_done, info, decision

    @staticmethod
    def strip_reviewer_metadata_from_chat(
        chat_messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Mask and remove reviewer rationale from trajectory messages.

        Ensures that only clean student-perspective actions enter the SFT training dataset.
        """
        clean_messages: list[dict[str, str]] = []
        for msg in chat_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Filter out any internal reviewer notes or unexpected roles
            if role in ("system", "user", "assistant", "tool"):
                clean_messages.append({"role": role, "content": content})

        return clean_messages


class ReviewedSkyRLEnvironment:
    """Gymnasium-compatible multi-turn environment wrapper with integrated HarnessReviewer."""

    def __init__(
        self,
        env: SkyRLSQLEnvironment,
        reviewer: HarnessReviewer,
    ) -> None:
        self.env = env
        self.reviewer = reviewer

    def reset(self, instance: dict[str, Any]) -> str:
        """Reset environment and reviewer state for a new episode."""
        self.reviewer.reset()
        return self.env.reset(instance)

    def step(self, model_completion: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Step environment with pre-execution interception."""
        obs, reward, is_done, info, _ = self.reviewer.step_env(self.env, model_completion)
        return obs, reward, is_done, info

    @property
    def history(self) -> list[dict[str, str]]:
        """Access student-visible conversational history."""
        return self.env.history

    @property
    def clean_chat_history(self) -> list[dict[str, str]]:
        """Access trajectory sanitized of reviewer thoughts."""
        return self.reviewer.strip_reviewer_metadata_from_chat(self.env.history)
