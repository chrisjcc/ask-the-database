"""Hierarchical verifiable reward rubric composite for SQL reinforcement learning."""

from dataclasses import dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.environment.test_suite_evaluator import TestSuiteEvaluator
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator
from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator


@dataclass(frozen=True)
class RewardOutput:
    """Consolidated reward output and telemetry payload."""

    total_reward: float
    is_valid_syntax: bool
    is_valid_execution: bool
    is_execution_equivalent: bool
    is_test_suite_equivalent: bool
    diagnostic_payload: dict[str, Any]


class SQLRewardRubric:
    """Framework-independent composite reward evaluator for SQL RLVR."""

    def __init__(self, config: DictConfig | dict[str, Any] | None = None) -> None:
        default_config = {
            "float_tolerance": 1e-4,
            "use_test_suite": True,
            "num_mutations": 4,
            "weights": {
                "syntax_penalty": -1.0,
                "exec_penalty": -0.5,
                "schema_bonus": 0.2,
                "base_execution_reward": 0.5,
                "test_suite_reward": 0.5,
            },
        }
        base_cfg = OmegaConf.create(default_config)
        if config is not None:
            user_cfg = OmegaConf.create(config) if isinstance(config, dict) else config
            self.cfg: DictConfig = cast(DictConfig, OmegaConf.merge(base_cfg, user_cfg))
        else:
            self.cfg = base_cfg

        self.syntax_validator = SQLSyntaxValidator()
        self.execution_evaluator = ExecutionEquivalenceEvaluator(
            float_tolerance=float(self.cfg.get("float_tolerance", 1e-4))
        )
        self.use_test_suite: bool = bool(self.cfg.get("use_test_suite", True))
        num_muts = int(self.cfg.get("num_mutations", 4))
        self.test_suite_evaluator = (
            TestSuiteEvaluator(num_mutations=num_muts) if self.use_test_suite else None
        )

        # Rubric Weights
        self.w_syntax_penalty: float = float(self.cfg.weights.syntax_penalty)
        self.w_exec_penalty: float = float(self.cfg.weights.exec_penalty)
        self.w_schema_bonus: float = float(self.cfg.weights.schema_bonus)
        self.w_base_exec: float = float(self.cfg.weights.base_execution_reward)
        self.w_ts_exec: float = float(self.cfg.weights.test_suite_reward)

    def evaluate(
        self,
        candidate_sql: str,
        gold_sql: str,
        db_path: str,
        requires_strict_order: bool = False,
    ) -> RewardOutput:
        """Calculate the total hierarchical reward and diagnostic telemetry."""
        # 1. Tier 1: Syntax Validation
        syntax_res = self.syntax_validator.validate(candidate_sql)
        if not syntax_res.is_valid:
            return RewardOutput(
                total_reward=self.w_syntax_penalty,
                is_valid_syntax=False,
                is_valid_execution=False,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "syntax_error",
                    "error_type": syntax_res.error_type,
                    "error_message": syntax_res.error_message,
                },
            )

        # 2. Tier 2: Execution Feasibility
        env = SQLiteEnvironment(db_path)
        gen_exec = env.execute(candidate_sql)
        gold_exec = env.execute(gold_sql)

        if not gen_exec.success:
            return RewardOutput(
                total_reward=self.w_exec_penalty,
                is_valid_syntax=True,
                is_valid_execution=False,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "execution_error",
                    "error_type": gen_exec.error_type,
                    "error_message": gen_exec.error_message,
                },
            )

        # 3. Tier 3: Execution Equivalence on Base Database
        exec_cmp = self.execution_evaluator.compare(gen_exec, gold_exec, requires_strict_order)
        if not exec_cmp.is_equivalent:
            # Bounded Invariant: query with wrong result can never receive net positive reward
            net_score = min(0.0, self.w_exec_penalty + self.w_schema_bonus)
            return RewardOutput(
                total_reward=net_score,
                is_valid_syntax=True,
                is_valid_execution=True,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "semantic_mismatch",
                    "reason": exec_cmp.error_reason,
                    "gold_rows": exec_cmp.gold_row_count,
                    "gen_rows": exec_cmp.gen_row_count,
                },
            )

        # 4. Tier 4: Test-Suite Verification across Mutated Clones
        if self.use_test_suite and self.test_suite_evaluator is not None:
            ts_res = self.test_suite_evaluator.evaluate_test_suite(
                db_path, candidate_sql, gold_sql, requires_strict_order
            )
            if not ts_res.passed_all_suites:
                # Passed base DB but failed mutated states: spurious false positive!
                return RewardOutput(
                    total_reward=self.w_base_exec,  # Partial reward (e.g. +0.5)
                    is_valid_syntax=True,
                    is_valid_execution=True,
                    is_execution_equivalent=True,
                    is_test_suite_equivalent=False,
                    diagnostic_payload={
                        "stage": "test_suite_failure",
                        "suites_passed": ts_res.suites_passed,
                        "failed_index": ts_res.failed_suite_index,
                        "failure_reason": ts_res.failure_reason,
                    },
                )

        # Passed all verification tiers: Full Reward (+1.0)
        return RewardOutput(
            total_reward=self.w_base_exec + self.w_ts_exec,
            is_valid_syntax=True,
            is_valid_execution=True,
            is_execution_equivalent=True,
            is_test_suite_equivalent=True,
            diagnostic_payload={
                "stage": "full_correctness",
                "gold_rows": exec_cmp.gold_row_count,
            },
        )
