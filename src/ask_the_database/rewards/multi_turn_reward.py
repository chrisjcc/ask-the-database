"""Multi-turn reward evaluation with step discount and self-correction bonuses."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ask_the_database.rewards.rubric import RewardOutput, SQLRewardRubric


@dataclass(frozen=True)
class MultiTurnRewardOutput:
    """Detailed decomposition of multi-turn reward calculation."""

    total_reward: float
    base_reward: float
    turn_discount: float
    recovery_bonus: float
    had_intermediate_error: bool
    is_execution_equivalent: bool
    rubric_output: RewardOutput | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class MultiTurnRewardEvaluator:
    """Evaluates intermediate tool steps and terminal multi-turn trajectories."""

    def __init__(self, config: DictConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self.cfg: DictConfig = OmegaConf.create({})
        elif isinstance(config, dict):
            self.cfg = OmegaConf.create(config)
        else:
            self.cfg = config

        self.rubric = SQLRewardRubric(self.cfg)
        self.step_penalty: float = float(self.cfg.get("step_penalty", -0.05))
        self.recovery_bonus: float = float(self.cfg.get("recovery_bonus", 0.3))
        self.had_intermediate_error: bool = False

    def reset(self) -> None:
        """Reset trajectory-specific state for a new episode."""
        self.had_intermediate_error = False

    def evaluate_step_detailed(
        self,
        action_name: str,
        tool_output: str,
        final_query: str | None,
        gold_sql: str,
        db_path: str,
        turn: int,
        is_terminal: bool,
    ) -> MultiTurnRewardOutput:
        """Calculate step reward with full diagnostic breakdown."""
        penalty_magnitude = abs(self.step_penalty)

        # 1. Non-terminal intermediate tool step
        if not is_terminal:
            has_error = (
                "Database Error" in tool_output
                or "Error:" in tool_output
                or "Execution error:" in tool_output
            )
            if has_error:
                self.had_intermediate_error = True
            step_reward = -penalty_magnitude
            return MultiTurnRewardOutput(
                total_reward=step_reward,
                base_reward=0.0,
                turn_discount=step_reward,
                recovery_bonus=0.0,
                had_intermediate_error=self.had_intermediate_error,
                is_execution_equivalent=False,
                diagnostics={"step_type": "intermediate", "action": action_name},
            )

        # 2. Terminal step without final query (e.g. max turns budget exhausted)
        if final_query is None or not final_query.strip():
            return MultiTurnRewardOutput(
                total_reward=-1.0,
                base_reward=-1.0,
                turn_discount=0.0,
                recovery_bonus=0.0,
                had_intermediate_error=self.had_intermediate_error,
                is_execution_equivalent=False,
                diagnostics={"error": "Exhausted turn budget without submitting final answer."},
            )

        # 3. Terminal step with final query
        base_res = self.rubric.evaluate(final_query, gold_sql, db_path)
        terminal_reward = base_res.total_reward

        # Turn discount: 1-turn incurs 0 discount; turns > 1 incur penalty_magnitude per additional turn
        turn_discount = penalty_magnitude * max(0, turn - 1)
        discounted_reward = terminal_reward - turn_discount

        # Recovery bonus: awarded if intermediate error occurred but final query is execution-equivalent
        recovery_bonus = 0.0
        if base_res.is_execution_equivalent and self.had_intermediate_error:
            recovery_bonus = self.recovery_bonus
            discounted_reward += recovery_bonus

        total_reward = max(-1.0, min(1.0, discounted_reward))

        return MultiTurnRewardOutput(
            total_reward=total_reward,
            base_reward=terminal_reward,
            turn_discount=turn_discount,
            recovery_bonus=recovery_bonus,
            had_intermediate_error=self.had_intermediate_error,
            is_execution_equivalent=base_res.is_execution_equivalent,
            rubric_output=base_res,
            diagnostics={
                "turn": turn,
                "action": action_name,
                "had_intermediate_error": self.had_intermediate_error,
                "recovery_applied": recovery_bonus > 0.0,
            },
        )

    def evaluate_step(
        self,
        action_name: str,
        tool_output: str,
        final_query: str | None,
        gold_sql: str,
        db_path: str,
        turn: int,
        is_terminal: bool,
    ) -> float:
        """Calculate step reward incorporating discount and self-correction bonus."""
        output = self.evaluate_step_detailed(
            action_name=action_name,
            tool_output=tool_output,
            final_query=final_query,
            gold_sql=gold_sql,
            db_path=db_path,
            turn=turn,
            is_terminal=is_terminal,
        )
        return output.total_reward
