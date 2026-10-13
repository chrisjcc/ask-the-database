"""Multi-turn reward evaluation with step discount and self-correction bonuses."""

from typing import Any

from omegaconf import DictConfig, OmegaConf

from ask_the_database.rewards.rubric import SQLRewardRubric


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
        # Non-terminal intermediate tool step
        if not is_terminal:
            if (
                "Database Error" in tool_output
                or "Error:" in tool_output
                or "Execution error:" in tool_output
            ):
                self.had_intermediate_error = True
            return self.step_penalty

        # Terminal step (Final Answer or max turns reached)
        if final_query is None:
            return -1.0  # Exhausted budget without submitting final answer

        base_res = self.rubric.evaluate(final_query, gold_sql, db_path)
        terminal_reward = base_res.total_reward

        # Apply turn discount: penalize more turns (turn 1 has no discount, turns > 1 incur penalty)
        discounted_reward = terminal_reward + (self.step_penalty * max(0, turn - 1))

        # Apply recovery bonus if model fixed an earlier runtime error and achieved execution equivalence
        if base_res.is_execution_equivalent and self.had_intermediate_error:
            discounted_reward += self.recovery_bonus

        return max(-1.0, min(1.0, discounted_reward))
