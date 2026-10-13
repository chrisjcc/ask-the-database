"""SkyRL-Gym multi-turn database interaction environment adapter."""

from typing import Any

from omegaconf import DictConfig, OmegaConf

from ask_the_database.agent.interaction_state import InteractionState
from ask_the_database.agent.prompt_formatter import parse_agent_action
from ask_the_database.agent.tool_registry import DatabaseToolRegistry
from ask_the_database.rewards.multi_turn_reward import MultiTurnRewardEvaluator


class SkyRLSQLEnvironment:
    """Gymnasium-compatible multi-turn SQL environment for SkyRL-Agent."""

    def __init__(
        self,
        config: DictConfig | dict[str, Any] | None = None,
        reward_evaluator: MultiTurnRewardEvaluator | None = None,
    ) -> None:
        if config is None:
            self.cfg: DictConfig = OmegaConf.create({})
        elif isinstance(config, dict):
            self.cfg = OmegaConf.create(config)
        else:
            self.cfg = config

        self.max_turns: int = int(self.cfg.get("max_turns", 5))
        self.reward_evaluator: MultiTurnRewardEvaluator = (
            reward_evaluator if reward_evaluator is not None else MultiTurnRewardEvaluator(self.cfg)
        )
        self.current_turn: int = 0
        self.history: list[dict[str, str]] = []
        self.registry: DatabaseToolRegistry | None = None
        self.gold_sql: str = ""
        self.db_path: str = ""
        self.db_id: str = ""
        self.question: str = ""
        self.state: InteractionState | None = None

    def reset(self, instance: dict[str, Any]) -> str:
        """Reset environment with a new benchmark question and database."""
        self.current_turn = 0
        self.gold_sql = str(instance.get("gold_sql", ""))
        self.db_path = str(instance.get("db_path", ""))
        self.db_id = str(instance.get("db_id", "default_db"))
        self.question = str(instance.get("question", ""))

        self.registry = DatabaseToolRegistry(self.db_path)
        self.reward_evaluator.reset()

        self.state = InteractionState(
            question=self.question,
            db_id=self.db_id,
            db_path=self.db_path,
            gold_sql=self.gold_sql,
            max_turns=self.max_turns,
        )
        self.history = self.state.to_chat_history()
        return self._format_observation()

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Extract tool call or final answer from LLM completion."""
        return parse_agent_action(text)

    def step(self, model_completion: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Advance the environment by executing the model's action."""
        self.current_turn += 1
        action_name, action_payload = self._parse_action(model_completion)

        assert self.registry is not None, "Environment must be reset() before calling step()."
        tool_res = self.registry.execute_action(action_name, action_payload)

        is_done = tool_res.is_terminal or (self.current_turn >= self.max_turns)

        # Compute multi-turn reward
        reward = self.reward_evaluator.evaluate_step(
            action_name=action_name,
            tool_output=tool_res.output,
            final_query=tool_res.final_query,
            gold_sql=self.gold_sql,
            db_path=self.db_path,
            turn=self.current_turn,
            is_terminal=is_done,
        )

        if self.state is not None:
            self.state.add_turn(
                action=action_name,
                action_input=action_payload,
                observation=tool_res.output,
                is_terminal=tool_res.is_terminal,
            )
            self.history = self.state.to_chat_history()

        info: dict[str, Any] = {
            "turn": self.current_turn,
            "action": action_name,
            "action_payload": action_payload,
            "tool_output": tool_res.output,
            "is_terminal": tool_res.is_terminal,
            "max_turns_exceeded": self.current_turn >= self.max_turns and not tool_res.is_terminal,
            "had_intermediate_error": self.reward_evaluator.had_intermediate_error,
        }

        return self._format_observation(), reward, is_done, info

    def _format_observation(self) -> str:
        """Render conversational message history into string observation."""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)
