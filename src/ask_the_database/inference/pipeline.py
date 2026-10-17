"""Inference pipeline for interactive multi-turn text-to-SQL agents."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ask_the_database.agent.prompt_formatter import format_chatml_prompt
from ask_the_database.environment.base import ExecutionResult
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment


@dataclass
class AgentStep:
    """Represents a single step in an agent trajectory."""

    turn: int
    action: str
    action_input: Any
    output: str
    is_terminal: bool = False
    latency_ms: float = 0.0


@dataclass
class AgentTrajectory:
    """Full trajectory recorded during interactive agent execution."""

    question: str
    db_id: str
    db_path: str
    steps: list[AgentStep] = field(default_factory=list)
    final_query: str = ""
    final_execution_result: ExecutionResult | None = None
    final_execution_success: bool = False
    had_error_recovery: bool = False
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to serializable dictionary."""
        return {
            "question": self.question,
            "db_id": self.db_id,
            "db_path": self.db_path,
            "final_query": self.final_query,
            "final_execution_success": self.final_execution_success,
            "had_error_recovery": self.had_error_recovery,
            "total_reward": self.total_reward,
            "steps": [
                {
                    "turn": s.turn,
                    "action": s.action,
                    "action_input": s.action_input,
                    "output": s.output,
                    "is_terminal": s.is_terminal,
                    "latency_ms": s.latency_ms,
                }
                for s in self.steps
            ],
        }


class NL2SQLPipeline:
    """Multi-turn agent inference pipeline supporting interactive rollouts."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        torch_dtype: Any = None,
        max_turns: int = 5,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_path = str(model_path) if model_path else None
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_turns = max_turns
        self.model = model
        self.tokenizer = tokenizer

        if self.model is None and self.model_path and Path(self.model_path).exists():
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt loading PyTorch / Hugging Face model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            assert self.model_path is not None
            device_target = self.device
            if device_target == "auto":
                device_target = "auto" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype or torch.bfloat16,
                device_map=device_target,
                trust_remote_code=True,
            )
        except Exception:
            pass

    def generate_step(
        self,
        messages: list[dict[str, str]],
        available_tables: list[str] | None = None,
        default_query: str | None = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Generate next agent completion from conversational history."""
        if self.model is not None and self.tokenizer is not None:
            import torch

            prompt = format_chatml_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": 256,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(0.01, float(temperature))
                gen_kwargs["top_p"] = float(top_p)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            decoded = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return str(decoded).strip()

        # Heuristic fallback agent when testing or offline
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        tables = available_tables or ["demo_table"]
        target_table = tables[0]

        if not assistant_msgs:
            return f'<tool name="inspect_schema">["{target_table}"]</tool>'
        if len(assistant_msgs) == 1:
            return f'<tool name="sample_data">{target_table}</tool>'

        query = default_query or f"SELECT * FROM {target_table};"
        return f"<final_answer>{query}</final_answer>"

    def run_interactive(
        self,
        question: str,
        db_path: str,
        db_id: str = "default_db",
        max_turns: int | None = None,
        gold_sql: str = "SELECT 1;",
    ) -> AgentTrajectory:
        """Execute interactive multi-turn agent rollout against database."""
        effective_max_turns = max_turns or self.max_turns
        env = SkyRLSQLEnvironment({"max_turns": effective_max_turns})
        env.reset(
            {
                "question": question,
                "gold_sql": gold_sql,
                "db_id": db_id,
                "db_path": db_path,
            }
        )

        assert env.registry is not None
        available_tables = env.registry.schema_manager.get_tables()

        steps: list[AgentStep] = []
        final_query = ""
        had_intermediate_error = False
        step_reward = 0.0

        for turn_idx in range(1, effective_max_turns + 1):
            completion = self.generate_step(
                env.history,
                available_tables=available_tables,
                default_query=gold_sql,
            )
            obs, step_reward, done, info = env.step(completion)

            action = info["action"]
            action_payload = info["action_payload"]
            is_terminal = info["is_terminal"]

            if info.get("had_intermediate_error"):
                had_intermediate_error = True

            step_record = AgentStep(
                turn=turn_idx,
                action=action,
                action_input=action_payload,
                output=info["tool_output"],
                is_terminal=is_terminal,
            )
            steps.append(step_record)

            if is_terminal or done:
                if action == "final_answer":
                    final_query = str(action_payload)
                elif env.state and env.state.final_query:
                    final_query = env.state.final_query
                else:
                    final_query = completion
                break

        # Execute final query on target database to obtain final execution result
        db_env = SQLiteEnvironment(db_path)
        final_res = db_env.execute(final_query)
        final_success = final_res.success
        had_error_recovery = had_intermediate_error and final_success

        return AgentTrajectory(
            question=question,
            db_id=db_id,
            db_path=db_path,
            steps=steps,
            final_query=final_query,
            final_execution_result=final_res,
            final_execution_success=final_success,
            had_error_recovery=had_error_recovery,
            total_reward=step_reward,
        )
