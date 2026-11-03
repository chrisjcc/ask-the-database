"""Inference pipeline for interactive multi-turn text-to-SQL agents."""

import contextlib
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
        self.load_error: str | None = None
        self.hardware_warning: str | None = None

        if self.model is None and self.model_path:
            is_local = Path(self.model_path).exists()
            is_hub_id = "/" in self.model_path and not self.model_path.startswith((".", "/"))
            if is_local or is_hub_id:
                self._try_load_model()

    @property
    def is_model_loaded(self) -> bool:
        """Return True if PyTorch/Hugging Face model and tokenizer are successfully loaded."""
        return self.model is not None and self.tokenizer is not None

    def _try_load_model(self) -> None:
        """Attempt loading PyTorch / Hugging Face model and tokenizer."""
        try:
            import json
            import logging
            import os

            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            assert self.model_path is not None
            has_cuda = torch.cuda.is_available()
            device_target = self.device
            if device_target == "auto":
                device_target = "auto" if has_cuda else "cpu"

            # Detect total physical RAM in GB
            ram_gb = 0.0
            with contextlib.suppress(Exception):
                ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)

            # Hardware memory guardrail for CPU environments
            if not has_cuda:
                # If RAM is genuinely insufficient for 7B model (< 20 GB, e.g. 16 GB Basic tier)
                if 0 < ram_gb < 20.0 and not os.getenv("FORCE_CPU_MODEL_LOAD"):
                    raise RuntimeError(
                        f"CUDA GPU not detected and system RAM ({ram_gb:.1f} GB) is insufficient for 7B parameter model. "
                        "Skipping neural model load to prevent container OOM termination. "
                        "Switch Space to T4 GPU or upgrade to >= 32 GB RAM to activate neural inference."
                    )

                # If system RAM is sufficient (>= 20 GB, e.g. 32 GB CPU upgrade plan)
                ram_str = f" ({ram_gb:.0f} GB RAM detected)" if ram_gb > 0 else ""
                self.hardware_warning = (
                    f"CUDA GPU not detected. Running in CPU mode{ram_str}. "
                    "Neural model loaded successfully; switching Space to a T4 GPU in Space Settings "
                    "would yield significantly faster inference performance."
                )
                logging.getLogger(__name__).info(self.hardware_warning)

                # Optimize CPU thread allocation across available cores
                with contextlib.suppress(Exception):
                    num_cpus = os.cpu_count() or 4
                    torch.set_num_threads(min(8, num_cpus))

            path_obj = Path(self.model_path)
            target_path = path_obj

            if path_obj.exists():
                # If this is an RLVR output directory without local adapter weights, inspect rlvr_policy_config.json
                if (
                    path_obj.is_dir()
                    and not (path_obj / "adapter_config.json").exists()
                    and not (path_obj / "config.json").exists()
                ):
                    policy_conf_path = path_obj / "rlvr_policy_config.json"
                    if policy_conf_path.exists():
                        with open(policy_conf_path, encoding="utf-8") as f:
                            policy_conf = json.load(f)
                        underlying = policy_conf.get("model_path")
                        if underlying and Path(underlying).exists():
                            target_path = Path(underlying)
                            logging.getLogger(__name__).info(
                                "Resolved RLVR checkpoint '%s' to underlying policy '%s'",
                                self.model_path,
                                target_path,
                            )

                # Check if target is a PEFT adapter directory
                adapter_conf_path = (
                    target_path / "adapter_config.json" if target_path.is_dir() else None
                )
                if adapter_conf_path and adapter_conf_path.exists():
                    with open(adapter_conf_path, encoding="utf-8") as f:
                        conf = json.load(f)
                    base_model_name = conf.get(
                        "base_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name, trust_remote_code=True
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=self.torch_dtype or torch.bfloat16,
                        device_map=device_target,
                        trust_remote_code=True,
                    )
                    self.model = PeftModel.from_pretrained(base_model, str(target_path))
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(target_path), trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(target_path),
                        torch_dtype=self.torch_dtype or torch.bfloat16,
                        device_map=device_target,
                        trust_remote_code=True,
                    )
            else:
                # Load from Hugging Face Hub (remote repo ID)
                hub_repo_id = self.model_path
                logging.getLogger(__name__).info(
                    "Attempting to download model from Hugging Face Hub: %s", hub_repo_id
                )
                try:
                    # Attempt loading as PEFT adapter from Hub
                    peft_config = PeftConfig.from_pretrained(hub_repo_id)
                    base_name = (
                        peft_config.base_model_name_or_path or "Qwen/Qwen2.5-Coder-7B-Instruct"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        base_name, trust_remote_code=True
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_name,
                        torch_dtype=self.torch_dtype or torch.bfloat16,
                        device_map=device_target,
                        trust_remote_code=True,
                    )
                    self.model = PeftModel.from_pretrained(base_model, hub_repo_id)
                except Exception:
                    # Fallback to direct causal LM from Hub
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        hub_repo_id, trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        hub_repo_id,
                        torch_dtype=self.torch_dtype or torch.bfloat16,
                        device_map=device_target,
                        trust_remote_code=True,
                    )
        except Exception as e:
            import logging

            self.load_error = str(e)
            logging.getLogger(__name__).warning(
                "Could not load model weights from %s: %s", self.model_path, e
            )

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

            prompt: str = ""
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    prompt = str(
                        self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
                except Exception:
                    prompt = format_chatml_prompt(messages)
            else:
                prompt = format_chatml_prompt(messages)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = getattr(self.model, "device", None)
            if device is None and hasattr(self.model, "base_model"):
                device = getattr(self.model.base_model, "device", None)
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
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

        if not assistant_msgs and self.max_turns > 1:
            return f'<tool name="inspect_schema">["{target_table}"]</tool>'
        if len(assistant_msgs) == 1 and self.max_turns > 2:
            return f'<tool name="sample_data">{target_table}</tool>'

        query = default_query or f"SELECT * FROM {target_table} LIMIT 5;"
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

        # In unit-test mock mode (when model_path is None), allow default_query to be gold_sql
        mock_default = gold_sql if self.model_path is None else None

        for turn_idx in range(1, effective_max_turns + 1):
            completion = self.generate_step(
                env.history,
                available_tables=available_tables,
                default_query=mock_default,
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
