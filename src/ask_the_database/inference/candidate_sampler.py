"""Multi-candidate generation and SQLite sandbox pre-filtering for LLM-as-a-Verifier.

Implements batched sampling of N candidate SQL queries at temperature T=0.7
and deterministic pre-filtering via isolated read-only SQLite sandbox execution.
"""

import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ask_the_database.environment.base import ExecutionResult
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.training.dataset import extract_sql_from_completion

logger = logging.getLogger(__name__)


def clean_candidate_sql(text: Any) -> str:
    """Extract and sanitize candidate SQL from raw generation output.

    Handles XML tags (<final_answer>), markdown blocks (```sql ... ```),
    and raw query strings.

    Args:
        text: Raw generation string or message sequence.

    Returns:
        Clean, stripped SQL query string.
    """
    if isinstance(text, list):
        if text and isinstance(text[-1], dict) and "content" in text[-1]:
            raw_str = str(text[-1]["content"])
        else:
            raw_str = str(text)
    else:
        raw_str = str(text)

    # 1. Check for XML tag <final_answer>
    xml_match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>", raw_str, re.DOTALL | re.IGNORECASE
    )
    if xml_match:
        return xml_match.group(1).strip().rstrip(";")

    # 2. Check for markdown code blocks via standard dataset utility
    extracted = extract_sql_from_completion(raw_str)
    return extracted.strip().rstrip(";")


@dataclass
class CandidateQuery:
    """Represents a single generated candidate SQL query and its execution diagnostics."""

    candidate_id: str
    sql: str
    raw_completion: str
    execution_result: ExecutionResult | None = None
    is_valid_syntax: bool = True
    is_executable: bool = True
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "sql": self.sql,
            "raw_completion": self.raw_completion,
            "is_valid_syntax": self.is_valid_syntax,
            "is_executable": self.is_executable,
            "execution_time_ms": self.execution_time_ms,
            "execution_success": (
                self.execution_result.success if self.execution_result is not None else None
            ),
            "metadata": self.metadata,
        }


class CandidateSampler:
    """Samples parallel SQL candidate completions and pre-filters them via SQLite sandbox."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any = None,
        tokenizer: Any = None,
        generator_fn: Callable[[str, int, float], list[str]] | None = None,
        num_candidates: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        timeout_seconds: float = 2.0,
        device: str = "auto",
        torch_dtype: Any = None,
    ) -> None:
        """Initialize candidate sampler.

        Args:
            model_path: Optional path to Hugging Face or local model checkpoint.
            model: Optional pre-loaded PyTorch/HF model instance.
            tokenizer: Optional pre-loaded tokenizer instance.
            generator_fn: Optional custom generator function (prompt, n, temp) -> list[raw_completions].
            num_candidates: Default number of candidate completions N (default: 5).
            temperature: Sampling temperature T (default: 0.7).
            top_p: Nucleus sampling probability cutoff (default: 0.95).
            max_new_tokens: Maximum tokens to generate per candidate.
            timeout_seconds: Sandbox execution timeout limit in seconds (default: 2.0s).
            device: PyTorch device string or 'auto'.
            torch_dtype: PyTorch tensor dtype.
        """
        self.model_path = str(model_path) if model_path is not None else None
        self.model = model
        self.tokenizer = tokenizer
        self.generator_fn = generator_fn
        self.num_candidates = num_candidates
        self.temperature = max(0.01, float(temperature))
        self.top_p = float(top_p)
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.device = device
        self.torch_dtype = torch_dtype

        if self.model is None and self.model_path and Path(self.model_path).exists():
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt loading PyTorch / Hugging Face model and tokenizer."""
        try:
            import json

            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            assert self.model_path is not None
            device_target = self.device
            if device_target == "auto":
                device_target = "cuda" if torch.cuda.is_available() else "cpu"

            path_obj = Path(self.model_path)
            target_path = path_obj

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
                        logger.info(
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
                    device_map=device_target if device_target != "cpu" else None,
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
                    device_map=device_target if device_target != "cpu" else None,
                    trust_remote_code=True,
                )
            if device_target == "cpu" and hasattr(self.model, "to"):
                self.model.to("cpu")
            logger.info("Successfully loaded model from %s on %s", self.model_path, device_target)
        except Exception as e:
            logger.warning("Could not load model weights from %s: %s", self.model_path, e)

    def build_generation_prompt(self, question: str, schema_ddl: str) -> str:
        """Format standard ChatML generation prompt for text-to-SQL."""
        system_msg = (
            "You are an expert SQL assistant. Given a database schema DDL and natural language question, "
            "write the exact, correct SQLite query that answers the question. "
            "Output your query enclosed inside ```sql ... ``` code block."
        )
        user_msg = f"[Database Schema DDL]:\n{schema_ddl}\n\n[User Question]:\n{question}"
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_candidates(
        self,
        question: str,
        schema_ddl: str,
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[CandidateQuery]:
        """Generate N candidate SQL completions.

        Args:
            question: Natural language question.
            schema_ddl: Database schema DDL string.
            n: Number of candidates to sample (defaults to self.num_candidates).
            temperature: Sampling temperature (defaults to self.temperature).

        Returns:
            List of N CandidateQuery instances.
        """
        num_cands = n if n is not None and n > 0 else self.num_candidates
        temp = temperature if temperature is not None else self.temperature
        prompt = self.build_generation_prompt(question, schema_ddl)

        # 1. Custom generator callable
        if self.generator_fn is not None:
            raw_completions = self.generator_fn(prompt, num_cands, temp)
            candidates: list[CandidateQuery] = []
            for idx, raw in enumerate(raw_completions):
                sql = clean_candidate_sql(raw)
                candidates.append(
                    CandidateQuery(
                        candidate_id=f"cand_{idx}",
                        sql=sql,
                        raw_completion=raw,
                        metadata={"source": "generator_fn", "index": idx},
                    )
                )
            return candidates

        # 2. PyTorch / Transformers model generation
        if self.model is not None and self.tokenizer is not None:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=max(0.01, float(temp)),
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=num_cands,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )

            input_len = inputs["input_ids"].shape[1]
            candidates = []
            for idx, seq in enumerate(outputs):
                decoded = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                sql = clean_candidate_sql(decoded)
                candidates.append(
                    CandidateQuery(
                        candidate_id=f"cand_{idx}",
                        sql=sql,
                        raw_completion=decoded,
                        metadata={"source": "transformers", "index": idx},
                    )
                )
            return candidates

        # 3. Fallback heuristic candidate generation for offline / testing
        return self._generate_fallback_candidates(question, schema_ddl, num_cands)

    def _generate_fallback_candidates(
        self,
        question: str,
        schema_ddl: str,
        num_cands: int,
    ) -> list[CandidateQuery]:
        """Generate synthetic heuristic candidates for offline testing."""
        # Find first table in DDL
        tables = re.findall(r"create\s+table\s+([a-zA-Z0-9_]+)", schema_ddl, re.IGNORECASE)
        table_name = tables[0] if tables else "data_table"

        # Find columns for that table
        columns = re.findall(
            r"([a-zA-Z0-9_]+)\s+(?:INT|TEXT|FLOAT|INTEGER|REAL|VARCHAR)", schema_ddl, re.IGNORECASE
        )
        col = columns[0] if columns else "id"

        base_candidates = [
            f"SELECT * FROM {table_name}",
            f"SELECT {col} FROM {table_name} LIMIT 10",
            f"SELECT COUNT(*) FROM {table_name}",
            f"SELECT DISTINCT {col} FROM {table_name}",
            f"SELECT {col} FROM {table_name} ORDER BY {col} DESC",
        ]

        candidates: list[CandidateQuery] = []
        for idx in range(num_cands):
            sql = base_candidates[idx % len(base_candidates)]
            raw = f"```sql\n{sql}\n```"
            candidates.append(
                CandidateQuery(
                    candidate_id=f"cand_{idx}",
                    sql=sql,
                    raw_completion=raw,
                    metadata={"source": "fallback_heuristic", "index": idx},
                )
            )
        return candidates

    def filter_candidates_in_sandbox(
        self,
        candidates: Sequence[CandidateQuery],
        db_path: str,
    ) -> tuple[list[CandidateQuery], list[CandidateQuery]]:
        """Execute candidates in SQLite sandbox and discard broken/timeout queries.

        Args:
            candidates: Sequence of generated CandidateQuery objects.
            db_path: Path to SQLite database file.

        Returns:
            Tuple of (surviving_candidates, discarded_candidates).
        """
        if not candidates:
            return [], []

        if not os.path.exists(db_path):
            logger.warning("Database path %s does not exist; skipping sandbox execution.", db_path)
            return list(candidates), []

        env = SQLiteEnvironment(db_path=db_path, timeout_seconds=self.timeout_seconds)
        surviving: list[CandidateQuery] = []
        discarded: list[CandidateQuery] = []

        for cand in candidates:
            start_t = time.perf_counter()
            exec_res = env.execute(cand.sql)
            latency_ms = (time.perf_counter() - start_t) * 1000

            cand.execution_result = exec_res
            cand.execution_time_ms = latency_ms

            if exec_res.error_type in ("SYNTAX", "SECURITY"):
                cand.is_valid_syntax = False
                cand.is_executable = False
                discarded.append(cand)
            elif (
                exec_res.error_type == "TIMEOUT"
                or exec_res.execution_time_ms > self.timeout_seconds * 1000
            ):
                cand.is_executable = False
                discarded.append(cand)
            elif not exec_res.success:
                # Operational / runtime execution error (e.g. table or column not found)
                cand.is_executable = False
                discarded.append(cand)
            else:
                # Query executed cleanly within bounds
                cand.is_valid_syntax = True
                cand.is_executable = True
                surviving.append(cand)

        # Safety Fallback: If all candidates fail sandbox execution, retain candidate 0 marked invalid
        if not surviving and candidates:
            logger.warning(
                "All %d candidates failed sandbox execution. Preserving candidate 0 as fallback.",
                len(candidates),
            )
            fallback = candidates[0]
            fallback.is_executable = False
            surviving.append(fallback)

        return surviving, discarded

    def sample_and_filter(
        self,
        question: str,
        schema_ddl: str,
        db_path: str | None = None,
        n: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[CandidateQuery], list[CandidateQuery]]:
        """Convenience method to generate N candidates and filter via SQLite sandbox."""
        cands = self.generate_candidates(question, schema_ddl, n=n, temperature=temperature)
        if db_path and os.path.exists(db_path):
            return self.filter_candidates_in_sandbox(cands, db_path)
        return cands, []
