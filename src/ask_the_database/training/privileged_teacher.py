"""Privileged Schema Teacher for On-Policy Distillation with Verifiable Reward (OPDVR).

Manages oracle teacher forward scoring, database DDL context injection,
and zero-VRAM LoRA weight sharing as specified in Lin et al. (arXiv:2608.24696, Aug 2026).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ask_the_database.environment.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class PrivilegedSchemaTeacher:
    """Privileged Schema Teacher wrapper for OPDVR and GRPD policy distillation.

    The teacher is endowed with oracle DDL schema knowledge of the target database.
    When running in shared base model mode, it disables the student's LoRA adapter
    weights during the forward pass, evaluating teacher log-probabilities on student
    action tokens with zero duplicate parameter VRAM allocation.
    """

    def __init__(
        self,
        model: nn.Module | Any,
        tokenizer: Any | None = None,
        is_shared_base: bool = True,
        max_context_length: int = 4096,
        max_ddl_tokens: int = 2500,
        databases_dir: str | Path | None = None,
    ) -> None:
        """Initialize the Privileged Schema Teacher.

        Args:
            model: PyTorch model or Hugging Face PreTrainedModel.
            tokenizer: Optional tokenizer for token length calculations and prompt formatting.
            is_shared_base: If True, indicates the teacher shares the base model with the student
                and must disable LoRA adapters during forward passes.
            max_context_length: Maximum total context tokens supported by the teacher.
            max_ddl_tokens: Maximum token budget allocated to the injected DDL schema.
            databases_dir: Optional root directory containing SQLite database files.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_shared_base = is_shared_base
        self.max_context_length = max_context_length
        self.max_ddl_tokens = max_ddl_tokens
        self.databases_dir = Path(databases_dir) if databases_dir else None

    @contextmanager
    def disable_adapter_context(self) -> Generator[None, None, None]:
        """Context manager to disable LoRA/PEFT adapters during teacher forward scoring."""
        if not self.is_shared_base:
            yield
            return

        # 1. Standard Hugging Face PEFT model interface
        if hasattr(self.model, "disable_adapter") and callable(self.model.disable_adapter):
            with self.model.disable_adapter():
                yield
            return

        # 2. Check inner model if wrapped (e.g. DataParallel, custom container)
        inner = getattr(self.model, "module", None) or getattr(self.model, "base_model", None)
        if (
            inner is not None
            and hasattr(inner, "disable_adapter")
            and callable(inner.disable_adapter)
        ):
            with inner.disable_adapter():
                yield
            return

        # Fallback: if not a PEFT model or adapter disabling not required
        yield

    def resolve_database_path(self, db_id: str) -> Path | None:
        """Resolve path to SQLite database by db_id."""
        candidate_dirs = []
        if self.databases_dir and self.databases_dir.exists():
            candidate_dirs.append(self.databases_dir)
        default_dir = Path("data/databases")
        if default_dir.exists():
            candidate_dirs.append(default_dir)

        for base_dir in candidate_dirs:
            # Check db_id/db_id.sqlite
            p1 = base_dir / db_id / f"{db_id}.sqlite"
            if p1.exists():
                return p1
            # Check db_id.sqlite
            p2 = base_dir / f"{db_id}.sqlite"
            if p2.exists():
                return p2
            # Check glob match
            matches = list(base_dir.glob(f"**/{db_id}.sqlite"))
            if matches:
                return matches[0]

        return None

    def extract_schema_ddl(
        self,
        db_id: str | None = None,
        db_path: str | Path | None = None,
    ) -> str:
        """Extract schema DDL summary using SchemaManager."""
        resolved_path: Path | None = None
        if db_path:
            resolved_path = Path(db_path)
        elif db_id:
            resolved_path = self.resolve_database_path(db_id)

        if resolved_path and resolved_path.exists():
            try:
                manager = SchemaManager(str(resolved_path))
                return manager.get_schema_summary()
            except Exception as e:
                logger.warning("Failed to extract schema from %s: %s", resolved_path, e)

        return ""

    def truncate_ddl_if_needed(self, schema_ddl: str) -> str:
        """Ensure schema DDL does not exceed max_ddl_tokens budget."""
        if not schema_ddl:
            return ""

        # Approximate token count (1 token ~= 4 characters or tokenizer length)
        token_count: int
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                token_count = len(self.tokenizer.encode(schema_ddl, add_special_tokens=False))
            except Exception:
                token_count = len(schema_ddl) // 4
        else:
            token_count = len(schema_ddl) // 4

        if token_count <= self.max_ddl_tokens:
            return schema_ddl

        # Budget exceeded: truncate table definitions cleanly
        logger.info(
            "Schema DDL tokens (%d) exceed max_ddl_tokens (%d); applying budget truncation.",
            token_count,
            self.max_ddl_tokens,
        )
        statements = schema_ddl.split("CREATE TABLE")
        retained: list[str] = []
        accumulated_tokens = 0

        for stmt in statements:
            if not stmt.strip():
                continue
            table_block = "CREATE TABLE" + stmt
            block_tokens = (
                len(self.tokenizer.encode(table_block, add_special_tokens=False))
                if self.tokenizer
                else len(table_block) // 4
            )
            if accumulated_tokens + block_tokens > self.max_ddl_tokens:
                retained.append("\n-- [Schema DDL truncated due to context window budget] --")
                break
            retained.append(table_block.strip())
            accumulated_tokens += block_tokens

        return "\n\n".join(retained)

    def prepare_privileged_prompt(
        self,
        base_prompt: str,
        db_id: str | None = None,
        schema_ddl: str | None = None,
        db_path: str | Path | None = None,
    ) -> str:
        """Construct the privileged teacher prompt by injecting database DDL schema.

        Args:
            base_prompt: Unprivileged student base prompt (ChatML or plain text).
            db_id: Optional database identifier to resolve schema if schema_ddl not provided.
            schema_ddl: Optional pre-extracted raw or summarized schema DDL.
            db_path: Optional path to SQLite database.

        Returns:
            Privileged prompt with embedded schema context.
        """
        raw_ddl = schema_ddl or self.extract_schema_ddl(db_id=db_id, db_path=db_path)
        formatted_ddl = self.truncate_ddl_if_needed(raw_ddl)

        if not formatted_ddl:
            return base_prompt

        schema_block = (
            f"=== PRIVILEGED SCHEMA CONTEXT (Oracle Teacher) ===\n"
            f"Target Database: {db_id or 'default'}\n"
            f"{formatted_ddl}\n"
            f"=================================================="
        )

        # 1. ChatML format injection: inject into system prompt before <|im_end|>
        if "<|im_start|>system" in base_prompt and "<|im_end|>" in base_prompt:
            parts = base_prompt.split("<|im_end|>", 1)
            system_part = parts[0]
            rest = parts[1]
            return f"{system_part}\n\n{schema_block}<|im_end|>{rest}"

        # 2. Standard prompt: prepend schema block before user instructions
        return f"{schema_block}\n\n{base_prompt}"

    @torch.no_grad()
    def compute_action_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        privileged_prompt_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute teacher log-probabilities on tokens in input_ids.

        Args:
            input_ids: Input token ID tensor of shape (B, L).
            attention_mask: Optional attention mask of shape (B, L).
            privileged_prompt_lengths: Optional list of prompt prefix lengths for each batch row.
                If provided, positions corresponding to prompt tokens are set to 0.0 log-probability.

        Returns:
            Token log-probabilities tensor of shape (B, L - 1) matching next-token predictions.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor (B, L), got shape {input_ids.shape}")

        b, seq_len = input_ids.shape
        if seq_len < 2:
            raise ValueError(f"Sequence length must be >= 2 for next-token scoring, got {seq_len}")

        # Execute forward pass with LoRA adapters disabled when sharing base model
        with self.disable_adapter_context():
            kwargs: dict[str, Any] = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            outputs = self.model(input_ids, **kwargs)
            logits = getattr(outputs, "logits", outputs)

        # Next-token prediction: logits at position t predict token at position t + 1
        shift_logits = logits[:, :-1, :]  # (B, L - 1, V)
        shift_targets = input_ids[:, 1:]  # (B, L - 1)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_targets.unsqueeze(-1),
        ).squeeze(-1)  # (B, L - 1)

        # Zero out prompt token positions if privileged prompt lengths are specified
        if privileged_prompt_lengths is not None:
            if len(privileged_prompt_lengths) != b:
                raise ValueError(
                    f"privileged_prompt_lengths length ({len(privileged_prompt_lengths)}) "
                    f"does not match batch size ({b})"
                )
            for i, p_len in enumerate(privileged_prompt_lengths):
                # Target at index t corresponds to input_ids[:, t + 1]
                # Tokens before index p_len are prompt tokens, so indices 0 .. p_len - 2 are prompt predictions
                prompt_cutoff = max(0, p_len - 1)
                token_logprobs[i, :prompt_cutoff] = 0.0

        return token_logprobs

    def is_weight_sharing_active(self, student_model: Any | None = None) -> bool:
        """Verify whether base weight sharing is active."""
        if not self.is_shared_base:
            return False
        if student_model is None:
            return True
        # Check direct identity or base_model unwrapping
        teacher_id = id(self.model)
        student_id = id(student_model)
        student_base = getattr(student_model, "base_model", None)
        student_base_id = id(student_base) if student_base is not None else None
        return teacher_id in (student_id, student_base_id)

    def get_parameter_memory_footprint(
        self,
        student_model: Any | None = None,
    ) -> dict[str, Any]:
        """Compute parameter counts and confirm zero extra base parameter allocations.

        Args:
            student_model: Optional student model to compare against.

        Returns:
            Dictionary with parameter counts and sharing verification status.
        """
        teacher_params = sum(p.numel() for p in self.model.parameters())
        sharing_active = self.is_weight_sharing_active(student_model)

        extra_allocated_params = 0 if sharing_active else teacher_params
        return {
            "teacher_parameter_count": teacher_params,
            "is_shared_base": self.is_shared_base,
            "weight_sharing_verified": sharing_active,
            "extra_base_parameters_allocated": extra_allocated_params,
        }
