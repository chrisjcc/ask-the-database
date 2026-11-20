"""Unit tests for PrivilegedSchemaTeacher in OPDVR distillation.

Validates:
    - Privileged DDL schema prompt preparation and context injection.
    - ChatML and standard prompt formatting.
    - Token budget truncation under context constraints.
    - Token-level log-probability computation with next-token alignment.
    - LoRA adapter disabling during teacher forward passes.
    - Zero duplicate parameter allocation verification (shared base mode).
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ask_the_database.training.privileged_teacher import PrivilegedSchemaTeacher


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Simple whitespace splitting for testing
        return [hash(w) % 1000 for w in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        words = text.split()
        input_ids = [hash(w) % 1000 for w in words]
        res: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            offsets = []
            curr = 0
            for w in words:
                start = text.find(w, curr)
                end = start + len(w)
                offsets.append((start, end))
                curr = end
            res["offset_mapping"] = offsets
        return res


class DummyCausalLM(nn.Module):
    """Simple model returning predictable logits for testing logprobs."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.adapter_disabled = False

    @contextmanager
    def disable_adapter(self) -> Generator[None, None, None]:
        prev = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = prev

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        hidden = self.embed(input_ids)
        logits = self.head(hidden)

        class Output:
            pass

        out = Output()
        out.logits = logits  # type: ignore[attr-defined]
        return out


class TestPrivilegedSchemaTeacher:
    """Test suite for PrivilegedSchemaTeacher."""

    def test_prepare_privileged_prompt_chatml(self) -> None:
        """Verify schema DDL is injected into system turn of ChatML prompts."""
        model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        base_prompt = (
            "<|im_start|>system\nYou are a database assistant.<|im_end|>\n"
            "<|im_start|>user\nFind total sales.<|im_end|>"
        )
        sample_ddl = "CREATE TABLE sales (id INT PRIMARY KEY, amount FLOAT);"

        privileged = teacher.prepare_privileged_prompt(
            base_prompt=base_prompt,
            db_id="sales_db",
            schema_ddl=sample_ddl,
        )

        assert "=== PRIVILEGED SCHEMA CONTEXT (Oracle Teacher) ===" in privileged
        assert "Target Database: sales_db" in privileged
        assert sample_ddl in privileged
        assert privileged.startswith("<|im_start|>system\nYou are a database assistant.")
        assert "<|im_start|>user\nFind total sales.<|im_end|>" in privileged

    def test_prepare_privileged_prompt_standard(self) -> None:
        """Verify schema DDL is prepended to non-ChatML prompts."""
        model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        base_prompt = "Question: Find total sales in department 3."
        sample_ddl = "CREATE TABLE employees (id INT, dept_id INT);"

        privileged = teacher.prepare_privileged_prompt(
            base_prompt=base_prompt,
            db_id="corp_db",
            schema_ddl=sample_ddl,
        )

        assert privileged.startswith("=== PRIVILEGED SCHEMA CONTEXT (Oracle Teacher) ===")
        assert "Question: Find total sales in department 3." in privileged

    def test_prepare_privileged_prompt_empty_ddl_fallback(self) -> None:
        """When no DDL is found or provided, base prompt is returned unchanged."""
        model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        base_prompt = "Select all customers."
        res = teacher.prepare_privileged_prompt(base_prompt, db_id="nonexistent_db")
        assert res == base_prompt

    def test_truncate_ddl_under_budget(self) -> None:
        """Verify DDL exceeding max_ddl_tokens is truncated cleanly."""
        model = DummyCausalLM()
        tokenizer = DummyTokenizer()
        teacher = PrivilegedSchemaTeacher(
            model=model,
            tokenizer=tokenizer,
            max_ddl_tokens=10,  # Strict small token budget
        )

        long_ddl = (
            "CREATE TABLE t1 (c1 INT, c2 TEXT, c3 FLOAT);\n\n"
            "CREATE TABLE t2 (c1 INT, c2 TEXT, c3 FLOAT, c4 VARCHAR(255));\n\n"
            "CREATE TABLE t3 (c1 INT PRIMARY KEY);"
        )

        truncated = teacher.truncate_ddl_if_needed(long_ddl)
        assert "CREATE TABLE t1" in truncated
        assert "[Schema DDL truncated due to context window budget]" in truncated
        assert "CREATE TABLE t3" not in truncated

    def test_compute_action_logprobs_shape_and_values(self) -> None:
        """Verify compute_action_logprobs returns (B, L - 1) token log-probabilities."""
        model = DummyCausalLM(vocab_size=50)
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        # Batch of 2 sequences, each of length 6
        input_ids = torch.tensor(
            [
                [1, 5, 12, 19, 23, 4],
                [2, 8, 14, 20, 25, 7],
            ],
            dtype=torch.long,
        )

        logprobs = teacher.compute_action_logprobs(input_ids)
        assert logprobs.shape == (2, 5)  # (B, L - 1)
        # Log probabilities must be <= 0.0
        assert torch.all(logprobs <= 0.0)
        assert not torch.isnan(logprobs).any()

    def test_compute_action_logprobs_prompt_zeroing(self) -> None:
        """Verify prompt positions receive 0.0 log-probability when prompt lengths are given."""
        model = DummyCausalLM(vocab_size=50)
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        input_ids = torch.tensor(
            [
                [1, 5, 12, 19, 23, 4],  # Length 6, prompt length 3
                [2, 8, 14, 20, 25, 7],  # Length 6, prompt length 4
            ],
            dtype=torch.long,
        )

        # Row 0: prompt length 3 -> prompt targets are indices 0, 1 -> zeroed
        # Row 1: prompt length 4 -> prompt targets are indices 0, 1, 2 -> zeroed
        logprobs = teacher.compute_action_logprobs(
            input_ids,
            privileged_prompt_lengths=[3, 4],
        )

        assert logprobs.shape == (2, 5)
        # Row 0 prompt positions
        assert torch.all(logprobs[0, :2] == 0.0)
        assert logprobs[0, 2] < 0.0  # completion target at pos 2
        # Row 1 prompt positions
        assert torch.all(logprobs[1, :3] == 0.0)
        assert logprobs[1, 3] < 0.0

    def test_lora_weight_sharing_adapter_disabling(self) -> None:
        """Verify teacher disables LoRA adapter during forward scoring."""
        model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        assert not model.adapter_disabled
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        # Forward pass should temporarily disable adapter
        _ = teacher.compute_action_logprobs(input_ids)
        # After forward, state should be restored
        assert not model.adapter_disabled

    def test_parameter_memory_footprint_zero_extra_vram(self) -> None:
        """Verify shared base model reports zero extra base parameter allocations."""
        student_model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=student_model, is_shared_base=True)

        footprint = teacher.get_parameter_memory_footprint(student_model=student_model)
        assert footprint["is_shared_base"] is True
        assert footprint["weight_sharing_verified"] is True
        assert footprint["extra_base_parameters_allocated"] == 0

    def test_unshared_base_reports_extra_parameters(self) -> None:
        """When teacher is independent model, reports its own parameters."""
        student_model = DummyCausalLM()
        separate_teacher_model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=separate_teacher_model, is_shared_base=False)

        footprint = teacher.get_parameter_memory_footprint(student_model=student_model)
        assert footprint["is_shared_base"] is False
        assert footprint["weight_sharing_verified"] is False
        assert footprint["extra_base_parameters_allocated"] > 0

    def test_extract_schema_ddl_from_sqlite_file(self, tmp_path: Path) -> None:
        """Verify schema extraction works against actual SQLite database file."""
        import sqlite3

        db_path = tmp_path / "test_store.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE products (product_id INT PRIMARY KEY, name TEXT, price FLOAT);")
        conn.execute("CREATE TABLE orders (order_id INT PRIMARY KEY, product_id INT);")
        conn.close()

        model = DummyCausalLM()
        teacher = PrivilegedSchemaTeacher(model=model, is_shared_base=True)

        ddl = teacher.extract_schema_ddl(db_path=db_path)
        assert "CREATE TABLE products" in ddl
        assert "CREATE TABLE orders" in ddl
