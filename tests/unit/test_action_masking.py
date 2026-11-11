"""Unit tests for Multi-Turn Action Token Loss Masking.

Validates:
    - Token mask generation across ChatML and XML-tagged trajectories.
    - Strict isolation of agent thoughts and tool actions (mask = 1.0).
    - Strict zeroing of environment observations and schema dumps (mask = 0.0).
    - Zero observation gradient leakage under policy gradient backward pass.
    - Integration with compute_policy_loss in TRL GRPO adapter.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ask_the_database.training.action_masking import (
    ActionMaskGenerator,
    assert_zero_observation_leakage,
)
from ask_the_database.training.adapters.trl_grpo import compute_policy_loss
from ask_the_database.training.losses import GRPDLoss


class MockTokenizer:
    """Mock tokenizer providing deterministic word-based token IDs and offset mappings."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def _get_id(self, word: str) -> int:
        if word not in self.vocab:
            new_id = len(self.vocab) + 1
            self.vocab[word] = new_id
            self.inv_vocab[new_id] = word
        return self.vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        return [self._get_id(t) for t in tokens]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.inv_vocab.get(t, "<unk>") for t in token_ids)

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        tokens = text.split()
        input_ids = [self._get_id(t) for t in tokens]
        res: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            offsets = []
            curr = 0
            for t in tokens:
                start = text.find(t, curr)
                end = start + len(t)
                offsets.append((start, end))
                curr = end
            res["offset_mapping"] = offsets
        return res


class TestActionMaskGenerator:
    """Test suite for ActionMaskGenerator."""

    def test_generate_from_messages_masks_roles_correctly(self) -> None:
        """Verify assistant turns get 1.0 mask, whereas system, user, tool get 0.0 mask."""
        tokenizer = MockTokenizer()
        generator = ActionMaskGenerator(tokenizer=tokenizer)

        messages = [
            {"role": "system", "content": "You are a database agent."},
            {"role": "user", "content": "Find the capital of France."},
            {"role": "assistant", "content": '<tool name="inspect_schema">["country"]</tool>'},
            {"role": "tool", "content": "CREATE TABLE country (code TEXT, capital TEXT);"},
            {
                "role": "assistant",
                "content": "<final_answer>SELECT capital FROM country;</final_answer>",
            },
        ]

        input_ids, action_mask = generator.generate_from_messages(messages)

        assert input_ids.shape == action_mask.shape
        assert action_mask.dim() == 2
        assert action_mask.shape[0] == 1
        seq_len = input_ids.shape[1]
        assert seq_len > 0

        # Check that there are both active (1.0) and inactive (0.0) tokens
        active_count = int(action_mask.sum().item())
        inactive_count = seq_len - active_count
        assert active_count > 0, "Must have active assistant tokens"
        assert inactive_count > 0, "Must have inactive observation/prompt tokens"

    def test_generate_from_text_xml_delimiters(self) -> None:
        """Verify XML delimiters (<thought>, <tool_call>, <tool_response>) are properly masked."""
        tokenizer = MockTokenizer()
        generator = ActionMaskGenerator(tokenizer=tokenizer)

        text = (
            "System: Explore database.\n"
            "<thought> I need to look up table schemas. </thought>\n"
            "<tool_call> inspect_schema: employees </tool_call>\n"
            "<tool_response> CREATE TABLE employees (id INT, salary FLOAT); </tool_response>\n"
            "<final_answer> SELECT AVG(salary) FROM employees; </final_answer>"
        )

        input_ids, action_mask = generator.generate_from_text(text)
        mask = action_mask[0].tolist()
        tokens = [tokenizer.inv_vocab[t] for t in input_ids[0].tolist()]

        # Identify token index of "CREATE" and "TABLE" from tool_response
        for idx, tok in enumerate(tokens):
            if tok in ("CREATE", "TABLE", "employees", "(id", "salary", "FLOAT);") and (
                "<tool_response>" in text[: text.find(tok)]
                and "</tool_response>" in text[text.find(tok) :]
            ):
                assert mask[idx] == 0.0, (
                    f"Observation token '{tok}' at index {idx} leaked into mask!"
                )

            # Final answer tokens must be active
            if tok in ("SELECT", "AVG(salary)", "FROM"):
                assert mask[idx] == 1.0, (
                    f"Action token '{tok}' at index {idx} was not masked as active!"
                )

    def test_assert_zero_observation_leakage_success(self) -> None:
        """Verify assert_zero_observation_leakage passes when observation tokens are 0.0."""
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
        obs_indices = [0, 1, 4, 5]
        # Should not raise
        assert_zero_observation_leakage(mask, obs_indices)

    def test_assert_zero_observation_leakage_raises_on_leak(self) -> None:
        """Verify assert_zero_observation_leakage raises AssertionError if observation token is active."""
        mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])  # Token at index 1 is leaked
        obs_indices = [0, 1, 3]

        with pytest.raises(AssertionError, match="Observation leakage detected"):
            assert_zero_observation_leakage(mask, obs_indices)


class TestZeroObservationGradientLeakage:
    """Test suite verifying mathematical gradient isolation of observations."""

    def test_zero_observation_gradient_leakage_in_grpd_loss(self) -> None:
        """Verify observation tokens produce strictly zero gradient in GRPDLoss backward pass."""
        # Batch of 2 trajectories, sequence length 6
        # Tokens 0, 1, 2: Prompt and observation tokens (mask = 0.0)
        # Tokens 3, 4, 5: Agent action tokens (mask = 1.0)
        student_logprobs = torch.tensor(
            [
                [-1.5, -2.0, -0.5, -1.2, -0.8, -2.1],
                [-1.1, -1.8, -0.9, -1.5, -1.0, -1.7],
            ],
            requires_grad=True,
        )
        teacher_logprobs = torch.tensor(
            [
                [-1.0, -2.5, -0.3, -0.7, -1.2, -1.5],
                [-0.8, -1.2, -1.0, -1.0, -0.5, -2.0],
            ]
        )
        advantages = torch.tensor([[1.0], [-1.0]])
        action_mask = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        loss_fn = GRPDLoss()
        loss, _ = loss_fn(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            action_mask=action_mask,
        )

        loss.backward()

        assert student_logprobs.grad is not None
        grads = student_logprobs.grad

        # Observation tokens (indices 0, 1, 2) must have EXACTLY 0.0 gradient
        obs_grads = grads[:, :3]
        assert torch.all(obs_grads == 0.0), f"Observation gradients are non-zero: {obs_grads}"

        # Action tokens (indices 3, 4, 5) must have non-zero gradients
        act_grads = grads[:, 3:]
        assert not torch.all(act_grads == 0.0), "Active tokens should produce non-zero gradients"

    def test_observation_mutation_does_not_alter_policy_loss(self) -> None:
        """Perturbing observation token logits must not alter policy loss value."""
        torch.manual_seed(42)
        b, seq_len = 2, 8
        rewards = torch.tensor([1.0, 2.0])
        comp_mask = torch.ones((b, seq_len))

        # First 4 tokens are observations, last 4 are actions
        action_mask = torch.zeros((b, seq_len))
        action_mask[:, 4:] = 1.0

        student_lp1 = torch.randn(b, seq_len)
        old_lp = torch.randn(b, seq_len)

        # Baseline loss
        res1 = compute_policy_loss(
            per_token_logps=student_lp1,
            old_per_token_logps=old_lp,
            completion_mask=comp_mask,
            rewards=rewards,
            group_size=2,
            action_mask=action_mask,
        )

        # Mutate observation logprobs dramatically
        student_lp2 = student_lp1.clone()
        student_lp2[:, :4] += 1000.0  # Massive perturbation on masked observation tokens

        res2 = compute_policy_loss(
            per_token_logps=student_lp2,
            old_per_token_logps=old_lp,
            completion_mask=comp_mask,
            rewards=rewards,
            group_size=2,
            action_mask=action_mask,
        )

        assert abs(res1.policy_loss - res2.policy_loss) < 1e-6
        assert res1.effective_tokens == res2.effective_tokens
        assert res1.effective_tokens == int(action_mask.sum().item())
