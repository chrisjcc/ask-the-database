"""Unit tests for BaseRLAdapter and RolloutBatch."""

from dataclasses import FrozenInstanceError

import pytest
from omegaconf import DictConfig, OmegaConf

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch


def test_rollout_batch_valid_creation() -> None:
    """Verify standard creation of RolloutBatch with valid fields."""
    batch = RolloutBatch(
        prompts=["What is the average age?", "Count all students"],
        completions=["SELECT AVG(age) FROM t", "SELECT COUNT(*) FROM s"],
        gold_sqls=["SELECT AVG(age) FROM t", "SELECT COUNT(*) FROM s"],
        db_paths=["/path/to/db1.sqlite", "/path/to/db2.sqlite"],
        metadata=[{"id": 1}, {"id": 2}],
    )
    assert batch.batch_size == 2
    assert len(batch.prompts) == 2
    assert batch.metadata[0]["id"] == 1


def test_rollout_batch_immutability() -> None:
    """Verify that RolloutBatch is frozen and immutable."""
    batch = RolloutBatch(
        prompts=["Q1"],
        completions=["C1"],
        gold_sqls=["G1"],
        db_paths=["/path/to/db.sqlite"],
    )
    with pytest.raises(FrozenInstanceError):
        batch.prompts = ["Q2"]  # type: ignore[misc]


def test_rollout_batch_length_mismatch() -> None:
    """Verify that RolloutBatch raises ValueError on mismatched sequence lengths."""
    with pytest.raises(ValueError, match="Batch lengths mismatch"):
        RolloutBatch(
            prompts=["Q1", "Q2"],
            completions=["C1"],  # only 1 completion
            gold_sqls=["G1", "G2"],
            db_paths=["/path/to/db.sqlite", "/path/to/db.sqlite"],
        )

    with pytest.raises(ValueError, match="Batch lengths mismatch"):
        RolloutBatch(
            prompts=["Q1"],
            completions=["C1"],
            gold_sqls=["G1", "G2"],
            db_paths=["/path/to/db.sqlite"],
        )

    with pytest.raises(ValueError, match="Batch lengths mismatch"):
        RolloutBatch(
            prompts=["Q1"],
            completions=["C1"],
            gold_sqls=["G1"],
            db_paths=["/path/to/db1.sqlite", "/path/to/db2.sqlite"],
        )

    with pytest.raises(ValueError, match="Metadata length"):
        RolloutBatch(
            prompts=["Q1"],
            completions=["C1"],
            gold_sqls=["G1"],
            db_paths=["/path/to/db.sqlite"],
            metadata=[{"id": 1}, {"id": 2}],
        )


def test_cannot_instantiate_base_rl_adapter() -> None:
    """Verify that BaseRLAdapter cannot be instantiated directly."""
    cfg = OmegaConf.create({"learning_rate": 1e-4})
    assert isinstance(cfg, DictConfig)
    rubric = SQLRewardRubric()
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseRLAdapter(config=cfg, rubric=rubric)  # type: ignore[abstract]


class ConcreteTestAdapter(BaseRLAdapter):
    """Concrete implementation of BaseRLAdapter for unit testing."""

    def __init__(self, config: DictConfig, rubric: SQLRewardRubric) -> None:
        super().__init__(config, rubric)
        self.initialized = False
        self.checkpoints: list[str] = []

    def initialize_models(self) -> None:
        self.initialized = True

    def compute_rewards(self, batch: RolloutBatch) -> list[float]:
        return [1.0 for _ in batch.completions]

    def train_step(self, batch: RolloutBatch) -> dict[str, float]:
        rewards = self.compute_rewards(batch)
        return {"loss": 0.42, "mean_reward": sum(rewards) / len(rewards)}

    def save_checkpoint(self, output_dir: str) -> None:
        self.checkpoints.append(output_dir)


def test_concrete_adapter_execution() -> None:
    """Verify that a concrete adapter implements all contract methods correctly."""
    cfg = OmegaConf.create({"learning_rate": 1e-4})
    assert isinstance(cfg, DictConfig)
    rubric = SQLRewardRubric()
    adapter = ConcreteTestAdapter(config=cfg, rubric=rubric)

    assert not adapter.initialized
    adapter.initialize_models()
    assert adapter.initialized

    batch = RolloutBatch(
        prompts=["Query 1", "Query 2"],
        completions=["SELECT 1", "SELECT 2"],
        gold_sqls=["SELECT 1", "SELECT 2"],
        db_paths=["/db1.sqlite", "/db2.sqlite"],
    )
    metrics = adapter.train_step(batch)
    assert metrics["loss"] == 0.42
    assert metrics["mean_reward"] == 1.0

    adapter.save_checkpoint("/models/chk1")
    assert adapter.checkpoints == ["/models/chk1"]
