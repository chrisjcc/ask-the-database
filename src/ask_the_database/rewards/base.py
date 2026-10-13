"""Base abstractions for verifiable reward components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagnosticPayload:
    """Telemetry and diagnostic metrics for reward and trajectory logging."""

    stage: str
    details: dict[str, Any] = field(default_factory=dict)


class BaseRewardScorer(ABC):
    """Abstract interface for reward scoring components."""

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> float:
        """Compute scalar reward value."""
        pass
