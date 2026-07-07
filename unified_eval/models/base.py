from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelInput:
    images: dict[str, np.ndarray]
    proprio: np.ndarray | None = None
    language: str | None = None
    timestep: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    actions: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


class UnifiedPolicyModel(ABC):
    @abstractmethod
    def reset(self, task: str | None = None, seed: int | None = None) -> None:
        """Reset per-episode policy state."""

    @abstractmethod
    def predict(self, obs: ModelInput) -> ModelOutput:
        """Predict one action or an action chunk for a normalized observation."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return serializable model metadata."""

    def close(self) -> None:
        """Release model resources."""
