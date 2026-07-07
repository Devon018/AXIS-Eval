from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from unified_eval.models.base import ModelInput


class BenchmarkAdapter(ABC):
    @abstractmethod
    def reset(self, task_id: str | int | None = None, seed: int | None = None) -> Any:
        """Start a new episode and return the raw benchmark observation."""

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Step the benchmark environment."""

    @abstractmethod
    def convert_obs_to_model_input(self, raw_obs: Any, timestep: int) -> ModelInput:
        """Convert benchmark-native observation into UnifiedPolicyModel input."""

    @abstractmethod
    def convert_action_to_env_action(self, action: np.ndarray) -> Any:
        """Convert model action into benchmark-native action."""

    @abstractmethod
    def get_success(self, raw_obs: Any, reward: float, done: bool, info: dict[str, Any]) -> bool:
        """Return the benchmark success signal for the current state."""

    @abstractmethod
    def get_task_info(self) -> dict[str, Any]:
        """Return serializable metadata for the active task."""

    @abstractmethod
    def render_frame(self) -> np.ndarray:
        """Return one RGB frame for rollout video."""

    @abstractmethod
    def close(self) -> None:
        """Release benchmark resources."""

    def list_task_ids(self) -> Sequence[str | int]:
        return []

    def get_benchmark_info(self) -> dict[str, Any]:
        return self.get_task_info()
