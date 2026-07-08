from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unified_eval.benchmarks.base import BenchmarkAdapter
from unified_eval.models.base import ModelInput


class RoboCasaAdapter(BenchmarkAdapter):
    def __init__(
        self,
        env_id: str = "robocasa/PickPlaceCounterToCabinet",
        split: str = "pretrain",
        task_suite: str | None = None,
        task_ids: Sequence[str] | None = None,
        num_episodes: int | None = None,
        max_episode_steps: int | None = None,
        save_video: bool | None = None,
        seed: int = 0,
        robocasa_root: str | None = None,
        robosuite_root: str | None = None,
        render_mode: str | None = None,
        image_keys: Sequence[str] = (
            "video.robot0_agentview_left",
            "robot0_agentview_left_image",
            "agentview_image",
            "image",
        ),
        wrist_image_keys: Sequence[str] = (
            "video.robot0_eye_in_hand",
            "robot0_eye_in_hand_image",
            "wrist_image",
        ),
        proprio_key: str | None = None,
        proprio_keys: Sequence[str] | None = None,
        action_dim: int | None = None,
        pad_short_actions: bool = True,
        **env_kwargs: Any,
    ):
        self.env_id = env_id
        self.split = split
        self.task_suite = task_suite
        self.config_task_ids = list(task_ids) if task_ids is not None else [env_id]
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.save_video = save_video
        self.seed = int(seed)
        self.robocasa_root = Path(robocasa_root).expanduser() if robocasa_root else None
        self.robosuite_root = Path(robosuite_root).expanduser() if robosuite_root else None
        self.render_mode = render_mode
        self.image_keys = tuple(image_keys)
        self.wrist_image_keys = tuple(wrist_image_keys)
        self.proprio_key = proprio_key
        self.proprio_keys = tuple(proprio_keys) if proprio_keys else None
        self.action_dim = action_dim
        self.pad_short_actions = bool(pad_short_actions)
        self.env_kwargs = env_kwargs
        self.env = None
        self.last_obs = None
        self.active_task_id: str | None = None
        self._gym = None
        self._setup_robocasa()

    def reset(self, task_id: str | int | None = None, seed: int | None = None) -> Any:
        self.close()
        self.active_task_id = str(task_id) if task_id is not None else self.env_id
        actual_seed = self.seed if seed is None else int(seed)
        make_kwargs = dict(self.env_kwargs)
        make_kwargs.setdefault("split", self.split)
        make_kwargs.setdefault("seed", actual_seed)
        self.env = self._gym.make(self.active_task_id, **make_kwargs)
        reset_result = self.env.reset(seed=actual_seed)
        self.last_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return self.last_obs

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("RoboCasa env is not initialized; call reset first")
        if isinstance(action, dict):
            result = self.env.step(action)
        else:
            result = self.env.step(np.asarray(action, dtype=np.float32))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        self.last_obs = obs
        return obs, float(reward), bool(done), dict(info or {})

    def convert_obs_to_model_input(self, raw_obs: Any, timestep: int) -> ModelInput:
        obs = self._as_dict(raw_obs)
        base_key = self._first_existing(obs, self.image_keys)
        wrist_key = self._first_existing(obs, self.wrist_image_keys, required=False)
        images = {"agentview": self._rgb_image(obs[base_key])}
        if wrist_key is not None:
            images["wrist"] = self._rgb_image(obs[wrist_key])
        else:
            images["wrist"] = np.zeros_like(images["agentview"])
        proprio = self._proprio(obs)
        return ModelInput(
            images=images,
            proprio=proprio,
            language=self._language(obs),
            timestep=timestep,
            meta={"benchmark": "robocasa", "task_id": self.active_task_id, "camera_names": list(images)},
        )

    def convert_action_to_env_action(self, action: np.ndarray) -> Any:
        if self.env is None:
            raise RuntimeError("RoboCasa env is not initialized; call reset first")
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        action_space = self.env.action_space
        if hasattr(action_space, "spaces"):
            return self._dict_action_from_flat(arr, action_space)

        target_dim = self.action_dim
        if target_dim is None:
            target_dim = int(np.prod(action_space.shape))
        if arr.size < target_dim:
            if not self.pad_short_actions:
                raise ValueError(f"RoboCasa action dim {arr.size} is smaller than env action dim {target_dim}")
            arr = np.pad(arr, (0, target_dim - arr.size))
        arr = arr[:target_dim]
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if low.size == target_dim and high.size == target_dim:
            arr = np.clip(arr, low, high)
        return arr.reshape(action_space.shape)

    def get_success(self, raw_obs: Any, reward: float, done: bool, info: dict[str, Any]) -> bool:
        for key in ("success", "task_success", "is_success"):
            if key in info:
                return bool(info[key])
        return bool(done and reward > 0)

    def get_task_info(self) -> dict[str, Any]:
        return {
            "benchmark": "robocasa",
            "task_id": self.active_task_id or self.env_id,
            "task_name": self.active_task_id or self.env_id,
            "language": self._current_language() or self.active_task_id or self.env_id,
            "env_id": self.env_id,
            "split": self.split,
            "task_suite": self.task_suite,
        }

    def render_frame(self) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("No RoboCasa observation has been produced yet")
        obs = self._as_dict(self.last_obs)
        base_key = self._first_existing(obs, self.image_keys)
        return self._rgb_image(obs[base_key])

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def list_task_ids(self) -> Sequence[str]:
        return self.config_task_ids

    def get_benchmark_info(self) -> dict[str, Any]:
        info = self.get_task_info()
        info["adapter"] = "RoboCasaAdapter"
        return info

    def _setup_robocasa(self) -> None:
        if self.robocasa_root:
            if not self.robocasa_root.exists():
                raise FileNotFoundError(f"robocasa_root does not exist: {self.robocasa_root}")
            root = str(self.robocasa_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        if self.robosuite_root:
            if not self.robosuite_root.exists():
                raise FileNotFoundError(f"robosuite_root does not exist: {self.robosuite_root}")
            root = str(self.robosuite_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        import gymnasium as gym

        import robosuite  # noqa: F401

        self._import_robocasa()

        self._gym = gym

    @staticmethod
    def _import_robocasa() -> None:
        real_numpy_version = np.__version__
        if real_numpy_version != "2.2.5":
            # RoboCasa 1.0.1 hard-checks this exact string at import time.
            # robosuite and numba must be imported before this temporary shim.
            np.__version__ = "2.2.5"
        try:
            import robocasa  # noqa: F401
        finally:
            np.__version__ = real_numpy_version

    @staticmethod
    def _as_dict(raw_obs: Any) -> dict[str, Any]:
        if not isinstance(raw_obs, dict):
            raise TypeError(f"RoboCasa observation must be a dict, got {type(raw_obs).__name__}")
        return raw_obs

    @staticmethod
    def _first_existing(obs: dict[str, Any], keys: Sequence[str], required: bool = True) -> str | None:
        for key in keys:
            if key in obs:
                return key
        if required:
            raise KeyError(f"None of the configured observation keys exist: {list(keys)}; available={sorted(obs)}")
        return None

    @staticmethod
    def _rgb_image(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"RoboCasa image must be HxWx3 RGB, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _proprio(self, obs: dict[str, Any]) -> np.ndarray | None:
        if self.proprio_keys:
            parts = []
            for key in self.proprio_keys:
                if key not in obs:
                    raise KeyError(f"Configured RoboCasa proprio_keys entry missing: {key}")
                parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
            return np.concatenate(parts)
        if self.proprio_key:
            if self.proprio_key not in obs:
                raise KeyError(f"Configured RoboCasa proprio_key missing: {self.proprio_key}")
            return np.asarray(obs[self.proprio_key], dtype=np.float32).reshape(-1)
        parts = []
        for key, value in obs.items():
            lowered = key.lower()
            if any(token in lowered for token in ("state", "eef", "qpos", "gripper")):
                arr = np.asarray(value)
                if np.issubdtype(arr.dtype, np.number):
                    parts.append(arr.astype(np.float32).reshape(-1))
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _dict_action_from_flat(self, action: np.ndarray, action_space: Any) -> dict[str, Any]:
        spaces = action_space.spaces
        target_dim = 0
        for subspace in spaces.values():
            shape = getattr(subspace, "shape", None)
            target_dim += int(np.prod(shape)) if shape is not None else 1
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < target_dim:
            if not self.pad_short_actions:
                raise ValueError(f"RoboCasa action dim {arr.size} is smaller than env action dim {target_dim}")
            arr = np.pad(arr, (0, target_dim - arr.size))
        arr = arr[:target_dim]

        env_action: dict[str, Any] = {}
        offset = 0
        for key, subspace in spaces.items():
            shape = getattr(subspace, "shape", None)
            if shape is None:
                value = int(round(float(arr[offset])))
                if hasattr(subspace, "n"):
                    value = int(np.clip(value, 0, int(subspace.n) - 1))
                env_action[key] = value
                offset += 1
                continue

            size = int(np.prod(shape))
            value = arr[offset : offset + size].reshape(shape).astype(np.float32)
            low = getattr(subspace, "low", None)
            high = getattr(subspace, "high", None)
            if low is not None and high is not None:
                value = np.clip(value, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
            env_action[key] = value
            offset += size
        return env_action

    def _current_language(self) -> str | None:
        if self.last_obs is None:
            return None
        try:
            return self._language(self._as_dict(self.last_obs))
        except Exception:
            return None

    @staticmethod
    def _language(obs: dict[str, Any]) -> str | None:
        for key in ("language", "instruction", "task", "task_description", "annotation.human.task_description"):
            value = obs.get(key)
            if isinstance(value, str):
                return value
        return None
