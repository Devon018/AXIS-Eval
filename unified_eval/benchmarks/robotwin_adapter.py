from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from unified_eval.benchmarks.base import BenchmarkAdapter
from unified_eval.models.base import ModelInput
from unified_eval.utils.imports import import_object


class RoboTwinAdapter(BenchmarkAdapter):
    def __init__(
        self,
        robotwin_root: str | None = None,
        environment_factory: str | None = None,
        task_config: str = "demo_clean",
        task_ids: Sequence[str] | None = None,
        seed: int = 0,
        image_keys: Sequence[str] = ("head_camera", "front_camera", "camera_0", "image"),
        left_wrist_keys: Sequence[str] = ("left_camera", "left_wrist_camera", "left_wrist_image"),
        right_wrist_keys: Sequence[str] = ("right_camera", "right_wrist_camera", "right_wrist_image"),
        state_keys: Sequence[str] = ("joint_action_vector", "joint_action", "qpos", "state"),
        action_type: str = "qpos",
        action_dim: int | None = None,
        pad_short_actions: bool = True,
        **factory_kwargs: Any,
    ):
        self.robotwin_root = Path(robotwin_root).expanduser() if robotwin_root else None
        self.environment_factory = environment_factory
        self.task_config = task_config
        self.config_task_ids = list(task_ids) if task_ids is not None else []
        self.seed = int(seed)
        self.image_keys = tuple(image_keys)
        self.left_wrist_keys = tuple(left_wrist_keys)
        self.right_wrist_keys = tuple(right_wrist_keys)
        self.state_keys = tuple(state_keys)
        self.action_type = action_type
        self.action_dim = action_dim
        self.pad_short_actions = bool(pad_short_actions)
        self.factory_kwargs = factory_kwargs
        self.env = None
        self.last_obs = None
        self.active_task_id: str | None = None
        self._task_args: dict[str, Any] | None = None
        self._episode_id = 0
        self._original_cwd = Path.cwd()
        self._setup_robotwin()

    def reset(self, task_id: str | int | None = None, seed: int | None = None) -> Any:
        self.close()
        if self.robotwin_root is not None:
            os.chdir(self.robotwin_root)
        self.active_task_id = str(task_id) if task_id is not None else (self.config_task_ids[0] if self.config_task_ids else None)
        if not self.active_task_id:
            raise ValueError("RoboTwin task_id is required")
        actual_seed = self.seed if seed is None else int(seed)
        if self.environment_factory:
            factory = import_object(self.environment_factory)
            kwargs = dict(self.factory_kwargs)
            kwargs.setdefault("task_name", self.active_task_id)
            kwargs.setdefault("task_config", self.task_config)
            kwargs.setdefault("seed", actual_seed)
            self.env = factory(**kwargs)
            if hasattr(self.env, "reset"):
                reset_result = self.env.reset()
                self.last_obs = reset_result if reset_result is not None else self.env.get_obs()
            else:
                self.last_obs = self.env.get_obs()
        else:
            self.env = self._make_official_task(self.active_task_id)
            self._task_args = self._official_task_args(self.active_task_id)
            self._task_args.update(self.factory_kwargs)
            self._task_args["eval_mode"] = True
            self._task_args["collect_data"] = False
            self._task_args["eval_video_log"] = False
            self.env.setup_demo(now_ep_num=self._episode_id, seed=actual_seed, is_test=True, **self._task_args)
            self._episode_id += 1
            self.last_obs = self.env.get_obs()
        return self.last_obs

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("RoboTwin env is not initialized; call reset first")
        if hasattr(self.env, "take_action"):
            result = self.env.take_action(np.asarray(action, dtype=np.float32), action_type=self.action_type)
        elif hasattr(self.env, "step"):
            result = self.env.step(np.asarray(action, dtype=np.float32))
        else:
            raise AttributeError("RoboTwin env must expose take_action(...) or step(...)")
        if isinstance(result, tuple) and len(result) >= 4:
            obs, reward, done, info = result[:4]
        else:
            obs = self.env.get_obs()
            reward = float(getattr(self.env, "reward", 0.0))
            done = bool(getattr(self.env, "success", False))
            info = {}
        self.last_obs = obs
        return obs, float(reward), bool(done), dict(info or {})

    def convert_obs_to_model_input(self, raw_obs: Any, timestep: int) -> ModelInput:
        obs = self._as_dict(raw_obs)
        base_key = self._first_existing(obs, self.image_keys)
        left_key = self._first_existing(obs, self.left_wrist_keys, required=False)
        right_key = self._first_existing(obs, self.right_wrist_keys, required=False)
        images = {"agentview": self._rgb_image(obs[base_key])}
        images["wrist"] = self._rgb_image(obs[left_key]) if left_key else np.zeros_like(images["agentview"])
        if right_key:
            images["right_wrist"] = self._rgb_image(obs[right_key])
        return ModelInput(
            images=images,
            proprio=self._state(obs),
            language=self._instruction(),
            timestep=timestep,
            meta={"benchmark": "robotwin", "task_id": self.active_task_id, "camera_names": list(images)},
        )

    def convert_action_to_env_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_dim is not None:
            if arr.size < self.action_dim:
                if not self.pad_short_actions:
                    raise ValueError(f"RoboTwin action dim {arr.size} is smaller than expected {self.action_dim}")
                arr = np.pad(arr, (0, self.action_dim - arr.size))
            arr = arr[: self.action_dim]
        return arr

    def get_success(self, raw_obs: Any, reward: float, done: bool, info: dict[str, Any]) -> bool:
        for key in ("success", "task_success", "is_success"):
            if key in info:
                return bool(info[key])
        if self.env is not None and hasattr(self.env, "check_success"):
            return bool(self.env.check_success())
        return bool(done)

    def get_task_info(self) -> dict[str, Any]:
        return {
            "benchmark": "robotwin",
            "task_id": self.active_task_id,
            "task_name": self.active_task_id,
            "language": self._instruction(),
            "task_config": self.task_config,
            "robotwin_root": str(self.robotwin_root) if self.robotwin_root else None,
            "environment_factory": self.environment_factory,
        }

    def render_frame(self) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("No RoboTwin observation has been produced yet")
        obs = self._as_dict(self.last_obs)
        base_key = self._first_existing(obs, self.image_keys)
        return self._rgb_image(obs[base_key])

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close_env"):
            self.env.close_env()
        elif self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None
        if self.robotwin_root is not None and Path.cwd() != self._original_cwd:
            os.chdir(self._original_cwd)

    def list_task_ids(self) -> Sequence[str]:
        return self.config_task_ids

    def get_benchmark_info(self) -> dict[str, Any]:
        info = self.get_task_info()
        info["adapter"] = "RoboTwinAdapter"
        return info

    def _setup_robotwin(self) -> None:
        if self.robotwin_root is None:
            return
        if not self.robotwin_root.exists():
            raise FileNotFoundError(f"RoboTwin root does not exist: {self.robotwin_root}")
        root = str(self.robotwin_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        desc_utils = str(self.robotwin_root / "description" / "utils")
        if desc_utils not in sys.path:
            sys.path.insert(0, desc_utils)
        os.chdir(self.robotwin_root)

    def _make_official_task(self, task_name: str):
        if self.robotwin_root is None:
            raise ValueError("robotwin_root is required when environment_factory is unset")
        task_class = import_object(f"envs.{task_name}:{task_name}")
        return task_class()

    def _official_task_args(self, task_name: str) -> dict[str, Any]:
        if self.robotwin_root is None:
            raise ValueError("robotwin_root is required when environment_factory is unset")
        config_dir = self.robotwin_root / "task_config"
        task_config_path = config_dir / f"{self.task_config}.yml"
        with task_config_path.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args["task_name"] = task_name
        args["task_config"] = self.task_config

        with (config_dir / "_camera_config.yml").open("r", encoding="utf-8") as f:
            camera_config = yaml.safe_load(f)
        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        with (config_dir / "_embodiment_config.yml").open("r", encoding="utf-8") as f:
            embodiment_types = yaml.safe_load(f)
        embodiment = args.get("embodiment")
        if len(embodiment) == 1:
            args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
            args["right_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
            args["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
            args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("RoboTwin embodiment items should be length 1 or 3")
        args["left_robot_file"] = self._resolve_robot_path(args["left_robot_file"])
        args["right_robot_file"] = self._resolve_robot_path(args["right_robot_file"])
        args["left_embodiment_config"] = self._read_robot_config(args["left_robot_file"])
        args["right_embodiment_config"] = self._read_robot_config(args["right_robot_file"])
        return args

    def _resolve_robot_path(self, robot_file: str) -> str:
        path = Path(robot_file)
        if not path.is_absolute():
            path = self.robotwin_root / path
        return str(path)

    @staticmethod
    def _read_robot_config(robot_file: str) -> dict[str, Any]:
        with (Path(robot_file) / "config.yml").open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _instruction(self) -> str | None:
        if self.env is not None and hasattr(self.env, "get_instruction"):
            return str(self.env.get_instruction())
        return self.active_task_id

    @staticmethod
    def _as_dict(raw_obs: Any) -> dict[str, Any]:
        if not isinstance(raw_obs, dict):
            raise TypeError(f"RoboTwin observation must be a dict, got {type(raw_obs).__name__}")
        obs = dict(raw_obs)
        nested = raw_obs.get("observation")
        if isinstance(nested, dict):
            for camera_name, camera_payload in nested.items():
                if not isinstance(camera_payload, dict):
                    continue
                for payload_key, payload_value in camera_payload.items():
                    obs[f"{camera_name}_{payload_key}"] = payload_value
                    obs[f"{camera_name}/{payload_key}"] = payload_value
                    if payload_key == "rgb":
                        obs[camera_name] = payload_value
        joint_action = raw_obs.get("joint_action")
        if isinstance(joint_action, dict):
            for key, value in joint_action.items():
                obs[f"joint_action_{key}"] = value
        return obs

    @staticmethod
    def _first_existing(obs: dict[str, Any], keys: Sequence[str], required: bool = True) -> str | None:
        for key in keys:
            if key in obs:
                return key
        if required:
            raise KeyError(f"None of the configured RoboTwin observation keys exist: {list(keys)}; available={sorted(obs)}")
        return None

    @staticmethod
    def _rgb_image(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"RoboTwin image must be HxWx3 RGB, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _state(self, obs: dict[str, Any]) -> np.ndarray | None:
        for key in self.state_keys:
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32).reshape(-1)
        numeric_parts = []
        for key, value in obs.items():
            if any(token in key.lower() for token in ("qpos", "joint", "state", "eef", "gripper")):
                arr = np.asarray(value)
                if np.issubdtype(arr.dtype, np.number):
                    numeric_parts.append(arr.astype(np.float32).reshape(-1))
        if not numeric_parts:
            return None
        return np.concatenate(numeric_parts, axis=0)
