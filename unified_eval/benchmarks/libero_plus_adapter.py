from __future__ import annotations

import contextlib
import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unified_eval.benchmarks.base import BenchmarkAdapter
from unified_eval.models.base import ModelInput


LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoPlusAdapter(BenchmarkAdapter):
    def __init__(
        self,
        task_suite: str = "libero_object",
        task_ids: Sequence[int] | None = None,
        seed: int = 0,
        resolution: int = 256,
        num_steps_wait: int = 10,
        libero_root: str | None = None,
        libero_config_path: str | None = None,
        action_dim: int = 7,
        **_: Any,
    ):
        self.task_suite_name = task_suite
        self.config_task_ids = list(task_ids) if task_ids is not None else None
        self.seed = int(seed)
        self.resolution = int(resolution)
        self.num_steps_wait = int(num_steps_wait)
        self.libero_root = Path(libero_root).expanduser() if libero_root else None
        self.libero_config_path = Path(libero_config_path).expanduser() if libero_config_path else None
        self.action_dim = int(action_dim)
        self.env = None
        self.task_suite = None
        self.task = None
        self.task_id: int | None = None
        self.initial_states = None
        self.last_obs = None
        self._benchmark = None
        self._get_libero_path = None
        self._OffScreenRenderEnv = None
        self._setup_libero()
        self.task_suite = self._make_task_suite(self.task_suite_name)

    def reset(self, task_id: str | int | None = None, seed: int | None = None) -> Any:
        if task_id is None:
            ids = self.list_task_ids()
            if not ids:
                raise ValueError(f"No tasks available in LIBERO suite {self.task_suite_name}")
            task_index = int(ids[0])
        else:
            task_index = int(task_id)
        if task_index < 0 or task_index >= self.task_suite.n_tasks:
            raise ValueError(f"LIBERO task_id {task_index} out of range for {self.task_suite_name}")
        self.close()
        self.task_id = task_index
        self.task = self.task_suite.get_task(task_index)
        self.initial_states = self.task_suite.get_task_init_states(task_index)
        if len(self.initial_states) == 0:
            raise RuntimeError(f"LIBERO task {task_index} has no initial states")
        actual_seed = self.seed if seed is None else int(seed)
        self.env = self._make_env(self.task, actual_seed)
        self.env.reset()
        init_index = actual_seed % len(self.initial_states)
        self.last_obs = self.env.set_init_state(self.initial_states[init_index])
        done = False
        reward = 0.0
        info: dict[str, Any] = {}
        for _ in range(self.num_steps_wait):
            self.last_obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION.tolist())
            if done:
                break
        return self.last_obs

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("LIBERO env is not initialized; call reset first")
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32).tolist())
        self.last_obs = obs
        return obs, float(reward), bool(done), dict(info or {})

    def convert_obs_to_model_input(self, raw_obs: Any, timestep: int) -> ModelInput:
        obs = raw_obs
        images = {
            "agentview": self._rgb_image(obs["agentview_image"]),
            "wrist": self._rgb_image(obs["robot0_eye_in_hand_image"]),
        }
        proprio = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                self._quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        return ModelInput(
            images=images,
            proprio=proprio,
            language=str(self.task.language),
            timestep=timestep,
            meta={"benchmark": "libero_plus", "task_id": self.task_id, "camera_names": list(images)},
        )

    def convert_action_to_env_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.action_dim:
            raise ValueError(f"LIBERO action dim {arr.size} is smaller than expected {self.action_dim}")
        return arr[: self.action_dim]

    def get_success(self, raw_obs: Any, reward: float, done: bool, info: dict[str, Any]) -> bool:
        return bool(done)

    def get_task_info(self) -> dict[str, Any]:
        if self.task is None:
            return {
                "benchmark": "libero_plus",
                "task_suite": self.task_suite_name,
                "num_tasks": getattr(self.task_suite, "n_tasks", None),
            }
        return {
            "benchmark": "libero_plus",
            "task_suite": self.task_suite_name,
            "task_id": self.task_id,
            "task_name": getattr(self.task, "name", None),
            "language": str(self.task.language),
            "num_tasks": getattr(self.task_suite, "n_tasks", None),
            "libero_root": str(self.libero_root) if self.libero_root else None,
        }

    def render_frame(self) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("No LIBERO observation has been produced yet")
        return self._rgb_image(self.last_obs["agentview_image"])

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def list_task_ids(self) -> Sequence[int]:
        if self.config_task_ids is not None:
            return self.config_task_ids
        if self.task_suite is None:
            return []
        return list(range(min(1, int(self.task_suite.n_tasks))))

    def get_benchmark_info(self) -> dict[str, Any]:
        info = self.get_task_info()
        info["adapter"] = "LiberoPlusAdapter"
        return info

    def _setup_libero(self) -> None:
        if self.libero_root:
            sys.path.insert(0, str(self.libero_root))
        if self.libero_config_path:
            os.environ["LIBERO_CONFIG_PATH"] = str(self.libero_config_path)
        os.environ.setdefault("MUJOCO_GL", "egl")
        self._patch_torch_load_for_init_states()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self._benchmark = benchmark
        self._get_libero_path = get_libero_path
        self._OffScreenRenderEnv = OffScreenRenderEnv

    @staticmethod
    def _patch_torch_load_for_init_states() -> None:
        try:
            import torch
        except ImportError:
            return
        if getattr(torch.load, "_axis_eval_libero_patched", False):
            return
        original_load = torch.load

        def load_with_legacy_default(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        load_with_legacy_default._axis_eval_libero_patched = True  # type: ignore[attr-defined]
        torch.load = load_with_legacy_default  # type: ignore[assignment]

    def _make_task_suite(self, name: str):
        with contextlib.redirect_stdout(io.StringIO()):
            suite_cls = self._benchmark.get_benchmark_dict()[name]
            return suite_cls()

    def _make_env(self, task, seed: int):
        task_bddl_file = Path(self._get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = self._OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=self.resolution,
            camera_widths=self.resolution,
        )
        env.seed(seed)
        return env

    @staticmethod
    def _rgb_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        arr = np.ascontiguousarray(arr[::-1, ::-1])
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).copy()
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(float(den), 0.0):
            return np.zeros(3, dtype=np.float32)
        return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)
