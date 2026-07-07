from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from unified_eval.models.base import ModelInput, ModelOutput, UnifiedPolicyModel


class Pi05Adapter(UnifiedPolicyModel):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        policy_config: str = "pi05_libero",
        host: str = "localhost",
        port: int = 8000,
        auto_start_server: bool = False,
        openpi_root: str | None = None,
        openpi_python: str | None = None,
        openpi_client_path: str | None = None,
        server_log_path: str = "tmp/pi05_policy_server.log",
        server_start_timeout: int = 900,
        image_size: int = 224,
        base_image_key: str = "agentview",
        left_wrist_image_key: str | None = "wrist",
        right_wrist_image_key: str | None = None,
        state_dim: int = 7,
        allow_zero_proprio: bool = False,
        action_key: str = "actions",
        xla_mem_fraction: float = 0.85,
        extra_pythonpath: list[str] | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.policy_config = policy_config
        self.host = host
        self.port = int(port)
        self.auto_start_server = bool(auto_start_server)
        self.openpi_root = Path(openpi_root).expanduser() if openpi_root else None
        self.openpi_python = openpi_python
        self.openpi_client_path = openpi_client_path
        self.server_log_path = Path(server_log_path)
        self.server_start_timeout = int(server_start_timeout)
        self.image_size = int(image_size)
        self.base_image_key = base_image_key
        self.left_wrist_image_key = left_wrist_image_key
        self.right_wrist_image_key = right_wrist_image_key
        self.state_dim = int(state_dim)
        self.allow_zero_proprio = bool(allow_zero_proprio)
        self.action_key = action_key
        self.xla_mem_fraction = float(xla_mem_fraction)
        self.extra_pythonpath = extra_pythonpath or []
        self._server_proc: subprocess.Popen | None = None
        self._task: str | None = None

        self._validate_checkpoint()
        self._prepare_imports()
        try:
            if self.auto_start_server:
                self._start_server()
            self._client = self._make_client()
        except Exception:
            self.close()
            raise

    def reset(self, task: str | None = None, seed: int | None = None) -> None:
        self._task = task
        if seed is not None:
            np.random.seed(seed)

    def predict(self, obs: ModelInput) -> ModelOutput:
        element = self._to_openpi_element(obs)
        result = self._client.infer(element)
        if self.action_key not in result:
            raise KeyError(f"Pi0.5 response does not contain action key {self.action_key!r}; keys={sorted(result)}")
        actions = np.asarray(result[self.action_key], dtype=np.float32)
        return ModelOutput(actions=actions, info={"backend": "openpi_websocket"})

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": "pi0.5",
            "checkpoint_path": self.checkpoint_path,
            "policy_config": self.policy_config,
            "device": self.device,
            "backend": "openpi_websocket",
            "host": self.host,
            "port": self.port,
            "auto_start_server": self.auto_start_server,
            "openpi_root": str(self.openpi_root) if self.openpi_root else None,
            "openpi_python": self.openpi_python,
        }

    def close(self) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
                self._server_proc.wait(timeout=20)

    def _validate_checkpoint(self) -> None:
        if self.checkpoint_path.startswith("gs://"):
            return
        path = Path(self.checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Pi0.5 checkpoint path does not exist: {path}")
        has_params = (path / "params").exists() or path.name == "params"
        has_metadata = (path / "_CHECKPOINT_METADATA").exists() or (path / "params" / "_METADATA").exists()
        if not has_params and not has_metadata:
            raise FileNotFoundError(
                f"Pi0.5 checkpoint path does not look like an OpenPI checkpoint: {path}"
            )

    def _prepare_imports(self) -> None:
        paths = []
        if self.openpi_client_path:
            paths.append(str(Path(self.openpi_client_path).expanduser()))
        if self.openpi_root:
            paths.extend(
                [
                    str(self.openpi_root / "packages" / "openpi-client" / "src"),
                    str(self.openpi_root / "src"),
                ]
            )
        paths.extend(self.extra_pythonpath)
        for path in reversed(paths):
            if path and path not in sys.path:
                sys.path.insert(0, path)

    def _make_client(self):
        from openpi_client import websocket_client_policy

        return websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def _to_openpi_element(self, obs: ModelInput) -> dict[str, Any]:
        from openpi_client import image_tools

        base_image = self._image(obs.images, self.base_image_key)
        base_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(base_image, self.image_size, self.image_size)
        )
        element: dict[str, Any] = {
            "observation/image": base_image,
            "observation/state": self._state(obs),
            "prompt": obs.language or self._task or "",
        }
        if self.left_wrist_image_key:
            element["observation/wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self._image(obs.images, self.left_wrist_image_key), self.image_size, self.image_size)
            )
        if self.right_wrist_image_key:
            element["observation/right_wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    self._image(obs.images, self.right_wrist_image_key), self.image_size, self.image_size
                )
            )
        element["meta"] = dict(obs.meta)
        return element

    @staticmethod
    def _image(images: dict[str, np.ndarray], key: str) -> np.ndarray:
        if key not in images:
            raise KeyError(f"ModelInput.images is missing required camera {key!r}; available={sorted(images)}")
        image = np.asarray(images[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera {key!r} must be HxWx3 RGB, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _state(self, obs: ModelInput) -> np.ndarray:
        if obs.proprio is None:
            if not self.allow_zero_proprio:
                raise ValueError("Pi0.5 requires proprio/state; set allow_zero_proprio only for explicit smoke runs")
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
        if state.size < self.state_dim:
            if not self.allow_zero_proprio:
                raise ValueError(f"Pi0.5 state has dim {state.size}, expected at least {self.state_dim}")
            state = np.pad(state, (0, self.state_dim - state.size))
        if state.size > self.state_dim:
            state = state[: self.state_dim]
        return state.astype(np.float32)

    def _start_server(self) -> None:
        if self.openpi_root is None:
            raise ValueError("openpi_root is required when auto_start_server=true")
        if self.openpi_python is None:
            raise ValueError("openpi_python is required when auto_start_server=true")
        serve_script = self.openpi_root / "scripts" / "serve_policy.py"
        if not serve_script.exists():
            raise FileNotFoundError(f"OpenPI serve_policy.py not found: {serve_script}")
        if self._port_open():
            raise RuntimeError(f"{self.host}:{self.port} is already open; refusing to attach to an unknown policy server")

        self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.server_log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(self.openpi_root / "src"),
                str(self.openpi_root / "packages" / "openpi-client" / "src"),
                env.get("PYTHONPATH", ""),
            ]
        )
        if self.device.startswith("cuda"):
            device_id = self.device.split(":", 1)[1] if ":" in self.device else env.get("CUDA_VISIBLE_DEVICES", "0")
            env["CUDA_VISIBLE_DEVICES"] = device_id
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.xla_mem_fraction)
        cmd = [
            self.openpi_python,
            str(serve_script),
            "--port",
            str(self.port),
            "policy:checkpoint",
            f"--policy.config={self.policy_config}",
            f"--policy.dir={self.checkpoint_path}",
        ]
        self._server_proc = subprocess.Popen(cmd, cwd=self.openpi_root, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        deadline = time.time() + self.server_start_timeout
        while time.time() < deadline:
            if self._port_open():
                return
            if self._server_proc is not None and self._server_proc.poll() is not None:
                raise RuntimeError(
                    f"Pi0.5 policy server exited early with code {self._server_proc.returncode}; "
                    f"log={self.server_log_path}"
                )
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for Pi0.5 policy server on {self.host}:{self.port}; log={self.server_log_path}")

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=2):
                return True
        except OSError:
            return False
