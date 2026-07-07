from __future__ import annotations

import itertools
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from unified_eval.benchmarks.base import BenchmarkAdapter
from unified_eval.benchmarks.libero_plus_adapter import LiberoPlusAdapter
from unified_eval.benchmarks.robocasa_adapter import RoboCasaAdapter
from unified_eval.benchmarks.robotwin_adapter import RoboTwinAdapter
from unified_eval.logging.result_logger import ResultLogger
from unified_eval.logging.video_logger import VideoLogger
from unified_eval.models.base import UnifiedPolicyModel
from unified_eval.models.pi05_adapter import Pi05Adapter
from unified_eval.utils.config import write_config


BENCHMARKS = {
    "libero_plus": LiberoPlusAdapter,
    "robotwin": RoboTwinAdapter,
    "robocasa": RoboCasaAdapter,
}

MODELS = {
    "pi0.5": Pi05Adapter,
    "pi05": Pi05Adapter,
}


class EvalRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.experiment = config.get("experiment", {})
        self.benchmark_cfg = config.get("benchmark", {})
        self.model_cfg = config.get("model", {})
        self.logging_cfg = config.get("logging", {})
        output_dir = Path(self.experiment["output_dir"]).expanduser()
        self.output_dir = output_dir if output_dir.is_absolute() else output_dir.resolve()

    def run(self) -> dict[str, Any]:
        benchmark = self._make_benchmark()
        model = self._make_model()
        logger = ResultLogger(self.output_dir)
        try:
            write_config(self.output_dir / "config.yaml", self.config)
            logger.write_json("model_info.json", model.get_model_info())
            logger.write_json("benchmark_info.json", benchmark.get_benchmark_info())
            report = self._run_episodes(benchmark, model, logger)
        finally:
            logger.close()
            benchmark.close()
            model.close()
        return report

    def _make_benchmark(self) -> BenchmarkAdapter:
        name = self.benchmark_cfg.get("name")
        if name not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {name}")
        kwargs = {k: v for k, v in self.benchmark_cfg.items() if k != "name"}
        return BENCHMARKS[name](**kwargs)

    def _make_model(self) -> UnifiedPolicyModel:
        name = self.model_cfg.get("name")
        if name not in MODELS:
            raise ValueError(f"Unsupported model: {name}")
        kwargs = {k: v for k, v in self.model_cfg.items() if k != "name"}
        return MODELS[name](**kwargs)

    def _episode_task_ids(self, benchmark: BenchmarkAdapter) -> list[str | int | None]:
        num_episodes = int(self.benchmark_cfg.get("num_episodes", 1))
        task_ids = self.benchmark_cfg.get("task_ids")
        if task_ids is None:
            task_ids = list(benchmark.list_task_ids())
        if task_ids is None or len(task_ids) == 0:
            task_ids = [None]
        return list(itertools.islice(itertools.cycle(task_ids), num_episodes))

    def _run_episodes(
        self,
        benchmark: BenchmarkAdapter,
        model: UnifiedPolicyModel,
        logger: ResultLogger,
    ) -> dict[str, Any]:
        benchmark_name = str(self.benchmark_cfg["name"])
        model_name = str(self.model_cfg["name"])
        max_steps = int(self.benchmark_cfg.get("max_episode_steps", 1))
        save_video = bool(self.logging_cfg.get("save_video", self.benchmark_cfg.get("save_video", True)))
        base_seed = int(self.benchmark_cfg.get("seed", 0))
        video_fps = int(self.logging_cfg.get("video_fps", 10))

        for episode_id, task_id in enumerate(self._episode_task_ids(benchmark)):
            seed = base_seed + episode_id
            task_info: dict[str, Any] = {}
            record = self._base_record(benchmark_name, model_name, task_id, seed, episode_id, max_steps)
            video_logger = VideoLogger(
                self.output_dir / "videos" / self._video_name(task_id, seed, episode_id),
                fps=video_fps,
                enabled=save_video,
            )
            raw_obs = None
            total_reward = 0.0
            done = False
            success = False
            steps = 0
            try:
                raw_obs = benchmark.reset(task_id=task_id, seed=seed)
                task_info = benchmark.get_task_info()
                record.update(
                    {
                        "task_name": task_info.get("task_name") or task_info.get("name"),
                        "language": task_info.get("language"),
                    }
                )
                model.reset(task=record.get("language"), seed=seed)
                first_frame = benchmark.render_frame()
                video_logger.add_frame(first_frame)

                for timestep in range(max_steps):
                    model_input = benchmark.convert_obs_to_model_input(raw_obs, timestep=timestep)
                    model_output = model.predict(model_input)
                    action = np.asarray(model_output.actions)
                    if action.ndim == 2:
                        action = action[0]
                    env_action = benchmark.convert_action_to_env_action(action)
                    raw_obs, reward, done, info = benchmark.step(env_action)
                    total_reward += float(reward)
                    steps = timestep + 1
                    video_logger.add_frame(benchmark.render_frame())
                    success = benchmark.get_success(raw_obs, reward, done, info)
                    if done:
                        break
            except Exception as exc:
                record["error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
            finally:
                video_path = video_logger.save()
                if video_path:
                    record["video_path"] = str(Path(video_path).relative_to(self.output_dir))
                record.update(
                    {
                        "success": bool(success),
                        "num_steps": steps,
                        "done": bool(done),
                        "total_reward": total_reward,
                    }
                )
                if task_info:
                    record.setdefault("task_name", task_info.get("task_name") or task_info.get("name"))
                    record.setdefault("language", task_info.get("language"))
                logger.append_episode(record)

        report = logger.write_report(benchmark=benchmark_name, model=model_name)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    @staticmethod
    def _base_record(
        benchmark_name: str,
        model_name: str,
        task_id: str | int | None,
        seed: int,
        episode_id: int,
        max_steps: int,
    ) -> dict[str, Any]:
        return {
            "benchmark": benchmark_name,
            "model": model_name,
            "task_id": str(task_id) if task_id is not None else None,
            "task_name": None,
            "language": None,
            "seed": seed,
            "episode_id": episode_id,
            "success": False,
            "num_steps": 0,
            "max_episode_steps": max_steps,
            "done": False,
            "total_reward": 0.0,
            "video_path": None,
            "error": None,
        }

    @staticmethod
    def _video_name(task_id: str | int | None, seed: int, episode_id: int) -> str:
        safe_task = str(task_id) if task_id is not None else "task"
        safe_task = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe_task)
        return f"{safe_task}_seed_{seed}_episode_{episode_id}.mp4"
