from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


class ResultLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self._episodes_file = self.episodes_path.open("w", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def write_json(self, filename: str, data: dict[str, Any]) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def append_episode(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self._episodes_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._episodes_file.flush()

    def close(self) -> None:
        if not self._episodes_file.closed:
            self._episodes_file.close()

    def write_report(self, benchmark: str, model: str) -> dict[str, Any]:
        num_episodes = len(self.records)
        num_success = sum(int(bool(r.get("success"))) for r in self.records)
        num_errors = sum(int(r.get("error") is not None) for r in self.records)
        videos_saved = sum(int(bool(r.get("video_path"))) for r in self.records)
        lengths = [float(r.get("num_steps") or 0) for r in self.records]
        report = {
            "benchmark": benchmark,
            "model": model,
            "num_episodes": num_episodes,
            "num_success": num_success,
            "success_rate": num_success / max(num_episodes, 1),
            "mean_episode_length": sum(lengths) / max(len(lengths), 1),
            "num_errors": num_errors,
            "videos_saved": videos_saved,
        }
        self.write_json("report.json", report)
        self._write_csv("report.csv", self.records)
        return report

    def _write_csv(self, filename: str, records: Iterable[dict[str, Any]]) -> None:
        rows = list(records)
        path = self.output_dir / filename
        fieldnames = [
            "benchmark",
            "model",
            "task_id",
            "task_name",
            "language",
            "seed",
            "episode_id",
            "success",
            "num_steps",
            "max_episode_steps",
            "done",
            "total_reward",
            "video_path",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
