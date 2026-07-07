#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize unified_eval report.json files.")
    parser.add_argument("--input", required=True, help="Directory containing benchmark output directories.")
    parser.add_argument("--output", default=None, help="Summary JSON path. Defaults to <input>/summary.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else input_dir / "summary.json"
    reports = sorted(input_dir.glob("*/report.json"))
    if not reports:
        raise FileNotFoundError(f"No report.json files found under {input_dir}")

    summary: dict[str, Any] = {"model": None, "benchmarks": {}}
    for report_path in reports:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        benchmark = report["benchmark"]
        summary["model"] = summary["model"] or report.get("model")
        summary["benchmarks"][benchmark] = {
            "num_episodes": report.get("num_episodes", 0),
            "success_rate": report.get("success_rate", 0.0),
            "videos_saved": report.get("videos_saved", 0),
            "num_errors": report.get("num_errors", 0),
            "report_path": str(report_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
