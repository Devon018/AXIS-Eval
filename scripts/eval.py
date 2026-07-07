#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_eval.runners.eval_runner import EvalRunner
from unified_eval.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified Pi0.5 evaluation on one benchmark.")
    parser.add_argument("--config", required=True, help="Path to a YAML evaluation config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Optional dotted override, e.g. --set benchmark.num_episodes=1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    EvalRunner(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
