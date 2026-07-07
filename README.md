# AXIS-Eval

Unified evaluation codebase for running a trained Pi0.5 / OpenPI policy on
multiple robot simulation benchmarks with one model interface, one runner, and
one report format.

Supported benchmarks:

| Benchmark | Adapter | Example config | Video | Report |
| --- | --- | --- | --- | --- |
| LIBERO-Plus | `LiberoPlusAdapter` | `configs/libero_plus_pi05.yaml` | mp4 | JSON, CSV |
| RoboTwin2.0 | `RoboTwinAdapter` | `configs/robotwin_pi05.yaml` | mp4 | JSON, CSV |
| RoboCasa | `RoboCasaAdapter` | `configs/robocasa_pi05.yaml` | mp4 | JSON, CSV |

The main goal is not to improve success rate. The goal is to load a trained
policy checkpoint, execute real rollouts in different benchmark environments,
and save comparable artifacts:

- per-episode `episodes.jsonl`
- aggregate `report.json`
- tabular `report.csv`
- rollout videos under `videos/`
- cross-benchmark `summary.json`

## Runtime Contract

The verified runtime for this repository is:

```bash
ssh Axis-A100
cd /root/workspace/dihong/AXIS-Eval
source /root/miniconda3/etc/profile.d/conda.sh
conda activate axis-eval
```

Do not run benchmark evaluation locally. Local files can be edited and synced,
but rollouts, dependency checks, and generated outputs should run on `Axis-A100`
inside the `axis-eval` environment.

## Repository Layout

```text
unified_eval/
├── models/
│   ├── base.py              # UnifiedPolicyModel, ModelInput, ModelOutput
│   └── pi05_adapter.py      # OpenPI websocket policy adapter
├── benchmarks/
│   ├── base.py              # BenchmarkAdapter interface
│   ├── libero_plus_adapter.py
│   ├── robotwin_adapter.py
│   └── robocasa_adapter.py
├── runners/
│   └── eval_runner.py       # config-driven rollout loop
├── logging/
│   ├── result_logger.py     # episodes.jsonl, report.json, report.csv
│   └── video_logger.py      # mp4 writer
└── utils/
    ├── config.py
    └── imports.py

configs/
├── libero_plus_pi05.yaml
├── robotwin_pi05.yaml
└── robocasa_pi05.yaml

scripts/
├── eval.py
└── summarize.py
```

## Installation

The repository only contains the unified runner and adapters. Large benchmark
repositories, simulator assets, and OpenPI checkpoints are external and should
not be committed here.

Install the project-level Python utilities:

```bash
python -m pip install -r requirements.txt
```

The current `Axis-A100` environment has already been prepared with the required
benchmark dependencies and assets for evaluation. For a new machine, set up the
external dependencies first, then update the config paths:

| Component | Config field | Verified path on `Axis-A100` |
| --- | --- | --- |
| OpenPI source | `model.openpi_root` | `/root/workspace/dihong/axis-training/third_party/openpi` |
| OpenPI Python | `model.openpi_python` | `/root/miniconda3/envs/axis-training/bin/python` |
| OpenPI client | `model.openpi_client_path` | `/root/workspace/dihong/axis-training/third_party/openpi/packages/openpi-client/src` |
| LIBERO-Plus | `benchmark.libero_root` | `/root/workspace/dihong/axis-training/third_party/LIBERO-plus` |
| LIBERO config | `benchmark.libero_config_path` | `/root/workspace/dihong/axis-training/tmp/libero_plus_config` |
| RoboTwin2.0 | `benchmark.robotwin_root` | `/root/workspace/dihong/AXIS-Eval/tmp/vendor/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e` |
| RoboCasa | `benchmark.robocasa_root` | `/root/workspace/dihong/AXIS-Eval/tmp/vendor/robocasa-be22d659b02db8f6d7f3a3c3edc742934fdcbaae` |
| RoboCasa robosuite | `benchmark.robosuite_root` | `/root/workspace/dihong/AXIS-Eval/tmp/vendor/robosuite-85abee228d1c43ab1939bce33028099945d453b4` |

If remote downloads fail because proxy variables point to a bad proxy, retry
with proxy variables cleared for that command:

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    python -m pip install -r requirements.txt
```

## Using A Trained Policy

Each benchmark is driven by one YAML file. To evaluate a trained policy:

1. Choose the closest config under `configs/`.
2. Set `model.checkpoint_path` to the trained checkpoint.
3. Set `model.policy_config` to the matching OpenPI policy config.
4. Set a unique `model.port` if running several policy servers.
5. Set `experiment.output_dir` to a new output directory.
6. Set benchmark task IDs, episode count, and max steps.
7. Run `scripts/eval.py`.

Example:

```bash
python scripts/eval.py \
  --config configs/libero_plus_pi05.yaml \
  --set model.checkpoint_path=/path/to/your/checkpoint \
  --set model.policy_config=pi05_libero \
  --set experiment.output_dir=outputs/my_libero_eval \
  --set benchmark.num_episodes=5
```

The Pi0.5 adapter validates the checkpoint path. When
`model.auto_start_server: true`, it starts:

```text
<openpi_python> <openpi_root>/scripts/serve_policy.py \
  --port <port> \
  policy:checkpoint \
  --policy.config=<policy_config> \
  --policy.dir=<checkpoint_path>
```

The runner connects through the OpenPI websocket client and sends a normalized
`ModelInput`:

- RGB images as `H x W x 3`, `uint8`
- optional proprio/state vector
- language prompt
- timestep
- benchmark metadata

## Running Evaluations

Run one benchmark:

```bash
python scripts/eval.py --config configs/libero_plus_pi05.yaml
python scripts/eval.py --config configs/robotwin_pi05.yaml
python scripts/eval.py --config configs/robocasa_pi05.yaml
```

Summarize multiple benchmark outputs:

```bash
python scripts/summarize.py --input outputs/
```

## Configuration Reference

### `experiment`

```yaml
experiment:
  name: libero_plus_pi05_eval
  output_dir: outputs/libero_plus_pi05
```

- `name`: human-readable run name.
- `output_dir`: directory for config snapshots, reports, and videos. Relative
  paths are resolved from the repository root.

### `benchmark`

Common fields:

```yaml
benchmark:
  name: libero_plus
  task_ids: [0]
  num_episodes: 5
  max_episode_steps: 600
  save_video: true
  seed: 0
```

- `name`: one of `libero_plus`, `robotwin`, `robocasa`.
- `task_ids`: task list. The runner cycles through this list until
  `num_episodes` episodes have run.
- `num_episodes`: total episodes for this config invocation.
- `max_episode_steps`: per-episode horizon.
- `save_video`: enables mp4 rollout videos.
- `seed`: base seed. Episode `i` uses `seed + i`.

LIBERO-Plus-specific fields:

```yaml
benchmark:
  task_suite: libero_object
  num_steps_wait: 10
  resolution: 256
  libero_root: /root/workspace/dihong/axis-training/third_party/LIBERO-plus
  libero_config_path: /root/workspace/dihong/axis-training/tmp/libero_plus_config
```

RoboTwin2.0-specific fields:

```yaml
benchmark:
  robotwin_root: /root/workspace/dihong/AXIS-Eval/tmp/vendor/RoboTwin-...
  task_config: demo_clean
  environment_factory: null
  action_type: qpos
  action_dim: 14
```

When `environment_factory` is `null`, the adapter imports RoboTwin's official
`envs.<task_id>:<task_id>` class and calls `setup_demo(...)` with the selected
task config. You can point `environment_factory` to a custom factory using
`module.submodule:function_name`.

RoboCasa-specific fields:

```yaml
benchmark:
  env_id: robocasa/PickPlaceCounterToCabinet
  split: pretrain
  robocasa_root: /root/workspace/dihong/AXIS-Eval/tmp/vendor/robocasa-...
  robosuite_root: /root/workspace/dihong/AXIS-Eval/tmp/vendor/robosuite-...
  obj_registries: [lightwheel]
```

The example RoboCasa config uses the lightwheel object registry because those
fixtures and objects are installed in the verified remote asset directory.

### `model`

```yaml
model:
  name: pi0.5
  checkpoint_path: /data/openpi_cache/openpi-assets/checkpoints/pi05_libero
  policy_config: pi05_libero
  device: cuda:0
  host: localhost
  port: 18000
  auto_start_server: true
  openpi_root: /root/workspace/dihong/axis-training/third_party/openpi
  openpi_python: /root/miniconda3/envs/axis-training/bin/python
  openpi_client_path: /root/workspace/dihong/axis-training/third_party/openpi/packages/openpi-client/src
  server_log_path: tmp/libero_plus_pi05_server.log
  image_size: 224
  base_image_key: agentview
  left_wrist_image_key: wrist
  state_dim: 8
```

- `checkpoint_path`: trained OpenPI checkpoint directory.
- `policy_config`: OpenPI policy config name used by `serve_policy.py`.
- `device`: CUDA device used by the OpenPI policy server.
- `port`: websocket server port. Use different ports for concurrent runs.
- `server_log_path`: policy server logs. Check this first when startup fails.
- `image_size`: image size sent to OpenPI after resize-with-pad.
- `base_image_key`, `left_wrist_image_key`, `right_wrist_image_key`: camera keys
  expected by the model adapter.
- `state_dim`: proprio/state dimension passed to the policy.

### `logging`

```yaml
logging:
  save_report: true
  save_video: true
  video_fps: 10
```

## Outputs

Each benchmark run writes:

```text
outputs/<benchmark>_pi05/
├── config.yaml
├── model_info.json
├── benchmark_info.json
├── episodes.jsonl
├── report.json
├── report.csv
└── videos/
    └── <task>_seed_<seed>_episode_<episode>.mp4
```

Example episode record:

```json
{
  "benchmark": "libero_plus",
  "model": "pi0.5",
  "task_id": "0",
  "task_name": "put_the_mug_on_the_plate",
  "language": "put the mug on the plate",
  "seed": 0,
  "episode_id": 0,
  "success": false,
  "num_steps": 600,
  "max_episode_steps": 600,
  "done": false,
  "total_reward": 0.0,
  "video_path": "videos/0_seed_0_episode_0.mp4",
  "error": null
}
```

Example aggregate report:

```json
{
  "benchmark": "libero_plus",
  "model": "pi0.5",
  "num_episodes": 5,
  "num_success": 0,
  "success_rate": 0.0,
  "mean_episode_length": 600.0,
  "num_errors": 0,
  "videos_saved": 5
}
```

`scripts/summarize.py` gathers all `*/report.json` files under an input
directory and writes:

```json
{
  "model": "pi0.5",
  "benchmarks": {
    "libero_plus": {
      "num_episodes": 5,
      "success_rate": 0.0,
      "videos_saved": 5,
      "num_errors": 0,
      "report_path": "outputs/libero_plus_pi05/report.json"
    }
  }
}
```

## Error Handling

An exception in one episode does not terminate the entire benchmark run. The
runner records the error and traceback in `episodes.jsonl`, saves any frames
already collected, and continues with the next episode.

The aggregate report counts failed episodes in `num_errors`.

## Action And State Notes

This repository intentionally does not try to unify robot embodiments. Each
benchmark adapter converts the model action into the environment-native action:

- LIBERO-Plus uses a 7D continuous action.
- RoboTwin can use a 14D `qpos` action and pads shorter policy actions with
  zeros when `pad_short_actions` is enabled.
- RoboCasa's Gym wrapper uses a dict action space; the adapter splits a flat
  policy action into the required `action.*` keys.

Padding shorter actions keeps cross-embodiment rollouts runnable. It is not a
success-rate optimization.

## Troubleshooting

Checkpoint path fails:

- Confirm `model.checkpoint_path` exists on the remote machine.
- For OpenPI checkpoints, the directory should contain `params/` or checkpoint
  metadata.

Policy server fails to start:

- Check `model.server_log_path`.
- Confirm `model.port` is free.
- Confirm `model.openpi_python` and `model.openpi_root` point to the OpenPI
  environment that can run `scripts/serve_policy.py`.

Port is already open:

- The adapter refuses to attach to an unknown existing policy server.
- Use a new port or stop the stale process.

Benchmark import or asset error:

- Check the benchmark root path in the config.
- Check that benchmark assets are installed under the configured root.
- Keep large vendor trees and generated outputs under `tmp/`; do not commit
  them.

RoboCasa import checks:

- RoboCasa 1.0.1 has a hard NumPy version string check at import time. The
  adapter imports `robosuite` first, temporarily shims the NumPy version string
  during `robocasa` import, then restores the real version. Do not change the
  global NumPy install only to satisfy that import assertion without retesting
  LIBERO and RoboTwin.

GitHub or remote downloads fail:

- Verify SSH identity or proxy settings explicitly.
- If a proxy returns `403`, retry the specific command with proxy variables
  cleared instead of changing dependency versions.

## Development

To add a new model:

1. Implement `UnifiedPolicyModel`.
2. Add it to `MODELS` in `unified_eval/runners/eval_runner.py`.
3. Create a config under `configs/`.
4. Verify with a short remote validation run on `Axis-A100`.

To add a new benchmark:

1. Implement `BenchmarkAdapter`.
2. Add it to `BENCHMARKS` in `unified_eval/runners/eval_runner.py`.
3. Provide a benchmark config.
4. Verify observation conversion, action conversion, video rendering, and report
   writing with a short remote validation run.

Before committing:

```bash
python -m compileall -q unified_eval scripts
git status --short
```

Generated outputs, videos, logs, caches, benchmark assets, and vendor checkouts
belong in `tmp/` or `outputs/` and should stay out of Git.
