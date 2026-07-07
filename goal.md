# 项目需求书：Unified Evaluation Codebase for LIBERO-Plus, RoboTwin2.0, RoboCasa

## 1. 项目目标

搭建一套统一的机器人策略评测代码库，支持在 **LIBERO-Plus、RoboTwin2.0、RoboCasa** 三个 benchmark 上使用同一个模型接口进行 evaluation。

本项目不关注训练、不关注 success rate 优化、不关注 embodiment 统一。
核心目标是：

```text
加载一个 Pi0.5 checkpoint，在三个 benchmark 上完成 rollout，并输出统一格式的 report 和 rollout video。
```

---

## 2. 核心需求

项目需要完成以下功能：

1. 统一模型接口 `UnifiedPolicyModel`
2. 统一 benchmark 接口 `BenchmarkAdapter`
3. 接入三个 benchmark：

   * LIBERO-Plus
   * RoboTwin2.0
   * RoboCasa
4. 接入 Pi0.5 模型 checkpoint
5. 支持统一 evaluation runner
6. 支持 rollout video 保存
7. 支持统一 report 输出
8. 支持 YAML 配置启动评测

---

## 3. 非目标

本项目不需要完成：

1. 不需要训练 Pi0.5
2. 不需要 fine-tune 模型
3. 不需要保证三个 benchmark 上 success rate 高
4. 不需要统一 robot embodiment
5. 不需要支持真实机器人
6. 不需要做 distributed evaluation
7. 不需要做复杂 failure analysis
8. 不需要接入多个模型，第一版只接 Pi0.5

---

## 4. 项目结构

```text
unified_eval/
│
├── models/
│   ├── base.py
│   └── pi05_adapter.py
│
├── benchmarks/
│   ├── base.py
│   ├── libero_plus_adapter.py
│   ├── robotwin_adapter.py
│   └── robocasa_adapter.py
│
├── runners/
│   └── eval_runner.py
│
├── logging/
│   ├── result_logger.py
│   └── video_logger.py
│
├── configs/
│   ├── libero_plus_pi05.yaml
│   ├── robotwin_pi05.yaml
│   └── robocasa_pi05.yaml
│
├── scripts/
│   ├── eval.py
│   └── summarize.py
│
└── outputs/
```

---

## 5. 统一 Model Interface

所有模型都需要继承同一个接口。

```python
class UnifiedPolicyModel:
    def reset(self, task: str | None = None, seed: int | None = None):
        pass

    def predict(self, obs: "ModelInput") -> "ModelOutput":
        pass

    def get_model_info(self) -> dict:
        pass
```

### ModelInput

```python
@dataclass
class ModelInput:
    images: dict[str, np.ndarray]
    proprio: np.ndarray | None = None
    language: str | None = None
    timestep: int | None = None
    meta: dict = field(default_factory=dict)
```

要求：

```text
images: 多相机 RGB 图像，格式为 H x W x 3，RGB，uint8
proprio: 机器人状态，可选
language: 当前任务语言指令
timestep: 当前 episode step
meta: benchmark、task_id、camera 名称等额外信息
```

### ModelOutput

```python
@dataclass
class ModelOutput:
    actions: np.ndarray
    info: dict = field(default_factory=dict)
```

要求：

```text
actions 可以是单步 action，也可以是 action chunk
shape: [action_dim] 或 [horizon, action_dim]
```

---

## 6. Pi0.5 Adapter

需要实现：

```text
models/pi05_adapter.py
```

功能包括：

1. 加载 Pi0.5 checkpoint
2. 接收统一格式 `ModelInput`
3. 将 RGB image、proprio、language 转换为 Pi0.5 所需输入
4. 调用 Pi0.5 inference
5. 输出统一格式 `ModelOutput`
6. 支持 action chunk
7. 支持每个 episode reset
8. 记录模型信息

示例接口：

```python
class Pi05Adapter(UnifiedPolicyModel):
    def __init__(self, checkpoint_path: str, device: str = "cuda", **kwargs):
        pass

    def reset(self, task: str | None = None, seed: int | None = None):
        pass

    def predict(self, obs: ModelInput) -> ModelOutput:
        pass

    def get_model_info(self) -> dict:
        return {
            "model_name": "pi0.5",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
        }
```

---

## 7. 统一 Benchmark Interface

三个 benchmark 都需要实现同一个接口。

```python
class BenchmarkAdapter:
    def reset(self, task_id: str | None = None, seed: int | None = None):
        pass

    def step(self, action: np.ndarray):
        pass

    def convert_obs_to_model_input(self, raw_obs, timestep: int) -> ModelInput:
        pass

    def convert_action_to_env_action(self, action: np.ndarray) -> np.ndarray:
        pass

    def get_success(self, raw_obs, reward, done, info) -> bool:
        pass

    def get_task_info(self) -> dict:
        pass

    def render_frame(self) -> np.ndarray:
        pass

    def close(self):
        pass
```

---

## 8. 三个 Benchmark Adapter

需要分别实现：

```text
benchmarks/libero_plus_adapter.py
benchmarks/robotwin_adapter.py
benchmarks/robocasa_adapter.py
```

每个 adapter 负责：

1. 初始化对应 benchmark 环境
2. 加载 task
3. reset 环境
4. step 环境
5. 获取 RGB 图像
6. 获取 proprio
7. 获取 language instruction
8. 转换为统一 `ModelInput`
9. 将模型输出 action 转换为环境 action
10. 获取 success 信息
11. 提供 rollout video frame

注意：
不同 benchmark 的 action space 不需要强行统一。每个 benchmark adapter 自己负责把 Pi0.5 输出的 action 转成当前环境可以执行的 action。

---

## 9. Eval Runner

需要实现：

```text
runners/eval_runner.py
```

功能：

1. 读取 config
2. 初始化 benchmark adapter
3. 初始化 Pi0.5 model adapter
4. 对指定 task 进行 rollout
5. 保存每个 episode 的结果
6. 保存 rollout video
7. 生成 report

核心流程：

```python
obs = benchmark.reset(task_id=task_id, seed=seed)
model.reset(task=task_language, seed=seed)

for t in range(max_episode_steps):
    model_input = benchmark.convert_obs_to_model_input(obs, timestep=t)
    model_output = model.predict(model_input)

    action = model_output.actions
    if action.ndim == 2:
        action = action[0]

    env_action = benchmark.convert_action_to_env_action(action)
    obs, reward, done, info = benchmark.step(env_action)

    frame = benchmark.render_frame()
    video_logger.add_frame(frame)

    success = benchmark.get_success(obs, reward, done, info)

    if done:
        break
```

---

## 10. 配置文件

每个 benchmark 一个配置文件。

### LIBERO-Plus

```yaml
experiment:
  name: libero_plus_pi05_eval
  output_dir: outputs/libero_plus_pi05

benchmark:
  name: libero_plus
  task_suite: libero_plus
  task_ids: null
  num_episodes: 5
  max_episode_steps: 600
  save_video: true

model:
  name: pi0.5
  checkpoint_path: /path/to/pi05/checkpoint
  device: cuda

logging:
  save_report: true
  save_video: true
```

### RoboTwin2.0

```yaml
experiment:
  name: robotwin_pi05_eval
  output_dir: outputs/robotwin_pi05

benchmark:
  name: robotwin
  task_suite: default
  task_ids: null
  num_episodes: 5
  max_episode_steps: 600
  save_video: true

model:
  name: pi0.5
  checkpoint_path: /path/to/pi05/checkpoint
  device: cuda

logging:
  save_report: true
  save_video: true
```

### RoboCasa

```yaml
experiment:
  name: robocasa_pi05_eval
  output_dir: outputs/robocasa_pi05

benchmark:
  name: robocasa
  task_suite: default
  task_ids: null
  num_episodes: 5
  max_episode_steps: 1000
  save_video: true

model:
  name: pi0.5
  checkpoint_path: /path/to/pi05/checkpoint
  device: cuda

logging:
  save_report: true
  save_video: true
```

---

## 11. CLI

需要支持以下命令：

```bash
python scripts/eval.py --config configs/libero_plus_pi05.yaml
python scripts/eval.py --config configs/robotwin_pi05.yaml
python scripts/eval.py --config configs/robocasa_pi05.yaml
```

也需要支持统一运行三个 benchmark：

```bash
python scripts/eval.py --config configs/libero_plus_pi05.yaml
python scripts/eval.py --config configs/robotwin_pi05.yaml
python scripts/eval.py --config configs/robocasa_pi05.yaml
python scripts/summarize.py --input outputs/
```

---

## 12. 输出结果

每个 benchmark evaluation 后需要生成：

```text
outputs/
└── benchmark_name_pi05/
    ├── config.yaml
    ├── model_info.json
    ├── benchmark_info.json
    ├── episodes.jsonl
    ├── report.json
    ├── report.csv
    └── videos/
        ├── task_000_seed_0_episode_0.mp4
        ├── task_001_seed_0_episode_0.mp4
        └── ...
```

---

## 13. Episode Report 格式

每个 episode 保存一条记录：

```json
{
  "benchmark": "libero_plus",
  "model": "pi0.5",
  "task_id": "task_000",
  "task_name": "put_the_mug_on_the_plate",
  "language": "put the mug on the plate",
  "seed": 0,
  "episode_id": 0,
  "success": false,
  "num_steps": 600,
  "max_episode_steps": 600,
  "done": false,
  "total_reward": 0.0,
  "video_path": "videos/task_000_seed_0_episode_0.mp4",
  "error": null
}
```

说明：

```text
success 可以记录，但不作为验收重点。
验收重点是 rollout 是否完成、report 是否输出、video 是否保存。
```

---

## 14. 总 Report 格式

每个 benchmark 输出一个 `report.json`：

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

最终 `summarize.py` 需要汇总三个 benchmark：

```json
{
  "model": "pi0.5",
  "benchmarks": {
    "libero_plus": {
      "num_episodes": 5,
      "success_rate": 0.0,
      "videos_saved": 5
    },
    "robotwin": {
      "num_episodes": 5,
      "success_rate": 0.0,
      "videos_saved": 5
    },
    "robocasa": {
      "num_episodes": 5,
      "success_rate": 0.0,
      "videos_saved": 5
    }
  }
}
```

---

## 15. Error Handling

评测过程中某个 episode 出错时，不应直接终止整个 benchmark。

需要记录：

```json
{
  "benchmark": "robotwin",
  "task_id": "task_003",
  "episode_id": 0,
  "error": "action_dim_mismatch",
  "traceback": "...",
  "video_path": "videos/task_003_seed_0_episode_0.mp4"
}
```

要求：

1. 单个 episode 出错后继续下一个 episode
2. 错误写入 `episodes.jsonl`
3. 已经生成的视频仍然保存
4. report 中统计 `num_errors`

---

## 16. 验收标准

项目完成后，以下命令必须能运行：

```bash
python scripts/eval.py --config configs/libero_plus_pi05.yaml
python scripts/eval.py --config configs/robotwin_pi05.yaml
python scripts/eval.py --config configs/robocasa_pi05.yaml
python scripts/summarize.py --input outputs/
```

运行完成后必须得到：

```text
outputs/libero_plus_pi05/report.json
outputs/libero_plus_pi05/report.csv
outputs/libero_plus_pi05/videos/*.mp4

outputs/robotwin_pi05/report.json
outputs/robotwin_pi05/report.csv
outputs/robotwin_pi05/videos/*.mp4

outputs/robocasa_pi05/report.json
outputs/robocasa_pi05/report.csv
outputs/robocasa_pi05/videos/*.mp4

outputs/summary.json
```

验收重点：

1. 三个 benchmark 都可以初始化
2. Pi0.5 checkpoint 可以加载
3. 每个 benchmark 至少完成若干 rollout
4. 每个 rollout 可以保存视频
5. 每个 benchmark 可以输出 report
6. 最终可以汇总三个 benchmark 的结果
7. 不要求 success rate 达到任何数值

---

## 17. 最终交付物

最终需要交付：

```text
1. 统一 evaluation codebase
2. LIBERO-Plus adapter
3. RoboTwin2.0 adapter
4. RoboCasa adapter
5. Pi0.5 model adapter
6. EvalRunner
7. Video logger
8. Result logger
9. 三个 benchmark 的 config
10. summarize.py 汇总脚本
11. README 使用说明
```

README 至少包含：

```text
1. 如何安装依赖
2. 如何配置 Pi0.5 checkpoint
3. 如何分别运行三个 benchmark
4. 输出文件在哪里
5. 如何查看 rollout video
6. 如何查看 report
```

---

## 18. 一句话总结

本项目的目标是：

```text
用统一的模型接口加载 Pi0.5 checkpoint，
在 LIBERO-Plus、RoboTwin2.0、RoboCasa 三个 benchmark 上完成 rollout，
并统一输出 report 和 rollout video。
```
