# xarm-aio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 xarm 右臂（x_air 硬件栈）构建对接官方 openpi websocket 推理服务的推理客户端工程，并交付 openpi 侧 transforms/配置示例。

**Architecture:** 分层结构对标 piper-aio：硬件层 `xarm_operator.py`（xarm_can CAN 直控 + ROS2 相机订阅），推理层 `inference/`（OpenpiClient + chunk 同步主循环 + 动作滤波），`openpi_config/` 交付给官方 openpi 仓库的 policy transforms 与训练配置片段。全程单臂 8 维（7 关节 + 1 夹爪，弧度，绝对位置）。

**Tech Stack:** Python 3.10+，numpy，opencv-python，pyyaml，matplotlib，openpi-client（官方），rclpy（系统 ROS2），xarm_can（x_air 编译产物），pytest。

**Spec:** `docs/superpowers/specs/2026-07-02-xarm-openpi-inference-design.md`

## Global Constraints

- 工程根目录：`/home/lft-vlai2/Documents/csy/xarm-aio`（已 git init）。所有相对路径以此为根。
- state/action 恒为 8 维：`[joint1..joint7, gripper]`，单位弧度，绝对位置。夹爪范围 -1.0（全开）~ 0.0（闭合）。
- 关节限位（来自 x_air URDF）：lower `[-1.3, -0.1, -1.5, 0.0, -1.5, -0.7, -1.5]`，upper `[3.4, 1.7, 1.5, 2.4, 1.5, 0.7, 1.5]`。
- MIT 控制增益：KP `[240, 240, 240, 240, 24, 31, 25]`，KD `[3, 3, 3, 3, 0.2, 0.2, 0.2]`；夹爪 KP 16.0 / KD 0.3；回 home 时夹爪 KP 10.0 / KD 0.5。
- 网络观测格式（客户端 → 服务端）：`{"state": float32 (8,), "images": {"cam_high": uint8 (3,224,224), "cam_right_wrist": uint8 (3,224,224)}, "prompt": str}`；服务端返回 `{"actions": (chunk,8+)}`，客户端取 `[:, :8]`。
- 相机 topic 默认值：`/cam_chest/cam_chest/color/image_raw`、`/cam_wrist_right/cam_wrist_right/color/image_rect_raw`。
- 测试文件与模块同目录（piper-aio 惯例），从 `inference/` 目录内用相对导入运行 pytest。
- 不引入 pistar 相关内容（无 `adv_ind`、无 rollout HDF5 录制）。
- 提交信息末尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

## 依赖环境说明（执行前读）

- `rclpy`/`sensor_msgs` 来自系统 ROS2 环境（x_air 使用的同一套），运行前 `source /opt/ros/<distro>/setup.bash`。**单元测试不依赖 ROS2**（相机类的测试只测 dry-run 假实现）。
- `xarm_can` 是 x_air 编译的 C++ 扩展，仅真机模式 import（函数内延迟导入）。**单元测试不依赖它**。
- `openpi-client` 安装：`pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"`。Task 2 的测试需要它（用其 `image_tools`），若安装失败改为本地复制 `resize_with_pad`/`convert_to_uint8`（见 Task 2 备注）。

## File Structure

```
xarm-aio/
├── requirements.txt
├── .gitignore
├── README.md                       # Task 7
├── xarm_operator.py                # Task 4：硬件层（CAN + 相机）
├── test_xarm_operator.py           # Task 4
├── inference/
│   ├── action_filter.py            # Task 1：EMA/滑窗/chunk 过渡插值（单臂 8 维版）
│   ├── test_action_filter.py       # Task 1
│   ├── clients.py                  # Task 2：OpenpiClient
│   ├── test_clients.py             # Task 2
│   ├── utils.py                    # Task 3：任务配置加载/夹爪后处理/键盘交互
│   ├── test_utils.py               # Task 3
│   ├── task_configs.yaml           # Task 3
│   ├── infer_sync.py               # Task 5：主循环
│   └── test_infer_smoke.py         # Task 5：dry-run 冒烟测试
└── openpi_config/
    ├── xarm_policy.py              # Task 6：XarmInputs/Outputs（拷入 openpi 仓库用）
    └── README.md                   # Task 6：TrainConfig 片段 + 训练/serve 命令
```

---

### Task 1: 项目骨架 + 动作滤波模块（单臂 8 维）

**Files:**
- Create: `requirements.txt`, `.gitignore`
- Create: `inference/action_filter.py`
- Test: `inference/test_action_filter.py`

**Interfaces:**
- Produces:
  - `ActionFilter(alpha: float)`，方法 `reset()`、`apply(action: np.ndarray) -> np.ndarray`（EMA，alpha≥1.0 直通）
  - `MovingAverageFilter(window: int)`，同上接口（window≤1 直通）
  - `interpolate_chunk_transition(prev_action, raw_action, step_idx: int, total_steps: int) -> np.ndarray`
  - `plot_filter_comparison(raw, filt, alpha, save_path)`、`save_filter_trajectories(raw, filt, alpha, save_path)`（单臂版，raw/filt 形状 `(steps, 8)`）
  - `DIM_LABELS = ["joint1", ..., "joint7", "gripper"]`

来源：`piper-aio/inference/action_filter.py`，改动点：去掉 left/right 双臂参数（单臂只传一个 action 数组）、维度 7→8、`DIM_LABELS` 改为 7 关节 + gripper。

- [ ] **Step 1: 建骨架文件**

`requirements.txt`：

```
numpy
opencv-python
pyyaml
matplotlib
pytest
openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client
```

`.gitignore`：

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 2: 写失败测试** `inference/test_action_filter.py`

```python
import numpy as np

from action_filter import (
    ActionFilter,
    MovingAverageFilter,
    interpolate_chunk_transition,
    plot_filter_comparison,
    save_filter_trajectories,
)


def test_ema_passthrough_when_alpha_one():
    f = ActionFilter(alpha=1.0)
    action = np.arange(8, dtype=np.float64)
    out = f.apply(action)
    np.testing.assert_array_equal(out, action)


def test_ema_smooths_second_sample():
    f = ActionFilter(alpha=0.5)
    first = np.zeros(8)
    second = np.ones(8)
    f.apply(first)
    out = f.apply(second)
    np.testing.assert_allclose(out, np.full(8, 0.5))


def test_ema_reset_clears_state():
    f = ActionFilter(alpha=0.5)
    f.apply(np.zeros(8))
    f.reset()
    out = f.apply(np.ones(8))
    np.testing.assert_allclose(out, np.ones(8))


def test_moving_average_passthrough_when_window_one():
    f = MovingAverageFilter(window=1)
    action = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(f.apply(action), action)


def test_moving_average_window_math():
    f = MovingAverageFilter(window=3)
    f.apply(np.zeros(8))
    f.apply(np.ones(8))
    out = f.apply(np.full(8, 2.0))
    np.testing.assert_allclose(out, np.ones(8))  # mean(0,1,2)


def test_interpolate_chunk_transition_first_step():
    prev = np.zeros(8)
    raw = np.ones(8)
    out = interpolate_chunk_transition(prev, raw, step_idx=0, total_steps=5)
    np.testing.assert_allclose(out, np.full(8, 0.2))


def test_interpolate_chunk_transition_last_step_equals_raw():
    prev = np.zeros(8)
    raw = np.ones(8)
    out = interpolate_chunk_transition(prev, raw, step_idx=4, total_steps=5)
    np.testing.assert_allclose(out, raw)


def test_plot_and_save_trajectories(tmp_path):
    steps = 5
    raw = np.random.rand(steps, 8)
    filt = np.random.rand(steps, 8)
    png = tmp_path / "cmp.png"
    npz = tmp_path / "cmp.npz"

    plot_filter_comparison(raw, filt, alpha=0.5, save_path=str(png))
    save_filter_trajectories(raw, filt, alpha=0.5, save_path=str(npz))

    assert png.exists() and png.stat().st_size > 0
    data = np.load(npz)
    np.testing.assert_array_equal(data["raw"], raw)
    np.testing.assert_array_equal(data["filt"], filt)
    assert float(data["alpha"]) == 0.5
```

- [ ] **Step 3: 运行确认失败**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_action_filter.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'action_filter'`

- [ ] **Step 4: 实现** `inference/action_filter.py`

```python
from collections import deque

import numpy as np


class MovingAverageFilter:
    """Sliding-window moving average filter applied to published actions.

    `window == 1` is a passthrough (no filtering). Larger windows smooth more
    but add lag (lag ≈ window/2 steps).
    """

    def __init__(self, window: int):
        self.window = max(1, window)
        self.buf = None

    def reset(self):
        self.buf = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self.window <= 1:
            return action
        if self.buf is None:
            self.buf = deque([action.astype(np.float64).copy()], maxlen=self.window)
        else:
            self.buf.append(action.astype(np.float64).copy())
        return np.mean(self.buf, axis=0)


class ActionFilter:
    """Exponential moving average filter applied to published actions.

    `alpha == 1.0` is a passthrough (no filtering). Smaller `alpha` smooths
    more but adds lag. The same `alpha` is applied to every dimension
    (arm joints and gripper).
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self.alpha >= 1.0:
            return action
        if self.prev is None:
            self.prev = action.astype(np.float64).copy()
        else:
            self.prev = self.alpha * action + (1 - self.alpha) * self.prev
        return self.prev.copy()


def interpolate_chunk_transition(prev_action, raw_action, step_idx, total_steps):
    """Linearly ramp from `prev_action` toward `raw_action`.

    `step_idx` is 0-indexed within the transition window; at
    `step_idx == total_steps - 1` the result equals `raw_action`.
    """
    weight = (step_idx + 1) / total_steps
    return (1 - weight) * prev_action + weight * raw_action


DIM_LABELS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]


def save_filter_trajectories(raw, filt, alpha, save_path):
    """Save raw/filtered action trajectories (steps, 8) to an `.npz` file."""
    np.savez(save_path, raw=raw, filt=filt, alpha=np.array(alpha))


def plot_filter_comparison(raw, filt, alpha, save_path):
    """Save a raw-vs-filtered action comparison plot. `raw`/`filt`: (steps, 8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(DIM_LABELS)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), squeeze=False)
    steps = np.arange(raw.shape[0])
    for col in range(ncols):
        ax = axes[0][col]
        ax.plot(steps, raw[:, col], "k--", label="raw")
        ax.plot(steps, filt[:, col], "b-", label="filtered")
        ax.set_title(DIM_LABELS[col])
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Action filter comparison (alpha={alpha})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 5: 运行确认通过**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_action_filter.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add requirements.txt .gitignore inference/action_filter.py inference/test_action_filter.py
git commit -m "feat: add project scaffold and single-arm action filters"
```

---

### Task 2: OpenpiClient（官方 openpi websocket 客户端封装）

**Files:**
- Create: `inference/clients.py`
- Test: `inference/test_clients.py`

**Interfaces:**
- Consumes: 官方 `openpi_client`（`image_tools`, `websocket_client_policy`）
- Produces:
  - `STATE_DIM = 8`
  - `build_observation(payload: dict) -> dict`：payload 为 `{"cam_high": HWC uint8, "cam_right_wrist": HWC uint8, "state": (8,), "instruction": str}`，返回网络观测（见 Global Constraints 格式）
  - `OpenpiClient(host: str, port: int)`，方法 `predict_action(payload) -> np.ndarray (chunk,8)`、`warmup() -> None`

备注：若 openpi-client 安装失败，把 `piper-aio/inference/utils.py` 里的 `resize_with_pad`/`convert_to_uint8`（第 352-380 行）复制进 `clients.py` 替代 `image_tools`，其余不变。

- [ ] **Step 1: 写失败测试** `inference/test_clients.py`

```python
import numpy as np

from clients import STATE_DIM, build_observation, OpenpiClient


def _payload():
    return {
        "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "instruction": "pick up the object",
    }


def test_build_observation_format():
    obs = build_observation(_payload())

    assert set(obs.keys()) == {"state", "images", "prompt"}
    assert obs["state"].shape == (STATE_DIM,)
    assert obs["state"].dtype == np.float32
    assert set(obs["images"].keys()) == {"cam_high", "cam_right_wrist"}
    for img in obs["images"].values():
        assert img.shape == (3, 224, 224)
        assert img.dtype == np.uint8
    assert obs["prompt"] == "pick up the object"


class _FakeWsClient:
    def __init__(self):
        self.received = []

    def infer(self, obs):
        self.received.append(obs)
        return {"actions": np.ones((50, 32), dtype=np.float32)}


def test_predict_action_truncates_to_state_dim():
    client = OpenpiClient.__new__(OpenpiClient)  # 跳过 __init__ 的真实连接
    client.client = _FakeWsClient()

    actions = client.predict_action(_payload())

    assert actions.shape == (50, STATE_DIM)


def test_warmup_sends_valid_observation():
    client = OpenpiClient.__new__(OpenpiClient)
    fake = _FakeWsClient()
    client.client = fake

    client.warmup()

    obs = fake.received[0]
    assert obs["state"].shape == (STATE_DIM,)
    assert set(obs["images"].keys()) == {"cam_high", "cam_right_wrist"}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_clients.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'clients'`（若报 `No module named 'openpi_client'`，先 `pip install -r ../requirements.txt`）

- [ ] **Step 3: 实现** `inference/clients.py`

```python
import numpy as np
from openpi_client import image_tools, websocket_client_policy

STATE_DIM = 8
IMAGE_KEYS = ("cam_high", "cam_right_wrist")


def build_observation(payload: dict) -> dict:
    """Convert raw robot payload into the wire observation for the openpi server.

    payload keys: "cam_high"/"cam_right_wrist" (HWC uint8), "state" (8,), "instruction".
    """
    images = {}
    for name in IMAGE_KEYS:
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(payload[name], 224, 224))
        images[name] = img.transpose(2, 0, 1)

    return {
        "state": np.asarray(payload["state"], dtype=np.float32),
        "images": images,
        "prompt": payload["instruction"],
    }


def _random_observation() -> dict:
    return {
        "state": np.ones((STATE_DIM,), dtype=np.float32),
        "images": {
            name: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
            for name in IMAGE_KEYS
        },
        "prompt": "do something",
    }


class OpenpiClient:
    def __init__(self, host: str, port: int) -> None:
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)

    def predict_action(self, payload: dict) -> np.ndarray:
        response = self.client.infer(build_observation(payload))
        actions = np.asarray(response["actions"])
        return actions[:, :STATE_DIM]

    def warmup(self) -> None:
        self.client.infer(_random_observation())
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_clients.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add inference/clients.py inference/test_clients.py
git commit -m "feat: add openpi websocket client wrapper for xarm"
```

---

### Task 3: 任务配置 + 夹爪后处理 + 键盘交互

**Files:**
- Create: `inference/task_configs.yaml`
- Create: `inference/utils.py`
- Test: `inference/test_utils.py`

**Interfaces:**
- Produces:
  - `get_config(args) -> dict`：需要 `args.task`, `args.max_publish_step`, `args.chunk_size`；返回 keys：`episode_len`, `state_dim`(恒 8), `right0`(list[8]), `action_postprocess`, `task`, `language_instruction`, `chunk_size`, `action_filter_alpha`(默认 1.0), `action_filter_type`(默认 "ema"), `action_filter_window`(默认 5), `chunk_transition_steps`(默认 0)
  - `process_action(task: str, action: np.ndarray (8,)) -> np.ndarray (8,)`：按任务配置对 `action[7]`（夹爪）应用 below/above → set/add 规则
  - `check_keyboard_input() -> str | None`（非阻塞读一个字符）
  - `handle_interactive_mode(task_time: float) -> "continue" | "reset" | "quit"`

来源：`piper-aio/inference/utils.py` 的对应函数。改动：单臂 8 维、夹爪索引 6→7、去掉 `_apply_gripper_rules` 里 piper 特有的 `max(0, ...)` 钳位（xarm 夹爪负值为张开，钳位交给硬件层 `clip_action`）。

- [ ] **Step 1: 写** `inference/task_configs.yaml`

```yaml
# 每个任务一个条目。right0 为 8 维 home 位（7 关节 + 夹爪，弧度）。
# 可选字段：
#   action_filter_alpha: 0.5        # EMA，(0,1]，1.0=不滤波，越小越平滑
#   action_filter_type: moving_average  # 配合 action_filter_window 使用
#   action_filter_window: 5
#   chunk_transition_steps: 10      # chunk 边界线性过渡步数，0=禁用
#   action_postprocess:             # 夹爪后处理规则（作用于 action[7]）
#     gripper:
#       - when: below / above
#         threshold: -0.5
#         set: -1.0                 # 或 add: 0.1

example_task:
  language_instruction: "pick up the object"
  right0: [-0.1413366903181501, 0.14400701915007197, -0.2534905012588702,
           0.8703364614328226, 0.012397955291065799, 0.12722209506370596,
           0.9061951628900591, 0.0]
  chunk_transition_steps: 10
```

（`right0` 前 7 维取自 `x_air/src/lerobot_collector/xarm_deploy_direct.py` 的 `initial_positions`，夹爪 0.0=闭合。）

- [ ] **Step 2: 写失败测试** `inference/test_utils.py`

```python
import argparse

import numpy as np
import pytest

from utils import get_config, process_action


def _args(task="example_task"):
    return argparse.Namespace(task=task, max_publish_step=1000, chunk_size=50)


def test_get_config_returns_required_fields():
    config = get_config(_args())

    assert config["state_dim"] == 8
    assert len(config["right0"]) == 8
    assert config["language_instruction"] == "pick up the object"
    assert config["chunk_size"] == 50
    assert config["episode_len"] == 1000
    assert config["action_filter_alpha"] == 1.0
    assert config["action_filter_type"] == "ema"
    assert config["chunk_transition_steps"] == 10


def test_get_config_unknown_task_raises():
    with pytest.raises(ValueError):
        get_config(_args(task="no_such_task"))


def test_process_action_gripper_rule_below_set(monkeypatch):
    import utils

    monkeypatch.setitem(
        utils.TASK_CONFIGS,
        "gripper_rule_task",
        {
            "language_instruction": "x",
            "right0": [0.0] * 8,
            "action_postprocess": {
                "gripper": [{"when": "below", "threshold": -0.5, "set": -1.0}]
            },
        },
    )
    action = np.zeros(8)
    action[7] = -0.6

    out = process_action("gripper_rule_task", action)

    assert out[7] == -1.0
    assert np.all(out[:7] == 0.0)


def test_process_action_no_rules_is_identity():
    action = np.arange(8, dtype=np.float64)
    out = process_action("example_task", action)
    np.testing.assert_array_equal(out, action)
```

- [ ] **Step 3: 运行确认失败**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_utils.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 4: 实现** `inference/utils.py`

```python
import select
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TASK_CONFIG_PATH = Path(__file__).with_name("task_configs.yaml")


def _load_task_configs():
    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as file:
        task_configs = yaml.safe_load(file)
    if not isinstance(task_configs, dict):
        raise ValueError(f"Invalid task config format in {TASK_CONFIG_PATH}")
    return task_configs


TASK_CONFIGS = _load_task_configs()


def get_config(args):
    task_config = TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {args.task}")

    language_instruction = task_config.get("language_instruction")
    right0 = task_config.get("right0")
    if language_instruction is None or right0 is None:
        raise ValueError(f"Task config for {args.task} is missing required fields")
    if len(right0) != 8:
        raise ValueError(f"right0 for {args.task} must have 8 dims, got {len(right0)}")

    return {
        "episode_len": args.max_publish_step,
        "state_dim": 8,
        "right0": right0,
        "action_postprocess": task_config.get("action_postprocess", {}),
        "task": args.task,
        "language_instruction": language_instruction,
        "chunk_size": args.chunk_size,
        "action_filter_alpha": task_config.get("action_filter_alpha", 1.0),
        "action_filter_type": task_config.get("action_filter_type", "ema"),
        "action_filter_window": task_config.get("action_filter_window", 5),
        "chunk_transition_steps": task_config.get("chunk_transition_steps", 0),
    }


def _apply_gripper_rules(gripper_value, rules):
    for rule in rules:
        condition = rule.get("when")
        threshold = rule.get("threshold")
        if condition == "below" and not gripper_value < threshold:
            continue
        if condition == "above" and not gripper_value > threshold:
            continue

        if "set" in rule:
            gripper_value = rule["set"]
        if "add" in rule:
            gripper_value += rule["add"]
    return gripper_value


def process_action(task, action):
    action = action.copy()
    task_config = TASK_CONFIGS.get(task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {task}")
    rules = task_config.get("action_postprocess", {}).get("gripper", [])
    action[7] = _apply_gripper_rules(action[7], rules)
    return action


def check_keyboard_input():
    """Check if a key was pressed without blocking.

    Returns None when stdin is not a tty (e.g. under pytest), where
    select() on the replaced stdin object would raise.
    """
    try:
        if not sys.stdin.isatty():
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except (OSError, ValueError):
        return None
    return None


def handle_interactive_mode(task_time):
    """Handle interactive mode when space is pressed.

    Returns: 'continue' to resume, 'reset' to restart, 'quit' to stop.
    """
    print("\n" + "=" * 50)
    print(f"Task time: {time.time() - task_time:.1f} s")
    print("INTERACTIVE MODE")
    print("  'c' - Continue running")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    print("=" * 50)

    while True:
        key = sys.stdin.read(1).lower()
        if key == "c":
            print("Continuing...")
            return "continue"
        elif key == "r":
            print("Restarting...")
            return "reset"
        elif key == "q":
            print("Stopping...")
            return "quit"
```

- [ ] **Step 5: 运行确认通过**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_utils.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add inference/task_configs.yaml inference/utils.py inference/test_utils.py
git commit -m "feat: add task configs, gripper postprocess and keyboard helpers"
```

---

### Task 4: 硬件层 `xarm_operator.py`（CAN 直控 + 相机 + dry-run）

**Files:**
- Create: `xarm_operator.py`（工程根目录）
- Test: `test_xarm_operator.py`（工程根目录）

**Interfaces:**
- Produces:
  - `clip_action(action: np.ndarray (8,)) -> np.ndarray (8,)`（纯函数：关节 + 夹爪限位）
  - `HOME_POSITION: np.ndarray (8,)`
  - `XarmOperator(can_interface: str = "can1", dry_run: bool = False)`：
    - `read_state() -> np.ndarray (8,)`
    - `send_action(action: (8,), kp=None, kd=None) -> None`（内部先 `clip_action`）
    - `go_home(target: (8,) | None = None, nstep: int = 220, step_dt: float = 0.01) -> None`
    - `shutdown() -> None`
  - `CameraStreamer(topic_map: dict[str, str])`：`get_images(max_age_s: float = 0.5) -> dict[str, np.ndarray HWC uint8 RGB] | None`、`stop()`
  - `FakeCameraStreamer()`：同接口，返回随机 (480,640,3) uint8 图像

来源：`x_air/src/lerobot_collector/xarm_deploy_direct.py`（电机初始化、MIT 发送、回 home、ROS 图像转换）。`xarm_can` 与 `rclpy` 都在**方法/构造函数内延迟导入**，dry-run 路径完全不 import 它们。相机返回 HWC uint8 RGB（不做 CHW/归一化——那是 `clients.build_observation` 的职责）。

- [ ] **Step 1: 写失败测试** `test_xarm_operator.py`

```python
import numpy as np

from xarm_operator import (
    HOME_POSITION,
    FakeCameraStreamer,
    XarmOperator,
    clip_action,
)


def test_clip_action_bounds():
    action = np.array([-5.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5])
    out = clip_action(action)

    assert out[0] == -1.3   # joint1 lower
    assert out[1] == 1.7    # joint2 upper
    assert out[3] == 1.0    # 未越界不变
    assert out[7] == 0.0    # gripper upper


def test_clip_action_gripper_lower():
    action = np.zeros(8)
    action[7] = -2.0
    assert clip_action(action)[7] == -1.0


def test_dry_run_send_then_read_roundtrip():
    op = XarmOperator(dry_run=True)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, -0.5])

    op.send_action(target)
    state = op.read_state()

    assert state.shape == (8,)
    np.testing.assert_allclose(state, target)


def test_dry_run_send_clips_before_apply():
    op = XarmOperator(dry_run=True)
    action = np.zeros(8)
    action[0] = -99.0

    op.send_action(action)

    assert op.read_state()[0] == -1.3


def test_go_home_dry_run_reaches_target():
    op = XarmOperator(dry_run=True)
    op.send_action(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5]))

    op.go_home(nstep=5, step_dt=0.0)

    np.testing.assert_allclose(op.read_state(), HOME_POSITION, atol=1e-9)


def test_fake_camera_streamer_returns_images():
    cams = FakeCameraStreamer()
    images = cams.get_images()

    assert set(images.keys()) == {"cam_high", "cam_right_wrist"}
    for img in images.values():
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
    cams.stop()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python -m pytest test_xarm_operator.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'xarm_operator'`

- [ ] **Step 3: 实现** `xarm_operator.py`

```python
"""xarm hardware layer: direct CAN control (xarm_can) + ROS2 camera streaming.

`xarm_can` and `rclpy` are imported lazily so the dry-run path (and unit
tests) work without hardware or a ROS2 environment.
"""

import threading
import time

import numpy as np

# 关节限位 (rad)，来自 x_air URDF
JOINT_LIMITS_LOWER = np.array([-1.3, -0.1, -1.5, 0.0, -1.5, -0.7, -1.5])
JOINT_LIMITS_UPPER = np.array([3.4, 1.7, 1.5, 2.4, 1.5, 0.7, 1.5])
# 夹爪限位 (rad)：-1.0 全开，0.0 闭合
GRIPPER_LOWER, GRIPPER_UPPER = -1.0, 0.0

# MIT 控制增益（x_air xarm_deploy_direct.py 现值）
DEFAULT_KP = [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0]
DEFAULT_KD = [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2]
GRIPPER_KP, GRIPPER_KD = 16.0, 0.3
HOME_GRIPPER_KP, HOME_GRIPPER_KD = 10.0, 0.5

HOME_POSITION = np.array([
    -0.1413366903181501, 0.14400701915007197, -0.2534905012588702,
    0.8703364614328226, 0.012397955291065799, 0.12722209506370596,
    0.9061951628900591, 0.0,
])

CAMERA_TOPICS = {
    "cam_high": "/cam_chest/cam_chest/color/image_raw",
    "cam_right_wrist": "/cam_wrist_right/cam_wrist_right/color/image_rect_raw",
}


def clip_action(action: np.ndarray) -> np.ndarray:
    """Clip an 8-dim action to joint and gripper limits."""
    action = np.asarray(action, dtype=np.float64).copy()
    action[:7] = np.clip(action[:7], JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    action[7] = np.clip(action[7], GRIPPER_LOWER, GRIPPER_UPPER)
    return action


class _MockArm:
    """Dry-run stand-in for the CAN hardware: positions track commands."""

    def __init__(self):
        self.positions = HOME_POSITION.copy()


class XarmOperator:
    def __init__(self, can_interface: str = "can1", dry_run: bool = False):
        self.dry_run = dry_run
        if dry_run:
            self._mock = _MockArm()
            return

        import xarm_can as oa

        self._oa = oa
        self.arm = oa.XArm(can_interface, True)  # True = CAN-FD
        motor_types = [
            oa.MotorType.DM8009, oa.MotorType.DM8009,
            oa.MotorType.DM4340, oa.MotorType.DM4340,
            oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
        ]
        send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
        self.arm.init_arm_motors(motor_types, send_ids, recv_ids)
        self.arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.arm.enable_all()
        time.sleep(0.1)
        self.arm.recv_all()
        self.arm.refresh_all()
        self.arm.recv_all()
        n_motors = len(self.arm.get_arm().get_motors())
        print(f"CAN hardware initialized on {can_interface}: {n_motors} arm motors")

    def read_state(self) -> np.ndarray:
        if self.dry_run:
            return self._mock.positions.copy()

        self.arm.refresh_all()
        self.arm.recv_all()
        joints = [m.get_position() for m in self.arm.get_arm().get_motors()]
        grippers = self.arm.get_gripper().get_motors()
        gripper = grippers[0].get_position() if grippers else 0.0
        return np.array(joints + [gripper], dtype=np.float64)

    def send_action(self, action, kp=None, kd=None, gripper_kp=GRIPPER_KP, gripper_kd=GRIPPER_KD):
        action = clip_action(action)
        if self.dry_run:
            self._mock.positions = action
            return

        kp = kp if kp is not None else DEFAULT_KP
        kd = kd if kd is not None else DEFAULT_KD
        oa = self._oa
        arm_params = [
            oa.MITParam(kp[i], kd[i], float(action[i]), 0.0, 0.0) for i in range(7)
        ]
        self.arm.get_arm().mit_control_all(arm_params)
        self.arm.get_gripper().mit_control_all(
            [oa.MITParam(gripper_kp, gripper_kd, float(action[7]), 0.0, 0.0)]
        )
        # 等待 CAN 总线处理后接收反馈（参考 x_air 遥操作脚本）
        time.sleep(0.0002)
        self.arm.recv_all()

    def go_home(self, target=None, nstep: int = 220, step_dt: float = 0.01):
        """Smoothly interpolate from the current position to `target`."""
        target = clip_action(HOME_POSITION if target is None else np.asarray(target))
        current = self.read_state()
        for step in range(nstep):
            alpha = (step + 1) / nstep
            self.send_action(
                (1 - alpha) * current + alpha * target,
                gripper_kp=HOME_GRIPPER_KP,
                gripper_kd=HOME_GRIPPER_KD,
            )
            if step_dt > 0:
                time.sleep(step_dt)

    def shutdown(self):
        if self.dry_run:
            return
        print("Disabling motors...")
        self.arm.disable_all()
        self.arm.recv_all()


def _ros_image_to_numpy(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image into an HWC uint8 RGB array."""
    height, width = msg.height, msg.width
    encoding = msg.encoding
    img_array = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return img_array.reshape((height, width, 3))
    if encoding == "bgr8":
        return img_array.reshape((height, width, 3))[:, :, ::-1]
    if encoding == "rgba8":
        return img_array.reshape((height, width, 4))[:, :, :3]
    if encoding == "bgra8":
        return img_array.reshape((height, width, 4))[:, :, [2, 1, 0]]
    if encoding == "mono8":
        return np.repeat(img_array.reshape((height, width, 1)), 3, axis=2)
    raise ValueError(f"Unsupported image encoding: {encoding}")


class CameraStreamer:
    """Subscribes to camera topics on a background rclpy thread and caches
    the latest frame per camera."""

    def __init__(self, topic_map: dict | None = None):
        import rclpy
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        topic_map = topic_map or CAMERA_TOPICS
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("xarm_aio_cameras")
        self._lock = threading.Lock()
        self._frames = {}
        self._stamps = {}

        def make_callback(name):
            def callback(msg):
                img = _ros_image_to_numpy(msg)
                with self._lock:
                    self._frames[name] = img
                    self._stamps[name] = time.monotonic()
            return callback

        self._names = list(topic_map.keys())
        for name, topic in topic_map.items():
            self._node.create_subscription(Image, topic, make_callback(name), 10)

        self._thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._thread.start()

    def get_images(self, max_age_s: float = 0.5):
        """Return the latest frames, or None if any camera is missing/stale."""
        now = time.monotonic()
        with self._lock:
            for name in self._names:
                if name not in self._frames:
                    print(f"Camera {name} has no frame yet")
                    return None
                if now - self._stamps[name] > max_age_s:
                    print(f"Camera {name} frame is stale ({now - self._stamps[name]:.2f}s)")
                    return None
            return {name: self._frames[name].copy() for name in self._names}

    def stop(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class FakeCameraStreamer:
    """Dry-run camera source returning random images."""

    def get_images(self, max_age_s: float = 0.5):
        return {
            name: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
            for name in CAMERA_TOPICS
        }

    def stop(self):
        pass
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python -m pytest test_xarm_operator.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add xarm_operator.py test_xarm_operator.py
git commit -m "feat: add xarm hardware operator with CAN control, cameras and dry-run"
```

---

### Task 5: 主循环 `infer_sync.py` + dry-run 冒烟测试

**Files:**
- Create: `inference/infer_sync.py`
- Test: `inference/test_infer_smoke.py`

**Interfaces:**
- Consumes：Task 1-4 的全部接口（`OpenpiClient`, `get_config`, `process_action`, `check_keyboard_input`, `handle_interactive_mode`, `ActionFilter`, `MovingAverageFilter`, `interpolate_chunk_transition`, `plot_filter_comparison`, `save_filter_trajectories`, `XarmOperator`, `CameraStreamer`, `FakeCameraStreamer`）
- Produces：
  - `model_inference(args, config, operator, cameras, policy=None) -> None`（`policy=None` 时内部创建 `OpenpiClient(args.host, args.port)`；测试注入假 policy）
  - CLI 入口 `main()`

执行顺序（每步）：chunk 边界过渡插值 → 滤波 → （硬件层内部）限位 → 高频细分插值下发。步频 `--publish_rate`（默认 30Hz），每步内按 `--mit_rate`（默认 500Hz）细分为 `sub_steps = round(mit_rate / publish_rate)` 次 MIT 发送，从上一步已发布位置线性插值到本步动作。

- [ ] **Step 1: 写失败测试** `inference/test_infer_smoke.py`

```python
import argparse
import builtins
import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xarm_operator import XarmOperator, FakeCameraStreamer
from infer_sync import model_inference


class FakePolicy:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.infer_calls = 0

    def predict_action(self, payload):
        self.infer_calls += 1
        # 常数 chunk：向 home 附近的固定目标移动
        target = np.linspace(0.1, 0.8, 8)
        return np.tile(target, (self.chunk_size, 1))

    def warmup(self):
        pass


def _args():
    return argparse.Namespace(
        task="example_task",
        max_publish_step=4,
        chunk_size=2,
        publish_rate=100,
        mit_rate=200,
        host="127.0.0.1",
        port=8000,
        max_episodes=1,
        auto_start=True,
        plot_filter=False,
        save_dir="",
        dry_run=True,
    )


def test_smoke_dry_run(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    args = _args()

    from utils import get_config

    config = get_config(args)
    config["episode_len"] = args.max_publish_step

    operator = XarmOperator(dry_run=True)
    cameras = FakeCameraStreamer()
    policy = FakePolicy(chunk_size=args.chunk_size)

    model_inference(args, config, operator, cameras, policy=policy)

    # episode_len=4, chunk_size=2 → 推理 2 次
    assert policy.infer_calls == 2
    # 循环结束后机械臂停在最后动作（限位内）
    state = operator.read_state()
    assert state.shape == (8,)
    assert np.all(state[:7] >= -1.5) and np.all(state[:7] <= 3.4)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_infer_smoke.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'infer_sync'`

- [ ] **Step 3: 实现** `inference/infer_sync.py`

```python
import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from action_filter import (
    ActionFilter,
    MovingAverageFilter,
    interpolate_chunk_transition,
    plot_filter_comparison,
    save_filter_trajectories,
)
from clients import OpenpiClient
from utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from xarm_operator import CameraStreamer, FakeCameraStreamer, XarmOperator

shutdown_event = threading.Event()


def _on_sigint(signum, frame):
    shutdown_event.set()


def get_observation(operator, cameras):
    images = cameras.get_images()
    if images is None:
        return None
    return {"images": images, "state": operator.read_state()}


def inference_fn(policy, operator, cameras, config):
    observation = get_observation(operator, cameras)
    if observation is None:
        return None

    payload = {
        "cam_high": observation["images"]["cam_high"],
        "cam_right_wrist": observation["images"]["cam_right_wrist"],
        "state": observation["state"],
        "instruction": config["language_instruction"],
    }
    start_time = time.perf_counter()
    actions = policy.predict_action(payload)
    print(f"Model inference time: {(time.perf_counter() - start_time) * 1000:.1f} ms")
    return actions


def maybe_save_filter_plot(args, config, raw_hist, filt_hist):
    if not args.plot_filter or not raw_hist:
        return
    save_dir = args.save_dir if args.save_dir else "/tmp"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(save_dir, f"action_filter_{config['task']}_{timestamp}")
    plot_filter_comparison(
        np.array(raw_hist), np.array(filt_hist), config["action_filter_alpha"], base + ".png"
    )
    save_filter_trajectories(
        np.array(raw_hist), np.array(filt_hist), config["action_filter_alpha"], base + ".npz"
    )
    print(f"Saved action filter comparison to {base}.png / .npz")


def model_inference(args, config, operator, cameras, policy=None):
    if policy is None:
        policy = OpenpiClient(host=args.host, port=args.port)

    chunk_size = config["chunk_size"]
    right0 = np.asarray(config["right0"])

    operator.go_home(right0)
    print("Warmup the server...")
    policy.warmup()
    print("Server warmed up")

    if not args.auto_start:
        input("Press enter to continue")
    task_time = time.time()

    if config["action_filter_type"] == "moving_average":
        action_filter = MovingAverageFilter(config["action_filter_window"])
    else:
        action_filter = ActionFilter(config["action_filter_alpha"])

    sub_steps = max(1, round(args.mit_rate / args.publish_rate))
    step_period = 1.0 / args.publish_rate
    episodes_done = 0

    try:
        while not shutdown_event.is_set():
            t = 0
            action_filter.reset()
            action_buffer = None
            raw_hist, filt_hist = [], []
            chunk_transition_steps = config["chunk_transition_steps"]
            transition_counter = chunk_transition_steps
            prev_action = None
            prev_published = None
            episode_closed = False

            while t < config["episode_len"] and not shutdown_event.is_set():
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        maybe_save_filter_plot(args, config, raw_hist, filt_hist)
                        episode_closed = True
                        operator.go_home(right0)
                        if not args.auto_start:
                            input("Press enter to continue")
                        task_time = time.time()
                        break
                    elif result == "quit":
                        maybe_save_filter_plot(args, config, raw_hist, filt_hist)
                        return

                if t % chunk_size == 0:
                    action_buffer = inference_fn(policy, operator, cameras, config)
                    if action_buffer is None:
                        print("Observation unavailable, ending episode")
                        break
                    assert action_buffer.shape[0] >= chunk_size, (
                        f"Action chunk length {action_buffer.shape[0]} < {chunk_size}"
                    )
                    if t > 0:
                        transition_counter = 0

                step_start = time.perf_counter()
                action = process_action(config["task"], action_buffer[t % chunk_size].copy())
                raw_hist.append(action.copy())

                if (
                    chunk_transition_steps > 0
                    and transition_counter < chunk_transition_steps
                    and prev_action is not None
                ):
                    action = interpolate_chunk_transition(
                        prev_action, action, transition_counter, chunk_transition_steps
                    )
                    transition_counter += 1

                action = action_filter.apply(action)
                filt_hist.append(action.copy())
                prev_action = action.copy()

                start = prev_published if prev_published is not None else action
                for s in range(sub_steps):
                    alpha = (s + 1) / sub_steps
                    operator.send_action((1 - alpha) * start + alpha * action)
                    target_time = step_start + (s + 1) * step_period / sub_steps
                    sleep_s = target_time - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                prev_published = action.copy()

                t += 1
                print("Published Step", t)

            if not episode_closed:
                maybe_save_filter_plot(args, config, raw_hist, filt_hist)
            episodes_done += 1
            if args.max_episodes and episodes_done >= args.max_episodes:
                return
            if shutdown_event.is_set():
                return
    finally:
        operator.go_home(right0)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name in task_configs.yaml")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="openpi server host")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument("--max_publish_step", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--chunk_size", type=int, default=50, help="Action chunk size")
    parser.add_argument("--publish_rate", type=int, default=30, help="Action step rate (Hz)")
    parser.add_argument("--mit_rate", type=int, default=500, help="MIT command rate (Hz)")
    parser.add_argument("--can_interface", type=str, default="can1", help="CAN interface name")
    parser.add_argument("--max_episodes", type=int, default=0, help="Stop after N episodes (0 = infinite)")
    parser.add_argument("--auto_start", action="store_true", default=False, help="Skip enter-to-continue prompts")
    parser.add_argument("--plot_filter", action="store_true", default=False, help="Save raw-vs-filtered plot per episode")
    parser.add_argument("--save_dir", type=str, default="", help="Directory for --plot_filter outputs")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Mock hardware and cameras")
    parser.add_argument(
        "--cam_high_topic", type=str, default="/cam_chest/cam_chest/color/image_raw"
    )
    parser.add_argument(
        "--cam_right_wrist_topic",
        type=str,
        default="/cam_wrist_right/cam_wrist_right/color/image_rect_raw",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    operator = XarmOperator(can_interface=args.can_interface, dry_run=args.dry_run)
    if args.dry_run:
        cameras = FakeCameraStreamer()
    else:
        cameras = CameraStreamer(
            {"cam_high": args.cam_high_topic, "cam_right_wrist": args.cam_right_wrist_topic}
        )

    import termios
    import tty

    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, operator, cameras)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        operator.shutdown()
        cameras.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio/inference && python -m pytest test_infer_smoke.py -v`
Expected: 1 passed

- [ ] **Step 5: 全量测试回归**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python -m pytest test_xarm_operator.py inference/ -v`
Expected: 全部通过（8 + 3 + 4 + 6 + 1 = 22 passed）

注意：从根目录运行 `inference/` 下的测试时，pytest 的 rootdir 插入机制会把 `inference/` 加进 `sys.path`（无 `__init__.py` 的目录按文件目录插入），若出现导入错误则分两条命令分别在两个目录运行。

- [ ] **Step 6: CLI dry-run 手工验证**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python inference/infer_sync.py --task example_task --dry_run --auto_start --max_episodes 1 --max_publish_step 6 --chunk_size 3 --publish_rate 100 --mit_rate 200 2>&1 | tail -5`

预期：打印 `Published Step 1..6` 后正常退出（policy 未注入时会尝试连接 websocket——所以此步预期**连接失败报错**属正常，除非本机有 server。若无 server，验证改为：报错信息是连接类错误而非代码错误即可）。

- [ ] **Step 7: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add inference/infer_sync.py inference/test_infer_smoke.py
git commit -m "feat: add chunked sync inference loop with dry-run smoke test"
```

---

### Task 6: openpi 侧交付物 `openpi_config/`

**Files:**
- Create: `openpi_config/xarm_policy.py`
- Create: `openpi_config/README.md`

**Interfaces:**
- Produces（供官方 openpi 仓库使用，本工程内只做语法检查）：
  - `make_xarm_example() -> dict`
  - `XarmInputs(action_dim: int, model_type: ModelType)`：`{"state","images","prompt"}` → `{"state"(pad 到 action_dim), "image"{base_0_rgb, left_wrist_0_rgb(置零+mask False), right_wrist_0_rgb}, "image_mask", "prompt", ["actions"(pad)]}`
  - `XarmOutputs()`：`{"actions": data["actions"][:, :8]}`

参考 `pistar/src/openpi/policies/piper_policy.py` 的结构 + 官方 openpi Libero policy 的 `pad_to_dim`/`model_type` 用法。

- [ ] **Step 1: 写** `openpi_config/xarm_policy.py`

```python
"""xarm single right-arm policy transforms for official openpi.

Copy this file to `openpi/src/openpi/policies/xarm_policy.py` in your openpi
checkout. See README.md in this directory for the matching TrainConfig.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xarm_example() -> dict:
    """Creates a random input example for the xarm policy."""
    return {
        "state": np.ones((8,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _convert_image(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class XarmInputs(transforms.DataTransformFn):
    """Inputs for the xarm single right-arm policy.

    Expected inputs:
    - images: dict with "cam_high" and "cam_right_wrist"
    - state: [8] (7 joints + 1 gripper, rad, absolute)
    - actions: [action_horizon, 8]
    """

    # The pi0 model's action dimension; state/actions are padded to this.
    action_dim: int = 32

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        state = transforms.pad_to_dim(
            np.asarray(data["state"], dtype=np.float32), self.action_dim
        )

        base_image = _convert_image(data["images"]["cam_high"])
        right_wrist = _convert_image(data["images"]["cam_right_wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(
                np.asarray(data["actions"], dtype=np.float32), self.action_dim
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XarmOutputs(transforms.DataTransformFn):
    """Outputs for the xarm policy: keep the first 8 action dims."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8], dtype=np.float32)}
```

- [ ] **Step 2: 写** `openpi_config/README.md`

````markdown
# openpi 侧配置（xarm 右臂单臂）

本目录文件用于官方 openpi 仓库（https://github.com/Physical-Intelligence/openpi）。
数据集为 x_air `lerobot_collector` 采集的 LeRobot 数据集，字段：
`observation.images.cam_chest`、`observation.images.cam_wrist_right`、
`observation.state` (8,)、`action` (8,)。

## 1. 安装 transforms

```bash
cp xarm_policy.py <openpi>/src/openpi/policies/xarm_policy.py
```

## 2. 注册数据配置与训练配置

在 `<openpi>/src/openpi/training/config.py` 中：

（a）加 import：

```python
import openpi.policies.xarm_policy as xarm_policy
```

（b）加 DataConfigFactory（放在其他 `LeRobot*DataConfig` 附近）：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotXarmDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_chest",
                            "cam_right_wrist": "observation.images.cam_wrist_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

（c）在 `_CONFIGS` 列表中加 TrainConfig（pi0 LoRA 微调）：

```python
TrainConfig(
    name="pi0_xarm_right_lora",
    model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    data=LeRobotXarmDataConfig(
        repo_id="<your_hf_user>/<your_xarm_dataset>",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    ema_decay=None,
),
```

注意：
- `prompt_from_task=True` 要求数据集每个 episode 有 task 字符串；若采集时没写，
  改用 `base_config=DataConfig(default_prompt="<你的任务指令>")`（与
  `task_configs.yaml` 中 `language_instruction` 保持一致）。
- 字段名以你的 openpi 版本为准，若 `DataConfigFactory`/`ModelTransformFactory`
  签名有变，参照同文件里 Libero 的写法对齐。

## 3. 计算归一化统计量

```bash
cd <openpi>
uv run scripts/compute_norm_stats.py --config-name pi0_xarm_right_lora
```

## 4. 训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_xarm_right_lora \
    --exp-name=xarm_right_v1 --overwrite
```

## 5. 启动推理服务

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_xarm_right_lora \
    --policy.dir=checkpoints/pi0_xarm_right_lora/xarm_right_v1/29999
```

客户端（机器人主机）：

```bash
cd xarm-aio
python inference/infer_sync.py --task <task_name> --host <GPU_IP> --port 8000
```
````

- [ ] **Step 3: 语法检查**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python -m py_compile openpi_config/xarm_policy.py && echo OK`
Expected: `OK`（openpi 依赖不需要安装，py_compile 只查语法）

- [ ] **Step 4: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add openpi_config/
git commit -m "feat: add openpi-side xarm policy transforms and train config guide"
```

---

### Task 7: 顶层 README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes：全部前序任务的 CLI 与文件路径

- [ ] **Step 1: 写** `README.md`

````markdown
# xarm-aio

xarm 右臂（x_air 硬件栈）× 官方 [openpi](https://github.com/Physical-Intelligence/openpi)
推理适配。对标 piper-aio 的推理部分。

```
x_air 采集(LeRobot 数据集) → openpi 微调(见 openpi_config/) → serve_policy.py(GPU)
                                                                    ↓ websocket
             infer_sync.py(机器人主机) → xarm_operator(CAN 直控 + ROS2 相机) → xarm
```

- state/action：8 维（7 关节 + 1 夹爪，弧度，绝对位置），夹爪 -1.0(开)~0.0(合)
- 观测：`cam_high`（胸前 cam_chest）+ `cam_right_wrist`（右腕），224×224 CHW uint8
- 服务端返回 50 步 action chunk，客户端 30Hz 逐步执行、500Hz MIT 细分插值下发

## 安装（机器人主机）

```bash
source /opt/ros/<distro>/setup.bash        # rclpy / sensor_msgs
# xarm_can：x_air 编译的 C++ 扩展，确保在 PYTHONPATH（参考 x_air/src/xarm_can/python）
pip install -r requirements.txt
```

## 训练与服务端

见 [openpi_config/README.md](openpi_config/README.md)。

## 运行

```bash
# 1. 启动相机（x_air）
#    ros2 launch multi_realsense multi_cameras.launch.py
# 2. 真机推理
python inference/infer_sync.py --task example_task --host <GPU_IP> --port 8000
# 3. 无硬件联调（mock CAN + 随机图像，需 server 在跑）
python inference/infer_sync.py --task example_task --dry_run --host <GPU_IP> --port 8000
```

运行中按空格进入交互模式：`c` 继续 / `r` 回 home 重来 / `q` 退出。
退出时自动回 home 并失能电机。

常用参数：`--publish_rate 30`（步频）、`--mit_rate 500`（MIT 下发频率）、
`--chunk_size 50`、`--plot_filter --save_dir <dir>`（保存滤波对比图）、
`--auto_start`（跳过回车确认）、`--max_episodes N`。

任务配置在 `inference/task_configs.yaml`（指令、home 位、滤波、夹爪后处理）。

## 测试

```bash
python -m pytest test_xarm_operator.py -v
cd inference && python -m pytest . -v
```
````

- [ ] **Step 2: 验证测试全绿**

Run: `cd /home/lft-vlai2/Documents/csy/xarm-aio && python -m pytest test_xarm_operator.py -v && cd inference && python -m pytest . -v`
Expected: 全部通过

- [ ] **Step 3: Commit**

```bash
cd /home/lft-vlai2/Documents/csy/xarm-aio
git add README.md
git commit -m "docs: add top-level README"
```

---

## 真机验收清单（人工，不在自动化任务内）

1. `source` ROS2 环境，启动 x_air 相机 launch，`ros2 topic hz` 确认两路图像
2. GPU 机器按 `openpi_config/README.md` 完成微调并启动 serve
3. 先 `--dry_run` 连真 server 验证观测/动作维度往返
4. 真机低速验证：`--publish_rate 10 --mit_rate 200`，确认回 home、chunk 执行、空格交互、Ctrl-C 安全退出
5. 恢复默认频率跑完整任务
