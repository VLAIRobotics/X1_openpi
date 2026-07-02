# xarm-aio：xarm 机械臂 × 官方 openpi 推理适配设计

日期：2026-07-02
状态：已确认

## 背景与目标

piper-aio 是 piper 机械臂适配 openpi 框架（pistar 分叉）的推理部分。本工程 xarm-aio
为 xarm 机械臂（x_air 硬件栈）做同样的事，但对接**官方 openpi**
（https://github.com/Physical-Intelligence/openpi），不涉及 pistar。

- 部署形态：右臂单臂，2 相机（cam_chest 胸前 + cam_wrist_right 右腕）
- state/action：8 维（7 关节 + 1 夹爪，弧度，绝对位置）
- 服务端尚不存在：本工程同时交付 openpi 侧 transforms 与训练/推理配置示例，
  客户端观测格式与之对齐
- 工程位置：`/home/lft-vlai2/Documents/csy/xarm-aio`（独立仓库，不改动 x_air 与 piper-aio）

## 参考代码

| 来源 | 复用内容 |
|------|----------|
| `piper-aio/inference/infer_sync.py` | chunk 推理主循环、滤波、chunk 边界插值、高频细分下发 |
| `piper-aio/inference/clients.py` | OpenpiClient（websocket 客户端封装） |
| `piper-aio/inference/action_filter.py` + 单测 | EMA / 滑动窗口滤波、chunk 过渡插值（与臂无关，原样移植） |
| `piper-aio/inference/task_configs.yaml` | 按任务配置的组织方式 |
| `x_air/src/lerobot_collector/xarm_deploy_direct.py` | CAN 直控（xarm_can MIT 控制）、关节/夹爪限位、平滑回 home、ROS2 相机订阅 |

## 1. 总体数据流

```
x_air 采集(现有，不动) → LeRobot 数据集(cam_chest / cam_wrist_right / state8 / action8)
                              ↓
官方 openpi + 本工程 xarm transforms/TrainConfig → 微调 → checkpoint
                              ↓
openpi scripts/serve_policy.py（websocket，GPU 机器）
                              ↓
xarm-aio/inference/infer_sync.py（机器人主机）→ CAN 直控 xarm
```

**不需要数据转换脚本**：x_air 录制的就是 LeRobot 格式，openpi 侧用 repack
transform 对齐字段名即可。

## 2. 接口约定（两头对齐的核心）

客户端每次推理发送：

```python
{
  "state":  float32 (8,),        # 7 关节 + 1 夹爪，弧度，绝对位置
  "images": {
    "cam_high":        uint8 (3,224,224),   # cam_chest，resize_with_pad 后 CHW
    "cam_right_wrist": uint8 (3,224,224),   # cam_wrist_right
  },
  "prompt": "任务指令",
}
```

服务端返回 `{"actions": (chunk_size≥50, 8)}`，绝对关节位置。无 pistar 的 `adv_ind`。

openpi 侧 transforms：

- `XarmInputs`：state 8 维 pad 到模型 action 维度；`cam_high → base_0_rgb`、
  `cam_right_wrist → right_wrist_0_rgb`；左腕图像置零 + mask 掉
- `XarmOutputs`：取 `actions[:, :8]`

## 3. 硬件层 `xarm_operator.py`

从 `xarm_deploy_direct.py` 抽取，一个类统一管硬件与相机：

- CAN 初始化：电机型号/ID 表（DM8009×2, DM4340×2, DM4310×3 + 夹爪 DM4310）、
  使能、回调模式，照搬现有代码
- `read_state() → (8,)`：refresh_all + recv_all，7 关节 + 夹爪位置
- `send_action(action8)`：关节限位（URDF 值）+ 夹爪限位（-1.0~0.0 rad）裁剪后
  MIT 下发，KP/KD 沿用现值（[240,240,240,240,24,31,25] / [3,3,3,3,0.2,0.2,0.2]，
  夹爪 16.0/0.3）
- `go_home()`：220 步插值平滑回 home（照搬）
- 相机：rclpy 后台线程 spin，订阅
  `/cam_chest/cam_chest/color/image_raw` 与
  `/cam_wrist_right/cam_wrist_right/color/image_rect_raw`，
  缓存最新帧 + 时间戳，`get_images()` 带过期检查
- `shutdown()`：失能电机

## 4. 推理层 `inference/`

- `clients.py`：`OpenpiClient`，基于官方 `openpi_client.websocket_client_policy`。
  相比 piper-aio 版本：去掉 `adv_ind` 与双臂分支，图像 key 为
  `cam_high / cam_right_wrist`，state 8 维
- `action_filter.py` + 单测：原样移植
- `infer_sync.py` 主循环，结构同 piper-aio：
  - 每 `chunk_size`(默认 50) 步推理一次，步频 30Hz
  - 每步内部细分插值，以 ~500Hz 发 MIT 命令（对应 piper-aio 的
    `publish_rate_high`，取 x_air 的 MIT 通信节奏）
  - 顺序：chunk 边界过渡插值 → 滤波（EMA/滑窗）→ 限位 → 下发（与 piper-aio 一致）
  - 键盘交互沿用 piper-aio 模式（空格暂停 → 继续/重置/退出）
- `task_configs.yaml`：`language_instruction`、`right0`（home 位，默认取
  deploy 脚本 `initial_positions`）、滤波参数、可选夹爪后处理

## 5. `openpi_config/`（交付给官方 openpi 仓库的文件）

官方 openpi 的 TrainConfig 必须注册在其自身 `config.py`，故以"待拷贝文件 + 说明"交付：

- `xarm_policy.py`：`XarmInputs` / `XarmOutputs` transforms →
  拷到 `openpi/src/openpi/policies/`
- `README.md`：
  - `TrainConfig` 代码片段（如 `pi0_xarm_right_lora`，含 repack transforms、
    LoRA 配置）
  - norm stats 计算命令
  - 训练命令、serve 命令

## 6. 错误处理与测试

- 图像/状态缺失或过期（>0.5s）→ 跳过该步并告警，不发旧命令
- websocket 断连 → 保持当前位置（MIT 持续发最后位置），报错退出前不失能
- SIGINT / 退出 → 停止执行 → 失能电机（沿用 x_air cleanup 逻辑）
- 测试：
  - `action_filter` 单测（随移植带入）
  - client 观测构建单测（key、维度、dtype）
  - `--dry-run` 模式（mock CAN 硬件），无臂环境可联调 server

## 目录结构

```
xarm-aio/
├── xarm_operator.py       # 硬件层：CAN 直控 + ROS2 相机订阅
├── inference/
│   ├── infer_sync.py      # 主循环：chunk 推理 → 滤波 → 高频插值下发
│   ├── clients.py         # OpenpiClient（官方 openpi_client）
│   ├── action_filter.py   # EMA/滑窗 + chunk 边界插值（移植）
│   ├── test_action_filter.py
│   └── task_configs.yaml
├── openpi_config/
│   ├── xarm_policy.py     # XarmInputs/Outputs transforms
│   └── README.md          # TrainConfig 示例 + 训练/serve 命令
├── docs/superpowers/specs/
└── README.md
```
