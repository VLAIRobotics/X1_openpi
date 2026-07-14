# X1_openpi

面向 xArm 右臂（`x_max_sdk` 硬件栈）的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 训练与真机推理适配。GPU 工作站负责 OpenPI 训练和 websocket 策略服务；机器人主机负责 ROS 2 相机观测、CAN 状态读取和动作下发。

```text
x_max_sdk 采集 LeRobot 数据集 → OpenPI 微调（GPU 工作站）→ 策略服务 :8000
                                                               ↓ websocket
              ROS 2 相机 + CAN 机器人主机 → infer_sync.py → xArm 右臂
```

> 安全提示：本工程通过 CAN 直接控制真实机械臂。首次使用新 checkpoint 时，急停必须可触及；先完成 dry-run，再以低步数、低发布频率进行真机测试。

## 1. 约定与变量

所有命令默认工程位于 `~/X1_openpi`，ROS 2 发行版为 Humble。以下值需要按现场替换：

| 变量 | 含义 |
| --- | --- |
| `<GPU_IP>` | GPU 工作站地址；同机运行时用 `127.0.0.1` |
| `<STEP>` | 要加载的 checkpoint step 目录，例如 `20000` |
| `<D435_CHEST_SERIAL>` | 胸部 RealSense 序列号 |
| `<D405_RIGHT_WRIST_SERIAL>` | 右腕 RealSense 序列号 |

机器人与数据约定：

- 机械臂：xArm 右臂，默认 CAN 接口为 `can1`。
- 图像：`cam_high` 对应胸部相机；`cam_right_wrist` 对应右腕相机。
- state/action：8 维，顺序为 `[joint1, ..., joint7, gripper]`；单位均为弧度，动作是绝对位置。
- 夹爪范围：`-1.1` 约为全开，`0.0` 为闭合。
- 客户端会将图像 resize/pad 至 `224×224`、转换为 `CHW uint8`；服务端返回 50 步 action chunk，客户端默认以 30 Hz 执行、500 Hz MIT 插值下发。

示例任务位于 [`inference/task_configs.yaml`](inference/task_configs.yaml)：

```yaml
x1_stack_green_cube:
  language_instruction: "stack_green_cube"
  right0: [-0.14133669435977936, 0.444007009267807, -0.05349050089716911,
           1.600000023841858, 0.012397955171763897, 0.12722209095954895,
           0.0061951628886163235, -1.100000023841858]
  chunk_transition_steps: 10
```

`language_instruction` 必须与训练配置中的 `default_prompt` 相同；`right0` 是该任务的 8 维 home 位。

## 2. 机器人主机环境

本节只在机器人主机执行。GPU 工作站不需要 `xarm_can` 或 ROS 2 环境。

### 2.1 创建 xarm 环境

x_max 的 `xarm` 环境要求 Python 3.10 和 LeRobot `v0.4.0`。先在机器人主机完成以下步骤：

```bash
conda create -n xarm python=3.10 -y
conda activate xarm

git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
git checkout v0.4.0
pip install -e .

# 解决 Conda 的 libstdc++ / OpenSSL 版本冲突
conda install -c conda-forge openssl=3.2 libcurl -y
```

### 2.2 注册 x_max SDK 的 `xarm_can`

将 x_max_sdk 发布包放在 `~/x_max_sdk`。它提供与设备平台匹配的 `xarm_can` 扩展；扩展的 Python ABI 必须与当前 Python 3.10 一致。

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/x_max_sdk.sh" <<'EOF'
export PYTHONPATH="$HOME/x_max_sdk/publish/lerobot_collector${PYTHONPATH:+:$PYTHONPATH}"
EOF

conda deactivate
conda activate xarm
source /opt/ros/humble/setup.bash
python -c "import xarm_can, rclpy; print('xarm_can:', xarm_can.__file__)"
```

### 2.3 创建并安装 X1_openpi 环境

从已验证的 `xarm` 环境克隆独立环境，避免影响 x_max_sdk 的数据采集/部署环境：

```bash
conda create -n X1_openpi --clone xarm
conda activate X1_openpi

cd ~/X1_openpi
source /opt/ros/humble/setup.bash
pip install -r requirements.txt
```

若 `openpi-client` 无法通过网络安装，使用随本仓库子模块提供的源码安装：

```bash
cd ~/X1_openpi/openpi
pip install hatchling editables
pip install --no-build-isolation -e packages/openpi-client

python -c "from openpi_client import image_tools, websocket_client_policy; print('openpi_client ok')"
```

## 3. OpenPI 训练（GPU 工作站）

`openpi/` 是固定到官方 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 的子模块。xArm 适配不使用自定义 OpenPI fork；在官方 `config.py` 中注册下方已验证的训练配置。

### 3.1 初始化 OpenPI

```bash
cd ~/X1_openpi
git submodule update --init --recursive
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

训练数据必须是 LeRobot 数据集，且包含：

- `observation.images.cam_chest`
- `observation.images.cam_wrist_right`
- `observation.state`，形状 `(8,)`
- `action`，形状 `(8,)`

若使用不同的数据集或任务文本，只替换下方的 `repo_id`、`asset_id` 与 `default_prompt`；图像和 state/action 映射保持不变。

### 3.2 注册 xArm 训练配置

编辑 `~/X1_openpi/openpi/src/openpi/training/config.py`，在 `_CONFIGS = [` 列表中加入：

```python
TrainConfig(
    name="pi05_x1_stack_green_cube",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="x1_stack_green_cube",
        assets=AssetsConfig(asset_id="x1_stack_green_cube"),
        default_prompt="stack the green cube",
        adapt_to_pi=False,
        use_delta_joint_actions=False,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_chest",
                            "cam_right_wrist": "observation.images.cam_wrist_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    num_train_steps=20_000,
    batch_size=16,
    save_interval=1000,
    log_interval=10,
),
```

上例的 `default_prompt` 是 `"stack the green cube"`。如使用当前任务配置中的 `"stack_green_cube"`，训练前必须将 `default_prompt` 改为同一字符串。

### 3.3 计算统计量并训练

```bash
cd ~/X1_openpi/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_x1_stack_green_cube

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_x1_stack_green_cube \
  --exp-name=xarm_right_v1 --overwrite
```

checkpoint 输出路径为 `~/X1_openpi/openpi/checkpoints/pi05_x1_stack_green_cube/xarm_right_v1/<STEP>`。

## 4. 部署与真机推理

### 4.1 启动策略服务（GPU 工作站）

```bash
cd ~/X1_openpi/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_x1_stack_green_cube \
  --policy.dir=checkpoints/pi05_x1_stack_green_cube/xarm_right_v1/<STEP>
```

服务加载完成后监听 `0.0.0.0:8000`。

### 4.2 检查并启动相机（机器人主机）

先确认胸部和右腕 RealSense 的序列号：

```bash
source /opt/ros/humble/setup.bash
rs-enumerate-devices
```

确认相机用途后，填入检查到的序列号并启动：

```bash
ros2 launch multi_realsense multi_cameras.launch.py \
  serial_chest:=<D435_CHEST_SERIAL> \
  serial_right:=<D405_RIGHT_WRIST_SERIAL>
```

在另一个终端确认两路图像稳定发布：

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /cam_chest/cam_chest/color/image_raw
ros2 topic hz /cam_wrist_right/cam_wrist_right/color/image_raw
```

### 4.3 Dry-run 联通测试（机器人主机）

不连接 CAN 或相机，只验证 websocket、observation payload 和 action chunk：

```bash
cd ~/X1_openpi
conda activate X1_openpi

python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --dry_run \
  --host <GPU_IP> \
  --port 8000 \
  --auto_start \
  --max_episodes 1 \
  --max_publish_step 20
```

日志出现 `Server warmed up` 和连续的 `Published Step` 后，才可进入真机测试。

### 4.4 首次真机测试（机器人主机）

相机和策略服务均已启动后，以低步数和低频率运行。首次测试不要使用 `--auto_start`，程序会等待操作员按回车确认：

```bash
cd ~/X1_openpi
conda activate X1_openpi
source /opt/ros/humble/setup.bash

python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --host <GPU_IP> \
  --port 8000 \
  --max_episodes 10 \
  --max_publish_step 10 \
  --publish_rate 30 \
  --mit_rate 500 \
  --camera_wait_s 20 \
```

脚本会初始化 CAN、等待相机帧、回到任务 home 位并执行策略。退出时会回 home 并失能电机。确认关节方向、夹爪方向和动作幅度正常后，再逐步提高 `--max_publish_step`。

## 5. 常见问题

### `No module named xarm_can`

重新激活 `X1_openpi` 环境，并检查 x_max SDK 激活脚本与扩展路径：

```bash
conda activate X1_openpi
echo "$PYTHONPATH"
python -c "import xarm_can; print(xarm_can.__file__)"
```
