# xarm-aio

xarm 右臂（x_air 硬件栈）× 官方 [openpi](https://github.com/Physical-Intelligence/openpi)
推理适配。对标 piper-aio 的推理部分。

```
x_air 采集(LeRobot 数据集) → openpi 微调(见 openpi_config/) → serve_policy.py(GPU)
                                                                    ↓ websocket
             infer_sync.py(机器人主机) → xarm_operator(CAN 直控 + ROS2 相机) → xarm
```

安全提示：本工程会直接通过 CAN 控制真实机械臂。第一次运行新 checkpoint 时，请把急停放在手边，先用 `--dry_run` 联通，再用低步数、低发布频率真机测试。

## 硬件与数据约定

- 机械臂：xArm 右臂，`x_air` / `x_max_sdk` 硬件栈。
- 控制：`xarm_can` 直控 CAN-FD，默认 CAN 接口 `can1`。
- 相机：
  - `cam_high`：`/cam_chest/cam_chest/color/image_raw`
  - `cam_right_wrist`：`/cam_wrist_right/cam_wrist_right/color/image_raw`
- state/action：8 维，`[joint1..joint7, gripper]`。
- 单位：关节和夹爪都是弧度，动作是绝对位置。
- 夹爪范围：`-1.1` 约为全开，`0.0` 为闭合。
- openpi 输入图像：客户端会 resize/pad 到 `224x224`，并转成 `CHW uint8`。
- 服务端返回 50 步 action chunk，客户端 30Hz 逐步执行、500Hz MIT 细分插值下发。

当前示例任务在 [`inference/task_configs.yaml`](inference/task_configs.yaml)：

```yaml
x1_stack_green_cube:
  language_instruction: "stack_green_cube0625"
  right0: [-0.14133669435977936, 0.444007009267807, -0.05349050089716911,
           1.600000023841858, 0.012397955171763897, 0.12722209095954895,
           0.0061951628886163235, -1.100000023841858]
  chunk_transition_steps: 10
```

`language_instruction` 要和 openpi 训练/服务端使用的 prompt 保持一致。

## 环境准备

机器人主机建议使用独立环境：

```bash
conda create -n xarm-aio --clone xarm
conda activate xarm-aio
```

安装 Python 依赖：

```bash
cd ~/xarm-aio
source /opt/ros/<distro>/setup.bash        # rclpy / sensor_msgs
# xarm_can：x_air 编译的 C++ 扩展，确保在 PYTHONPATH（参考 x_air/src/xarm_can/python）
pip install -r requirements.txt
```

如果 `openpi-client` 通过网络安装失败，使用本地 openpi 仓库安装：

```bash
conda activate xarm-aio
cd ~/openpi
pip install hatchling editables
pip install --no-build-isolation -e packages/openpi-client
# 或（无 hatchling 时）把 <openpi>/packages/openpi-client/src 写入 site-packages 的 .pth
```

验证：

```bash
python -c "from openpi_client import image_tools, websocket_client_policy; print('openpi_client ok')"
```

如果机器人主机不能联网，也可以临时把 openpi-client 源码加入 `PYTHONPATH`：

```bash
export PYTHONPATH=$HOME/openpi/packages/openpi-client/src:$PYTHONPATH
python -c "from openpi_client import image_tools, websocket_client_policy; print('openpi_client ok')"
```

真机模式还需要确保当前 Python 能导入 `xarm_can`。用能跑通 `x_max_sdk` 部署脚本的环境最稳。如果报 `No module named xarm_can`，需要把 x_max_sdk 发布包中的 `xarm_can` 扩展路径加入 `PYTHONPATH`。

## 训练与服务端

见 [openpi_config/README.md](openpi_config/README.md)。

## 服务端启动

在 GPU/工作站上启动 openpi websocket 服务：

```bash
cd ~/Documents/csy/openpi

uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_x1_stack_green_cube \
  --policy.dir=../checkpoints/pi05_x1_stack_green_cube/x1_stack_green_cube_0702/5000
```

看到模型加载完成后，服务会监听 `0.0.0.0:8000`。

可以在客户端机器上测试健康检查：

```bash
curl --noproxy '*' http://<GPU_IP>:8000/healthz
```

正常返回：

```text
OK
```

如果客户端和服务端在同一台机器，`<GPU_IP>` 可以用 `127.0.0.1`。

## 关闭代理

如果机器上开了 clash、HTTP proxy 或 shell 里有代理变量，websocket 可能被代理转发导致握手失败。运行客户端前建议在当前终端关闭代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY ws_proxy wss_proxy WS_PROXY WSS_PROXY
export NO_PROXY=127.0.0.1,localhost,<GPU_IP>
export no_proxy=127.0.0.1,localhost,<GPU_IP>
```

如果服务端就在本机：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

## 启动流程

终端一：启动相机。

```bash
source /opt/ros/humble/setup.bash
ros2 launch multi_realsense multi_cameras.launch.py \
  serial_chest:=_146322070813 \
  serial_right:=_352122271543
```

终端二：启动推理脚本。

```bash
cd ~/xarm-aio
conda activate xarm-aio
source /opt/ros/humble/setup.bash

python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --host 192.168.1.101 \
  --port 8000 \
  --max_episodes 10 \
  --max_publish_step 500 \
  --publish_rate 30 \
  --mit_rate 500 \
  --camera_wait_s 20 \
  --plot_filter \
  --save_dir /tmp/xarm-infer
```

## Dry-Run 联通测试

先不连接硬件，只验证 xarm-aio 客户端、openpi server、payload/action chunk 是否匹配：

```bash
cd ~/xarm-aio
conda activate xarm-aio

python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --dry_run \
  --host <GPU_IP> \
  --port 8000 \
  --auto_start \
  --max_episodes 1 \
  --max_publish_step 20
```

成功时会看到：

```text
Warmup the server...
Server warmed up
Model inference time: ...
Published Step 1
...
Published Step 20
```

dry-run 通过后，再进入真机联调。

## 相机检查

在机器人主机上启动 ROS2 和 RealSense：

```bash
source /opt/ros/<distro>/setup.bash
ros2 launch multi_realsense multi_cameras.launch.py \
  serial_chest:=_<D435_CHEST_SERIAL> \
  serial_right:=_<D405_RIGHT_WRIST_SERIAL>
```

按当前日志里的设备，对应关系应是：

```bash
ros2 launch multi_realsense multi_cameras.launch.py \
  serial_chest:=_146322070813 \
  serial_right:=_352122271543
```

`xarm-aio` 默认订阅的是右腕相机 `cam_wrist_right`，所以不要把右腕序列号传给 `serial_left`。如果 launch 输出 `跳过 cam_wrist_right - 未指定序列号`，推理端会报 `Camera cam_right_wrist has no frame yet`。

另一个终端检查两个 topic：

```bash
source /opt/ros/<distro>/setup.bash

ros2 topic hz /cam_chest/cam_chest/color/image_raw
ros2 topic hz /cam_wrist_right/cam_wrist_right/color/image_raw
```

两个 topic 都稳定发布后再跑真机推理。

## 真机推理

第一次真机测试不要加 `--auto_start`，先用低步数和低发布频率：

```bash
cd ~/xarm-aio
conda activate xarm-aio
source /opt/ros/<distro>/setup.bash

python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --host <GPU_IP> \
  --port 8000 \
  --max_episodes 1 \
  --max_publish_step 10 \
  --publish_rate 10 \
  --mit_rate 500 \
  --camera_wait_s 20
```

脚本流程：

1. 初始化 CAN 硬件并使能电机。
2. 初始化 ROS2 相机订阅。
3. warmup openpi server。
4. 等待两个相机都收到画面。
5. 等待你按回车。
6. 机械臂平滑回 `right0`。
7. 执行策略动作。
8. 退出时回 home 并失能电机。

确认关节方向、夹爪方向和动作幅度都正常后，可以逐步增加：

```bash
--max_publish_step 30
--publish_rate 30
```

## 运行中控制

运行中按空格进入交互模式：

- `c`：继续执行。
- `r`：回 home 并重新开始。
- `q`：退出。

`Ctrl+C` 也会触发清理流程。正常退出时会回 home 并调用 `operator.shutdown()` 失能电机。

## 常用参数

- `--task`：任务名，必须存在于 `inference/task_configs.yaml`。
- `--host`：openpi server 地址。
- `--port`：openpi server 端口，默认 `8000`。
- `--dry_run`：mock CAN 和相机，只连接 openpi server。
- `--max_episodes`：最多执行几个 episode，`0` 表示无限。
- `--max_publish_step`：每个 episode 最多发布多少步。
- `--publish_rate`：策略动作发布频率，默认 `30Hz`。
- `--mit_rate`：MIT 细分下发频率，默认 `500Hz`。
- `--can_interface`：CAN 接口名，默认 `can1`。
- `--plot_filter`：保存 raw/filter action 对比图。
- `--save_dir`：保存滤波图和 `.npz` 的目录。
- `--chunk_size 50`。
- `--auto_start`：跳过回车确认。

任务配置在 `inference/task_configs.yaml`（指令、home 位、滤波、夹爪后处理）。

## 常见问题

### `ModuleNotFoundError: No module named 'openpi_client'`

当前 xarm-aio Python 环境没装 openpi-client：

```bash
conda activate xarm-aio
cd ~/openpi
pip install hatchling editables
pip install --no-build-isolation -e packages/openpi-client
```

或临时使用：

```bash
export PYTHONPATH=$HOME/openpi/packages/openpi-client/src:$PYTHONPATH
```

### `Cannot import 'hatchling.build'`

当前环境缺构建后端：

```bash
pip install hatchling
```

### `ModuleNotFoundError: No module named 'editables'`

当前环境缺 editable 安装依赖：

```bash
pip install editables
```

### `No module named xarm_can`

当前环境没有 x_max_sdk/x_air 的 `xarm_can` Python 扩展。使用能跑通 `x_max_sdk/publish/lerobot_collector/xarm_deploy_direct.py` 的 Python 环境，或把对应 `.so` 所在目录加入 `PYTHONPATH`。

### 相机无画面或 stale

确认相机 launch 已启动，并检查 topic：

```bash
ros2 topic hz /cam_chest/cam_chest/color/image_raw
ros2 topic hz /cam_wrist_right/cam_wrist_right/color/image_raw
```

也可以临时覆盖 topic：

```bash
python inference/infer_sync.py \
  --task x1_stack_green_cube \
  --host <GPU_IP> \
  --port 8000 \
  --cam_high_topic /your/chest/topic \
  --cam_right_wrist_topic /your/wrist/topic
```

## 本地测试

不接硬件时可以跑：

```bash
# 已 source ROS2 的终端需禁用其 pytest 插件自动加载（Python 版本不匹配会报 lark 错误）
cd ~/xarm-aio
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest test_xarm_operator.py -v
cd inference
python -m pytest . -v
```

这些测试不会导入 ROS2 或 `xarm_can`，适合改代码后快速确认基础逻辑。
