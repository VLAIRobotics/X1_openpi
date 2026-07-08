# xarm-aio

xarm 右臂（x_air 硬件栈）× 官方 [openpi](https://github.com/Physical-Intelligence/openpi)
推理适配。对标 piper-aio 的推理部分。

```
x_air 采集(LeRobot 数据集) → openpi 微调(见 openpi_config/) → serve_policy.py(GPU)
                                                                    ↓ websocket
             infer_sync.py(机器人主机) → xarm_operator(CAN 直控 + ROS2 相机) → xarm
```

- state/action：8 维（7 关节 + 1 夹爪，弧度，绝对位置），夹爪 -1.1(开)~0.0(合)
- 观测：`cam_high`（胸前 cam_chest）+ `cam_right_wrist`（右腕），224×224 CHW uint8
- 服务端返回 50 步 action chunk，客户端 30Hz 逐步执行、500Hz MIT 细分插值下发

## 安装（机器人主机）

```bash
source /opt/ros/<distro>/setup.bash        # rclpy / sensor_msgs
# xarm_can：x_air 编译的 C++ 扩展，确保在 PYTHONPATH（参考 x_air/src/xarm_can/python）
pip install -r requirements.txt
```

若 `openpi-client` 的 git 安装因网络失败，可改用本地 openpi/pistar 检出：

```bash
pip install --no-build-isolation -e <openpi>/packages/openpi-client
# 或（无 hatchling 时）把 <openpi>/packages/openpi-client/src 写入 site-packages 的 .pth
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
# 已 source ROS2 的终端需禁用其 pytest 插件自动加载（Python 版本不匹配会报 lark 错误）
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest test_xarm_operator.py -v
cd inference && python -m pytest . -v
```
