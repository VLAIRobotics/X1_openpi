# openpi 侧配置（xarm 右臂单臂）

本目录文件用于官方 openpi 仓库（https://github.com/Physical-Intelligence/openpi）。
数据集为 x_air `lerobot_collector` 采集的 LeRobot 数据集，字段：
`observation.images.cam_chest`、`observation.images.cam_wrist_right`、
`observation.state` (8,)、`action` (8,)、`task_index`。
其中 state/action 最后一维为夹爪位置，当前数据约为 -1.1（全开）~ 0.0（闭合）。

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
- `prompt_from_task=True` 会用数据中的 `task_index` 从 LeRobot `meta/tasks`
  取任务字符串并注入为 `prompt`，所以逐帧数据里没有 `prompt` 字段是正常的。
  若 `meta/tasks` 没有正确任务文本，
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
