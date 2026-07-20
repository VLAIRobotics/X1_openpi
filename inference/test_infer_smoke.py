import argparse
import builtins
import sys
import os

import numpy as np
import pytest

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
        camera_wait_s=15.0,
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


def test_model_inference_does_not_move_home_when_warmup_fails():
    from utils import get_config

    class WarmupFailPolicy:
        def warmup(self):
            raise RuntimeError("server unavailable")

    class FakeOperator:
        def __init__(self):
            self.go_home_calls = []

        def go_home(self, target=None):
            self.go_home_calls.append(target)

    args = _args()
    config = get_config(args)
    operator = FakeOperator()

    with pytest.raises(RuntimeError, match="server unavailable"):
        model_inference(args, config, operator, FakeCameraStreamer(), policy=WarmupFailPolicy())

    assert operator.go_home_calls == []


def test_model_inference_does_not_move_home_when_camera_missing():
    from utils import get_config

    class FakeOperator:
        def __init__(self):
            self.go_home_calls = []

        def go_home(self, target=None):
            self.go_home_calls.append(target)

    class MissingCamera:
        def get_images(self, max_age_s=0.5, verbose=True):
            return None

    args = _args()
    args.dry_run = False
    args.camera_wait_s = 0.01
    config = get_config(args)
    operator = FakeOperator()

    with pytest.raises(RuntimeError, match="Camera frames unavailable"):
        model_inference(args, config, operator, MissingCamera(), policy=FakePolicy(args.chunk_size))

    assert operator.go_home_calls == []


def test_main_shutdowns_operator_if_camera_init_fails(monkeypatch):
    import infer_sync

    shutdown_calls = []

    class FakeOperator:
        def __init__(self, can_interface, dry_run, home_position=None):
            self.can_interface = can_interface
            self.dry_run = dry_run

        def shutdown(self):
            shutdown_calls.append((self.can_interface, self.dry_run))

    class FailingCamera:
        def __init__(self, topic_map):
            raise RuntimeError("camera init failed")

    monkeypatch.setattr(sys, "argv", ["infer_sync.py", "--task", "example_task"])
    monkeypatch.setattr(infer_sync, "XarmOperator", FakeOperator)
    monkeypatch.setattr(infer_sync, "CameraStreamer", FailingCamera)

    with pytest.raises(RuntimeError, match="camera init failed"):
        infer_sync.main()

    assert shutdown_calls == [("can1", False)]
