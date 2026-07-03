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
