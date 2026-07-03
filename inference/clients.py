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
