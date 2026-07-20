import numpy as np
from openpi_client import image_tools, websocket_client_policy

STATE_DIM = 8
IMAGE_KEYS = ("cam_high", "cam_right_wrist")


def build_observation(
    payload: dict,
    *,
    state_dim: int = STATE_DIM,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
) -> dict:
    """Convert raw robot payload into the wire observation for the openpi server.

    payload keys: camera names (HWC uint8), "state", "instruction".
    """
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (state_dim,):
        raise ValueError(f"Expected state with shape ({state_dim},), got {state.shape}")

    images = {}
    for name in image_keys:
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(payload[name], 224, 224))
        images[name] = img.transpose(2, 0, 1)

    return {
        "state": state,
        "images": images,
        "prompt": payload["instruction"],
    }


def _random_observation(
    *,
    state_dim: int = STATE_DIM,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
) -> dict:
    return {
        "state": np.ones((state_dim,), dtype=np.float32),
        "images": {
            name: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
            for name in image_keys
        },
        "prompt": "do something",
    }


class OpenpiClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        state_dim: int = STATE_DIM,
        image_keys: tuple[str, ...] = IMAGE_KEYS,
    ) -> None:
        self.state_dim = state_dim
        self.image_keys = image_keys
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)

    def predict_action(self, payload: dict) -> np.ndarray:
        state_dim = getattr(self, "state_dim", STATE_DIM)
        image_keys = getattr(self, "image_keys", IMAGE_KEYS)
        response = self.client.infer(
            build_observation(payload, state_dim=state_dim, image_keys=image_keys)
        )
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or actions.shape[1] < state_dim:
            raise ValueError(
                f"Expected actions with shape (chunk, >= {state_dim}), got {actions.shape}"
            )
        return actions[:, :state_dim]

    def warmup(self) -> None:
        state_dim = getattr(self, "state_dim", STATE_DIM)
        image_keys = getattr(self, "image_keys", IMAGE_KEYS)
        self.client.infer(_random_observation(state_dim=state_dim, image_keys=image_keys))
