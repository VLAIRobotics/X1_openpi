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
