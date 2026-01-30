"""RBY1 policy input/output adapters.

This module is intentionally a *thin* adapter layer:
- **Inputs**: convert an rby1 observation dict into the model's common input format
  (images + image_mask + state [+ prompt]).
- **Outputs**: convert the model's action tensor into a simpler dict that your
  rby1 control loop can consume.

You will still need to map the final `actions` vector into your robot SDK
(e.g., base velocity + arm joint velocities + gripper command).

Design goals:
- Keep the "public" input keys close to what robot code naturally produces.
- Make it easy to swap cameras / state layout without touching model code.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_rby1_example() -> dict:
    """Creates a random input example for the RBY1 policy."""
    return {
        "observation/state": np.random.randn(24).astype(np.float32),
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize image into uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Rby1Inputs(transforms.DataTransformFn):
    """Convert rby1 observations into the model input dict.

    Expected input keys (robot-side):
    - `observation/head_image`: RGB image, uint8/float, shape (H,W,3) or (3,H,W)
    - `observation/left_wrist_image`: optional RGB image
    - `observation/right_wrist_image`: optional RGB image
    - `observation/state`: 24D float array
    - `prompt`: optional instruction string

    Output keys (model-side; do not change):
    - `state`: float32[s]
    - `image`: dict with keys in openpi.models.model.IMAGE_KEYS
    - `image_mask`: dict[image_key,bool]
    - `prompt`: optional
    """

    # Determines which model will be used (affects image masking behavior for padding).
    model_type: _model.ModelType

    # If true, require a wrist image; otherwise pad with zeros when missing.

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/head_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # pi0 family currently standardizes on three image keys.
        images = {
            "head_0_rgb": head_image,
            "left_wrist_0_rgb": left_wrist_image,
            # If you have a right wrist cam on rby1, replace this with it.
            "right_wrist_0_rgb": right_wrist_image,
        }

        image_masks = {
            "head_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
        }

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected 1D state vector, got shape={state.shape}")

        inputs: dict = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Rby1Outputs(transforms.DataTransformFn):
    """Post-process model outputs for rby1 control code.

    By default this is a no-op except ensuring numpy arrays and optionally slicing dims.
    Use `action_dim` if you want to truncate the model's action vector.
    """

    # If set, truncate action vectors to this dimension.
    action_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For RBY1, we only return the first 24 actions (since the rest is padding).
        # For your own dataset, replace `24` with the action dimension of your dataset.
        return {"actions": data["actions"][:, :24]}

