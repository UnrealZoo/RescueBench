"""Editable user solution for the Rescue benchmark.

This file is the small interface a new user is expected to edit. It mirrors
the old competition style used by Apex and R2ZeroShot, but also exposes the
new benchmark info dict when the method signature asks for it.

Default input format:
    python benchmark/experiment.py --model solution --solution benchmark/example/solution.py

    reset(reference_text, reference_image, info=None)
        Called once at episode start. reference_image is a base64 PNG string
        or None. With --solution-input-format raw it is a BGR numpy array.

    predicts(observation, picked, info=None)
        Called every step. observation is a base64 PNG string by default.
        picked is the current environment carrying flag. It is the same idea
        as the old competition success argument.

Returned action dict:
    angular:     turn angle, normally -30 to 30. Negative turns left.
    velocity:    forward speed, normally -100 to 100. Positive moves forward.
    interaction: 0 no-op, 3 carry, 4 drop.
    viewport:    kept for Apex/R2 compatibility, currently ignored by adapter.
    head:        optional benchmark head action, 0 none, 1 right, 2 left.

If you port an old two-argument solution, this still works:
    def predicts(self, ob, success):
        ...

If you only want navigation and want the benchmark state machine to trigger
carry/drop/open_door, run with --no-passthrough and keep interaction at 0.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional


def decode_bgr_image(image: Any):
    """Return a BGR numpy image for either base64 or raw benchmark input."""

    if image is None:
        return None
    if not isinstance(image, str):
        return image

    cv2 = __import__("cv2")
    np = __import__("numpy")
    data = base64.b64decode(image)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode base64 PNG image")
    return decoded


def action(angular: float = 0.0, velocity: float = 0.0, interaction: int = 0) -> Dict[str, Any]:
    """Build an old competition style Unreal action dict."""

    return {
        "angular": float(angular),
        "velocity": float(velocity),
        "viewport": 0,
        "interaction": int(interaction),
    }


NOOP = action()
FORWARD = action(velocity=50.0)
BACKWARD = action(velocity=-50.0)
TURN_LEFT = action(angular=-20.0)
TURN_RIGHT = action(angular=20.0)
CARRY = action(interaction=3)
DROP = action(interaction=4)


class AlgSolution:
    """Replace the body of this class with your model logic."""

    def __init__(self):
        self.step = 0
        self.reference_text = ""
        self.reference_image_bgr = None
        self.last_info: Dict[str, Any] = {}

    def reset(
        self,
        reference_text: Optional[str] = None,
        reference_image: Any = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called once at the beginning of every episode."""

        self.step = 0
        self.reference_text = reference_text or ""
        self.reference_image_bgr = decode_bgr_image(reference_image)
        self.last_info = info or {}

    def predicts(
        self,
        observation: Any,
        picked: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return one action for the current observation.

        Useful info keys in the new benchmark include:
            Pose, picked, task_phase, target_pose, state_machine_state,
            reference_text, reference_image_path, level, point_id, episode_id,
            rescue_distance, place_distance, interaction_z_threshold.
        """

        self.step += 1
        self.last_info = info or {}
        image_bgr = decode_bgr_image(observation)

        # Replace this block with model inference. image_bgr is a BGR uint8 array
        # with shape [H, W, 3] in both base64 and raw input modes.
        _height, _width = image_bgr.shape[:2]
        _task_phase = self.last_info.get("task_phase", "find_injured")

        # Example policy: keep moving forward. A real passthrough solution should
        # return CARRY near the injured person and DROP near the stretcher.
        if picked:
            return FORWARD.copy()
        return FORWARD.copy()
