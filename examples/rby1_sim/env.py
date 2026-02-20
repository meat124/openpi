import logging
import pickle
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    import rby1_sdk as rby
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("rby1_sdk is required for RBY1SimEnvironment.") from exc

try:
    import zmq
except ImportError:  # pragma: no cover - optional dependency
    zmq = None


logger = logging.getLogger(__name__)


class RBY1SimEnvironment(_environment.Environment):
    """Minimal OpenPI environment for Rainbow Robotics `rby1-sim` Docker.

    The official simulator exposes the same gRPC interface as the real robot (default port 50051),
    so `rby1_sdk` can be used to send actions and read state.

    This environment provides:
    - **State**: 16-dim (14 arm joints + 2 gripper values [dummy zeros by default])
    - **Images**: 3 camera keys expected by OpenPI, filled with black images (uint8, CHW)
      so you can test the control loop even without a simulator camera stream.
    - **Action**: 16-dim (right 8 + left 8) -> joint position commands (scaled)
    """

    def __init__(
        self,
        *,
        robot_ip: str = "localhost:50051",
        prompt: str = "pick up the object",
        render_height: int = 224,
        render_width: int = 224,
        left_action_dim: int = 8,
        right_action_dim: int = 8,
        action_scale: float = 0.1,
        minimum_time: float = 5.0,
        # Image source. The official `rby1-sim` gRPC interface does not standardize camera streaming.
        # Use "dummy" to feed black images, or "zmq" to receive images from an external publisher.
        image_source: str = "dummy",  # "dummy" | "zmq"
        image_zmq_address: str | None = None,
    ) -> None:
        self._prompt = prompt
        self._render_height = int(render_height)
        self._render_width = int(render_width)
        self._left_action_dim = int(left_action_dim)
        self._right_action_dim = int(right_action_dim)
        self._action_scale = float(action_scale)
        self._minimum_time = float(minimum_time)

        self._image_source = image_source
        self._image_socket = None
        self._last_images: dict[str, np.ndarray] | None = None

        self._robot = self._create_robot(robot_ip)
        self._robot.connect()
        if not self._robot.is_connected():
            raise RuntimeError(f"Failed to connect to simulator at {robot_ip}")
        logger.info("Connected to RBY1 simulator at %s", robot_ip)

        # In the simulator, commands may be ignored until power/servo/control are enabled.
        self._ensure_control_ready()

        self._done = False

        # Pre-allocate dummy images (CHW) in uint8.
        self._dummy_chw = np.zeros((3, self._render_height, self._render_width), dtype=np.uint8)

        if self._image_source == "zmq":
            if zmq is None:
                raise ImportError("pyzmq is required when image_source='zmq'")
            if not image_zmq_address:
                raise ValueError("image_zmq_address is required when image_source='zmq'")
            context = zmq.Context.instance()
            socket = context.socket(zmq.SUB)
            socket.connect(image_zmq_address)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._image_socket = socket
            logger.info("Subscribed to image stream at %s", image_zmq_address)
        elif self._image_source != "dummy":
            raise ValueError(f"Unsupported image_source={image_source!r}. Use 'dummy' or 'zmq'.")

    def _create_robot(self, robot_ip: str) -> object:
        if hasattr(rby, "create_robot"):
            return rby.create_robot(robot_ip, "a")
        raise RuntimeError("Unable to construct RBY1 robot client from rby1_sdk.")

    def _ensure_control_ready(self) -> None:
        """Best-effort bringup sequence for simulator control."""
        dev_pat = ".*"

        # If control manager is faulted, try clearing it first.
        try:
            if hasattr(self._robot, "get_control_manager_state"):
                st = self._robot.get_control_manager_state()
                logger.info("Simulator control_manager_state before bringup: %s", st)
            if hasattr(self._robot, "reset_fault_control_manager"):
                self._robot.reset_fault_control_manager()
        except Exception as exc:
            logger.warning("reset_fault_control_manager failed (continuing): %s", exc)

        # Power on
        try:
            if hasattr(self._robot, "is_power_on") and hasattr(self._robot, "power_on"):
                if not self._robot.is_power_on(dev_pat):
                    logger.info("Simulator power_on(%r)", dev_pat)
                    self._robot.power_on(dev_pat)
        except Exception as exc:
            logger.warning("power_on failed (continuing): %s", exc)

        # Servo on
        try:
            if hasattr(self._robot, "is_servo_on") and hasattr(self._robot, "servo_on"):
                if not self._robot.is_servo_on(dev_pat):
                    logger.info("Simulator servo_on(%r)", dev_pat)
                    self._robot.servo_on(dev_pat)
        except Exception as exc:
            logger.warning("servo_on failed (continuing): %s", exc)

        # Control manager enable
        try:
            if hasattr(self._robot, "enable_control_manager"):
                logger.info("Simulator enable_control_manager()")
                self._robot.enable_control_manager()
        except Exception as exc:
            logger.warning("enable_control_manager failed (continuing): %s", exc)

        # Wait for control ready (if provided)
        try:
            if hasattr(self._robot, "wait_for_control_ready"):
                ok = self._robot.wait_for_control_ready(timeout_ms=5000)
                if not ok:
                    logger.warning("wait_for_control_ready timed out (continuing)")
        except Exception as exc:
            logger.warning("wait_for_control_ready failed (continuing): %s", exc)

        try:
            if hasattr(self._robot, "get_control_manager_state"):
                st = self._robot.get_control_manager_state()
                logger.info("Simulator control_manager_state after bringup: %s", st)
        except Exception:
            pass

    @override
    def reset(self) -> None:
        # Simulator may or may not implement reset; call if present.
        if hasattr(self._robot, "reset"):
            try:
                self._robot.reset()
            except Exception as exc:
                logger.warning("Simulator reset() failed (continuing): %s", exc)
        self._ensure_control_ready()
        self._done = False

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        head, left, right = self._get_images()
        obs = {
            "observation/head_image": head,
            "observation/left_wrist_image": left,
            "observation/right_wrist_image": right,
            "prompt": self._prompt,
        }

        obs["observation/state"] = self._get_joint_positions()
        return obs

    def _get_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (head, left_wrist, right_wrist) as uint8 CHW."""
        if self._image_source == "dummy":
            return self._dummy_chw, self._dummy_chw, self._dummy_chw

        assert self._image_socket is not None
        # Expect pickled dict with keys: head_image/left_wrist_image/right_wrist_image (HWC or CHW).
        # We keep the last valid images if a message is missing keys.
        msg = self._image_socket.recv()
        data = pickle.loads(msg)
        if not isinstance(data, dict):
            raise TypeError("ZMQ image message must be a pickled dict.")

        def _prep(img) -> np.ndarray:
            img = np.asarray(img)
            # Normalize to uint8 HWC then to CHW.
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = image_tools.resize_with_pad(img, self._render_height, self._render_width)
            img = image_tools.convert_to_uint8(img)
            return np.transpose(img, (2, 0, 1))  # HWC -> CHW

        updated: dict[str, np.ndarray] = {}
        for key in ("head_image", "left_wrist_image", "right_wrist_image"):
            if key in data:
                updated[key] = _prep(data[key])

        if self._last_images is None:
            # On first frame, require all three.
            missing = [k for k in ("head_image", "left_wrist_image", "right_wrist_image") if k not in updated]
            if missing:
                raise KeyError(f"ZMQ image message missing keys: {missing}")
            self._last_images = updated
        else:
            self._last_images.update(updated)

        assert self._last_images is not None
        return (
            self._last_images["head_image"],
            self._last_images["left_wrist_image"],
            self._last_images["right_wrist_image"],
        )

    def _get_joint_positions(self) -> np.ndarray:
        """Get 16-dim state: 14 arm joints + 2 gripper values (zeros)."""
        if not hasattr(self._robot, "get_state"):
            raise RuntimeError("rby1_sdk robot object must provide get_state()")
        qpos = self._robot.get_state().position
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        # Match the real-env convention: extract arm joints (indices 8-21).
        if qpos.size >= 22:
            qpos = qpos[8:22]
        # Append dummy gripper state (2 dims) if missing.
        if qpos.size == 14:
            qpos = np.concatenate([qpos, np.zeros((2,), dtype=np.float32)], axis=0)
        return qpos.astype(np.float32, copy=False)

    @override
    def apply_action(self, action: dict) -> None:
        if "actions" not in action:
            raise KeyError("Action dict missing 'actions' key")

        action_vec = np.asarray(action["actions"], dtype=np.float32).reshape(-1)
        expected = self._left_action_dim + self._right_action_dim
        if action_vec.size != expected:
            raise ValueError(f"Expected action dim {expected}, got {action_vec.size}")

        right_action = action_vec[: self._right_action_dim]
        left_action = action_vec[self._right_action_dim : expected]

        left_arm = left_action[:7] if left_action.size >= 7 else left_action
        right_arm = right_action[:7] if right_action.size >= 7 else right_action

        # Send joint position commands (scaled) via SDK.
        self._robot.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_right_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_position(self._action_scale * right_arm)
                        .set_minimum_time(self._minimum_time)
                    )
                    .set_left_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_position(self._action_scale * left_arm)
                        .set_minimum_time(self._minimum_time)
                    )
                )
            )
        )

    def close(self) -> None:
        # Best-effort shutdown.
        try:
            if hasattr(self._robot, "disconnect"):
                self._robot.disconnect()
        except Exception:
            pass
        try:
            if self._image_socket is not None:
                self._image_socket.close(0)
        except Exception:
            pass

