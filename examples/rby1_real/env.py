import logging
import pickle
from typing import Dict, Optional, Sequence

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.rby1_real.remote_gripper import Gripper

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pyrealsense2 is required for RBY1Environment. Install Intel RealSense SDK Python bindings."
    ) from exc

try:
    import rby1_sdk as rby
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("rby1_sdk is required for RBY1Environment.") from exc

try:
    import zmq
except ImportError:  # pragma: no cover - optional dependency for ZMQ state source
    zmq = None


logger = logging.getLogger(__name__)


class _RealsenseCamera:
    """Wrapper for Intel RealSense camera with auto-retry and warmup."""

    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if self._serial is not None:
            self._config.enable_device(self._serial)
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        self._started = False

    def __del__(self) -> None:
        self.stop()

    def start(self) -> None:
        if not self._started:
            logger.info("Starting camera %s...", self._serial)
            try:
                # Stop any existing pipeline before starting
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass  # Pipeline wasn't running
                
                self._pipeline.start(self._config)
                self._started = True
                
                # Warmup: discard initial frames for stable image quality
                for _ in range(30):
                    self._pipeline.wait_for_frames(timeout_ms=1000)
                    
            except Exception as e:
                logger.warning("Camera start/warmup failed for %s: %s", self._serial, e)
                self._started = False
                # Ensure pipeline is stopped on failure
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass

    def stop(self) -> None:
        if self._started:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            finally:
                self._started = False
                logger.info(f"Camera {self._serial} stopped.")

    def get_rgb_image(self) -> np.ndarray:
        if not self._started:
            self.start()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=3000)
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    raise RuntimeError("Received frames but no color frame found.")
                return np.asanyarray(color_frame.get_data())
            except RuntimeError as exc:
                logger.warning(
                    "Frame fetch failed (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError("Failed to get image after retries")


class RBY1Environment(_environment.Environment):
    """OpenPI environment for RBY1 robot with 3 cameras and remote gripper.
    
    State: 16-dim (14 arm joints + 2 gripper values)
    Action: 16-dim (7 joints + 1 gripper per arm)
    """

    def __init__(
        self,
        *,
        robot_ip: str,
        prompt: str = "pick up the object",
        render_height: int = 224,
        render_width: int = 224,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        cam_head_serial: Optional[str] = None,
        cam_left_serial: Optional[str] = None,
        cam_right_serial: Optional[str] = None,
        left_action_dim: int = 8,  # 7 joints + 1 gripper
        right_action_dim: int = 8,  # 7 joints + 1 gripper
        state_source: str = "robot",
        state_zmq_address: Optional[str] = None,
        state_indices: Optional[Sequence[int]] = None,
        gripper_state_key: Optional[str] = None,
        use_remote_gripper: bool = True,
        gripper: Optional[object] = None,
        robot: Optional[object] = None,
    ) -> None:
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._left_action_dim = left_action_dim
        self._right_action_dim = right_action_dim
        self._state_source = state_source
        self._state_indices = np.asarray(state_indices, dtype=int) if state_indices is not None else None
        self._gripper_state_key = gripper_state_key
        self._use_remote_gripper = use_remote_gripper
        self._gripper = gripper

        # Initialize cameras
        self._cameras: Dict[str, _RealsenseCamera] = {
            "cam_head": _RealsenseCamera(
                serial=cam_head_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "cam_left_wrist": _RealsenseCamera(
                serial=cam_left_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "cam_right_wrist": _RealsenseCamera(
                serial=cam_right_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
        }

        logger.info("Starting cameras...")
        for name, cam in self._cameras.items():
            try:
                cam.start()
                logger.info("Camera %s started.", name)
            except Exception as exc:
                logger.error("Failed to start camera %s: %s", name, exc)
                raise

        # Connect to robot
        self._robot = robot if robot is not None else self._create_robot(robot_ip)
        self._robot.connect()
        if self._robot.is_connected():
            logger.info("Robot connected successfully")
        else:
            raise RuntimeError("Failed to connect to robot")

        # Setup ZMQ socket if using external state source
        self._state_socket = None
        if self._state_source == "zmq":
            if zmq is None:
                raise ImportError("pyzmq is required when state_source='zmq'")
            if not state_zmq_address:
                raise ValueError("state_zmq_address is required when state_source='zmq'")
            context = zmq.Context.instance()
            socket = context.socket(zmq.SUB)
            socket.connect(state_zmq_address)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._state_socket = socket

        # Initialize remote gripper via UDP
        if self._use_remote_gripper and self._gripper is None:
            if Gripper is None:
                raise ImportError("remote_gripper module required when use_remote_gripper=True")
            
            logger.info("Initializing remote gripper...")
            self._gripper = Gripper()
            logger.info("Connecting to gripper at %s:%s", self._gripper.host, self._gripper.port)
            
            try:
                if not self._gripper.initialize(verbose=True):
                    logger.warning("Gripper ping failed. Server may not be running")
                    self._gripper = None
                    return
                
                if not self._gripper.homing():
                    logger.warning("Gripper homing failed")
                    self._gripper = None
                    return
                
                self._gripper.start()
                self._gripper.set_normalized_target(np.array([1.0, 1.0]))
                logger.info("Gripper initialized and opened")
                
            except Exception as exc:
                logger.warning("Gripper initialization failed: %s", exc)
                self._gripper = None

    def _create_robot(self, robot_ip: str) -> object:
        if hasattr(rby, "create_robot"):
            return rby.create_robot(robot_ip, "a")
        raise RuntimeError("Unable to construct RBY1 robot client from rby1_sdk.")

    @override
    def reset(self) -> None:
        if hasattr(self._robot, "reset"):
            self._robot.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        # Capture images from all cameras
        images = {}
        for name, camera in self._cameras.items():
            raw_img = camera.get_rgb_image()
            resized = image_tools.resize_with_pad(raw_img, self._render_height, self._render_width)
            resized = image_tools.convert_to_uint8(resized)
            images[name] = einops.rearrange(resized, "h w c -> c h w")
        
        robot_qpos = self._get_joint_positions()
        
        return {
            "images": images,
            "state": robot_qpos,
            "prompt": self._prompt,
        }

    def _get_joint_positions(self) -> np.ndarray:
        """Get 16-dim state: 14 arm joints + 2 gripper values."""
        if self._state_source == "zmq":
            return self._get_joint_positions_from_zmq()

        if not hasattr(self._robot, "get_state"):
            raise RuntimeError("rby1_sdk robot object must provide get_state()")
        qpos = self._robot.get_state().position
        qpos = qpos[8:22]  # Extract arm joints (indices 8-21)
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        qpos = self._append_gripper_state(qpos)
        return self._apply_state_indices(qpos)

    def _get_joint_positions_from_zmq(self) -> np.ndarray:
        if self._state_socket is None:
            raise RuntimeError("ZMQ state socket is not initialized.")

        msg = self._state_socket.recv()
        data = pickle.loads(msg)
        qpos = self._extract_state_field(data, "joint_positions")
        if qpos is None:
            raise RuntimeError("ZMQ state does not contain joint_positions.")

        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if self._gripper_state_key is not None:
            gripper = self._extract_state_field(data, self._gripper_state_key)
            if gripper is not None:
                gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
                qpos = np.concatenate([qpos, gripper], axis=0)
        else:
            qpos = self._append_gripper_state(qpos)
        return self._apply_state_indices(qpos)

    def _append_gripper_state(self, qpos: np.ndarray) -> np.ndarray:
        if self._gripper is None:
            return qpos
        try:
            gripper_state = self._gripper.get_state()
            gripper_state = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
            return np.concatenate([qpos, gripper_state], axis=0)
        except Exception as exc:
            logger.warning("Failed to fetch remote gripper state: %s", exc)
            return qpos

    def _extract_state_field(self, data, key: str):
        if isinstance(data, dict):
            return data.get(key)
        return getattr(data, key, None)

    def _apply_state_indices(self, qpos: np.ndarray) -> np.ndarray:
        if self._state_indices is None:
            return qpos
        return qpos[self._state_indices]

    @override
    def apply_action(self, action: dict) -> None:
        """Apply 16-dim action: split into left (8) and right (8) commands."""
        if "actions" not in action:
            raise KeyError("Action dict missing 'actions' key")

        action_vec = np.asarray(action["actions"], dtype=np.float32).reshape(-1)
        expected = self._left_action_dim + self._right_action_dim
        
        if action_vec.size != expected:
            logger.warning(
                "Action dimension mismatch (expected %s, got %s). Splitting in half",
                expected,
                action_vec.size,
            )
            mid = action_vec.size // 2
            right_action = action_vec[:mid]
            left_action = action_vec[mid:]
        else:
            right_action = action_vec[: self._right_action_dim]
            left_action = action_vec[self._right_action_dim : expected]

        self._send_joint_positions(left_action, right_action)

    def _send_joint_positions(self, left_action: np.ndarray, right_action: np.ndarray) -> None:
        """Send arm and gripper commands to robot."""
        logger.debug("Sending left: %s, right: %s", left_action, right_action)
        minimum_time = 5

        # Split arm (first 7) and gripper (8th element) commands
        left_arm = left_action[:7] if left_action.size >= 7 else left_action
        right_arm = right_action[:7] if right_action.size >= 7 else right_action
        left_gripper = left_action[7] if left_action.size > 7 else None
        right_gripper = right_action[7] if right_action.size > 7 else None

        # Send gripper commands via UDP (non-blocking)
        if self._gripper is not None and (left_gripper is not None or right_gripper is not None):
            try:
                gripper_target = self._gripper.get_target()
                if right_gripper is not None:
                    gripper_target[0] = float(right_gripper)
                if left_gripper is not None:
                    gripper_target[1] = float(left_gripper)
                self._gripper.set_normalized_target(gripper_target, wait_for_reply=False)
                logger.debug("Gripper: [%.3f, %.3f]", gripper_target[0], gripper_target[1])
            except Exception as exc:
                logger.warning("Failed to send gripper command: %s", exc)

        # Send arm commands (currently sending zero pose for safety)
        self._robot.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_right_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_position(np.zeros(7))
                        .set_minimum_time(minimum_time)
                    )
                    .set_left_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_position(np.zeros(7))
                        .set_minimum_time(minimum_time)
                    )
                )
            ),
            1,
        ).get()
        logger.debug("Commands sent to robot")

    def close(self) -> None:
        for camera in self._cameras.values():
            camera.stop()
