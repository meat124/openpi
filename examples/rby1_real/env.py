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
            connected_serials = []
            try:
                ctx = rs.context()
                for dev in ctx.query_devices():
                    try:
                        connected_serials.append(dev.get_info(rs.camera_info.serial_number))
                    except Exception:
                        continue
                del ctx  # 파이프라인 시작 전 context 명시적 해제
            except Exception:
                connected_serials = []

            if self._serial is not None and connected_serials and self._serial not in connected_serials:
                raise RuntimeError(
                    f"Camera serial {self._serial} not found. Connected RealSense serials: {connected_serials}"
                )

            try:
                # Stop any existing pipeline before starting
                try:
                    self._pipeline.stop()
                except Exception:
                    pass  # Pipeline wasn't running
                
                self._pipeline.start(self._config)
                self._started = True  # pipeline 시작 즉시 표시 (warmup 중 interrupt 시에도 stop() 동작)

                # Warmup: discard initial frames for stable image quality
                for _ in range(30):
                    self._pipeline.wait_for_frames(timeout_ms=1000)

            except Exception as e:
                logger.warning("Camera start/warmup failed for %s: %s", self._serial, e)
                # Ensure pipeline is stopped on failure
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._started = False
                raise RuntimeError(f"Failed to start camera {self._serial}: {e}") from e

    def stop(self) -> None:
        if self._started:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            finally:
                self._started = False
                logger.info(f"Camera {self._serial} stopped.")

    def get_rgb_image(self) -> np.ndarray:
        if not self._started:
            self.start()
        if not self._started:
            raise RuntimeError(f"Camera {self._serial} is not started")

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
                if "cannot be called before start" in str(exc).lower():
                    raise RuntimeError(f"Camera {self._serial} is not running. start() failed earlier.") from exc
                if attempt < max_retries - 1:
                    self._started = False
                    try:
                        self._pipeline.stop()
                    except RuntimeError:
                        pass
                    self.start()
                else:
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
        arm_command_priority: int = 1,
        arm_action_scale: float = 1.0,
        arm_minimum_time: float = 0.1,
        log_action_send: bool = False,
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
        self._arm_command_priority = int(arm_command_priority)
        self._arm_action_scale = arm_action_scale
        self._arm_minimum_time = arm_minimum_time
        self._log_action_send = bool(log_action_send)
        self._state_source = state_source
        self._state_indices = np.asarray(state_indices, dtype=int) if state_indices is not None else None
        self._gripper_state_key = gripper_state_key
        self._use_remote_gripper = use_remote_gripper
        self._gripper = gripper

        if self._arm_action_scale <= 0:
            raise ValueError("arm_action_scale must be > 0")
        if self._arm_minimum_time <= 0:
            raise ValueError("arm_minimum_time must be > 0")
        if self._arm_command_priority < 0:
            raise ValueError("arm_command_priority must be >= 0")

        # Initialize cameras
        self._cameras: Dict[str, _RealsenseCamera] = {
            "observation/head_image": _RealsenseCamera(
                serial=cam_head_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "observation/left_wrist_image": _RealsenseCamera(
                serial=cam_left_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "observation/right_wrist_image": _RealsenseCamera(
                serial=cam_right_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
        }

        logger.info("Starting cameras...")
        try:
            for name, cam in self._cameras.items():
                try:
                    cam.start()
                    logger.info("Camera %s started.", name)
                except Exception as exc:
                    logger.error("Failed to start camera %s: %s", name, exc)
                    raise
        except BaseException:
            # 초기화 도중 예외(KeyboardInterrupt 포함) 발생 시
            # 이미 start()된 카메라들을 모두 정리하고 re-raise
            for cam in self._cameras.values():
                try:
                    cam.stop()
                except Exception:
                    pass
            raise

        # Connect to robot
        self._robot = robot if robot is not None else self._create_robot(robot_ip)
        self._robot.connect()
        if self._robot.is_connected():
            logger.info("Robot connected successfully")
        else:
            raise RuntimeError("Failed to connect to robot")

        # Ensure robot is ready to accept motion commands.
        self._prepare_robot_for_control()

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

    def _prepare_robot_for_control(self) -> None:
        """Power/servo/control-manager initialization with best-effort recovery."""
        try:
            if hasattr(self._robot, "power_on"):
                self._robot.power_on(".*")
            if hasattr(self._robot, "servo_on"):
                self._robot.servo_on(".*")

            # Recover faulted control manager if needed.
            if hasattr(self._robot, "get_control_manager_state") and hasattr(self._robot, "reset_fault_control_manager"):
                cm_state = self._robot.get_control_manager_state().state
                major = getattr(getattr(rby, "ControlManagerState", None), "State", None)
                if major is not None:
                    fault_states = {getattr(major, "MajorFault", None), getattr(major, "MinorFault", None)}
                    if cm_state in fault_states:
                        logger.warning("Control manager fault detected. Trying reset...")
                        self._robot.reset_fault_control_manager()

            if hasattr(self._robot, "enable_control_manager"):
                self._robot.enable_control_manager()

            if hasattr(self._robot, "cancel_control"):
                try:
                    self._robot.cancel_control()
                except Exception:
                    pass

            if hasattr(self._robot, "wait_for_control_ready"):
                self._robot.wait_for_control_ready(1000)

            logger.info("Robot control manager prepared.")
        except Exception as exc:
            logger.warning("Failed to fully prepare robot control manager: %s", exc)

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
        observation = {}
        for name, camera in self._cameras.items():
            raw_img = camera.get_rgb_image()
            resized = image_tools.resize_with_pad(raw_img, self._render_height, self._render_width)
            resized = image_tools.convert_to_uint8(resized)
            observation[name] = einops.rearrange(resized, "h w c -> c h w")
        
        robot_qpos = self._get_joint_positions()

        observation["observation/state"] = robot_qpos
        observation["prompt"] = self._prompt
        
        return observation

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
            raise ValueError(f"Action dimension mismatch (expected {expected}, got {action_vec.size})")

        right_action = action_vec[: self._right_action_dim]
        left_action = action_vec[self._right_action_dim : expected]

        self._send_joint_positions(left_action, right_action)

    def _send_joint_positions(self, left_action: np.ndarray, right_action: np.ndarray) -> None:
        """Send arm and gripper commands to robot."""
        logger.debug("Sending left: %s, right: %s", left_action, right_action)

        q_before = None
        if self._log_action_send and hasattr(self._robot, "get_state"):
            try:
                q_before = np.asarray(self._robot.get_state().position, dtype=np.float64)[8:22].copy()
            except Exception:
                q_before = None

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

        # Send arm commands
        rv = self._robot.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_right_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time)))
                        .set_position(self._arm_action_scale * right_arm)
                        .set_minimum_time(self._arm_minimum_time)
                    )
                    .set_left_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time)))
                        .set_position(self._arm_action_scale * left_arm)
                        .set_minimum_time(self._arm_minimum_time)
                    )
                )
            ),
            self._arm_command_priority,
        ).get()

        finish_code = getattr(rv, "finish_code", None)
        finish_enum = getattr(getattr(rby, "RobotCommandFeedback", None), "FinishCode", None)
        if self._log_action_send:
            logger.info(
                "[send] priority=%s left_norm=%.4f right_norm=%.4f finish_code=%s",
                self._arm_command_priority,
                float(np.linalg.norm(left_arm)),
                float(np.linalg.norm(right_arm)),
                finish_code,
            )
        if finish_enum is not None and finish_code is not None and finish_code != finish_enum.Ok:
            logger.warning("Robot arm command finish_code=%s", finish_code)

            # Auto-recovery path when control manager is idle.
            if finish_code == getattr(finish_enum, "ControlManagerIdle", None):
                logger.warning("Control manager idle. Re-preparing robot and retrying command once...")
                self._prepare_robot_for_control()
                rv_retry = self._robot.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_body_command(
                            rby.BodyComponentBasedCommandBuilder()
                            .set_right_arm_command(
                                rby.JointPositionCommandBuilder()
                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time)))
                                .set_position(self._arm_action_scale * right_arm)
                                .set_minimum_time(self._arm_minimum_time)
                            )
                            .set_left_arm_command(
                                rby.JointPositionCommandBuilder()
                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time)))
                                .set_position(self._arm_action_scale * left_arm)
                                .set_minimum_time(self._arm_minimum_time)
                            )
                        )
                    ),
                    self._arm_command_priority,
                ).get()
                retry_code = getattr(rv_retry, "finish_code", None)
                if finish_enum is not None and retry_code is not None and retry_code != finish_enum.Ok:
                    logger.warning("Retry arm command failed: finish_code=%s", retry_code)

        if self._log_action_send and q_before is not None and hasattr(self._robot, "get_state"):
            try:
                time_wait = max(0.02, min(0.2, float(self._arm_minimum_time)))
                import time
                time.sleep(time_wait)
                q_after = np.asarray(self._robot.get_state().position, dtype=np.float64)[8:22].copy()
                dq = q_after - q_before
                logger.info(
                    "[send] q_delta_norm=%.6f q_before[0]=%.4f q_after[0]=%.4f",
                    float(np.linalg.norm(dq)),
                    float(q_before[0]),
                    float(q_after[0]),
                )
            except Exception as exc:
                logger.warning("[send] failed to read q after command: %s", exc)

        logger.debug("Commands sent to robot")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception as exc:
                logger.warning("Failed to stop gripper cleanly: %s", exc)

        if hasattr(self._robot, "disconnect"):
            try:
                self._robot.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect robot cleanly: %s", exc)

        for camera in self._cameras.values():
            camera.stop()
