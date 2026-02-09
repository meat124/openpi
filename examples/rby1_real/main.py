"""RBY1 robot control via OpenPI policy server.

Usage:
    python main.py --args.robot-ip "localhost:50051" --args.prompt "pick up the cup"
"""
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.rby1_real import env as _env


@dataclasses.dataclass
class Args:
    """Configuration for RBY1 control."""
    # Policy server
    host: str = "localhost"
    port: int = 8000
    action_horizon: int = 25

    # Episode settings
    num_episodes: int = 1
    max_episode_steps: int = 1000
    max_hz: float = 50.0

    # Robot
    robot_ip: str = "192.168.0.10"
    prompt: str = "pick up the object"

    # Camera serials
    cam_head_serial: str | None = "922612070040"
    cam_left_serial: str | None = "838212070714"
    cam_right_serial: str | None = "838212074317"

    # Camera settings
    render_height: int = 224
    render_width: int = 224
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Action dimensions: 7 joints + 1 gripper per arm
    left_action_dim: int = 8
    right_action_dim: int = 8

    state_source: str = "robot"
    state_zmq_address: str | None = None
    state_indices: list[int] | None = None
    gripper_state_key: str | None = None
    use_remote_gripper: bool = True


def main(args: Args) -> None:
    """Connect to policy server and run control loop."""
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger = logging.getLogger(__name__)
    logger.info("Connected to policy server: %s", ws_client_policy.get_server_metadata())

    env = _env.RBY1Environment(
        robot_ip=args.robot_ip,
        prompt=args.prompt,
        render_height=args.render_height,
        render_width=args.render_width,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        cam_head_serial=args.cam_head_serial,
        cam_left_serial=args.cam_left_serial,
        cam_right_serial=args.cam_right_serial,
        left_action_dim=args.left_action_dim,
        right_action_dim=args.right_action_dim,
        state_source=args.state_source,
        state_zmq_address=args.state_zmq_address,
        state_indices=args.state_indices,
        gripper_state_key=args.gripper_state_key,
        use_remote_gripper=args.use_remote_gripper,
    )

    try:
        runtime = _runtime.Runtime(
            environment=env,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=args.max_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error("Runtime error: %s", e)
        raise
    finally:
        logger.info("Closing environment...")
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)