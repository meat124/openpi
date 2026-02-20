"""Run OpenPI policy server actions on `rby1-sim` Docker.

Usage:
    python main.py --robot-ip "localhost:50051" --prompt "pick up the cup"
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.rby1_sim import env as _env


@dataclasses.dataclass
class Args:
    # Policy server
    host: str = "localhost"
    port: int = 8000
    action_horizon: int = 25

    # Episode settings
    num_episodes: int = 1
    # 0 disables the timeout (run until the env reports done or Ctrl+C).
    max_episode_steps: int = 0
    max_hz: float = 50.0

    # Simulator (gRPC)
    robot_ip: str = "localhost:50051"
    prompt: str = "pick up the object"

    # Observation image shape (dummy black frames)
    render_height: int = 224
    render_width: int = 224
    # Image source: "dummy" (black frames) or "zmq" (subscribe to external publisher).
    image_source: str = "dummy"
    image_zmq_address: str | None = None

    # Action dimensions: 7 joints + 1 gripper per arm
    left_action_dim: int = 8
    right_action_dim: int = 8

    # Command scaling/safety
    action_scale: float = 1
    minimum_time: float = 5.0


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger = logging.getLogger(__name__)
    logger.info("Connected to policy server: %s", ws_client_policy.get_server_metadata())

    env = _env.RBY1SimEnvironment(
        robot_ip=args.robot_ip,
        prompt=args.prompt,
        render_height=args.render_height,
        render_width=args.render_width,
        left_action_dim=args.left_action_dim,
        right_action_dim=args.right_action_dim,
        action_scale=args.action_scale,
        minimum_time=args.minimum_time,
        image_source=args.image_source,
        image_zmq_address=args.image_zmq_address,
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
    finally:
        logger.info("Closing environment...")
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)

