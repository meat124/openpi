import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    RBY1 = "rby1"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Override checkpoint directory for RBY1 (shortcut: used when env=rby1 and policy=Default).
    # e.g. "/home/hyunjin/rby1_ws/openpi/checkpoints/pi05_rby1/PuttingCupintotheDishV2/29999"
    rby1_checkpoint_dir: str | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.RBY1: Checkpoint(
        config="pi05_rby1",
        dir="/home/hyunjin/rby1_ws/openpi/checkpoints/pi05_rby1/PuttingCupintotheDishV2/29999",  # TODO
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            # If RBY1 env and rby1_checkpoint_dir is provided, override the default checkpoint dir.
            if args.env == EnvMode.RBY1 and args.rby1_checkpoint_dir is not None:
                checkpoint = DEFAULT_CHECKPOINT[EnvMode.RBY1]
                checkpoint = dataclasses.replace(checkpoint, dir=args.rby1_checkpoint_dir)
                return _policy_config.create_trained_policy(
                    _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=args.default_prompt
                )
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata or {})

    # Add explicit source info so websocket clients can verify which checkpoint is served.
    if isinstance(args.policy, Checkpoint):
        served_checkpoint = args.policy
        policy_loader = "explicit_checkpoint"
    else:
        served_checkpoint = DEFAULT_CHECKPOINT[args.env]
        if args.env == EnvMode.RBY1 and args.rby1_checkpoint_dir is not None:
            served_checkpoint = dataclasses.replace(served_checkpoint, dir=args.rby1_checkpoint_dir)
            policy_loader = "rby1_override"
        else:
            policy_loader = "default_for_env"

    policy_metadata.setdefault("served_env", args.env.value)
    policy_metadata.setdefault("policy_loader", policy_loader)
    policy_metadata.setdefault("checkpoint_config", served_checkpoint.config)
    policy_metadata.setdefault("checkpoint_dir", served_checkpoint.dir)

    logging.info(
        "Serving policy: env=%s loader=%s config=%s checkpoint_dir=%s",
        args.env.value,
        policy_loader,
        served_checkpoint.config,
        served_checkpoint.dir,
    )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
