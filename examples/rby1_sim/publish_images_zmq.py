"""ZMQ image publisher compatible with `RBY1SimEnvironment`.

`examples/rby1_sim/env.py` expects a ZMQ SUB stream where each message is a pickled dict
with keys: `head_image`, `left_wrist_image`, `right_wrist_image`.
Each image can be HWC or CHW; dtype can be uint8 or anything convertible to uint8.

Examples:
  # Publish a changing synthetic pattern at 10 Hz (default bind port 5555)
  uv run python examples/rby1_sim/publish_images_zmq.py

  # Then in another terminal, visualize:
  uv run python examples/rby1_sim/visualize_images.py --image-source zmq --image-zmq-address tcp://127.0.0.1:5555

  # Or use the sim runtime:
  uv run python examples/rby1_sim/main.py --image-source zmq --image-zmq-address tcp://127.0.0.1:5555
"""

from __future__ import annotations

import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np
import tyro

try:
    import zmq
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyzmq is required to run this publisher.") from exc


def _pattern_frame(h: int, w: int, t: float) -> np.ndarray:
    """Simple moving RGB pattern, uint8 HWC."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx + int(t * 60)) % 256
    g = (yy + int(t * 80)) % 256
    b = ((xx // 2 + yy // 2) + int(t * 40)) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _load_image_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


@dataclasses.dataclass
class Args:
    # Bind address for PUB socket.
    bind: str = "tcp://*:5555"

    hz: float = 10.0

    # Source of frames:
    # - "pattern": synthetic moving RGB pattern
    # - "folder": loop over images in `folder`
    source: str = "pattern"  # "pattern" | "folder"

    height: int = 224
    width: int = 224

    folder: Path = Path("./frames")
    folder_glob: str = "*.png"

    # If true, publish the same frame to all three cameras.
    same_for_all: bool = True


def main(args: Args) -> None:
    context = zmq.Context.instance()
    sock = context.socket(zmq.PUB)
    sock.bind(args.bind)
    print(f"Publishing ZMQ images on {args.bind} ({args.hz} Hz), source={args.source}")

    interval = 1.0 / max(args.hz, 1e-6)

    if args.source == "folder":
        paths = sorted(args.folder.glob(args.folder_glob))
        if not paths:
            raise FileNotFoundError(f"No images found: {args.folder}/{args.folder_glob}")
        idx = 0

    n = 0
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if args.source == "pattern":
                frame = _pattern_frame(args.height, args.width, t)
            elif args.source == "folder":
                frame = _load_image_rgb(paths[idx % len(paths)])
                idx += 1
            else:
                raise ValueError(f"Unsupported source={args.source!r}. Use 'pattern' or 'folder'.")

            if args.same_for_all:
                payload = {
                    "head_image": frame,
                    "left_wrist_image": frame,
                    "right_wrist_image": frame,
                }
            else:
                payload = {
                    "head_image": frame,
                    "left_wrist_image": np.flip(frame, axis=1),
                    "right_wrist_image": np.flip(frame, axis=0),
                }

            sock.send(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            n += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"Stopped. Published {n} frames.")
    finally:
        sock.close(0)


if __name__ == "__main__":
    main(tyro.cli(Args))

