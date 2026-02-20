"""Visualize images received by `RBY1SimEnvironment`.

This script pulls observations from the simulator environment and visualizes the three
camera streams (head/left_wrist/right_wrist).

Examples:
  - OpenCV windows (default):
      uv run python examples/rby1_sim/visualize_images.py --robot-ip localhost:50051

  - Subscribe to ZMQ image publisher (if you wired one up):
      uv run python examples/rby1_sim/visualize_images.py --image-source zmq --image-zmq-address tcp://127.0.0.1:5555

  - Save frames to disk (headless-safe):
      uv run python examples/rby1_sim/visualize_images.py --mode save --save-dir /tmp/rby1_frames
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import tyro

from examples.rby1_sim.env import RBY1SimEnvironment


def _chw_to_hwc_rgb(img_chw: np.ndarray) -> np.ndarray:
    img = np.asarray(img_chw)
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {img.shape}")
    if img.shape[0] not in (1, 3):
        raise ValueError(f"Expected 1 or 3 channels, got shape {img.shape}")
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@dataclasses.dataclass
class Args:
    robot_ip: str = "localhost:50051"
    prompt: str = "pick up the object"

    render_height: int = 224
    render_width: int = 224

    image_source: str = "zmq"  # "dummy" | "zmq"
    image_zmq_address: str = "tcp://127.0.0.1:5555"

    # Display mode:
    # - "cv2": three OpenCV windows
    # - "mpl": matplotlib 1x3 (works in some remote setups)
    # - "save": write PNGs to save_dir (headless-safe)
    mode: str = "mpl"  # "cv2" | "mpl" | "save"

    hz: float = 10.0
    max_frames: int = 0  # 0 = run forever

    save_dir: Path = Path("./rby1_frames")
    save_every: int = 1


def main(args: Args) -> None:
    env = RBY1SimEnvironment(
        robot_ip=args.robot_ip,
        prompt=args.prompt,
        render_height=args.render_height,
        render_width=args.render_width,
        image_source=args.image_source,
        image_zmq_address=args.image_zmq_address,
    )

    try:
        if args.mode == "cv2":
            import cv2

            # Some environments ship a headless OpenCV build (no GTK/Qt), where `imshow` is unavailable.
            # In that case, fall back to "save" mode automatically.
            try:
                cv2.namedWindow("rby1/_probe", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("rby1/_probe")
                cv2_gui_ok = True
            except Exception:
                cv2_gui_ok = False

            if not cv2_gui_ok:
                args.mode = "save"
                args.save_dir.mkdir(parents=True, exist_ok=True)
                print(
                    "OpenCV GUI support is not available (imshow not implemented). "
                    f"Falling back to saving PNGs under: {args.save_dir}. "
                    "You can also run with `--mode mpl` or `--mode save` explicitly."
                )
                # Continue into the "save" branch below.
            else:
                delay_ms = max(1, int(1000.0 / max(args.hz, 1e-6)))
                n = 0
                while args.max_frames <= 0 or n < args.max_frames:
                    obs = env.get_observation()
                    head = _chw_to_hwc_rgb(obs["observation/head_image"])
                    left = _chw_to_hwc_rgb(obs["observation/left_wrist_image"])
                    right = _chw_to_hwc_rgb(obs["observation/right_wrist_image"])

                    cv2.imshow("rby1/head (RGB)", cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
                    cv2.imshow("rby1/left_wrist (RGB)", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
                    cv2.imshow("rby1/right_wrist (RGB)", cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (ord("q"), 27):  # q or ESC
                        break
                    n += 1

                cv2.destroyAllWindows()
                return

        if args.mode == "mpl":
            import matplotlib.pyplot as plt

            plt.ion()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            ims = []
            titles = ["head", "left_wrist", "right_wrist"]
            for ax, title in zip(axs, titles, strict=True):
                ax.set_title(title)
                ax.axis("off")
                ims.append(ax.imshow(np.zeros((args.render_height, args.render_width, 3), dtype=np.uint8)))

            interval = 1.0 / max(args.hz, 1e-6)
            n = 0
            while args.max_frames <= 0 or n < args.max_frames:
                t0 = time.time()
                obs = env.get_observation()
                frames = [
                    _chw_to_hwc_rgb(obs["observation/head_image"]),
                    _chw_to_hwc_rgb(obs["observation/left_wrist_image"]),
                    _chw_to_hwc_rgb(obs["observation/right_wrist_image"]),
                ]
                for im, frame in zip(ims, frames, strict=True):
                    im.set_data(frame)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
                n += 1

                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)

        elif args.mode == "save":
            from PIL import Image

            args.save_dir.mkdir(parents=True, exist_ok=True)
            interval = 1.0 / max(args.hz, 1e-6)
            n = 0
            saved = 0
            while args.max_frames <= 0 or n < args.max_frames:
                t0 = time.time()
                obs = env.get_observation()
                frames = {
                    "head": _chw_to_hwc_rgb(obs["observation/head_image"]),
                    "left_wrist": _chw_to_hwc_rgb(obs["observation/left_wrist_image"]),
                    "right_wrist": _chw_to_hwc_rgb(obs["observation/right_wrist_image"]),
                }

                if (n % max(1, args.save_every)) == 0:
                    for name, frame in frames.items():
                        Image.fromarray(frame).save(args.save_dir / f"{saved:06d}_{name}.png")
                    saved += 1
                n += 1

                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)

        else:
            raise ValueError(f"Unsupported mode={args.mode!r}. Use 'cv2', 'mpl', or 'save'.")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))

