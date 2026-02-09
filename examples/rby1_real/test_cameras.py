"""Test and save RGB images from all 3 RealSense cameras.

Usage:
    python examples/rby1_real/test_cameras.py
    
Output:
    Saves images to debug_images/ directory
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found. Install Intel RealSense SDK.")
    exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Camera serials from config.yaml
CAMERAS = {
    "cam_head": "922612070040",
    "cam_left_wrist": "838212070714", 
    "cam_right_wrist": "838212074317",
}

WIDTH = 640
HEIGHT = 480
FPS = 30


def capture_image(serial: str, name: str) -> np.ndarray:
    """Capture one frame from RealSense camera."""
    logger.info(f"Capturing from {name} ({serial})...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    
    try:
        # Start pipeline
        pipeline.start(config)
        logger.info(f"  Pipeline started")
        
        # Warmup: skip first few frames
        for i in range(30):
            pipeline.wait_for_frames(timeout_ms=1000)
        logger.info(f"  Warmup complete")
        
        # Capture actual frame
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            raise RuntimeError(f"No color frame from {name}")
        
        # Convert to numpy array
        image = np.asanyarray(color_frame.get_data())
        logger.info(f"  ✓ Captured: {image.shape} dtype={image.dtype}")
        
        return image
        
    finally:
        pipeline.stop()
        logger.info(f"  Pipeline stopped")


def save_image(image: np.ndarray, output_path: Path):
    """Save numpy array as PNG image."""
    # Convert RGB numpy array to PIL Image
    pil_image = Image.fromarray(image, mode='RGB')
    pil_image.save(output_path)
    logger.info(f"  Saved to: {output_path}")


def main():
    # Create output directory
    output_dir = Path("debug_images")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*60)
    logger.info("RBY1 Camera Test")
    logger.info("="*60)
    
    results = {}
    
    # Capture from all cameras
    for name, serial in CAMERAS.items():
        try:
            image = capture_image(serial, name)
            
            # Save image
            filename = f"{name}_{timestamp}.png"
            output_path = output_dir / filename
            save_image(image, output_path)
            
            results[name] = {
                "success": True,
                "shape": image.shape,
                "path": output_path
            }
            
        except Exception as e:
            logger.error(f"Failed to capture from {name}: {e}")
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    logger.info("")
    logger.info("="*60)
    logger.info("Summary")
    logger.info("="*60)
    
    success_count = sum(1 for r in results.values() if r["success"])
    logger.info(f"Captured: {success_count}/{len(CAMERAS)} cameras")
    
    for name, result in results.items():
        if result["success"]:
            logger.info(f"  ✓ {name}: {result['shape']} -> {result['path']}")
        else:
            logger.info(f"  ✗ {name}: {result['error']}")
    
    if success_count == len(CAMERAS):
        logger.info("")
        logger.info("✓ All cameras working!")
        logger.info(f"Check images in: {output_dir.absolute()}")
    else:
        logger.warning("")
        logger.warning(f"Only {success_count}/{len(CAMERAS)} cameras working")


if __name__ == "__main__":
    main()
