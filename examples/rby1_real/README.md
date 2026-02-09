# RB-Y1 Real Inference

This example runs the OpenPI policy server locally and connects a real RB-Y1 robot client with 3 RealSense cameras.

## Files
- env.py: RB-Y1 environment (RealSense capture + robot I/O)
- main.py: Runtime loop connecting to the policy server

## Required setup
- Install openpi-client (packages/openpi-client)
- Install pyrealsense2
- Install/enable rby1_sdk in the same environment
- Ensure the 3 RealSense cameras are connected to this GPU PC

## Run
1) Start the policy server (RBY1)
```
uv run scripts/serve_policy.py --env RBY1
```
2) Run the RB-Y1 client
```
python examples/rby1_real/main.py \
  --robot-ip 192.168.0.10 \
  --cam-head-serial 922612070040 \
  --cam-left-serial 838212070714 \
  --cam-right-serial 838212074317
```

## Notes
- Images are resized to 224x224 and sent as uint8.
- The observation uses camera keys: cam_head, cam_left_wrist, cam_right_wrist.
- If the model expects cam_low, add a dummy image or update transforms accordingly.
- If your rby1_sdk API differs, update env.py:
  - _create_robot()
  - _get_joint_positions()
  - _send_joint_positions()
