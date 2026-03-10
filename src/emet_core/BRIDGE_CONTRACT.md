# Bridge contract (ZMQ server)

Any **robot bridge** or **simulator bridge** that talks to the Emet agent over ZMQ must implement the same contract. Robot bridges: **Stretch** → `stretch_ros2_bridge`, **Innate Mars** → `innate_mars_bridge`. Simulator example: MuJoCo server. This keeps the agent backend-agnostic.

## Base class

Subclass `emet.core.server.BaseZmqServer` and implement the abstract methods.

## Ports (defaults)

- **send_port** 4401: full observations (get_full_observation_message)
- **recv_port** 4402: actions (handle_action)
- **send_state_port** 4403: lightweight state (get_state_message)
- **send_servo_port** 4404: servo/visual stream (get_servo_message)

## Required methods

| Method | Returns | Purpose |
|--------|---------|--------|
| `get_control_mode()` | `str` | `"manipulation"`, `"navigation"`, or `"none"` |
| `get_full_observation_message()` | `dict` | Full obs (images, pose, joints, etc.) for logging / heavy clients |
| `get_state_message()` | `dict` | Small state (pose, joints, at_goal, etc.) for fast loop |
| `get_servo_message()` | `dict` | Images + poses + robot config for policy/visual servoing |
| `handle_action(action: dict)` | None | Execute one action from the client |
| `is_running()` | `bool` | True while the server should keep running |

## Message shapes (conventions)

Clients expect these keys when present. All bridges should use the same keys so the agent can treat backends interchangeably.

### get_state_message()

- `base_pose`: (x, y, theta) or 4x4 matrix
- `ee_pose`: 4x4 end-effector pose in world/map frame
- `joint_positions`: 1D array
- `joint_velocities`: 1D array
- `control_mode`: str
- `step`: int (optional)
- `at_goal`: bool (optional)
- `is_homed` / `is_runstopped`: bool (optional)

### get_servo_message()

- `ee/pose`: 4x4
- `robot/config`: joint position array
- `step`: int
- `head_cam/color_image`: compressed (e.g. jpg bytes)
- `head_cam/color_camera_K`: 3x3 numpy
- `head_cam/pose`: 4x4
- `head_cam/color_image/shape`: (H, W, C)
- `head_cam/image_scaling`: float
- Same pattern for `head_cam_left/`, `head_cam_right/`, `ee_cam/` when the robot has multiple head cams or an EE camera.
- Depth keys (`head_cam/depth_image`, `head_cam/depth_camera_K`, etc.) when depth is available.

### get_full_observation_message()

- At least: images (e.g. `rgb`, `depth` or per-cam), `camera_pose` or per-cam poses, `joint`, `joint_velocities`, `ee_pose`, `base_pose` or `gps`/`compass`, `step`, `recv_address`.

### handle_action(action)

- Common keys: `xyt`, `nav_relative`, `nav_blocking`, `joint`, `gripper`, `head_to`, `posture`, `control_mode`, `say` / `say_sync`, `base_velocity`. Bridges ignore unsupported keys.

## Adding a new backend

1. Depend only on **emet-core** (and your backend SDK: ROS2, mujoco, etc.).
2. Subclass `BaseZmqServer`, implement the six methods above.
3. Map your backend’s observations into the same dict keys; map action keys to your backend’s commands.
4. No need for torch, perception, or LLM in the bridge process.
