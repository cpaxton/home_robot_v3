# innate_mars_bridge

ROS2 bridge for the **Innate Mars** robot (from the [innate-os](https://github.com/innate-os) codebase). Exposes observations over ZMQ in the same style as `stretch_ros2_bridge`:

- **Pose**: base pose (x, y, theta) from `/odom`, optional TF lookups for map frame
- **Proprioception**: arm joint positions and velocities from `/mars/arm/state`
- **Cameras**:
  - Head left: `/mars/main_camera/left/image_raw` (+ camera_info)
  - Head right: `/mars/main_camera/right/image_raw` (+ camera_info)
  - EE (wrist) camera: `/mars/arm/image_raw`

## Prerequisites

- ROS2 Humble (or compatible)
- Innate Mars stack running (maurice_arm, maurice_cam, odom/TF)
- Python package `emet` (from this repo) for `BaseZmqServer` and compression

## Build

From a ROS2 workspace that includes this repo (or a symlink to `src/innate_mars_bridge`):

```bash
colcon build --packages-select innate_mars_bridge
source install/setup.bash
```

## Run

Start the ZMQ server (default: send 4401, recv 4402):

```bash
ros2 run innate_mars_bridge server
```

Or with the launch file:

```bash
ros2 launch innate_mars_bridge server.launch.py
```

Ensure the Innate Mars drivers are running (arm state, main camera left/right, arm camera, and `/odom` or TF) before starting the bridge.
