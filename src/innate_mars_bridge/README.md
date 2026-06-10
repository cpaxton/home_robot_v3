# innate_mars_bridge

ROS2 bridge for the **Innate Mars** robot ([innate-os](https://github.com/innate-inc/innate-os)). Exposes observations and **Nav2 navigation** over ZMQ (same contract as `stretch_ros2_bridge`).

**Upstream pin:** `main` @ [innate-inc/innate-os](https://github.com/innate-inc/innate-os) — verify topics with `uv run python scripts/audit_innate_os_topics.py`.

## Capabilities

- **Pose:** base `(x, y, theta)` from `/odom`, TF for cameras and `ee_link`
- **Proprioception:** `/mars/arm/state` → 10-DoF Emet joint vector (base + arm + gripper mimic)
- **Cameras:** head stereo (`/mars/main_camera/left|right/...`), wrist `/mars/arm/image_raw`
- **Navigation:** ZMQ `xyt` → Nav2 `navigate_to_pose` (innate-os `maurice_nav`)
- **Depth:** not on ZMQ; clients use DA3 (`dynav_innate_mars.yaml`)

## Prerequisites

- ROS2 Humble
- innate-os Mars stack (`maurice_bringup`, `maurice_cam`, `maurice_arm`, `maurice_nav`)
- `emet-core` (from this repo)

## Build

```bash
# Symlink or copy src/innate_mars_bridge into a ROS workspace src/
colcon build --packages-select innate_mars_bridge
source install/setup.bash
```

## Run

```bash
ros2 launch innate_mars_bridge server.launch.py
# or
ros2 run innate_mars_bridge server --local
```

Default ZMQ ports: send **4401**, recv **4402**, state **4403**, servo **4404**.

## Emet client

```bash
uv run emet run dynagraph --robot innate_mars --robot-ip <MARS_IP> \
  --dynav-config dynav_innate_mars.yaml --export runs/mars_hw
```

See [docs/robots/innate_mars.md](../docs/robots/innate_mars.md) and [docs/robots/innate_mars_hardware.md](../docs/robots/innate_mars_hardware.md).
