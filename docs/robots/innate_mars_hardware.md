# Innate Mars hardware bring-up (Dynagraph / DynaMem)

Use when physical Mars is on the network. Requires [innate-os](https://github.com/innate-inc/innate-os) on the robot and `innate_mars_bridge` built in a ROS 2 Humble workspace.

## Prerequisites

1. Robot running innate-os (`innate build`, drivers up: `maurice_bringup`, `maurice_cam`, `maurice_arm`, `maurice_nav` in **navigation** or **mapfree** mode).
2. Workstation or onboard PC: `colcon build --packages-select innate_mars_bridge`, `source install/setup.bash`.
3. Emet env: `uv sync` from `home_robot_v2` root.

## Bring-up sequence

| Step | Command | Pass criteria |
|------|---------|---------------|
| 1. Topic audit | `uv run python scripts/audit_innate_os_topics.py` | Expected `/mars/*` + `/odom` present |
| 2. Start bridge | `emet mars start --ip <host> --username jetson1` | ZMQ ports 4401–4404 listening |
| 3. Camera smoke | `uv run emet preview-cameras --source zmq --robot innate_mars --robot-ip <IP>` | Left/right head montage non-black |
| 4. DA3 tune | `uv run emet debug-da3-depth --robot innate_mars --robot-ip <IP>` | Depth + point cloud in Rerun; tune `da3_clip_max_m` / sky mask in `dynav_innate_mars.yaml` if needed |
| 5. Short map | `uv run emet run dynamem --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml -S` | `world/semantic_memory/pointcloud` grows |
| 6. Nav probe | Send small `xyt` via dynagraph explore or client; watch base move | `at_goal` true after goal |
| 7. Dynagraph export | `uv run emet run dynagraph --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml --export runs/mars_hw_001` | `floor_metrics.json` + graph export |

## Workstation shortcut

From the dev machine (after `uv sync`):

```bash
# First time or after bridge code changes:
emet mars start --ip herman --username jetson1 --deploy

# Routine restart (innate-os already running on robot):
emet mars start --ip herman --username jetson1

# Optional camera smoke:
emet mars start --ip herman --username jetson1 --preview

emet mars status --ip herman --username jetson1
emet mars stop --ip herman --username jetson1
```

Requires innate-os on the robot: `cd ~/innate-os && innate service start` (interactive sudo on first boot).

## Config notes

- **Depth:** Hardware has no ZMQ depth → `dynav_innate_mars.yaml` (`depth_source: auto`, `da3_stereo: true`).
- **Timeouts:** `export EMET_ZMQ_STARTUP_TIMEOUT=120` if cameras are slow to start.
- **Compare to sim:** Log `depth_source` and explored area; hardware has no `sim_object_placements` GT.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Black images | `maurice_cam` running; bridge waited for all three cameras |
| No base motion | `maurice_nav` mode; `ros2 action list \| grep navigate` |
| Sheared voxel map | `camera_K` / TF: run audit script; verify stereo poses in ZMQ |
| Phantom walls in map | Lower `da3_clip_max_m`, increase `da3_ignore_sky_fraction_top` |
