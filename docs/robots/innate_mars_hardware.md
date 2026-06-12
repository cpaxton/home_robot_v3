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
| 3. Camera smoke | `uv run emet capture --robot innate_mars --ip <IP>` (or `preview-cameras --source zmq …`) | Montage + `metadata.json` non-black; optional `--map` for one-frame Rerun map |
| 4. Live Rerun | `uv run emet stream --robot innate_mars --ip <IP>` | Cameras + MJCF mesh updating in Rerun |
| 5. Live map | `uv run emet stream --robot innate_mars --ip <IP> --map --dynav-config dynav_innate_mars.yaml` | Voxel map + semantic point cloud growing in Rerun |
| 6. Live graph | `uv run emet stream --robot innate_mars --ip <IP> --graph --dynav-config dynav_innate_mars.yaml` | Dynagraph scene graph + voxel map in Rerun |
| 7. DA3 tune | `uv run emet debug-da3-depth --robot innate_mars --robot-ip <IP>` | Depth + point cloud in Rerun; tune `da3_clip_max_m` / sky mask in `dynav_innate_mars.yaml` if needed |
| 8. Short map | `uv run emet run dynamem --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml -S` | `world/semantic_memory/pointcloud` grows |
| 9. Nav probe | Send small `xyt` via dynagraph explore or client; watch base move | `at_goal` true after goal |
| 10. Dynagraph export | `uv run emet run dynagraph --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml --export runs/mars_hw_001` | `floor_metrics.json` + graph export |

## Workstation shortcut

From the dev machine (after `uv sync`):

**`--deploy` / `emet deploy`** rsyncs `src/emet_core` → `~/emet/emet_core` (with `--delete`), syncs `src/innate_mars_bridge` into the ROS workspace, writes `~/emet/bridge_env.sh` (`PYTHONPATH=~/emet/emet_core:~/emet/src`), runs `colcon build`, then smoke-tests imports on the robot. Run this after any bridge or `emet_core` change; routine restarts can omit it.

```bash
# First time or after bridge / emet_core changes:
emet mars start --ip herman --username jetson1 --deploy

# Sync only (no bridge restart):
emet deploy --connection herman

# Routine restart (innate-os already running on robot):
emet mars start --ip herman --username jetson1

# Optional camera smoke:
emet mars start --ip herman --username jetson1 --preview

emet mars status --ip herman --username jetson1
emet mars stop --ip herman --username jetson1
```

Requires innate-os on the robot: `cd ~/innate-os && innate service start` (interactive sudo on first boot).

## Config notes

- **Depth:** Hardware has no hardware depth sensor → default `dynav_innate_mars.yaml` uses `depth_source: auto` + DA3 stereo on the **workstation** when ZMQ omits depth.
- **Onboard DA3 (Jetson):** Run inference on the robot to save workstation GPU:

  ```bash
  # One-time: sync perception code + install depth-anything-3 on Jetson
  emet mars start --connection herman --deploy --onboard-da3

  # Workstation mapping (uses ZMQ depth; no local DA3 load):
  uv run emet stream --map-only --connection herman
  ```

  The bridge sets `EMET_MARS_ONBOARD_DA3=1`, runs **DA3-SMALL** stereo on head cameras, and publishes JP2 `depth` on port 4401. With `depth_source: auto`, the workstation **never loads DA3** when depth is present.

  Tune on robot via env: `EMET_MARS_DA3_MODEL_ID`, `EMET_MARS_DA3_PROCESS_RES`, `EMET_MARS_DA3_INFER_EVERY_N` (default 2), `EMET_MARS_DA3_STEREO=1`.

- **Timeouts:** `export EMET_ZMQ_STARTUP_TIMEOUT=120` if cameras are slow to start.
- **Compare to sim:** Log `depth_source` and explored area; hardware has no `sim_object_placements` GT.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Black images | `maurice_cam` running; bridge waited for all three cameras |
| No base motion | `maurice_nav` mode; `ros2 action list \| grep navigate` |
| Sheared voxel map / sloped floor in Rerun | ``camera_K`` must match JPEG resolution. ``maurice_cam`` ``camera_info`` is often ~320×240 while streams are 640×480; bridge scales K to the decoded image (workstation client also auto-aligns). Restart stream after deploy. |
| Red/blue swapped in Rerun RGB | Bridge must convert ROS ``bgr8`` → RGB before ``to_jpg`` (``innate_mars_bridge/ros/camera.py``) |
| MJCF mesh head frozen / map vs model misaligned | ZMQ ``joint_head`` drives MJCF ``joint_head`` (bridge prefers ``base_link``→``head`` TF over a stale ``/mars/head/current_position``). Rerun MJCF replay uses TF-calibrated stereo mounts, not the sim table-forward hack. Sim meshes face **−base Y** (table-forward); hardware replay applies **+90° Z** so visuals match ROS **+X** forward (same as ``camera_pose`` / voxel map). Meshes use MJCF material colors (not flat white). Run ``uv run python scripts/debug_innate_mars_head_align.py --ip <host>`` (sim ≈0 mm; hardware target ≈0.5 mm in base frame). |
| Black / missing EE (wrist) camera in Rerun | ``ros2 topic info /mars/arm/image_raw -v`` — need **Publisher count ≥ 1** (``maurice_cam`` arm stream). Bridge fills black JPEGs when absent. ``emet view-bridge`` or montage third panel shows ``ee_cam/color_image``. |
