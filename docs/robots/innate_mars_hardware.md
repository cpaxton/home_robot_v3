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
| 3. Camera smoke | `uv run emet capture --robot innate_mars --ip <IP>` (or `preview-cameras --source zmq …`) | Montage + `metadata.json` non-black; optional `--backend voxel_only` for one-frame Rerun map |
| 4. Live Rerun | `uv run emet stream --robot innate_mars --ip <IP>` | Cameras + MJCF mesh updating in Rerun |
| 5. Live map | `uv run emet stream --robot innate_mars --ip <IP> --backend dynamem` | Voxel map + semantic point cloud growing in Rerun |
| 6. Live graph | `uv run emet stream --robot innate_mars --ip <IP> --backend dynagraph` | Dynagraph scene graph + voxel map in Rerun (see [known_issues.md](../known_issues.md) — stationary stream may inflate node count) |
| 7. DA3 tune | `uv run emet debug-da3-depth --robot innate_mars --robot-ip <IP>` | Depth + point cloud in Rerun; tune `da3_clip_max_m` / sky mask in `dynav_innate_mars.yaml` if needed |
| 8. Short map | `uv run emet run dynamem --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml -S` | `world/semantic_memory/pointcloud` grows |
| 9. Nav probe | Send small `xyt` via dynagraph explore or client; watch base move | `at_goal` true after goal |
| 10. Dynagraph export | `uv run emet run dynagraph --robot innate_mars --robot-ip <IP> --dynav-config dynav_innate_mars.yaml --export runs/mars_hw_001` | `floor_metrics.json` + graph export |
| 11. Discord chat + explore | See [Discord chat + explore](#discord-chat--explore-herman) below | Bot replies; base moves on “explore …” |

## Discord chat + explore (Herman)

Chat with the robot over **Discord** (text + photos/maps) while it explores. Voice / robot TTS are not required.

**Prereqs:** steps 1–2 (bridge up), ideally 3–4 (cameras), and Nav2 for base motion (step 9). Prefer **`--onboard-da3`** on the bridge so the workstation GPU is free for the chat LLM + caption VLM.

```bash
# Discord extra once: uv sync  (discord group) or see docs/discord_bot.md
export DISCORD_TOKEN=...

# One-time: pin agent YAML on the profile (persona name lives in the YAML)
emet connect save 192.168.1.43 --user jetson1 --name herman \
  --robot innate_mars --workspace ~/innate-os/ros2_ws --emet-dir ~/emet \
  --config configs/agent_innate_mars.yaml

# Unified Qwen2-VL-7B on Jetson :8000 for text + captions (see docs/llm_serve.md).
# Smoke: uv run emet llm health --host caliban && uv run emet llm smoke --host caliban
# Chat:  uv run emet run chat --host caliban --once "Reply with exactly: pong"
uv run emet run agent --connection herman --host caliban
# Optional ``--rerun`` for live 3D; Discord alone is enough for chat + maps.
# ``emet run`` does not inject a default ``--robot-ip``; ``--connection herman`` supplies host + config.
# Prefer ``--onboard-da3`` on the bridge so olympia VRAM stays for voxels (caption VLM is remote).
# Memory: ``agent.memory_backend: dynagraph`` (single Dynagraph plug-in; not open_vocab + GraphEQA).
```

Profile ``config`` is shared: with herman **active**, bare ``emet run dynamem`` / ``emet stream`` (no ``--config``) also load ``agent_innate_mars.yaml``. Pass an explicit ``--config`` or use a profile without ``config`` when switching to Stretch/sim on the same machine — see [cli.md](../cli.md) (`emet connect`).

Preset enables Discord + EQA captions (`agent.eqa: true`), persona ``agent.name: Herman``, and ``agent.llm: openai``. Point at the Orin with ``--host caliban`` (or ``EMET_LLM_HOST``) so text + VL use unified-7b on ``:8000``. Voxels stay local. Optional: `--rerun` for Rerun; `--llm qwen35-4B` to force a local text router.

| Prompt (in Discord) | Expected |
|---------------------|----------|
| “what do you see?” / “describe the scene” | Caption + live head photo (`describe_scene`) |
| “explore the house” / “look around” | Nav + mapping; Discord may show `*Exploring…*` progress |
| “where is the couch?” | Memory / graph answer after some mapping |

**Turn-blocking explore:** tool calls (including `explore`) run to completion before the next Discord/terminal message is handled. Messages typed mid-explore **queue** and run when explore finishes — not a live interrupt. Progress italics can still post during long tools.

**Plugged in / tethered (no drive):** set ``EMET_BASE_ROTATE_ONLY=1`` so the agent may only yaw in place (`rotate_base`, `scan_environment`) plus perceive (`describe_scene`, send_*). Absolute XY nav and `explore` are refused.

**ZMQ nav wait:** `GenericZmqClient.move_base_to` always sends `nav_blocking=false` so the robot recv thread is not stuck in Nav2/Spin (that made Discord look hung). When the agent requests `blocking=True`, the **client** waits on `at_goal` / `nav_timeout_s` instead.

**Not this path:** Habitat/eval **EQA_EPISODE** agentic GraphEQA (`eqa.agentic_verify`) is a different orchestrator mode from Discord **CHAT** — see [AGENT_RUN.md modes](../AGENT_RUN.md#skill-library-vs-orchestrator-modes) and [evaluation.md](../evaluation.md#agentic-grapheqa-verify--offline-tuning). Canonical agent docs: [AGENT_RUN.md](../AGENT_RUN.md). Bot setup: [discord_bot.md](../discord_bot.md).

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
  uv run emet stream --backend voxel_only --connection herman
  ```

  The bridge sets `EMET_MARS_ONBOARD_DA3=1`, runs **DA3-SMALL** stereo on head cameras, and publishes JP2 `depth` on port 4401. With `depth_source: auto`, the workstation **never loads DA3** when depth is present.

  Tune on robot via env: `EMET_MARS_DA3_MODEL_ID`, `EMET_MARS_DA3_PROCESS_RES`, `EMET_MARS_DA3_INFER_EVERY_N` (default 2), `EMET_MARS_DA3_STEREO=1`. Onboard speckle: `EMET_MARS_DA3_SPECKLE_OPEN_KERNEL` (Jetson default may still be `0`; workstation Mars overlay uses `3`). See [dynav_config.md](../dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in).

- **Timeouts:** `export EMET_ZMQ_STARTUP_TIMEOUT=120` if cameras are slow to start.
- **Compare to sim:** Log `depth_source` and explored area; hardware has no `sim_object_placements` GT.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Black images | `maurice_cam` running; bridge waited for all three cameras |
| No base motion / “Turning…” but robot still | Bridge uses `NavigateToPose` in the `map` frame. If `map→odom` TF is missing (`slam_toolbox` unconfigured / AMCL not publishing), goals hang then time out. Check: `ros2 run tf2_ros tf2_echo map odom` (must print transforms). In-place yaw can still work via Nav2 `/spin` (bridge prefers Spin for relative yaw-only). Restore localization, or `emet deploy` after bridge Spin fix. Verify: `ros2 action list \| grep -E 'navigate\|spin'`. |
| Sheared voxel map / sloped floor in Rerun | ``camera_K`` must match JPEG resolution. ``maurice_cam`` ``camera_info`` is often ~320×240 while streams are 640×480; bridge scales K to the decoded image (workstation client also auto-aligns). Restart stream after deploy. |
| Floating mid-air points in ``world/point_cloud`` | DA3 speckle / flying pixels on hardware. Mars defaults enable mild filters (`depth_speckle_open_kernel: 3`, `voxel_pcd_dbscan_min_samples: 6`) in [`configs/emet/default.yaml`](../configs/emet/default.yaml). Disable with ``--set mapping.filters.depth_speckle_open_kernel=0`` if real structure is eroded. See [dynav_config.md](../dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in). |
| Curved / bowed walls (RGB looks flat) | Usually DA3 + lighting/intrinsics, not stream misconfig. ``emet debug-da3-depth``; try ``DA3METRIC-LARGE`` or sensor depth in sim for A/B. [innate_mars.md](innate_mars.md). |
| Red/blue swapped in Rerun RGB | Bridge must convert ROS ``bgr8`` → RGB before ``to_jpg`` (``innate_mars_bridge/ros/camera.py``) |
| MJCF mesh head frozen / map vs model misaligned | **Hardware only:** ``joint_head`` inferred from ``camera_pose``; ``head_visual`` shifted forward (~70 mm). **Sim** keeps vanilla ``innate_mars.xml`` (head STL at neck pivot — may look slightly off vs cameras; arm/base replay unchanged). Hardware body gets **+90° Z** visual fix; sim does not. Meshes are dark semi-transparent; base sphere/arrow off by default. ``uv run python scripts/debug_innate_mars_head_align.py --ip <host>`` |
| Black / missing EE (wrist) camera in Rerun | ``ros2 topic info /mars/arm/image_raw -v`` — need **Publisher count ≥ 1** (``maurice_cam`` arm stream). Bridge fills black JPEGs when absent. ``emet view-bridge`` or montage third panel shows ``ee_cam/color_image``. |
