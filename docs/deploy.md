# Deploy (Stretch / Mars robot bridge + Jetson LLM)

Canonical how-to for pushing bridge code from this workstation onto a robot, or LLM/VLM
onto a LAN Jetson (e.g. AGX Orin). Flag reference: verify with `uv run emet deploy --help` and
`uv run emet mars --help`.

| Target | Command | Syncs / starts |
|--------|---------|----------------|
| **Stretch 3** | `emet deploy --robot stretch` | `src/emet_core` → `~/emet/emet_core`, `src/stretch_ros2_bridge` → `~/ament_ws/src/…`, `colcon build` |
| **Innate Mars** | `emet deploy --robot innate_mars` / `emet mars start --deploy` | same pattern with `innate_mars_bridge` → `~/innate-os/ros2_ws` |
| **LAN Orin (LLM/VLM)** | `emet deploy llm --host ORIN_HOST` | Jetson OpenAI server — **not** a robot bridge |

Do **not** confuse robot bridge deploy with the Orin LLM host: bridge = ZMQ cameras/nav; Orin = text+VL inference.

Hardware notes: [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md) · Stretch bring-up still uses Hello Robot home + `stretch_robot_home.py` (see README). Remote VL: [llm_serve.md](llm_serve.md).

Replace `STRETCH_IP`, `MARS_IP`, and `ORIN_HOST` with your LAN hostnames or IPs. Profile names (`stretch`, `mars`) are arbitrary — use whatever you like in `emet connect save … --name …`.

---

## One-time connection profile

Same `emet connect` flow for both robots — set `--robot` so bare `emet deploy` picks the right bridge.

```bash
# Stretch (native ament_ws)
uv run emet connect save STRETCH_IP --user hello-robot --name stretch \
  --robot stretch --workspace ~/ament_ws --emet-dir ~/emet

# Innate Mars (innate-os)
uv run emet connect save MARS_IP --user jetson1 --name mars \
  --robot innate_mars --workspace ~/innate-os/ros2_ws --emet-dir ~/emet \
  --config configs/agent_innate_mars.yaml

uv run emet connect list
uv run emet connect use stretch    # or: mars
uv run emet connect show
```

With an **active** profile, omit `--host` / `--ip` on `emet deploy`, `emet mars …`, `emet capture`, `emet stream`, `emet run agent`.

---

## Stretch bridge deploy + start

Prereq on the robot: ROS 2 Humble + `~/ament_ws`, Stretch drivers homed (`stretch_robot_home.py`).
Docker alternative (image from Hello Robot): `./scripts/run_stretch_ai_ros2_bridge_server.sh` on the robot — still fine; `emet deploy` is the **native** rsync + colcon path (same idea as Mars).

```bash
uv run emet connect use stretch

# Sync emet_core + stretch_ros2_bridge, colcon build:
uv run emet deploy --robot stretch
# or rely on profile robot=:
uv run emet deploy --connection stretch

# Deploy and start bridge (nohup on robot; log /tmp/emet-stretch-bridge.log):
uv run emet deploy --robot stretch --start-bridge

# Client smoke (workstation):
uv run emet capture --robot stretch --connection stretch
uv run emet stream --robot stretch --connection stretch
uv run emet run agent --robot stretch --robot-ip STRETCH_IP --rerun
```

Manual start on the robot (if you skipped `--start-bridge`):

```bash
source ~/emet/bridge_env.sh
cd ~/ament_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch stretch_ros2_bridge server.launch.py
# Stretch RE2 without gripper cam: server_no_d405.launch.py
```

---

## Mars bridge deploy + start

Prereq on the robot: innate-os up (`innate service start`), drivers including `maurice_cam` / `maurice_arm` / `maurice_nav`.

```bash
uv run emet connect use mars

# After bridge or emet_core changes (rsync + colcon + start):
uv run emet mars start --connection mars --deploy

# Routine restart (no rsync):
uv run emet mars start --connection mars

# Sync only (no bridge restart) — same as Stretch style:
uv run emet deploy --connection mars
# equivalent: uv run emet deploy --robot innate_mars --connection mars

# Health:
uv run emet mars status --connection mars
uv run emet capture --connection mars
```

`emet mars status` prints bridge/ZMQ health **and** a camera line (`head` / `wrist` / Arducam symlink). Bridge **ready** does not mean the wrist stream is live.

Optional: `--onboard-da3` (implies deploy) runs Depth Anything 3 on the Jetson; prefer this for Discord house chat so the workstation GPU stays free for voxels.

---

## Wrist camera (Mars Arducam)

EE / wrist frames come from ROS topic `/mars/arm/image_raw`, published by `maurice_cam` **ArmCameraDriver**. That driver looks for a V4L symlink under `/dev/v4l/by-id/` whose name contains **`Arducam`**.

If the Arducam is unplugged or not enumerated, you get:

```text
Camera symlink matching pattern 'Arducam' not found in /dev/v4l/by-id/
```

Symptoms on the workstation:

- `emet capture` → tiny/black `rgb_tertiary_camera_arm.jpg`
- `emet mars status` → `wrist down` · `no Arducam symlink`
- `/mars/arm/image_raw` **Publisher count: 0** (subscribers may still be >0)

**Fix (hardware / innate-os, not emet deploy):**

1. Reseat / plug the wrist USB camera; confirm a `*Arducam*` entry appears:

   ```bash
   ssh jetson1@MARS_IP 'ls -la /dev/v4l/by-id/'
   ```

2. Restart cameras (on the robot / via innate-os), e.g. relaunch `maurice_cam` (`camera_composable.launch.py`) or `innate service restart` per innate-os docs.

3. Confirm publishers:

   ```bash
   ssh jetson1@MARS_IP 'bash -lc "source ~/innate-os/ros2_ws/install/setup.bash; ros2 topic info /mars/arm/image_raw -v"'
   # Publisher count ≥ 1
   ```

4. Recapture:

   ```bash
   uv run emet capture --connection mars
   # rgb_tertiary_camera_arm.jpg should be ~tens of KB with non-zero mean
   ```

`emet deploy` / bridge restart **cannot** invent a missing USB camera. Head stereo (`3D_USB_Camera` → `/dev/video0|1`) can be healthy while the wrist is dark.

---

## Jetson LLM deploy (LAN Orin)

```bash
uv run emet deploy llm --host ORIN_HOST --profile unified-7b
uv run emet llm health --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST
```

Then chat / agent can use `--host ORIN_HOST` (see [AGENT_RUN.md](AGENT_RUN.md) / [innate_mars_hardware.md](robots/innate_mars_hardware.md#discord-chat--explore)).

---

## Quick checklist

| Step | Stretch | Mars |
|------|---------|------|
| Profile | `robot: stretch`, `~/ament_ws` | `robot: innate_mars`, innate-os ws |
| Deploy | `emet deploy --robot stretch` | `emet deploy --robot innate_mars` / `--connection mars` / `emet mars start --deploy` |
| Start | `--start-bridge` or `ros2 launch stretch_ros2_bridge …` | `emet mars start` (tmux) |
| Smoke | `emet capture --robot stretch` | `emet mars status` + `emet capture` |
| LLM (optional) | `emet deploy llm --host ORIN_HOST` → `emet llm health` green | same |

Profiles without `robot:` must pass `--robot` explicitly — bare `emet deploy` will not guess Stretch for a Mars connection.

---

## Related

- CLI: [cli.md](cli.md) (`emet deploy`, `emet mars`, `emet connect`)
- Hardware: [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md)
- Stretch install / ament symlink (legacy): [install_details.md](install_details.md)
- ZMQ / capture: [zmq_obs.md](zmq_obs.md)
- LLM: [llm_serve.md](llm_serve.md)
