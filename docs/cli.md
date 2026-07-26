# Emet CLI Tool

The `emet` CLI makes it easy to start simulations, run robot agents, sync dependencies, view logs, and run tests. It supports **tab completion** for bash, zsh, and fish (see [Tab completion](#tab-completion) below).

## Installation

After installing emet (`uv sync` or `pip install -e .`), the `emet` command is available:

```bash
uv run emet --help
```

### Run from this repo

Commands in this doc assume you are in the **project root** with dependencies synced (`uv sync`).

- Prefer **`uv run emet …`** so you use **this checkout’s** code and `.venv`.
- After `source .venv/bin/activate`, bare **`emet …`** is equivalent.
- If **`emet run dynagraph --help`** (via `uv run python -m emet.app.run_dynagraph --help`) lists **`--explore-loop`** but bare **`emet`** does not, your PATH points at another install (e.g. an old clone). Run `which emet` and stick to **`uv run emet`** here. Details: [TESTING.md](TESTING.md#run-from-this-repo).

## Quick Start

```bash
# 1. Start the MuJoCo simulation server (in one terminal)
uv run emet serve mujoco
# Innate Mars (default table + robot): same command with --robot
uv run emet serve --robot innate_mars --headless   # optional: default backend is mujoco

# 2. Run DynaMem with visual servoing (in another terminal)
uv run emet run dynamem --robot-ip 127.0.0.1 -S --visual-servo
# With Innate Mars sim, pass the same robot to the client:
uv run emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S

# 3. Or run mapping
uv run emet run mapping --robot-ip 127.0.0.1

# Optional: DA3 depth + point cloud in Rerun (sim must be running; depth-anything-3 is a default dep)
uv run emet debug-da3-depth --robot innate_mars
# Equivalent:
uv run emet run debug-da3-depth --robot innate_mars
```

If port 4401 is already in use: `uv run emet kill-mujoco-server` then retry, or `uv run emet serve mujoco --port-offset 100`.

## Commands

### `emet serve [backend]`

Start a simulation server.

| Backend | Description |
|---------|--------------|
| `mujoco` | MuJoCo ZMQ simulation (default) |
| `robocasa` | Shortcut for `mujoco --scene robocasa` (kitchen scenes) |
| `molmospaces` | Shortcut for `mujoco --scene ithor` (merge Molmo scene + robot, then ZMQ) |

The positional backend is optional (`emet serve` defaults to **mujoco**). Optional scene name after `molmospaces`: `emet serve molmospaces procthor-10k`.

**Options:**
- `--robot NAME` — Simulator robot (default `stretch` for table, Robocasa, and MolmoSpaces when omitted). Client apps (`emet run dynamem`, `emet run dynagraph`, `emet run agent`) may **omit `--robot`** when the running ZMQ server publishes `emet_robot_id`. Unified config: **`--config`** (default `configs/emet/default.yaml`; env **`EMET_CONFIG`**). Innate Mars depth tuning is under **`robots.innate_mars`** in that file (see [emet_config.md](emet_config.md)).
- `--scene NAME|PATH` — Scene selector (see table below)
- `--split` — `train` / `val` / `test` when `--scene` is a MolmoSpaces catalog name (default: `train`)
- `--index N` — Scene index within split when `--scene` is MolmoSpaces (default: `0`)
- `--install-scene-if-missing` — Download/link MolmoSpaces scene assets if missing
- `--robocasa-task NAME` — Robocasa task when `--scene robocasa` (default: PickPlaceCounterToCabinet)
- `--list-robocasa-tasks` — Print all Robocasa task names and exit (for use with `--robocasa-task`)
- `--headless` — Run without native viewer (use web at http://localhost:9090?url=ws://localhost:9877)
- `--port-offset N` — Add N to default ports (e.g. 100 → 4501–4504) when 4401 is busy
- `--seed N` — Random seed (default: 0)

**`--scene` values** (for `emet serve mujoco` and `emet run agent --start-sim`):

| Value | Launch kind | Notes |
|-------|-------------|--------|
| *(omit)* | Default table | Packaged `scene_environment.xml` (red cylinder, blue cube, wood floor) |
| `default`, `table` | Default table | Same as omit |
| `robocasa` | Robocasa kitchen | Use `--robocasa-task` (default `PickPlaceCounterToCabinet`); shortcut: `emet serve robocasa` |
| `ithor` | MolmoSpaces | iTHOR / AI2-THOR; use `--split` / `--index` / `--install-scene-if-missing` |
| `procthor-10k` | MolmoSpaces | ProcTHOR-10K |
| `procthor-objaverse` | MolmoSpaces | ProcTHOR + Objaverse objects |
| `holodeck-objaverse` | MolmoSpaces | Holodeck + Objaverse |
| `/path/to/scene.xml` | Custom MJCF | Merged or standalone scene file (must exist on disk) |

Molmo-only flags (`--split`, `--index`, `--install-scene-if-missing`) are ignored for other scene kinds.

**Removed flags** (use `--scene` instead): `--use-robocasa`, `--molmospaces-scene`, `--molmospaces-split`, `--molmospaces-index`, `--molmospaces-install`, `--scene-path`.

See [Simulation](simulation.md), [MolmoSpaces](molmospaces.md), and maintainer [simulation_modules.md](simulation_modules.md).

**Examples:**
```bash
emet serve                          # MuJoCo, default scene, Stretch
emet serve mujoco --headless        # No native viewer
emet serve --robot innate_mars --headless   # Innate Mars + default table (match client --robot)
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S --dynav-config dynav_innate_mars.yaml   # DynaMem + DA3 (no ZMQ depth)
emet serve mujoco --scene robocasa    # Robocasa scene
emet serve robocasa --robot galaxea_r1 --headless
emet serve molmospaces --headless   # Molmo iTHOR + stretch (default robot)
emet serve mujoco --scene ithor --index 0 --headless
emet serve mujoco --scene ithor --robot rby1 --headless   # Galaxea R1 on Molmo scene
```

---

### `emet dataset <subcommand>`

MolmoBot-Data and other learning dataset tools. See [datasets/molmobot.md](datasets/molmobot.md).

| Subcommand | Description |
|------------|-------------|
| `molmobot inspect` | Episode stats from H5 root or file |
| `molmobot list-trajs` | Trajectory keys in one batch H5 |
| `molmobot export-lerobot` | Export JSONL episodes for LeRobot preprocessing |
| `molmobot replay` | Open-loop joint replay against ZMQ sim |

---

### `emet robots <subcommand>`

Inspect **emet robot registry** backends (MJCF path, camera wiring, stereo pair). Complements `emet molmospaces list-robots` (wrapper static IDs).

| Subcommand | Description |
|------------|-------------|
| `list` | All canonical robots (`stretch`, `innate_mars`, `xlerobot`, `franka_fr3`, …) with MJCF + camera names |
| `info ROBOT` | Detailed spec vs MJCF camera ids, stereo right cam, planar base, spawn metadata |
| `preview-cameras ROBOT` | Shortcut for `emet preview-cameras --robot ROBOT` (forwards extra flags) |

**Examples:**
```bash
emet robots list
emet robots info xlerobot
emet robots preview-cameras xlerobot --source local --out /tmp/xlerobot_cams.png
emet robots preview-cameras xlerobot --source zmq --robot-ip 127.0.0.1
```

---

### `emet molmospaces <subcommand>`

MolmoSpaces scene setup, merge, passive sim, and offline maintainer tools. See [molmospaces.md](molmospaces.md), [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md), and [simulation_modules.md](simulation_modules.md).

| Subcommand | Wrapper? | Description |
|------------|----------|-------------|
| `list-robots` | No (core) | MolmoSpaces wrapper IDs **and** emet registry robots with MJCF (`xlerobot`, `franka_fr3`, …) |
| `list-scenes` | Yes | Scene names and split sizes |
| `install-scene` | Yes | Download/install a scene XML |
| `merge-scene` | Yes | Merge scene + robot MJCF (`-o` required) |
| `serve` | Yes | Passive MuJoCo (wrapper viewer; not ZMQ agent stack) |
| `build-occ-map` | No (core) | Orthographic occupancy PNG + meta for spawn QA |
| `write-spawn-metadata` | No (core) | Measure and write `molmospaces_spawn.json` |
| `export-nerfstudio` | No (core) | `transforms.json` from explore episode dir |

**Examples:**
```bash
emet molmospaces list-robots
emet molmospaces merge-scene --scene ithor --robot stretch -o /tmp/ithor_stretch.xml
emet molmospaces write-spawn-metadata --robot stretch --mjcf /tmp/ithor_stretch.xml
emet molmospaces build-occ-map /tmp/ithor_stretch.xml
emet molmospaces serve --scene ithor --viewer   # default robot: stretch
```

For **agent + tools**, use **`emet serve mujoco --scene ithor …`** (ZMQ) instead of passive `emet molmospaces serve`.

---

### `emet run <app> [options]`

Run a robot agent or app.

| App | Description |
|-----|--------------|
| `agent` | Embodied LLM agent + tools (optional Discord, opt-in Rerun). See [AGENT_RUN.md](AGENT_RUN.md). |
| `dynamem` | DynaMem navigation + manipulation |
| `graph-eqa` | Graph-based EQA memory (see [graph_eqa.md](graph_eqa.md)) |
| `dynagraph` | Graph EQA + merge/staleness ([dynagraph.md](dynagraph.md)); **`--explore-loop`**, **`--export`**, **`--question`** |
| `mapping` | 3D mapping and exploration |
| `grasp` | Grasp object (red cylinder demo) |
| `chat` | LLM chat with robot |
| `ai_pickup` | AI-powered pickup |
| `timing` | Network timing test |
| `debug-da3-depth` | Live DA3 depth + point cloud to Rerun (same as `emet debug-da3-depth …`) |

**Common options:**
- `--robot` — Robot backend (`stretch`, **`innate_mars`**, `rby1`, …). **Must match** the simulator: `emet serve --robot <name>` and `emet run dynamem --robot <name>` use the same registry key.
- `--robot-ip` / `--robot_ip` — Robot or simulator IP (default: 127.0.0.1)
- `--server-ip` / `--server_ip` — Server IP for AnyGrasp (dynamem)
- `-S, --skip` — Skip confirmations
- `--headless` — Run without display
- `--visual-servo` / `-V` / `--visual_servo` — Use visual servoing (dynamem)
- `--target-object` / `--target_object` — Target object (grasp)
- `--parameter-file` / `--parameter_file` — Planner config (e.g. sim_planner.yaml)

Unknown options (e.g. `--match-method`, `--rerun-debug`) are passed through to the underlying app.

**Examples:**
```bash
emet run agent --robot stretch --robot-ip 127.0.0.1
emet run agent --start-sim -c "describe the scene"
emet run agent --memory-backend dynagraph --config configs/agent_stretch_discord.yaml
emet run dynamem --robot-ip 127.0.0.1 -S
emet run dynamem -S --visual-servo --match-method class --rerun-debug
emet run mapping --robot-ip 127.0.0.1
emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
emet run timing --robot-ip 192.168.1.15 --headless
emet run debug-da3-depth --robot innate_mars --depth-source sensor
```

---

### `emet debug-da3-depth [options]`

Stream head camera(s) from the ZMQ server through **Depth Anything 3** (or sim depth) and log left RGB, colormapped depth, and a strided world-frame point cloud to **Rerun**. Uses the same `resolve_depth_map` path as DynaMem, so it is the quickest way to confirm intrinsics, poses, and stereo wiring before chasing exploration failures.

**Prerequisites:** Sim or robot bridge running (`emet serve mujoco --robot innate_mars --headless`). The `depth-anything-3` package is installed with the project (`uv sync`); first run may download model weights from Hugging Face.

**Useful options:**
- `--meshes` / `--no-meshes` (default: on) — log robot **visual meshes** from the robot MJCF under `da3/robot/mesh/…` in world frame (needs `mujoco` + `uv sync --extra sim`) so you can check alignment with the point cloud.
- `--depth-source da3` (default) — run DA3; `--depth-source sensor` uses rendered sim depth (no DA3) for A/B checks.
- `--model-id` — default `depth-anything/DA3-SMALL` for speed; use `DA3METRIC-LARGE` when you need metric calibration.
- `--process-res` — default `378` (faster); raise for sharper depth at more compute.
- `--da3-stereo` — default **off** (monocular DA3, matches dynav); pass `--da3-stereo` for two-view inference when wiring stereo depth.
- `--hz`, `--stride` — cap FPS and point-cloud density.

**Examples:**
```bash
emet debug-da3-depth --robot innate_mars
emet debug-da3-depth --robot innate_mars --depth-source sensor
emet debug-da3-depth --model-id depth-anything/DA3METRIC-LARGE --process-res 504 --hz 2
```

---

### `emet mars [start|status|stop]`

Deploy and manage the **Innate Mars ZMQ bridge** on a Jetson running innate-os. See [Innate Mars hardware bring-up](robots/innate_mars_hardware.md) for full recipes (`--deploy`, `--onboard-da3`, Herman connection profile).

```bash
emet mars start --ip herman --username jetson1 --deploy
emet mars status --connection herman
emet mars stop --connection herman
```

`start` and `status` print a compact one- or two-line summary (bridge state, ZMQ ports, optional ROS hint, suggested next command). Colors follow Click when stdout is a TTY; set `NO_COLOR=1` or `EMET_NO_COLOR=1` to disable.

---

### `emet connect [save|list|show]`

SSH / deploy **connection profiles** stored in ``~/.stretch/connection.json`` (field reference: ``src/emet/utils/connection.py``).

| Command | Purpose |
|---------|--------|
| `emet connect save HOST …` | Create or update a profile; default sets it **active** |
| `emet connect list` | List profiles and mark which is active |
| `emet connect show` | Print the active profile |

**`emet connect save` flags**

| Flag | Stored as | Notes |
|------|-----------|--------|
| `--user` / `-u` | `user` | SSH login (default `root`) |
| `--password` / `-p` | `password` | Optional; else `EMET_ROBOT_PASSWORD` or SSH keys at runtime |
| `--name` / `-n` | profile key | Default: hostname/IP |
| `--robot` | `robot` | Emet robot id (e.g. `innate_mars`) for CLI defaults |
| `--workspace` | `workspace` | Remote ROS2 workspace (Mars: `~/innate-os/ros2_ws`) |
| `--emet-dir` | `emet_dir` | Remote emet install root (default `~/emet`) |
| `--no-active` | — | Save without setting active or updating `robot_ip.txt` |

**Examples:**
```bash
emet connect save herman --user jetson1 --name herman \
  --robot innate_mars --workspace ~/innate-os/ros2_ws --emet-dir ~/emet
emet connect list
emet connect show
```

Used by `emet deploy`, `emet mars start`, `emet capture`, `emet stream`, and `emet preview-cameras` when `--ip` / `--host` is omitted (active profile, or `--connection NAME`).

---

### `emet capture` / `emet stream`

Both commands are **profile shortcuts** into the same runner ([`zmq_obs.md`](zmq_obs.md) · `emet.app.zmq_obs`): shared ZMQ resolution, mapping session, and backend factory. Open bugs: [`known_issues.md`](known_issues.md).

| | **`emet capture`** | **`emet stream`** |
|---|-------------------|-------------------|
| **Artifact save** | Always (montage + `metadata.json`) | Only with `--out-dir` |
| **Mapping** | One update when `--backend` is set | Loop at `--hz` until Ctrl+C or `--max-steps` |
| **No `--backend`** | Save frame and exit | localhost → cameras-only Rerun; remote → `dynamem` |
| **Rerun hold** | `--rerun-hold-s` (default 30s) after map | Continuous viewer |

Full architecture, backend table, and hardware examples: **[`docs/zmq_obs.md`](zmq_obs.md)**.

### `emet capture [options]`

One-shot **ZMQ smoke test** for any robot backend: subscribe once on the observation port (default **4401**), save a labeled camera montage + per-camera JPEGs + `metadata.json` (joints, poses, GPS/compass when present). Optional **`--backend dynamem`** or **`--backend voxel_only`** runs a single mapping `update()` and opens Rerun (same depth stack as `emet stream`).

**Defaults:** `--ip 127.0.0.1`, `--robot stretch`. With a saved connection (`emet connect save …`) and no `--ip`, uses the active profile host (and `robot:` from the profile when `--robot` is omitted).

| Flag | Meaning |
|------|--------|
| `--ip` / `--robot-ip` | ZMQ host (default localhost sim) |
| `--connection` | Saved profile name (overrides host when `--ip` omitted) |
| `--robot` | Backend (`stretch`, `innate_mars`, `rby1`, …) |
| `--out-dir` | Output dir (default `runs/capture/<robot>_<timestamp>/`) |
| `--backend` | After capture, one ``emet stream``-style update (any stream backend; Rerun unless `--no-rerun`) |
| `--dynav-config` | Dynav YAML for `--backend` (hardware Mars: auto `dynav_innate_mars.yaml`) |

**Examples:**
```bash
emet capture
emet capture --robot innate_mars --ip herman
emet capture --connection herman --backend voxel_only --no-rerun
emet capture --robot stretch --backend dynamem --no-rerun --out-dir /tmp/cap
```

---

### `emet stream [options]`

**Live** ZMQ → Rerun viewer: head/stereo/arm cameras, base pose, and MJCF mesh (when the robot spec provides one). Runs until Ctrl+C.

**Mapping backends** run a continuous ``agent.update()`` loop (same controllers as ``emet run dynamem`` / ``emet run dynagraph``, without rotate/explore/nav). Use ``--backend`` with the same names as paper evals:

| `--backend` | Stack |
|-------------|--------|
| `dynamem` | Voxel semantic map |
| `voxel_only` | Voxel + depth only (no SigLIP/YoloE/VLM; DA3 on hardware) |
| `graph_eqa` | Voxel + GraphEQA graph |
| `dynagraph` | Voxel + merged graph + VLM |
| `ground_truth` | Sim GT graph from `emet_session` |
| `svm` | Instance memory |
| `scene_graph` | Voxel + open-vocab scene graph |

| Situation | Behavior |
|-----------|----------|
| localhost, no `--backend` | Cameras + mesh only |
| remote host, no `--backend` | **Defaults to `--backend dynamem`** |
| `--cameras-only` | Cameras + mesh (no mapping), even on remote |

Updates run at **`--hz`** (default 1 Hz). Other flags:

| Flag | Meaning |
|------|--------|
| `--dynav-config` | Planner/dynav YAML (hardware Mars: auto `dynav_innate_mars.yaml`) |
| `--hz` | Update rate (default 1) |
| `--max-steps` | Stop after N updates (0 = Ctrl+C) |
| `--out-dir` | Optional: save one montage + `metadata.json` before streaming |
| `--cameras-only` | No mapping loop |
| `--compare-to-gt` | Dynagraph: overlay sim GT reference |
| `--headless` | Web server only, no auto-open browser |
| `--rerun-bind` | Listen on 0.0.0.0 for remote viewing |
| `--verbose` | Per-step status + DA3 INFO timing (default: quiet, status every 5s) |

**Examples:**
```bash
emet stream --cameras-only
emet stream --connection herman
emet stream --connection herman --backend voxel_only
emet stream --connection herman --backend dynagraph
emet stream --connection herman --backend graph_eqa
emet stream --robot stretch --backend svm
```

For DA3 depth debug (no voxel map), use ``emet debug-da3-depth`` instead.

---

### `emet preview-cameras [options]`

Build a **labeled horizontal montage** of the robot’s MuJoCo/ZMQ cameras (for Innate Mars: `head_left`, `head_right`, `camera_arm`) to check orientation, stereo wiring, and tabletop aim without running a full agent loop. Implements `emet.app.preview_robot_cameras`; options are passed through (see `emet preview-cameras -h`).

**Modes**
- **`--source local`** (default) — Load the same **merged** model as `emet serve mujoco` (`scene_environment.xml` + robot MJCF), render with MuJoCo at 640×480, and apply the same RGB postprocess as `RobosuiteZmqServer` (per robot: `RobotSpec.robosuite_rgb_depth_ops`; innate_mars uses **`flipud`** on MuJoCo `Renderer` output; robots with empty ops may still honor optional `EMET_ROBOSUITE_RENDER_FLIPUD`).
- **`--source zmq`** — Subscribe once on the **full observation** port (default **4401**, same as `GenericZmqClient`), decode JPEG fields, and montage. Requires a running sim or bridge. Newer `RobosuiteZmqServer` builds also attach a third JPEG (`rgb_tertiary`, `camera_name_tertiary`) when the spec lists a distinct third camera.

**Common options:** `--robot`, `--connection` (saved profile host/robot, same as `capture`/`stream`), `--out` (single PNG), `--max-cams`, `--row-height`, `--recv-port` / `--timeout-ms` (ZMQ), `--discord` (post the single montage; needs `DISCORD_TOKEN`, `EMET_DISCORD_CHANNEL`).

**Head nod capture (`--nod`, local only)**

Sweeps the sim **head hinge** (`joint_head` on Innate Mars) and writes **one montage PNG per pose** so you can scrub a nod in the image viewer or ffmpeg. Default motion is a full **bounce** (low → high → low) in exactly `--nod-frames` samples; use `--nod-motion once` for a one-way sweep only.

| Flag | Meaning |
|------|--------|
| `--nod-out-dir DIR` | PNG output directory (default: `./robot_cam_nod_<robot>`) |
| `--nod-joint` | Hinge name (default `joint_head`) |
| `--nod-low`, `--nod-high` | Joint limits in radians (defaults match URDF ± nod range) |
| `--nod-frames` | Number of captured poses |
| `--nod-motion bounce\|once` | `bounce` (default) or single stroke |
| `--nod-video PATH` | Optional stitched **mp4** (OpenCV `mp4v`) |
| `--nod-fps` | Frames per second for `--nod-video` |

`--discord` is not supported together with `--nod`. Set `EMET_PREVIEW_CAMERAS_OPENCV=1` (or `EMET_OPENCV_PREVIEW=1`) to open the last frame in an OpenCV window after a successful run.

**Examples:**
```bash
emet preview-cameras
emet preview-cameras --robot innate_mars --out /tmp/mars_cams.png
emet preview-cameras --robot xlerobot --out /tmp/xlerobot_cams.png
emet preview-cameras --source zmq --robot innate_mars --robot-ip 127.0.0.1
emet preview-cameras --source zmq --connection herman

emet preview-cameras --nod --nod-out-dir ./nod_caps --nod-frames 41 --nod-video ./nod.mp4
emet preview-cameras --nod --nod-motion once --nod-low -0.12 --nod-high 0.25 --nod-frames 25
```

See [Innate Mars](robots/innate_mars.md#camera-diagnostics-and-head-nod-preview) for MJCF head joint and ZMQ image keys.

---

### `emet sync [options]`

Sync dependencies. Uses `uv sync` if available, otherwise `pip install -e .`.

**Options:**
- `--all` — Request all common extras (same packages as default uv groups: dev, sim, hand_tracker, dynamem, da3); redundant with a plain `emet sync` when using uv.
- `-e, --extra EXTRA` — Include optional dependency extra (repeat for multiple); adds on top of default dependency groups unless you use `uv sync --no-default-groups` yourself.
- `--sim` — Include sim extra (MuJoCo pip deps). `./install.sh` defaults to **no** Robocasa clone; use `--sim`, `--all`, or `--profile=full` / `EMET_INSTALL_PROFILE=full` for legacy behavior
- `--dynamem` — Include dynamem (SAM-2)
- `--dev` — Include dev (pytest, ruff, mypy)
- `--hand-tracker` — Include hand_tracker (mediapipe)
- `--no-install` — Only sync lockfile, do not install emet

**Examples:**
```bash
emet sync
emet sync --all
emet sync -e sim -e dynamem
emet sync --all --hand-tracker
emet sync --no-install
```

With **uv**, `emet sync` runs `uv sync`, which installs **[tool.uv] default-groups** from `pyproject.toml` (dev, sim, hand_tracker, dynamem, da3). For base dependencies only: `uv sync --no-default-groups`. To skip SAM-2: `uv sync --no-group dynamem`.

---

### `emet show <path> [options]`

View Rerun logs (.rrd) or other visualization data.

**Options:**
- `--web` — Open in web viewer instead of native

**Examples:**
```bash
emet show data_0.rrd
emet show logs/run_001.rrd --web
```

---

### `emet test [options] [pytest-args...]`

Run tests with pytest. Uses coverage if pytest-cov is installed.

**Options:**
- `-v, --verbose` — Verbose output
- `--no-cov` — Disable coverage
- `--no-sim` — Set `RUN_SIM_TESTS=0` (skip MuJoCo integration tests)

Pytest options (`-k`, `-m`, `-x`, file paths, etc.) are forwarded to pytest. You can put them after the file list, for example:

```bash
uv run emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
```

(`emet test` uses Click `ignore_unknown_options` so pytest’s `-k` is not mistaken for an emet flag.)

**Examples:**
```bash
emet test
emet test -v
emet test --no-sim                    # skip sim tests (faster)
emet test src/test/cli/test_cli.py
emet test -k test_serve
emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
```

---

### `emet sqa3d <subcommand>`

Situated 3D QA on ScanNet replay (mesh or posed `.sens` RGB-D). Full guide: [sqa3d.md](sqa3d.md). GPU layout: [sqa3d_compute.md](sqa3d_compute.md).

| Subcommand | Description |
|------------|-------------|
| `info` | Print `SQA3D_DATA_DIR` / `SCANNET_ROOT` paths and file status |
| `verify` | Check annotations, meshes; optional `--run-embodied-smoke` |
| `list-questions` | List questions (`--split`, `--limit`) |
| `run-episode` | One embodied episode (Dynagraph or DynaMem) |
| `run-batch` | Batch episodes to JSONL (`-o` required) |
| `run-real-sweep` | Download assets (optional), real-VLM batch, score EM@1 |
| `plot-results` | TP/FP/FN breakdown + paper figures from episode JSONL |

**Shared options (embodied runs):**

| Flag | Default | Notes |
|------|---------|-------|
| `--method` | `dynagraph` | `dynagraph` = DynaMem map + GraphEQA; `dynamem` = voxel EQA only |
| `--profile` | `smoke` if `--mock-llm`, else `tuned` | `tuned` = 15 planning steps, real VLM |
| `--replay-mode` | `auto` | `auto` = `.sens` when on disk near anchor, else mesh; `sens` = require `.sens`; `mesh` = Open3D only |
| `--device` | `cuda` (real VLM) | `cpu` for fallback (slow) |
| `--scannet-root` | `SCANNET_ROOT` | Override ScanNet cache |
| `--data-dir` | `SQA3D_DATA_DIR` | Override SQA3D annotations |

**`run-batch`:** `--question-start` / `--question-end` slice the split list (not `question_id`). `--skip-missing-scenes` (default on). `--resume` skips completed `question_id`s in the output JSONL. `--isolate-episodes` (default **off**) — one subprocess per episode to free GPU between runs.

**`run-real-sweep`:** `--download` / `--no-download`, `--with-sens` (download posed RGB-D), `--output-dir` (default `/tmp/sqa3d_real_sweep`). `--isolate-episodes` default **on**. Writes `<method>_<split>_q<start>-<end>.jsonl` and `_eval.json`.

**Examples:**
```bash
# Data + smoke
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00
uv run emet sqa3d verify --run-embodied-smoke
uv run emet sqa3d run-episode --split train --question-id 220602000000 --mock-llm

# Posed ScanNet RGB (needs .sens on disk)
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens
uv run emet sqa3d run-episode --split val --question-id 220602000049 --replay-mode sens --profile tuned

# Real-VLM sweep (isolated subprocesses; use dedicated GPU — see sqa3d_compute.md)
uv run emet sqa3d run-real-sweep --split val --question-start 0 --question-end 30 \
  --replay-mode sens --no-download --output-dir /tmp/sqa3d_sweep
./scripts/run_sqa3d_gpu_sweep.sh --split val --question-start 0 --question-end 30 --replay-mode sens

# Full split on multiple GPUs (~linear speedup; merges shard JSONL → CSV)
./scripts/run_sqa3d_sharded_sweep.sh --split val --method dynagraph --all --gpus 0,1,2,3
SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh sqa3d-val

# Figures
uv run emet sqa3d plot-results -p /tmp/sqa3d_sweep/dynagraph_val_q0-30.jsonl -o /tmp/sqa3d_figs
```

---

### `emet eval-sqa3d`

Score SQA3D predictions (JSONL, episode JSONL, or `eqa_results.json`) with EM@1.

**Options:** `-p/--predictions` (required), `--split` (default `val`), `--data-dir`, `--questions-path`, `--annotations-path`, `-o/--output`, `--require-all`.

Episode JSONL from `emet sqa3d run-batch` / `run-real-sweep` is auto-detected (no separate gold file needed for per-row scoring).

**Examples:**
```bash
uv run emet eval-sqa3d -p /tmp/sqa3d_batch.jsonl --split val -o sqa3d_eval.json
uv run emet eval-sqa3d -p preds.jsonl --questions-path fixtures/q.json --annotations-path fixtures/a.json
```

---

### `emet robovista <subcommand>`

Offline robot-centric MCQ-VQA on HuggingFace [`sy-xie/robovista`](https://huggingface.co/datasets/sy-xie/robovista) (474 questions, static images). **Not** embodied navigation — accuracy is **not comparable** to HM-EQA. First Hub download caches ~1.1 GB of embedded images.

**Subcommands:** `info`, `run-batch`.

**`run-batch` options:** `--eqa-vl-family`, `--eqa-hf-model-id`, `--device`, `--domain` (repeatable), `--ability-type` (repeatable), `--max-questions`, `--mock-llm`, `--output-dir` (default `~/runs/emet/robovista/<timestamp>/`), `--resume`.

**Examples:**
```bash
uv run emet robovista info
uv run emet robovista run-batch --mock-llm --max-questions 5
uv run emet robovista run-batch --domain domestic --max-questions 50 \
  --eqa-vl-family qwen3_vl --device cuda
```

---

### `emet install <subcommand> [options]`

Install submodules, simulation extras, or full setup.

| Subcommand | Description |
|------------|-------------|
| `submodules` | Init and update git submodules (segment-anything-2, ok-robot) |
| `sim` | Install Robocasa and robosuite (clones into third_party) |
| `robocasa` | Same as `sim` |
| `menu` | Interactive text UI to manage sub-assets (submodules, sim, kitchen assets, MolmoSpaces) |
| `full` | Run full install (./install.sh) |
| `pre-commit` | Install pre-commit hooks (ruff, mypy, etc.) |

**`emet install submodules`**
- `--recursive` / `--no-recursive` — Recursively init nested submodules (default: recursive)

**`emet install sim`** / **`emet install robocasa`** (same)
- Clones robosuite and **Robocasa v0.2** (pinned for MuJoCo 3.3+ / numpy compatibility; main/v1.0 can conflict).
- `-d, --download-assets` — Download Robocasa kitchen assets
- `-a, --setup-macros` — Force run macro setup (overwrite existing); by default macros are set up only when missing

**`emet install full`**
- `-y, --yes` — Skip confirmation prompts
- `--sim` — Include simulation extras
- `--cpu` — CPU-only (skip SAM2)
- `--no-sam2` — Skip Segment Anything 2

**Examples:**
```bash
emet install menu                   # Interactive menu: status and install sub-assets
emet install submodules             # Init and update submodules
emet install sim                    # Install Robocasa, robosuite (third_party)
emet install robocasa               # Same as install sim
emet install sim -d -a              # With assets and force-overwrite macros
emet install full                   # Full install (uv, deps, sync; sim opt-in)
emet install full -y --sim        # Non-interactive + simulation (Robocasa)
emet install full -y --profile full   # Legacy: enable sim without --sim
emet install full --cpu             # CPU-only (no SAM2)
emet install menu                   # Rich plan wizard (needs dev extra / rich)
emet install pre-commit             # Install git hooks (requires emet sync --dev)
emet install pre-commit --run       # Install and run on all files
```

`emet install sim` runs `emet sync -e sim` afterward by default (use `--no-sync` to skip). The project’s `pyproject.toml` uses a uv override for numpy so that `sync -e sim -e dynamem` works; see [Simulation](simulation.md#troubleshooting).

**`emet clean`** — Remove third-party sim clones (robosuite, robosuite_models, robocasa) from `third_party/`. Use `-y` to skip the confirmation prompt. Re-run `emet install sim` to reinstall.

```bash
emet clean        # Remove sim third_party dirs (prompts to confirm)
emet clean -y     # Remove without prompting
```

---

### `emet jobs` (queued / running eval experiments)

Local job registry under `~/runs/emet/jobs/` (override with `EMET_JOBS_DIR`). Queue/smoke scripts register here; unmanaged eval PIDs are still shown via process scan.

| Subcommand | Role |
|------------|------|
| `emet jobs` / `emet jobs list` | Active registered jobs (+ unmanaged eval PIDs) |
| `emet jobs list --all` | Include done/failed/cancelled |
| `emet jobs status JOB_ID` | Human-readable record + progress/ETA + **viz paths** under `OUT/bundles/` / `figures/` (`--json` includes derived `progress`) |
| `emet jobs report [JOB_ID]` | Progress + per-episode score table + viz/feh hints (defaults to running/waiting job). `conf` shows `v=` verify-gate and `e=` EQA `Confidence:` (often `e=N` even on correct letters) |
| `emet jobs report [JOB_ID] --question ID [--arm agentic]` | Per-episode deep dive: question, pred/gold, verify vs EQA confidence, verify phrases + detector scores, stale re-verify / fallback-submit red flags, abstain reasons |
| `emet jobs cancel JOB_ID` | SIGTERM→SIGKILL job process tree; mark cancelled |
| `emet jobs logs JOB_ID [--tail N]` | Tail queue/orchestrator log |
| `emet jobs register …` | Scripts: create a record (prints job id) |
| `emet jobs update JOB_ID --status …` | Heartbeat / terminal status; optional `--units-done/--units-total/--phase/--current-id` |
| `emet jobs run --name NAME [--need-mib N] [--cpu-safe/--no-cpu-safe] [--gpu-exclusive/--no-gpu-exclusive] [--wait-pid P] -- CMD…` | Register + nohup wrapper (sets `EMET_JOB_ID`). With `--need-mib`, **cpu-safe** and **gpu-exclusive** default **on**. |

`emet jobs list` shows a **PROGRESS** column (units, phase, current id, ETA) from job meta and/or `OUT/progress.json`. Prefer this over bare `nohup` for multi-hour GPU evals.

```bash
uv run emet jobs
uv run emet jobs report              # defaults to running job; scores from OUT/*_q*.jsonl
uv run emet jobs report --question 88 # per-episode trace: phrases, verify scores, red flags
uv run emet jobs list --all
uv run emet jobs run --name dyn-improve-eqa --need-mib 14000 -- \
  ./scripts/run_dynagraph_dynamic_improve_smokes.sh ~/runs/emet/dynamic_exploration/eqa_out
uv run emet jobs status JOB_ID
uv run emet jobs update JOB_ID --units-done 8 --units-total 64 --phase classic --current-id 17
uv run emet jobs cancel JOB_ID
uv run emet jobs logs JOB_ID --tail 80
```

Related: [`emet eval`](#emet-eval-gpu-preflight--stale-cleanup) for GPU preflight / orphan cleanup (not the same as job cancel); [`emet hmeqa`](#emet-hmeqa-hm-eqa-h2h) for classic vs agentic launches.

### `emet eval` (GPU preflight / stale cleanup)

Canonical GPU preflight for paper evals and overnight smokes (Python implementation of [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh)).

| Subcommand | Role |
|------------|------|
| `emet eval status` | Free/total VRAM + compute apps (read-only) |
| `emet eval diagnose` | Habitat/HM-EQA readiness notes: empty apps ≠ EGL OK; flags empty `CUDA_VISIBLE_DEVICES`, missing `.venv-habitat`, recent `emet` segfault hints |
| `emet eval check [--need-mib N]` | Exit 1 if free VRAM &lt; N (default `NEED_MIB` or 12000) |
| `emet eval wait [--need-mib N]` | Block until free VRAM is stably above N |
| `emet eval kill-stale [--no-gpu] [--settle-sec S]` | SIGTERM→SIGKILL orphaned eval/sim/`uv run emet` trees |
| `emet eval affinity [--apply] [--pid P] [--json]` | Show/apply turbo-CPU exclusion mask |
| `emet eval recover [--need-mib N]` | `status` + `diagnose` + `wait` one-shot (post-crash / post-reboot) |

Skips the caller process ancestry and any PIDs in `EMET_GPU_PROTECT_PIDS`. See [evaluation.md](evaluation.md#gpu-preflight-all-overnight--vlm-jobs), [known_issues.md](known_issues.md#nvidia-driver-hang--cursor-agent-crash-during-stacked-gpu-evals), and [environment_variables.md](environment_variables.md).

```bash
uv run emet eval status
uv run emet eval diagnose
uv run emet eval affinity
uv run emet eval recover --need-mib 12000
uv run emet eval kill-stale
NEED_MIB=12000 uv run emet eval wait
uv run emet eval check --need-mib 14000
```

Related top-level scoring apps remain: `emet eval-dynagraph`, `emet eval-calibration`, `emet eval-sqa3d`.

### `emet hmeqa` (HM-EQA H2H)

Dogfood entrypoints for classic vs agentic-verify Dynagraph. Prefer these over hand-rolled `env … taskset … ./scripts/run_hmeqa_agentic_h2h.sh`.

| Command | Purpose |
|---------|---------|
| `emet hmeqa h2h [OUT] [--resume] [--arms …] [--ids …] [--preset paper-router] [--agentic-verifier none\|owlv2\|yoloe] [--require-verified\|--allow-unverified] [--agentic-router] [--crash-policy skip\|abort] [--streak-abort N]` | Launch via `emet jobs run --need-mib` (cpu-safe + gpu-exclusive) |
| `emet hmeqa resume [OUT] [--preset paper-router]` | Resolve latest OUT from status symlink if omitted; `RESUME=1` (retries empty per-qid jsonl) |
| `emet hmeqa overnight [--base DIR] [--skip-bal32] [--gate-min-acc 0.25]` | Holdout-8 → optional agentic retune → bal-32 in **one** `emet jobs` run (paper-router defaults) |
| `emet hmeqa status [OUT]` | Progress + scored counts + crash capsules |
| `emet hmeqa summarize [OUT]` | `scripts/summarize_hmeqa_agentic_h2h.py` |

`OUT/DONE` is written only when every arm×id unit has a non-empty scored jsonl. Partial batches (skipped native crashes, etc.) exit nonzero, mark the job `failed` / `INCOMPLETE`, and leave `STATUS` pointing at resume — not summarize.

Default crash policy is **skip** (settle + retry, continue). **`--streak-abort 2`** (default) aborts early after consecutive native crashes so a wedged driver does not burn the full batch.

**`--preset paper-router`** (on `h2h` / `resume`): sets owlv2 + allow-unverified + agentic-router where flags were left at Click defaults; explicit flags still win. Probe runs should omit the preset and keep `--require-verified`.

**`emet hmeqa overnight`** defaults to paper-router policy (owlv2, allow-unverified, router on). Inner phases call `run_hmeqa_agentic_h2h.sh` directly (no nested jobs). `scripts/run_hmeqa_overnight_ladder.sh` is a thin shim to this command.

Agentic validation uses `scripts/summarize_agentic_ladder.py`: it reports accuracy, selective risk/coverage, fused-verify precision, visibility at verify, path length, hypothesis count, abstention, false confirmation, and forced submits. Balanced-32 is blocked unless a 4+ episode probe has a nonzero fused verified-answer rate and zero forced submits.

```bash
uv run emet eval recover --need-mib 12000
uv run emet hmeqa overnight
# or a probe:
uv run emet hmeqa h2h ~/runs/emet/hmeqa_graph_probe --arms agentic \
  --ids 12,17,18,56 --agentic-verifier owlv2 --require-verified
uv run python scripts/summarize_agentic_ladder.py ~/runs/emet/hmeqa_graph_probe \
  --require-balanced32-gate
uv run emet hmeqa resume --preset paper-router
uv run emet hmeqa status
uv run emet jobs
```

### `emet kill-mujoco-server [options]`

Stop MuJoCo simulation server(s) so ports 4401–4404 are free. For broader orphan cleanup use **`emet eval kill-stale`**.

**Options:**
- `--port N` — Kill process on port N (default: 4401)
- `--all` — Kill all mujoco_server processes, then free ports 4401–4404

**Examples:**
```bash
emet kill-mujoco-server              # free port 4401
emet kill-mujoco-server --all       # stop all mujoco servers and free 4401–4404
emet kill-mujoco-server --port 4501 # if you used --port-offset 100
```

---

### Tab completion {#tab-completion}

#### `emet install-completion [options]`

Print shell completion script so that `emet`, subcommands, and options tab-complete.

**Options:**
- `-s, --shell {bash,zsh,fish}` — Shell (auto-detected from $SHELL if omitted)

**Setup:**
```bash
# Bash: add to ~/.bashrc
eval "$(emet install-completion --shell bash)"

# Zsh: add to ~/.zshrc
eval "$(emet install-completion --shell zsh)"

# Fish: add to ~/.config/fish/config.fish
emet install-completion --shell fish | source
```

Then restart your shell or `source` your config file. After that, `emet <TAB>` completes to `serve`, `kill-mujoco-server`, `run`, etc.

---

## Simulation workflow

### Stretch (default)

1. **Terminal 1** — Start the server:
   ```bash
   emet serve mujoco
   # or for Robocasa: emet serve mujoco --scene robocasa
   ```

2. **Terminal 2** — Run the agent or DynaMem:
   ```bash
   # Embodied LLM agent + tools (see AGENT_RUN.md)
   emet run agent --robot-ip 127.0.0.1
   # Or one terminal: emet run agent --start-sim -c "describe the scene"

   # DynaMem navigation + manipulation
   emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo
   ```

### Innate Mars (same ZMQ protocol, registry robot)

1. **Terminal 1** — MuJoCo sim with the Mars MJCF and default table scene:
   ```bash
   emet serve --robot innate_mars --headless
   # equivalent: emet serve mujoco --robot innate_mars --headless
   ```

2. **Terminal 2** — DynaMem (or `examples/mapping_innate_mars_sim.py`) with the same `--robot`:
   ```bash
   emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S --cpu-only
   ```

### Headless / no display

```bash
emet serve mujoco --headless
emet run dynamem --robot-ip 127.0.0.1 -S --headless
```

Then open http://localhost:9090?url=ws://localhost:9877 in a browser.

---

## Testing

Run the full test suite from the project root:

```bash
uv run emet test
```

**Master index:** [TESTING.md](TESTING.md) — all test docs, harnesses, and the Dynagraph graph+EQA gap.

**Dynagraph quick checks:**

```bash
# Unit (fast)
uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v

# Multi-robot Robocasa floor E2E (~20 min; needs sim extra)
uv run python src/test/app/run_dynagraph_multi_robot_e2e.py
```

See [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) for pass criteria and artefact paths.

**SQA3D + ScanNet** — see [`emet sqa3d`](#emet-sqa3d-subcommand) above and [sqa3d.md](sqa3d.md):

```bash
uv run emet test src/test/benchmarks/sqa3d/ -v
uv run python scripts/run_sqa3d_scannet_smoke.py
```

Run only CLI tests:

```bash
uv run emet test src/test/cli/
```

Verbose:

```bash
uv run emet test -v
```

Skip sim integration tests (faster):

```bash
uv run emet test --no-sim
```
