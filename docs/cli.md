# Emet CLI Tool

The `emet` CLI makes it easy to start simulations, run robot agents, sync dependencies, view logs, and run tests. It supports **tab completion** for bash, zsh, and fish (see [Tab completion](#tab-completion) below). Click groups live under `emet.cli_cmds`; [`src/emet/cli.py`](../src/emet/cli.py) is the registrar (`emet.cli:main`). Sim-heavy top-level commands (`export-sim-gt`, `eval-dynagraph`, `ovmm`, `debug-da3-depth`, …) are **lazy**: `emet jobs`, `emet eval`, and `emet --help` do not import MuJoCo.

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

If `emet serve mujoco` fails on `import scipy` / numpy ABI errors while `uv run python -c "import scipy"` works: leftover `python3.12` site-packages in a 3.10 `.venv` — see [pythonpath.md](pythonpath.md) and [known_issues.md](known_issues.md).

## Commands

### `emet serve [backend]`

Start a simulation server **or** an OpenAI-compatible LLM HTTP API.

Backends: `mujoco` (default), `robocasa`, `molmospaces`, `habitat`, **`llm`**.

**LLM (LAN OpenAI API)** — load Qwen (or other `get_llm_client` keys) on this host for a workstation to call:

```bash
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
emet serve llm --vl --host 0.0.0.0 --port 8001   # dual-port caption/EQA beside text
# LAN Orin example (unified-7b): text + VL both on Jetson :8000 — docs/llm_serve.md
export EMET_OPENAI_BASE_URL=http://ORIN_HOST:8000/v1
# or: export EMET_LLM_HOST=ORIN_HOST

# Remote health / smoke (pass --host; unified-7b text+VL on :8000)
uv run emet llm health --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST --vl-only
# Interactive / one-shot chat against LAN endpoints
uv run emet run chat --host ORIN_HOST --once "Reply with exactly: pong"
uv run emet run chat --host ORIN_HOST --vl --once "What color is the flag?" --image /path/to.jpg
emet run agent --llm openai
```

Details: [llm_serve.md](llm_serve.md) — **§1 Remote inference** and **§2 Testing LLMs** (Jetson unified-7b / dual-2b). Jetson install: [jetson.md](jetson.md).

Start a simulation server.

| Backend | Description |
|---------|--------------|
| `mujoco` | MuJoCo ZMQ simulation (default) |
| `robocasa` | Shortcut for `mujoco --scene robocasa` (kitchen scenes) |
| `molmospaces` | Shortcut for `mujoco --scene ithor` (merge Molmo scene + robot, then ZMQ) |
| `habitat` | Habitat-Sim EQA ZMQ (requires `.venv-habitat`) |

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

### `emet grasp-oracle`

Fake MolmoSpaces grasp predictor (ZMQ REP). Loads NPZ/JSON under `$MLSPACES_ASSETS_DIR/grasps` and returns world-frame 4×4 poses. Robot-agnostic; used by [`scripts/scripted_molmo_grasp_mp.py`](../scripts/scripted_molmo_grasp_mp.py).

| Option | Default | Description |
|--------|---------|-------------|
| `--bind` | `tcp://127.0.0.1:5558` | ZMQ REP bind address |
| `--grasps-dir` | `$MLSPACES_ASSETS_DIR/grasps` | Grasp asset root |
| `--tcp-frame` | `droid` | Gripper TCP correction (`droid` / `rum`) |

```bash
uv run emet grasp-oracle --bind tcp://127.0.0.1:5558
```

See [motion_planning.md](motion_planning.md#molmospaces-grasp-oracle-multi-robot).

---

### `emet run <app> [options]`

Run a robot agent or app.

| App | Description |
|-----|--------------|
| `agent` | Embodied LLM agent + tools (optional Discord, opt-in Rerun). See [AGENT_RUN.md](AGENT_RUN.md). |
| `dynamem` | DynaMem navigation + manipulation |
| `graph-eqa` | Graph-based EQA memory (see [graph_eqa.md](graph_eqa.md), [graph_memory.md](graph_memory.md)) |
| `dynagraph` | Graph EQA + merge/staleness ([dynagraph.md](dynagraph.md)); **`--explore-loop`**, **`--export`**, **`--question`** |
| `lazy-graph` | Same CLI as dynagraph with LazyGraph memory ([lazy_graph.md](lazy_graph.md)); shared [`graph_nav_cli.py`](../src/emet/app/graph_nav_cli.py) (`configure_graph_nav`) |
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
emet run agent --confirm-nav --rerun   # preview plan on map; y/n before base moves (real robot)
emet run dynamem --robot-ip 127.0.0.1 -S
emet run dynamem -S --visual-servo --match-method class --rerun-debug
emet run mapping --robot-ip 127.0.0.1
emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
emet run timing --robot-ip 192.168.1.15 --headless
emet run debug-da3-depth --robot innate_mars --depth-source sensor
```

Agent-specific flags (also see [AGENT_RUN.md](AGENT_RUN.md)): **`--confirm-nav`** / **`EMET_CONFIRM_NAV=1`** — show motion plan on the 2D map and wait for y/n (Discord posts the map PNG).

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

### `emet deploy` / `emet deploy llm`

**Operator guide:** [deploy.md](deploy.md) (Stretch + Mars bridges vs LAN Orin LLM, wrist Arducam checklist).

**Robot bridge:** sync `emet_core` + the bridge package for `--robot stretch` (`stretch_ros2_bridge` → `~/ament_ws`) or `--robot innate_mars` (`innate_mars_bridge` → innate-os workspace). Default robot comes from the active `emet connect` profile (`robot:` field), else `stretch`.

```bash
uv run emet deploy --robot stretch --start-bridge
uv run emet deploy --connection mars          # profile robot=innate_mars
uv run emet mars start --connection mars --deploy
```

**LAN LLM/VLM (AGX Orin, ~64 GiB unified memory):**

```bash
uv run emet deploy llm --host ORIN_HOST                         # unified-7b (Qwen2-VL-7B on :8000)
uv run emet deploy llm --host ORIN_HOST --profile dual-2b       # text :8000 + VL-2B :8001
uv run emet llm health --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST --vl-only
```

Details: [llm_serve.md](llm_serve.md). Shell helper: `./scripts/deploy_caliban_vl.sh --host ORIN_HOST --profile unified-7b` (script name is historical).

### `emet mars [start|status|stop]`

Deploy and manage the **Innate Mars ZMQ bridge** on a Jetson running innate-os. See [deploy.md](deploy.md) and [Innate Mars hardware bring-up](robots/innate_mars_hardware.md) for full recipes (`--deploy`, `--onboard-da3`). Stretch uses `emet deploy --robot stretch` (no `emet mars`).

```bash
emet mars start --ip MARS_IP --username jetson1 --deploy
emet mars status --connection mars
emet mars stop --connection mars
```

`start` and `status` print a compact summary (bridge state, ZMQ ports, optional ROS hint, **head/wrist camera line**, suggested next command). Colors follow Click when stdout is a TTY; set `NO_COLOR=1` or `EMET_NO_COLOR=1` to disable.

---

### `emet connect [save|list|show|use]`

SSH / deploy **connection profiles** stored in ``~/.stretch/connection.json`` (field reference: ``src/emet/utils/connection.py``).

| Command | Purpose |
|---------|--------|
| `emet connect save HOST …` | Create or update a profile; default sets it **active** |
| `emet connect list` | List profiles and mark which is active |
| `emet connect show` | Print the active profile |
| `emet connect use NAME` | Set active profile (switch Stretch ↔ Mars) |

**`emet connect save` flags**

| Flag | Stored as | Notes |
|------|-----------|--------|
| `--user` / `-u` | `user` | SSH login (default `root`; Stretch usually `hello-robot`, Mars `jetson1`) |
| `--password` / `-p` | `password` | Optional; else `EMET_ROBOT_PASSWORD` or SSH keys at runtime |
| `--name` / `-n` | profile key | Default: hostname/IP |
| `--robot` | `robot` | `stretch` or `innate_mars` — selects bridge for `emet deploy` |
| `--config` | `config` | Default unified YAML when apps leave `--config` at Click default (e.g. `configs/agent_innate_mars.yaml`) |
| `--workspace` | `workspace` | Remote ROS2 workspace (Stretch: `~/ament_ws`; Mars: `~/innate-os/ros2_ws`) |
| `--emet-dir` | `emet_dir` | Remote emet install root (default `~/emet`) |
| `--no-active` | — | Save without setting active or updating `robot_ip.txt` |

**Examples:**
```bash
emet connect save STRETCH_IP --user hello-robot --name stretch \
  --robot stretch --workspace ~/ament_ws
emet connect save MARS_IP --user jetson1 --name mars \
  --robot innate_mars --workspace ~/innate-os/ros2_ws --emet-dir ~/emet \
  --config configs/agent_innate_mars.yaml
emet connect list
emet connect use stretch
emet connect show
```

**Host / robot:** `emet deploy`, `emet mars start`, `emet capture`, `emet stream`, `emet run agent`, and `emet preview-cameras` use the active profile (or `--connection NAME`) when `--ip` / `--host` / `--robot-ip` is omitted.

**Profile `config` (YAML path):** Not agent-only. Any app that resolves unified config through the shared CLI helper (`emet run agent`, `emet run dynamem`, `emet run dynagraph`, `emet stream`, `emet capture`, …) uses the profile’s `config` whenever **`--config` / `-C` is left at the Click default**:

1. Explicit `--config` / `--agent-config` / `--dynav-config` wins.
2. Else profile `config` for `--connection NAME`, or for the **active** profile when `--connection` is omitted.
3. Else `EMET_CONFIG` / packaged default (`configs/emet/default.yaml`).

So with a Mars profile active and `config=configs/agent_innate_mars.yaml`, a bare `emet run dynamem` (no `--config`) also loads that agent preset — usually fine on a Mars-only workstation. For Stretch/sim on the same machine, pass an explicit `--config`, use a profile without `config`, or `--no-active` when saving the Mars profile.

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
emet capture --robot innate_mars --ip MARS_IP
emet capture --connection mars --backend voxel_only --no-rerun
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
| `static_graph` | Voxel + GraphEQA graph (zero-merge baseline; alias `graph_eqa`) |
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
emet stream --connection mars
emet stream --connection mars --backend voxel_only
emet stream --connection mars --backend dynagraph
emet stream --connection mars --backend static_graph
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
emet preview-cameras --source zmq --connection mars

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
emet test agent-regression            # Discord / Herman / agent pack (no sim)
emet test src/test/cli/test_cli.py
emet test -k test_serve
emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
# VLM-free multi-env nav/explore (Robocasa + OVMM kitchen + MolmoSpaces):
emet test -v src/test/simulation/test_multi_env_nav_explore_smoke.py
emet test -k robocasa_l1 src/test/simulation/test_multi_env_nav_explore_smoke.py
```

`emet test agent-regression` expands to the fixed no-sim pack used as the Discord/Herman gate (see `.cursor/rules/agent-discord-regression.mdc`).

---

### `emet ovmm <subcommand>`

OVMM find/full paper benchmarks and multi-env sweeps (Robocasa + MolmoSpaces). Deep dives: [ovmm_find_phase_benchmark.md](ovmm_find_phase_benchmark.md), [ovmm_full_benchmark.md](ovmm_full_benchmark.md), [paper_benchmarks.md](paper_benchmarks.md).

| Subcommand | Description |
|------------|-------------|
| `find` | Batch find-phase (FindObj / FindRec) across memory backends |
| `full` | Batch full OVMM (find + pick/place; `--manip-mode`) |
| `prepare` | Write `sim/` + `find_episodes.yaml` / `full_episodes.yaml` from a preset |
| `sweep` | `prepare` → find → full → `rates` (paper multi-env path) |
| `rates` | Aggregate `OUT/find` + `OUT/full` → `rates.json` (excludes bind/task-init fails) |
| `status` | Per-episode outcomes + bind-fail counts |

**Presets:** `configs/ovmm/sweeps/` (e.g. `molmo-robocasa`). Explicitly **no** `default_table`. Dynagraph find is the **same AgenticEQA loop** as HM-EQA (OVMM phrased as questions) — method in [dynagraph.md](dynagraph.md#method). `localize_text` is an investigate card and the scored XYZ (never camera pose). The harness does **not** pin episode YAML phrases. Preset `agentic_find: true`. `--oneshot-localize` / `emet ovmm full --oneshot-localize` is a leftover **mapping ablation**, not the product path. Agentic budget: `--agentic-max-rounds` / `--agentic-max-nav-steps` on `find`/`full` (or preset `defaults.agentic_max_rounds` / `agentic_max_nav_steps`). `--via-jobs` sets `EMET_ALLOW_SDPA_ATTN=1` so in-process VL uses SDPA (FA2 has hung with MuJoCo co-resident).

**Examples:**
```bash
# Multi-env dynagraph sweep (prefer --via-jobs on a shared GPU workstation)
uv run emet ovmm sweep --preset molmo-robocasa --backend dynagraph --via-jobs

# Stepwise
uv run emet ovmm prepare --preset molmo-robocasa --out ~/runs/emet/ovmm_molmo_robocasa/DATE
uv run emet ovmm find --episodes OUT/find_episodes.yaml --backend dynagraph --port-stride 4 \
  --output-dir OUT/find --no-scene-cache
uv run emet ovmm full --episodes OUT/full_episodes.yaml --backend dynagraph --manip-mode sim \
  --port-stride 4 --output-dir OUT/full
uv run emet ovmm rates --out OUT
uv run emet ovmm status --out OUT
```

Compatibility wrappers (same library path): `scripts/eval_ovmm_find_phases.py`, `scripts/eval_ovmm_full.py`.

TAMP clutter-clearance (MolmoSpaces iTHOR; not an `emet` subcommand): `scripts/eval_tamp_clutter.py` — [tamp_clutter.md](experiments/tamp_clutter.md).

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
| `paper` | Install `latexmk` and the TeX Live packages used by `paper/main.tex` (Ubuntu/apt) |
| `menu` | Interactive UI to manage sub-assets and optional paper tooling |
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
- `--paper` / `--no-paper` — Include paper tooling, or omit it from `--all`
- `--all` — Include simulation, MolmoSpaces, dynamem, and paper tooling

**Examples:**
```bash
emet install menu                   # Interactive menu: status and install sub-assets
emet install paper -y               # latexmk + TeX Live for ./paper/build.sh
emet install submodules             # Init and update submodules
emet install sim                    # Install Robocasa, robosuite (third_party)
emet install robocasa               # Same as install sim
emet install sim -d -a              # With assets and force-overwrite macros
emet install full                   # Default full profile (sim + MolmoSpaces)
emet install full -y --paper         # Full profile plus local paper toolchain
emet install full -y --all           # All bundles, including paper tooling
emet install full -y --sim           # Explicit simulation (Robocasa)
emet install full -y --profile full  # Explicit full profile
emet install full -y --profile jetson # Jetson Orin lean install (see docs/jetson.md)
emet install full --cpu             # CPU-only (no SAM2)
emet install menu                   # Rich plan wizard (needs dev extra / rich)
emet install pre-commit             # Install git hooks (requires emet sync --dev)
emet install pre-commit --run       # Install and run on all files
```

`emet install paper` uses `apt` and may request `sudo`; it installs `latexmk`,
`texlive-latex-extra`, and `texlive-bibtex-extra`. The toolchain is optional in
the normal full profile because TeX Live is large. `./paper/build.sh` falls back
to `texlive/texlive:latest` when Docker is available.

Jetson Orin / Tegra: `./scripts/install_jetson.sh -y` or `emet install full -y --profile jetson` (no sim/SAM2/Molmo; Python 3.10 via uv). Details: [jetson.md](jetson.md).

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
| `emet jobs report [JOB_ID]` | Progress + per-episode score table + viz/feh hints (defaults to running/waiting job). `conf` shows `v=` verify-gate and `e=` EQA `Confidence:` (often `e=N` even on correct letters). `--fail-only` lists incorrect rows; `--out-dir PATH` reports without a registry id |
| `emet jobs report [JOB_ID] --question ID [--arm agentic]` | Per-episode deep dive with sections: **view investigation** (eqa_history action/Unknown loops), **rooms** (merged/vlm/graph timeline, `Rooms:` line, MCQ targets, mismatch/redirects), router picks, investigate/station/explore, assess, verify, red flags. Flags: `--rooms` (rooms focus), `-s/--section`, `--brief`, `-v/--verbose`, `--json` |
| `emet jobs cancel JOB_ID` | SIGTERM→SIGKILL job process tree; mark cancelled; prints resume hint + warns if unmanaged eval PIDs remain |
| `emet jobs logs JOB_ID [--tail N]` | Tail queue/orchestrator log |
| `emet jobs register …` | Scripts: create a record (prints job id); optional `--description` / `-d` |
| `emet jobs update JOB_ID --status …` | Heartbeat / terminal status; optional `--units-done/--units-total/--phase/--current-id` / `--description` |
| `emet jobs run --name NAME [-d TEXT] [--need-mib N] [--cpu-safe/--no-cpu-safe] [--gpu-exclusive/--no-gpu-exclusive] [--wait-pid P] [--wait-timeout-sec S] [--lock-timeout-sec S] [--gpu-wait-max-rounds N] -- CMD…` | Start a detached supervisor that self-registers, sets `EMET_JOB_ID`, and runs the command. GPU-like commands and jobs with `--need-mib` default to **cpu-safe** + **gpu-exclusive**; exclusive jobs hold a host-wide `flock` for their full lifetime. **cpu-safe** pins via `python -m emet.utils.cpu_affinity` (not `emet eval affinity`) so the wrapper does not import MuJoCo after the previous sim job releases the lock. |

`emet jobs list` shows a **PROGRESS** column (units, phase, current id, ETA) from job meta and/or `OUT/progress.json`. Jobs with a `--description` / `-d` also show a **`why:`** line under the row (and in `emet jobs status`). The detached supervisor owns registration: if the invoking terminal or agent dies before spawn, no phantom queued record is created; if it dies after spawn, the supervisor registers and continues independently. The host-wide `flock` is the serialization authority for exclusive jobs; the launcher does not infer and wait on unrelated active GPU PIDs. Only explicit `--wait-pid` prerequisites are waited, and all PID, lock, and optional pre-command GPU waits are bounded (defaults: six hours for PID/lock, 120 GPU polling rounds). The canonical shared lock is `~/runs/emet/gpu.lock` (`EMET_GPU_LOCK`); `EMET_GPU_LOCK_FILE` is a compatibility alias. This applies equally to `emet hmeqa …`, `emet ovmm … --via-jobs`, and direct `emet jobs run`. Prefer it over bare `nohup` for multi-hour GPU evals.

```bash
uv run emet jobs
uv run emet jobs report              # defaults to running job; scores from OUT/*_q*.jsonl
uv run emet jobs report --fail-only
uv run emet jobs report --question 104 --rooms   # room timeline / Rooms: line audit
uv run emet jobs report --out-dir ~/runs/emet/hmeqa_… -q 104 -v
uv run emet jobs report --question 88 # full per-episode trace
uv run emet jobs list --all
uv run emet jobs run --name dyn-improve-eqa -d "owlv2 find; no confirm gate" --need-mib 14000 -- \
  ./scripts/run_dynagraph_dynamic_improve_smokes.sh ~/runs/emet/dynamic_exploration/eqa_out
uv run emet jobs status JOB_ID
uv run emet jobs update JOB_ID --units-done 8 --units-total 64 --phase classic --current-id 17
uv run emet jobs update JOB_ID -d "retag: compare confirm on vs off"
uv run emet jobs cancel JOB_ID
uv run emet jobs                 # confirm no unmanaged Habitat orphans after cancel
uv run emet jobs logs JOB_ID --tail 80
```

Tag an already-running job without restarting it:

```bash
uv run emet jobs update JOB_ID -d "owlv2 + EMET_EQA_ANSWERABLE_CONFIRM=0 (vs confirm-on restore)"
```

**Pause / resume (official):** there is no separate `pause` subcommand — cancel the managed job, then relaunch with the same OUT/`--base`.

```bash
# Pause a live overnight / H2H job (prefer this over raw kill / kill-stale):
uv run emet jobs cancel JOB_ID
uv run emet jobs                 # unmanaged emet-habitat leftovers?
uv run emet eval status          # GPU should clear

# Resume overnight ladder (same --base; skips validated COMPLETE-marker phases):
uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_… --job-name hmeqa-overnight

# Resume a single H2H OUT only (holdout8 or bal32 dir):
uv run emet hmeqa resume ~/runs/emet/hmeqa_overnight_…/bal32 --preset paper-router
```

Empty, partial, and unvalidated per-qid jsonl are retried on resume. Only units with a hash-validated `bundles/<arm>_q<id>/COMPLETE.json` marker are kept.

Related: [`emet eval`](#emet-eval-gpu-preflight--stale-cleanup) for GPU preflight / orphan cleanup (not the same as job cancel); [`emet hmeqa`](#emet-hmeqa-hm-eqa-h2h) for classic vs agentic launches; [`emet status`](#emet-status-recovery-log) after an agent death.

### `emet eval` (GPU preflight / stale cleanup)

Canonical GPU preflight for paper evals and overnight smokes (Python implementation of [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh)).

| Subcommand | Role |
|------------|------|
| `emet eval status` | Free/total VRAM + compute apps (read-only) |
| `emet eval diagnose` | Habitat/HM-EQA readiness notes: empty apps ≠ EGL OK; flags empty `CUDA_VISIBLE_DEVICES`, missing `.venv-habitat`, recent `emet` segfault hints |
| `emet eval check [--need-mib N]` | Exit 1 if free VRAM &lt; N (default `NEED_MIB` or 12000) |
| `emet eval wait [--need-mib N] [--max-rounds N]` | Wait until free VRAM is stably above N, bounded by 120 rounds by default |
| `emet eval kill-stale [--no-gpu] [--settle-sec S]` | SIGTERM→SIGKILL orphaned eval/sim/`uv run emet` trees |
| `emet eval affinity [--apply] [--pid P] [--json]` | Show/apply turbo-CPU exclusion mask (stdlib affinity helpers only; does not import OVMM/MuJoCo) |
| `emet eval recover [--need-mib N] [--max-rounds N]` | `status` + `diagnose` + bounded `wait` one-shot (post-crash / post-reboot) |

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

### `emet habitat` (Habitat-Sim / HM-EQA wrapper)

Requires `./scripts/install_habitat.sh` (``.venv-habitat``). **Never** run `run-episode` / EGL probes as blocking Cursor agent commands — use `safe-start` then `emet jobs` / `emet hmeqa`.

| Command | Purpose |
|---------|---------|
| `emet habitat info` | Data paths + asset status |
| `emet habitat safe-start [--need-mib N] [--question-id Q] [--smoke-episode]` | `eval recover` + **detached** jobs-wrapped `emet-habitat egl-probe` (no VLM). Exit 0 = queued, not EGL OK. Optional mock-llm episode also queued (waits behind probe). |
| `emet habitat egl-probe --force-inline` | Inline EGL only (dedicated terminal); agents are redirected to `safe-start` |
| `emet habitat list-questions` / `serve` / `run-episode` | Wrapper passthrough (prefer jobs for anything that loads Habitat) |

```bash
uv run emet habitat safe-start --need-mib 4000
uv run emet jobs status JOB   # wait until done
uv run emet jobs logs JOB --tail 40
# only after probe status=done and logs show EGL OK:
uv run emet hmeqa h2h --preset paper-router …
```

Related top-level scoring apps remain: `emet eval-dynagraph`, `emet eval-calibration`, `emet eval-sqa3d`.

### `emet status` (recovery log)

Per-checkout `STATUS.log` helpers (wraps `scripts/status_log.sh`). Prefer these after an agent death.

| Command | Purpose |
|---------|---------|
| `emet status tail [N]` | Last N lines of this checkout's STATUS.log |
| `emet status path` | Print STATUS.log path |
| `emet status latest` | Resolve `latest` OUT symlink |

```bash
uv run emet status tail
uv run emet status path
uv run emet status latest
```

Orchestrators still *source* `scripts/status_log.sh` to write records.

### `emet hmeqa` (HM-EQA H2H)

Dogfood entrypoints for classic vs agentic-verify Dynagraph. Prefer these over hand-rolled `env … taskset … ./scripts/run_hmeqa_agentic_h2h.sh`.

| Command | Purpose |
|---------|---------|
| `emet hmeqa h2h [OUT] [--resume] [--arms …] [--ids …] [-d TEXT] [--variant-config FILE] [--host HOST] [--vl-endpoint …] [--vl-port N] [--preset paper-router] [--eqa-hf-model-id …] [--eqa-vl-family …] [--eqa-vl-quantization int4\|int8\|float16\|bfloat16\|float32\|none] [--agentic-verifier none\|owlv2\|yoloe] [--require-verified\|--allow-unverified] [--agentic-router] [--use-hm3d-semantics\|--no-hm3d-semantics] [--enrich-labels\|--no-enrich-labels] [--action-progress-mode off\|shadow\|enforce] [--crash-policy skip\|abort] [--streak-abort N]` | Launch via `emet jobs run --need-mib` (cpu-safe + gpu-exclusive); `--variant-config` loads all nine variant axes from strict YAML; `-d` tags the job why; `--host` / `--vl-endpoint` inject remote answer VL into the job env |
| `emet hmeqa resume [OUT] [variant flags…]` | Resolve latest OUT if omitted; reuse its frozen variant/model/budgets/IDs, validate commit + dirty state + digest, then set `RESUME=1` |
| `emet hmeqa overnight [--base DIR] [--skip-bal32] [--gate-min-acc 0.25]` | Holdout-8 → optional agentic retune → bal-32 in **one** `emet jobs` run (paper-router defaults). Re-pass `--base` after cancel to resume (skips only phases with a validated JSON `DONE`; sets `RESUME=1` when validated or pending state exists) |
| `emet hmeqa status [OUT]` | Progress + scored counts + crash capsules |
| `emet hmeqa summarize [OUT]` | `scripts/summarize_hmeqa_agentic_h2h.py` |
| `emet hmeqa significance [OUT] [--from-summary …] [--json …]` | Paired McNemar / Wilcoxon / bootstrap on classic vs agentic |
| `emet hmeqa failures [OUT] [--from-summary …] [--json …]` | Offline classic_only / context-gap attribution (+ traces) |
| `emet hmeqa inspect [OUT] --qid N [--open rgb\|frames\|images\|frontier\|maps\|video]` | Episode score + assess/explore + copy-paste `feh`/`mpv` paths |
| `emet hmeqa inspect [OUT] --misses` | List incorrect scored episodes |
| `emet hmeqa ladder RUN_DIR… [-o …] [--require-balanced32-gate]` | Probe/holdout ladder metrics + optional balanced-32 gate |

Each episode first writes an isolated pending row. A zero-exit row is schema-checked, matched to arm/qid/method/error policy, paired with the exact expected debug bundle, copied into a staging directory, validated against the frozen artifact profile, hashed, and atomically renamed before `COMPLETE.json` is published. Aggregates, progress, and resume state are derived only from validated completion markers. Wrong answers are valid completed episodes; partial/multiple JSON values, nonzero exits, escaped/symlinked bundles, missing required artifacts, and hash mismatches do not commit. Object-crop mosaics are best-effort because an episode may have no usable instance crops; when the diagnostics inventory declares one, it is still copied and validated strictly.

`OUT/DONE` is atomic JSON, not a sentinel string. It is written only when every expected arm×id marker validates and records the run config digest plus aggregate SHA-256. Partial batches exit nonzero, mark the job `failed` / `INCOMPLETE`, and leave `STATUS` pointing at resume.

Default crash policy is **skip** (settle + retry, continue). **`--streak-abort 2`** (default) aborts early after consecutive native crashes so a wedged driver does not burn the full batch.

**Frozen A/B axes (on both `h2h` and `resume`):**

- `--decision-policy legacy|grounded_v2`
- `--use-hm3d-semantics|--no-hm3d-semantics`
- `--enrich-labels|--no-enrich-labels`
- `--graph-evidence-mode off|shadow|agent`
- `--room-history-mode off|shadow|agent`
- `--room-policy canonical|llm`
- `--room-target-hints|--no-room-target-hints`
- `--investigate-stamp|--no-investigate-stamp`
- `--attempt-ledger-mode off|shadow|agent`
- `--action-progress-mode off|shadow|enforce`
- `--variant-id ID`

On first launch, `--variant-config FILE` provides the complete variant block (`id`, decision, graph/history/ledger modes, action-progress mode, room policy/hints, and investigate stamp). Schema v2 rejects missing or unknown fields; legacy schema-v1 files remain readable and default the new action-progress field to `off`. Explicit variant flags win over file values; the manifest source map records the resolved file path plus SHA-256 for every value supplied by the file. Checked-in action-history controls are `configs/benchmarks/hmeqa_action_history_{shadow,agent}.yaml`; static-world retry-policy controls are `configs/benchmarks/hmeqa_action_progress_{shadow,enforce}.yaml`. Resume uses the frozen manifest and does not re-read the file.

For `grounded_v2`, ledger mode is also the dedicated action-history visibility switch: `off` neither persists rows nor renders history, `shadow` persists rows/artifacts but hides them, and `agent` additionally renders recent outcomes, loop flags, per-place attempt summaries/failure risk, global attempts, and mirrored attempt provenance. Room timeline and live approach affordances remain separate axes; see [attempt_ledger.md](attempt_ledger.md).

`action_progress_mode` is independent from ledger visibility and requires `agentic_decision_policy=grounded_v2`; invalid combinations fail before launch. `shadow` computes and traces `would_suppress` decisions but leaves cards and execution unchanged. `enforce` removes only an equivalent terminal/no-progress candidate whose action-specific target state is unchanged; alternate approaches, new views, material frontier/coverage changes, and partial navigation progress remain eligible. This initial policy assumes mostly static HM-EQA scenes. It is not a permanent failure blacklist or a general dynamic-world scheduler; see [attempt_ledger.md](attempt_ledger.md#static-world-action-progress-policy).

Legacy-compatible defaults are `legacy`, graph/history/ledger/action-progress `off`, `canonical`, target hints on, investigate stamps off, HM3D semantic labels off, per-question enrich labels off, and variant ID `legacy`. The two perception/oracle switches are independent frozen axes: enabling the HM3D semantic sensor does not implicitly seed enrich labels, and an explicit semantics-on request fails if its scene assets or annotated dataset config are unavailable. Model and budget controls (`--eqa-hf-model-id`, `--eqa-vl-family`, `--eqa-vl-quantization`, `--eqa-answer-max-new-tokens`, `--episode-timeout`, `--max-planning-steps`, `--max-movement-step`) are frozen with the IDs and are applied to the Habitat runtime. Manifest schema v4 freezes the artifact profile and action-progress axis into the config digest. Each new OUT gets `run_manifest.json` containing the full commit, dirty-tree state/digest, effective values and sources, deterministic config digest, canonical data/HM3D paths, and SHA-256 hashes for `questions.csv` and `scene_init_poses.csv`. HM3D meshes are too large to hash at launch, so the manifest freezes their root path but does not claim mesh-content identity. Resume fills omitted frozen flags from that manifest and refuses commit, dirty-tree, config, artifact-profile, or small-dataset hash mismatches; ambient policy, artifact, and lifecycle variables are not inherited by H2H children. Schema-v2/v3 manifests remain readable for analysis but fail closed on resume. Non-`off` progress runs additionally require parseable `agentic_summary.json` and `agentic_trace.jsonl` diagnostics whose runtime mode matches the manifest before an episode can publish completion. Operational controls such as cooldown, crash policy/streak, job description/name, VRAM threshold, coverage-figure IDs, and foreground mode may change safely on resume. Historical partial OUTs without a manifest fail closed.

**`--preset paper-router`** (on `h2h` / `resume`): sets `agentic_verifier=none` (Qwen `vlm_assess` is the verify gate) + allow-unverified + agentic-router where flags were left at Click defaults; explicit flags still win. It does **not** alter any frozen A/B axis above. Opt in OWL/YoloE with `--agentic-verifier owlv2|yoloe`. Probe runs can omit the preset and keep `--require-verified`. `run_hmeqa_agentic_h2h.sh` honors `EMET_EQA_AGENTIC_ROUTER` (default `0`); scored 2026-07-26 bal-32 used router off because the script previously hardcoded it. Larger-VLM ladder: `--eqa-hf-model-id Qwen/Qwen3-VL-32B-Instruct` (or `EQA_HF_MODEL_ID`) passes through to `emet-habitat run-episode`; see `docs/habitat/vlm_bakeoff.md` and `docs/experiments/agentic_scale.md`.

**Remote answer VL (LAN Orin):** direct and overnight launches construct the same allowlisted child environment. Parent-shell `export EMET_VL_ENDPOINT=…` is **not** inherited unless it is a frozen/explicit input. The child always receives `RESUME=0` or `RESUME=1`; validated job id, canonical lock path/inode-backed FD 9, canonical data paths, and credential inputs are preserved, while ambient policy/lifecycle controls are dropped. Pass **`--host ORIN_HOST`** (injects `EMET_LLM_HOST`, `EMET_OPENAI_BASE_URL`, `EMET_VL_ENDPOINT=openai@http://ORIN_HOST:8000/v1` for unified-7b) or **`--vl-endpoint openai@http://ORIN_HOST:8000/v1`**. Dual-2b: `--host ORIN_HOST --vl-port 8001`. Launch stderr prints the injected endpoint; episode jsonl records `vl_endpoint`. Habitat-Sim still needs local GPU — the Orin only offloads the answer VLM. See `docs/llm_serve.md`.

**`emet hmeqa overnight`** defaults to paper-router policy (`agentic_verifier=none`, allow-unverified, router on). Inner phases call `run_hmeqa_agentic_h2h.sh` directly through the process-tree lifecycle helper (no nested jobs), preserving FD 9 only after canonical lock-path/inode validation. Set `COPY_PAPER_FIGS=1` only when regenerating **holdout-8** paper figures (default off so bal-32 cannot overwrite them).

**Pause / resume overnight:** `emet jobs cancel JOB_ID`, confirm GPU idle (`emet jobs` / `emet eval status`), then:

```bash
uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_<stamp> --job-name hmeqa-overnight
```

That skips phases only when `OUT/DONE` parses and validates against every expected completion marker. A corrupt/plain-text `DONE` is incomplete. Partial `bal32/` runs use marker/pending state to set `RESUME=1`; unvalidated rows are regenerated from committed bundles. For a single H2H directory use `emet hmeqa resume OUT --preset paper-router` instead.

**`emet hmeqa ladder`** reports accuracy, selective risk/coverage, fused-verify precision, visibility at verify, path length, hypothesis count, abstention, false confirmation, and forced submits. Balanced-32 is blocked unless a 4+ episode probe has a nonzero fused verified-answer rate and zero forced submits.

```bash
uv run emet eval recover --need-mib 12000
uv run emet hmeqa overnight
# pause: uv run emet jobs cancel JOB_ID
# resume same ladder:
# uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_… --job-name hmeqa-overnight
# or a probe:
uv run emet hmeqa h2h ~/runs/emet/hmeqa_graph_probe --arms agentic \
  --ids 12,17,18,56 --agentic-verifier owlv2 --require-verified
uv run emet hmeqa ladder ~/runs/emet/hmeqa_graph_probe --require-balanced32-gate
uv run emet hmeqa significance ~/runs/emet/hmeqa_agentic_bal32_...
uv run emet hmeqa failures ~/runs/emet/hmeqa_agentic_bal32_...
uv run emet hmeqa inspect ~/runs/emet/hmeqa_holdout8_fix4_... --misses
uv run emet hmeqa inspect ~/runs/emet/hmeqa_holdout8_fix4_... --qid 105 --open rgb
uv run emet hmeqa resume --preset paper-router
uv run emet hmeqa status
uv run emet status tail
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
