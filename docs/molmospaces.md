# MolmoSpaces

[MolmoSpaces](https://github.com/allenai/molmospaces) is an open ecosystem for robot manipulation and navigation: indoor scenes (iTHOR, ProcTHOR, Holodeck), object models, and robot assets (e.g. **rby1** / Galaxea R1 family, Franka). Emet integrates MolmoSpaces so you can set up scenes, use their robots, and run or visualize results in simulation.

MolmoSpaces requires **mujoco 3.4** and **numpy>=2.2**, which conflict with the main emet environment (numpy<2). The main sim extra uses **mujoco>=3.4** so merged Molmo MJCFs load in the ZMQ server; numpy stays on the 1.x line in core. So the integration is split:

- **Core emet**: No dependency on molmo-spaces or mujoco 3.4. The `emet molmospaces` CLI lives in core; `list-robots` uses a static config. For `list-scenes`, `install-scene`, `merge-scene`, and `serve`, the CLI **delegates to a thin wrapper** via subprocess.
- **Sim YAML / one-terminal agent**: see [sim_configs.md](sim_configs.md) for `configs/sim/molmospaces_*.yaml` and `emet run agent --start-sim`.
- **Wrapper (emet-molmospaces)**: A separate package that depends on emet, molmo-spaces, mujoco>=3.4, and numpy>=2.2. It provides the `emet-molmospaces` console script and implements list-scenes, install-scene, merge-scene, and serve. Install it in a dedicated venv (e.g. `.venv-molmospaces`) or any env where you accept those deps.

## Install the MolmoSpaces wrapper

From the project root:

**Default (sim install):** `./install.sh -y` and `emet install full -y` also create **`.venv-molmospaces`** when `packages/emet_molmospaces` exists, so `emet serve mujoco --molmospaces-*` works without a second install. Use **`--no-molmospaces`** to skip that venv (lighter machine / CI).

**Wrapper only** (e.g. you used `--no-sim` earlier):

```bash
./install.sh --molmospaces -y
```

```bash
emet install full -y --molmospaces
# or full default install (includes MolmoSpaces with sim):
emet install full -y
```

This creates `.venv-molmospaces`, installs emet (no-deps) and then the wrapper from the repo (`pip install -e packages/emet_molmospaces`). The wrapper’s script `emet-molmospaces` will be at `.venv-molmospaces/bin/emet-molmospaces`. Core emet discovers it there, or via `MOLMOSPACES_PYTHON` (see below), or via `which emet-molmospaces` if on PATH.

Alternatively, install the **local** wrapper (there is no PyPI package named `emet-molmospaces`; do not use `pip install emet-molmospaces` alone). **molmo-spaces** is also **not on PyPI** as a simple `pip install molmo-spaces`; this repo installs it **from GitHub** (`allenai/molmospaces`). The MolmoSpaces venv must use **Python ≥3.11** (upstream requirement).

```bash
# from repo root, in a dedicated venv:
uv venv .venv-molmospaces --python 3.11
uv pip install --python .venv-molmospaces/bin/python --no-deps -e .
uv pip install --python .venv-molmospaces/bin/python -e packages/emet_molmospaces
```

Or use `./install.sh --molmospaces -y`, which creates `.venv-molmospaces` (Python 3.11+) and runs the equivalent steps.

Set the asset directory for scene install and serve:

```bash
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets
mkdir -p "$HOME/.cache/molmospaces/assets"
export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets"
```

Emet discovers the wrapper by: **first** `.venv-molmospaces` in the repo if that Python can `import molmo_spaces`, then `MOLMOSPACES_PYTHON` **only if** it passes the same import check, then `emet-molmospaces` on `PATH` if its sibling `python` can import MolmoSpaces. If you set `MOLMOSPACES_PYTHON` to an old conda env, **unset** it so the project venv is used. To force a specific env:

```bash
export MOLMOSPACES_PYTHON=/path/to/your/molmospaces/venv/bin/python
```

(Core looks for `emet-molmospaces` in that Python’s `bin` directory.)

## Commands

- **List robots** (no wrapper needed; uses static list in core):

  ```bash
  emet molmospaces list-robots
  ```

  Prints supported robot IDs (rby1, rby1m, franka_droid, franka_cap, etc.). Default is **rby1** (Galaxea R1 family).

- **List scenes** (delegates to wrapper):

  ```bash
  emet molmospaces list-scenes
  ```

  If the wrapper is not installed, the CLI exits with instructions to run `./install.sh --molmospaces` or install `packages/emet_molmospaces` editable into `.venv-molmospaces`. With the wrapper, prints scene names (ithor, procthor-10k, etc.) and split sizes.

- **Install a scene**:

  ```bash
  emet molmospaces install-scene --scene ithor --split train --index 0 --scene-path /tmp/ithor_scene.xml
  ```

  Delegates to the wrapper; downloads and installs the scene; optionally copies the scene XML to `--scene-path`.

- **Merge scene + robot (for ZMQ / agent)**:

  ```bash
  emet molmospaces merge-scene --scene ithor --split train --index 0 --robot rby1 \
    -o src/emet/assets/robot/galaxea_r1/molmospaces_rby1.xml
  ```

  Delegates to the wrapper: installs the scene if needed, merges the **rby1** (Galaxea R1) MJCF from core emet assets into the scene, and writes a **persistent** merged MJCF. Prefer writing **`-o`** under `src/emet/assets/robot/galaxea_r1/` (next to `galaxea_r1.xml`) so MuJoCo resolves robot meshes; `/tmp` breaks `assetdir="meshes"`. Use that path with **`emet serve mujoco`** in your **main** project environment (see below), not only with passive `emet molmospaces serve`.

- **Build iTHOR orthographic occupancy (debug / offline)**:

  ```bash
  emet molmospaces build-occ-map path/to/merged.xml
  ```

  Uses the vendored Molmo-style **`iTHORMap`** (orthographic segmentation render) and writes **`occupancy.png`** and **`occupancy_meta.json`** next to the MJCF (or under **`-o` / `--output-dir`**). Headless servers should set **`MUJOCO_GL=egl`** (or `osmesa`) so MuJoCo can render. This does not call the MolmoSpaces wrapper.

- **Run simulation (serve)**:

  ```bash
  emet molmospaces serve --scene ithor --robot rby1 --viewer
  ```

  Delegates to the wrapper: installs the scene (if needed), loads the MJCF, and runs MuJoCo. Use `--viewer` for the native MuJoCo viewer, `--headless` for no GUI. Optional `--rerun PORT` or `--rerun path.rrd` logs step data to Rerun.

  Examples:

  ```bash
  emet molmospaces serve --scene ithor --split train --index 1 --robot rby1 --viewer
  emet molmospaces serve --scene procthor-10k --headless --rerun 9876
  ```

## Agent chat and ZMQ (core emet)

Passive `emet molmospaces serve` only steps physics in the wrapper’s MuJoCo. To drive the robot and use **`emet run agent`** (LLM + tools), run the **ZMQ MuJoCo server** from the same environment you use for normal simulation (`uv sync` from the repo root installs default groups including **sim**).

**Recommended (one command):** `emet serve mujoco` can merge a MolmoSpaces scene with **rby1** (Galaxea R1) via the wrapper, then start the ZMQ server. You do **not** need to run `merge-scene` first.

1. Install assets and (if needed) the wrapper venv: `./install.sh --molmospaces -y`. If you do not set `MLSPACES_ASSETS_DIR` / `MLSPACES_CACHE_DIR`, core emet defaults them to sibling directories under `~/.cache/molmospaces/` (`assets` and `resource_cache`). Upstream requires those roots to differ and not nest inside each other.
2. From the project root with the **main** `.venv` (where `emet` and sim extras live):

   ```bash
   emet serve mujoco --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 --robot rby1 --headless
   ```

   This calls the wrapper’s `merge-scene`, writes the merged MJCF under **`src/emet/assets/robot/galaxea_r1/`** (a temp file named `molmospaces_merged_*.xml`), then starts `emet.simulation.mujoco_server` with `--robot` set to the same robot you merged (e.g. `rby1` or `galaxea_r1`). The file must live next to `galaxea_r1.xml` so MuJoCo resolves the robot’s `assetdir="meshes"`; writing the merge under `/tmp` breaks mesh loading. The merged file is kept on disk until the server **stops** (so iTHOR occupancy sampling can read the same path), then removed. If you omit `--robot`, the effective default for this path is **rby1** (Stretch has no bundled merge MJCF; passing `--robot stretch` is an error).

3. Run the agent:

   ```bash
   emet run agent --robot-ip 127.0.0.1 --robot rby1
   ```

**Optional: fixed path** — use `emet molmospaces merge-scene ... -o /path/to/merged.xml` if you want a stable file, then `emet serve mujoco --robot rby1 --scene-path /path/to/merged.xml`.

Use `--port-offset` on both server and agent if default ZMQ ports are busy. The agent uses **`GenericZmqClient`** for `rby1`, matching `emet run dynamem --robot rby1`.

### FAQ (ZMQ path vs wrapper `molmospaces serve`)

- **Why does the merged MJCF always live under `…/galaxea_r1/` even when I pass `--robot innate_mars`?**  
  MuJoCo resolves the robot’s `assetdir="meshes"` relative to the **main** MJCF file. Emet always writes the temp merge next to the vendored **Galaxea R1** XML so mesh paths resolve. That directory name is **not** a claim that the articulated model is always “Galaxea”; the merged robot follows `--robot` (bundled MJCF only).

- **Why does `--robot stretch` fail?**  
  There is no Stretch-in-iTHOR MJCF on the emet merge path. Use `rby1`, `galaxea_r1`, `innate_mars`, `rb_y1`, or `maurice`, or omit `--robot` (defaults to `rby1` when `--molmospaces-scene` is set).

- **Does `emet serve mujoco --molmospaces-scene` run Molmo’s learned policies?**  
  No. The ZMQ server uses **emet’s** `RobosuiteZmqServer`: MuJoCo dynamics, PD-style actuators from the MJCF, and the same ZMQ protocol as other non-Stretch sims. It is not the upstream MolmoSpaces Python control stack.

- **Post-load diagnostics (actuators, floor height, short `mj_step` probe):**  
  Pass `--debug-molmospaces-spawn` on `emet serve mujoco`, or set **`EMET_ROBOSUITE_POST_LOAD_DEBUG=1`**. After load, emet applies the MJCF **`home` keyframe** when present (e.g. Galaxea R1) while **preserving** the MolmoSpaces autoplace base pose, then stabilizes; this reduces arm/torso collapse from default compiled `qpos`.

**iTHOR spawn occupancy (ZMQ server):** For **`ithor`** scenes, free-joint XY search prefers points sampled from the same orthographic occupancy map (Molmo-style) before falling back to annulus/grid heuristics. Set **`EMET_MOLMOSPACES_OCC_MAP=0`** (or `false`) to disable. **`EMET_MOLMOSPACES_OCC_SEED`** seeds the occupancy free-point subsample (default `0`).

The ZMQ server publishes a static **`emet_session`** block (scene / robot / capabilities) on every message; see [zmq_session_metadata.md](zmq_session_metadata.md). `emet run molmospaces-explore` prefers this metadata for `episode.json` when the server reports a MolmoSpaces environment.

## Exploration dataset + NeRF (phase 1)

Record posed RGB (and optional depth) while the robot moves in a MolmoSpaces-backed scene. This uses the **same ZMQ workflow** as above: start the MuJoCo server in one terminal, then run the explorer in another.

### Two-terminal workflow

1. **Terminal A — sim** (main `uv` env, sim extras):

   ```bash
   emet serve mujoco --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 \
     --robot rby1 --headless
   ```

2. **Terminal B — explore + record**:

   ```bash
   emet run molmospaces-explore --robot-ip 127.0.0.1 \
     --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 \
     --output-dir ./data/molmo_ep_ithor_0 --steps 120 --capture-hz 2 \
     --export-transforms
   ```

   Add ``--robot rby1`` if you want to force a backend; if omitted, the explorer reads ``emet_robot_id`` from the running ZMQ server first.

By default the explorer also writes **`episode_rgb.mp4`** in the same output directory (requires a working OpenCV `VideoWriter` with `mp4v`). Use **`--no-mp4`** to disable.

Scene flags on the explore command are **defaults for metadata**; when the server publishes **`emet_session`**, `episode.json` prefers the server’s MolmoSpaces `environment` and embeds a JSON-safe copy of **`emet_session`** (robot, runtime, capabilities). Random goals use `--goal-x-min` / `--goal-x-max` / `--goal-y-min` / `--goal-y-max` (meters) and `--navigate-every` to throttle `move_base_to` calls.

Optional **`--with-graph-report`** builds a lightweight **GraphEQA** text report (`graph_report.txt`) for debugging; it does not affect NeRF files. Use **`--cpu-only`** with that flag on machines without a VLM GPU.

### On-disk layout (per episode)

| Path | Purpose |
|------|---------|
| `images/frame_XXXXXX.png` | Head RGB |
| `depths/frame_XXXXXX.npy` | Depth in meters (optional; `--no-depth` disables) |
| `metadata.jsonl` | One JSON per line: `image`, optional `depth`, `camera_pose` (4×4), `camera_K`, `gps`, `compass`, `seq_id` |
| `episode_rgb.mp4` | **Exploration video** (RGB timeline at `--mp4-fps`, OpenCV `mp4v`); written by default after a run. Pass **`--no-mp4`** to skip if your OpenCV lacks working `VideoWriter`. |
| `episode.json` | Scene/robot/split/index, step counts, optional `git_commit`, and **`rgb_mp4`** filename when an MP4 was written |

**Camera convention:** `camera_pose` is stored as emitted by the MuJoCo ZMQ server for the head camera (4×4 homogeneous). The **`emet molmospaces export-nerfstudio`** command maps each row to NERFStudio **`transform_matrix`** entries and sets **`camera_angle_x`** from intrinsics when available. For strict NeRF pipelines you may need an additional **OpenCV ↔ OpenGL** flip depending on the trainer; document any extra transform in your training config.

```bash
emet molmospaces export-nerfstudio --episode-dir ./data/molmo_ep_ithor_0
```

### MuJoCo version note

MolmoSpaces assets are built with **MuJoCo 3.4**; the main **`sim`** extra pins **mujoco>=3.4** for compatibility. If `emet serve mujoco` still fails to load a merged MJCF, report an asset compatibility issue. The wrapper venv is still required for **download/install/merge**; the server should run where **`emet` and sim extras** are installed.

### Simulation control (emet ZMQ vs MolmoSpaces upstream)

MolmoSpaces keeps articulated robots from collapsing at idle by **holding joint targets** and **rewriting actuator `ctrl` every physics sub-step** before `env.step` (see upstream `molmo_spaces/tasks/task.py`: inner loop calls `robot.compute_control()` then steps; `RBY1.update_control` / `compute_control` in `molmo_spaces/robots/rby1.py`; stationary joint targets in `molmo_spaces/controllers/joint_pos.py`).

The emet **`RobosuiteZmqServer`** path does not import that stack (different object graph and optional MolmoSpaces venv). It **mirrors the same semantics**: a per-actuator hold buffer aligned with `RobotSpec` joint/actuator names, refreshed when ZMQ sends `joint` targets or when post-load sync copies `qpos` into `ctrl`, then **re-applied before each `mj_step`** so PD actuators never sit on stale `ctrl` between client messages.

## Showing results

- **Viewer:** `emet molmospaces serve --viewer` opens the MuJoCo passive viewer.
- **Rerun:** Use `--rerun <port>` or `--rerun path.rrd`; then open the Rerun viewer (e.g. `http://localhost:9090?url=ws://localhost:9876` or load the RRD).

For a step-by-step **testing plan** (core tests, wrapper tests with mocks, optional integration), see **[docs/plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md)**.

### Quick verification (developers)

- **Core CLI tests** (no live MolmoSpaces import required for most cases):

  ```bash
  uv run emet test src/test/cli/test_molmospaces_cli.py
  ```

- **Optional**: tests that invoke the real wrapper need `RUN_MOLMOSPACES_TESTS=1` and a Python env where `import emet_molmospaces` succeeds (often `.venv-molmospaces` after `./install.sh --molmospaces`; you can run pytest with that interpreter).

- **Optional — base orientation after spawn + physics** (10 iTHOR FloorPlans, autoplace, ~900 `mj_step`, checks upright + small quaternion drift):

  ```bash
  RUN_MOLMOSPACES_TESTS=1 uv run emet test src/test/molmospaces/test_molmospaces_ithor_base_settle.py -v
  ```

  Tunables: `EMET_MOLMOSPACES_ORIENTATION_N` (default 10), `EMET_MOLMOSPACES_SETTLE_STEPS` (default 900), `EMET_MOLMOSPACES_ORIENTATION_MAX_DEG` (default 8), `EMET_MOLMOSPACES_MIN_UP_DOT` (default 0.92).

- **Wrapper package tests**:

  ```bash
  uv run pytest packages/emet_molmospaces/tests/ -q
  ```

- **Smoke (end-to-end load + ZMQ)**: from repo root, with sim extras and wrapper installed, expect the server to start without MJCF errors:

  ```bash
  timeout 25 uv run emet serve mujoco --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 --robot rby1 --headless
  ```

- **Optional explore + export smoke** (manual, two terminals; needs a running ZMQ server and patience for first obs):

  Terminal A: same `emet serve mujoco …` as in **Exploration dataset + NeRF** above until the server is ready.

  Terminal B (few steps, then NERFStudio export):

  ```bash
  uv run emet run molmospaces-explore --robot rby1 --output-dir ./tmp_molmo_explore_smoke \
    --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 \
    --steps 3 --no-mp4 --export-transforms
  ```

  Core tests also cover **`emet molmospaces export-nerfstudio`** on a synthetic episode directory (no sim).

## Troubleshooting

- **"Timeout waiting for observations/state from ZMQ server" (GenericZmqClient)**  
  The MuJoCo server must be running **before** you start `emet run …` clients. Merged Molmo scenes often need **30–90 seconds** after `emet serve mujoco` starts before the first ZMQ messages appear. The client now waits **60 seconds** by default (was 10). Increase further: `export EMET_ZMQ_STARTUP_TIMEOUT=120` or `emet run molmospaces-explore --zmq-startup-timeout 120 …`. Confirm **`--robot`** and **`--port-offset`** match on server and client.

- **"MolmoSpaces wrapper not found" / `pip install emet-molmospaces` fails**
  The wrapper is **not on PyPI**. From the repo root run `./install.sh --molmospaces -y`, or install editable: `uv pip install --no-deps -e .` and `uv pip install -e packages/emet_molmospaces` into `.venv-molmospaces` (see **Install the MolmoSpaces wrapper** above). Core emet discovers `.venv-molmospaces/bin/emet-molmospaces` or runs `python -m emet_molmospaces` from that venv. You can also set `MOLMOSPACES_PYTHON` to that venv’s `python` binary.

- **"MLSPACES_ASSETS_DIR not set"**
  Export `MLSPACES_ASSETS_DIR` to a directory where scene assets will be downloaded (e.g. `~/.cache/molmospaces/assets`).

- **"molmo_spaces not found"** (from the wrapper)
  The wrapper is running in an env that doesn’t have molmo-spaces. Use the venv where you installed emet-molmospaces (e.g. `.venv-molmospaces`) or set `MOLMOSPACES_PYTHON` so the core invokes the wrapper from that env.

- **`emet serve mujoco` fails to parse merged MJCF**
  See **MuJoCo version note** above. Confirm `--scene_path` points to the file written by `merge-scene` and that `--robot rby1` matches the merged robot.

- **Missing THOR object meshes (`…/objects/thor/...`) or broken `resource_cache` layout**
  Ensure `MLSPACES_ASSETS_DIR` and `MLSPACES_CACHE_DIR` match the env used for install/merge (sibling dirs). Re-run `emet serve mujoco --molmospaces-scene …` or `emet molmospaces merge-scene …` after `./install.sh --molmospaces -y`. Core applies `ensure_molmo_asset_layout_symlinks()` before load to link `scenes/objects` → `objects` and flatten versioned THOR folders where needed.

- **Robot meshes missing (`base_link.STL`, etc.) after merge**
  The merged wrapper XML must sit next to `galaxea_r1.xml` (not under `/tmp` alone). Using `emet serve mujoco --molmospaces-scene …` does this automatically; if you merge manually, pass `-o` under `src/emet/assets/robot/galaxea_r1/` or another path that keeps MuJoCo’s mesh resolution consistent with the packaged robot MJCF.

- **`uv pip install -e packages/emet_molmospaces` fails (Python 3.10 / emet not found)**
  Install the wrapper **only** into **`.venv-molmospaces`** (Python **≥3.11**), not the main `.venv` / conda env. Use `./install.sh --molmospaces -y` or the commands in **Install the MolmoSpaces wrapper** — do not run `uv pip install -e packages/emet_molmospaces` in the 3.10 environment.
