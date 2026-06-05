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
| `robocasa` | Shortcut for `mujoco --use-robocasa` (kitchen scenes) |
| `molmospaces` | Shortcut for `mujoco --molmospaces-scene …` (merge Molmo scene + robot, then ZMQ) |

The positional backend is optional (`emet serve` defaults to **mujoco**). Optional scene name after `molmospaces`: `emet serve molmospaces ithor`.

**Options:**
- `--robot NAME` — Simulator robot (default `stretch` for table, Robocasa, and MolmoSpaces when omitted). Use **`innate_mars`**, **`rby1`**, **`galaxea_r1`**, etc. for registry robots: loads **`scene_environment.xml`** (default table room: red cylinder, blue cube, floor) merged with that robot’s MJCF and starts the **generic ZMQ** sim (`RobosuiteZmqServer`) on ports **4401–4404**. Must match **`emet run dynamem --robot NAME`** (or `create_robot_client_from_cli`) on the client. DynaMem uses the same default **`dynav_config.yaml`** for all robots; for Innate Mars **without** ZMQ depth, add **`--dynav-config dynav_innate_mars.yaml`** (see `docs/robots/innate_mars.md`).
- `--use-robocasa` — Use Robocasa for scene generation (default task: PickPlaceCounterToCabinet)
- `--molmospaces-scene NAME` — MolmoSpaces scene (e.g. `ithor`); merges via wrapper then starts ZMQ (same as `emet serve molmospaces`)
- `--molmospaces-split` — `train` / `val` / `test` (default: `train`)
- `--molmospaces-index N` — Scene index within split (default: `0`)
- `--molmospaces-install` — Download/link scene assets if missing
- `--list-robocasa-tasks` — Print all Robocasa task names and exit (for use with `--robocasa-task`)
- `--headless` — Run without native viewer (use web at http://localhost:9090?url=ws://localhost:9877)
- `--scene-path PATH` — Path to MuJoCo scene XML
- `--port-offset N` — Add N to default ports (e.g. 100 → 4501–4504) when 4401 is busy
- `--seed N` — Random seed (default: 0)

See [Simulation](simulation.md), [MolmoSpaces](molmospaces.md), and maintainer [simulation_modules.md](simulation_modules.md).

**Examples:**
```bash
emet serve                          # MuJoCo, default scene, Stretch
emet serve mujoco --headless        # No native viewer
emet serve --robot innate_mars --headless   # Innate Mars + default table (match client --robot)
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S --dynav-config dynav_innate_mars.yaml   # DynaMem + DA3 (no ZMQ depth)
emet serve mujoco --use-robocasa    # Robocasa scene
emet serve robocasa --robot galaxea_r1 --headless
emet serve molmospaces --headless   # Molmo iTHOR + stretch (default robot)
emet serve mujoco --molmospaces-scene ithor --molmospaces-index 0 --headless
emet serve molmospaces --robot rby1 --headless   # Galaxea R1 on Molmo scene
```

---

### `emet molmospaces <subcommand>`

MolmoSpaces scene setup, merge, passive sim, and offline maintainer tools. See [molmospaces.md](molmospaces.md), [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md), and [simulation_modules.md](simulation_modules.md).

| Subcommand | Wrapper? | Description |
|------------|----------|-------------|
| `list-robots` | No (core) | Static robot IDs; default **stretch** when `--robot` omitted on serve |
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

For **agent + tools**, use **`emet serve mujoco --molmospaces-scene …`** (ZMQ) instead of passive `emet molmospaces serve`.

---

### `emet run <app> [options]`

Run a robot agent or app.

| App | Description |
|-----|--------------|
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

### `emet preview-cameras [options]`

Build a **labeled horizontal montage** of the robot’s MuJoCo/ZMQ cameras (for Innate Mars: `head_left`, `head_right`, `camera_arm`) to check orientation, stereo wiring, and tabletop aim without running a full agent loop. Implements `emet.app.preview_robot_cameras`; options are passed through (see `emet preview-cameras -h`).

**Modes**
- **`--source local`** (default) — Load the same **merged** model as `emet serve mujoco` (`scene_environment.xml` + robot MJCF), render with MuJoCo at 640×480, and apply the same RGB postprocess as `RobosuiteZmqServer` (per robot: `RobotSpec.robosuite_rgb_depth_ops`; innate_mars uses **`flipud`** on MuJoCo `Renderer` output; robots with empty ops may still honor optional `EMET_ROBOSUITE_RENDER_FLIPUD`).
- **`--source zmq`** — Subscribe once on the **full observation** port (default **4401**, same as `GenericZmqClient`), decode JPEG fields, and montage. Requires a running sim or bridge. Newer `RobosuiteZmqServer` builds also attach a third JPEG (`rgb_tertiary`, `camera_name_tertiary`) when the spec lists a distinct third camera.

**Common options:** `--robot`, `--out` (single PNG), `--max-cams`, `--row-height`, `--recv-port` / `--timeout-ms` (ZMQ), `--discord` (post the single montage; needs `DISCORD_TOKEN`, `EMET_DISCORD_CHANNEL`).

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
emet preview-cameras --source zmq --robot innate_mars --robot-ip 127.0.0.1

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

### `emet kill-mujoco-server [options]`

Stop MuJoCo simulation server(s) so ports 4401–4404 are free.

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
   # or for Robocasa: emet serve mujoco --use-robocasa
   ```

2. **Terminal 2** — Run the agent:
   ```bash
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
