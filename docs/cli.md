# Emet CLI Tool

The `emet` CLI makes it easy to start simulations, run robot agents, sync dependencies, view logs, and run tests. It supports **tab completion** for bash, zsh, and fish (see [Tab completion](#tab-completion) below).

## Installation

After installing emet (`uv sync` or `pip install -e .`), the `emet` command is available:

```bash
emet --help
```

## Quick Start

```bash
# 1. Start the MuJoCo simulation server (in one terminal)
emet serve mujoco
# Innate Mars (default table + robot): same command with --robot
emet serve --robot innate_mars --headless   # optional: default backend is mujoco

# 2. Run DynaMem with visual servoing (in another terminal)
emet run dynamem --robot-ip 127.0.0.1 -S --visual-servo
# With Innate Mars sim, pass the same robot to the client:
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S

# 3. Or run mapping
emet run mapping --robot-ip 127.0.0.1

# Optional: DA3 depth + point cloud in Rerun (sim must be running; depth-anything-3 is a default dep)
emet debug-da3-depth --robot innate_mars
# Equivalent:
emet run debug-da3-depth --robot innate_mars
```

If port 4401 is already in use: `emet kill-mujoco-server` then retry, or `emet serve mujoco --port-offset 100`.

## Commands

### `emet serve [backend]`

Start a simulation server.

| Backend | Description |
|---------|--------------|
| `mujoco` | MuJoCo simulation (default) |

The positional **`[mujoco|robocasa]`** is optional (`emet serve` defaults to **mujoco**).

**Options:**
- `--robot NAME` — Simulator robot (default `stretch`). Use **`innate_mars`**, **`rby1`**, **`galaxea_r1`**, etc. for registry robots: loads **`scene_environment.xml`** (default table room: red cylinder, blue cube, floor) merged with that robot’s MJCF and starts the **generic ZMQ** sim (`RobosuiteZmqServer`) on ports **4401–4404**. Must match **`emet run dynamem --robot NAME`** (or `create_robot_client_from_cli`) on the client. For DynaMem with Innate Mars + Depth Anything 3, pass **`--dynav-config dynav_innate_mars.yaml`** to **`emet run dynamem`** (see `docs/robots/innate_mars.md`).
- `--use-robocasa` — Use Robocasa for scene generation (default task: PickPlaceCounterToCabinet)
- `--list-robocasa-tasks` — Print all Robocasa task names and exit (for use with `--robocasa-task`)
- `--headless` — Run without native viewer (use web at http://localhost:9090?url=ws://localhost:9877)
- `--scene-path PATH` — Path to MuJoCo scene XML
- `--port-offset N` — Add N to default ports (e.g. 100 → 4501–4504) when 4401 is busy
- `--seed N` — Random seed (default: 0)

**Examples:**
```bash
emet serve                          # MuJoCo, default scene, Stretch
emet serve mujoco --headless        # No native viewer
emet serve --robot innate_mars --headless   # Innate Mars + default table (match client --robot)
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S --dynav-config dynav_innate_mars.yaml   # DynaMem + DA3
emet serve mujoco --use-robocasa    # Robocasa scene
```

---

### `emet run <app> [options]`

Run a robot agent or app.

| App | Description |
|-----|--------------|
| `dynamem` | DynaMem navigation + manipulation |
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
- `--hz`, `--stride` — cap FPS and point-cloud density.

**Examples:**
```bash
emet debug-da3-depth --robot innate_mars
emet debug-da3-depth --robot innate_mars --depth-source sensor
emet debug-da3-depth --model-id depth-anything/DA3METRIC-LARGE --process-res 504 --hz 2
```

---

### `emet sync [options]`

Sync dependencies. Uses `uv sync` if available, otherwise `pip install -e .`.

**Options:**
- `--all` — Install all common extras (sim, dynamem, dev) — MuJoCo, SAM-2, pytest, etc.
- `-e, --extra EXTRA` — Install extra (repeat for multiple)
- `--sim` — Include sim (MuJoCo, robocasa). `./install.sh` defaults to **no** sim; use `--sim`, `--all`, or `--profile=full` / `EMET_INSTALL_PROFILE=full` for legacy behavior
- `--dynamem` — Include dynamem (SAM-2)
- `--dev` — Include dev (pytest, black, mypy)
- `--hand-tracker` — Include hand_tracker (mediapipe)
- `--discord` — Include discord
- `--no-install` — Only sync lockfile, do not install emet

**Examples:**
```bash
emet sync
emet sync --all
emet sync -e sim -e dynamem
emet sync -e sim -e dynamem
emet sync --all --hand-tracker
emet sync --no-install
```

When using both `sim` and `dynamem`, uv applies a numpy override (see `[tool.uv] override-dependencies` in `pyproject.toml`) so Robocasa’s pin and SAM-2’s requirement resolve together.

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

Run the full test suite:
```bash
emet test
```

Run only CLI tests:
```bash
emet test src/test/cli/
```

Verbose with coverage:
```bash
emet test -v
```
