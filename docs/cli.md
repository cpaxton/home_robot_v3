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

# 2. Run DynaMem with visual servoing (in another terminal)
emet run dynamem --robot-ip 127.0.0.1 -S --visual-servo

# 3. Or run mapping
emet run mapping --robot-ip 127.0.0.1
```

If port 4401 is already in use: `emet kill-mujoco-server` then retry, or `emet serve mujoco --port-offset 100`.

## Commands

### `emet serve [backend]`

Start a simulation server.

| Backend | Description |
|---------|--------------|
| `mujoco` | MuJoCo simulation (default) |

**Options:**
- `--use-robocasa` — Use Robocasa for scene generation (default task: PickPlaceCounterToCabinet)
- `--list-robocasa-tasks` — Print all Robocasa task names and exit (for use with `--robocasa-task`)
- `--headless` — Run without native viewer (use web at http://localhost:9090?url=ws://localhost:9877)
- `--scene-path PATH` — Path to MuJoCo scene XML
- `--seed N` — Random seed (default: 0)

**Examples:**
```bash
emet serve                          # MuJoCo, default scene
emet serve mujoco --headless        # No native viewer
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

**Common options:**
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
```

---

### `emet sync [options]`

Sync dependencies. Uses `uv sync` if available, otherwise `pip install -e .`.

**Options:**
- `--all` — Install all common extras (sim, dynamem, dev) — MuJoCo, SAM-2, pytest, etc.
- `-e, --extra EXTRA` — Install extra (repeat for multiple)
- `--sim` — Include sim (MuJoCo, robocasa); sim is default for `./install.sh`, use `--no-sim` to skip
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

**Examples:**
```bash
emet test
emet test -v
emet test src/test/cli/test_cli.py
emet test -k test_serve
```

---

### `emet install <subcommand> [options]`

Install submodules, simulation extras, or full setup.

| Subcommand | Description |
|------------|-------------|
| `submodules` | Init and update git submodules (segment-anything-2, ok-robot) |
| `sim` | Install Robocasa and robosuite (clones into third_party) |
| `robocasa` | Same as `sim` |
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
emet install submodules              # Init and update submodules
emet install sim                    # Install Robocasa, robosuite (third_party)
emet install robocasa               # Same as install sim
emet install sim -d -a              # With assets and force-overwrite macros
emet install full                   # Full install (uv, deps, sync)
emet install full -y                # Non-interactive (sim included by default)
emet install full --cpu             # CPU-only (no SAM2)
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

1. **Terminal 1** — Start the server:
   ```bash
   emet serve mujoco
   # or for Robocasa: emet serve mujoco --use-robocasa
   ```

2. **Terminal 2** — Run the agent:
   ```bash
   emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo
   ```

3. **Headless** — If running without a display:
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
