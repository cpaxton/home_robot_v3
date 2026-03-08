# Emet CLI Tool

The `emet` CLI makes it easy to start simulations, run robot agents, sync dependencies, view logs, and run tests. It supports bash and zsh tab completion.

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

## Commands

### `emet serve [backend]`

Start a simulation server.

| Backend | Description |
|---------|--------------|
| `mujoco` | MuJoCo simulation (default) |

**Options:**
- `--use-robocasa` — Use Robocasa for scene generation
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
- `--robot-ip IP` — Robot or simulator IP (default: 127.0.0.1)
- `--server-ip IP` — Server IP for AnyGrasp (dynamem)
- `-S, --skip` — Skip confirmations
- `--headless` — Run without display
- `--visual-servo, -V` — Use visual servoing (dynamem)
- `--target-object OBJ` — Target object (grasp)
- `--parameter-file FILE` — Planner config (e.g. sim_planner.yaml)

**Examples:**
```bash
emet run dynamem --robot-ip 127.0.0.1 -S
emet run dynamem -S --visual-servo --match-method class
emet run mapping --robot-ip 127.0.0.1
emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
emet run timing --robot-ip 192.168.1.15 --headless
```

Extra arguments are passed through to the underlying app.

---

### `emet sync [options]`

Sync dependencies. Uses `uv sync` if available, otherwise `pip install -e .`.

**Options:**
- `-e, --extra EXTRA` — Install extra (repeat for multiple): `sim`, `dynamem`, `dev`
- `--no-install` — Only sync lockfile, do not install emet

**Examples:**
```bash
emet sync
emet sync -e sim -e dynamem
emet sync -e dev
emet sync --no-install
```

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
| `full` | Run full install (./install.sh) |

**`emet install submodules`**
- `--recursive` / `--no-recursive` — Recursively init nested submodules (default: recursive)

**`emet install sim`**
- `-d, --download-assets` — Download Robocasa kitchen assets
- `-a, --setup-macros` — Run Robocasa setup_macros.py

**`emet install full`**
- `-y, --yes` — Skip confirmation prompts
- `--sim` — Include simulation extras
- `--cpu` — CPU-only (skip SAM2)
- `--no-sam2` — Skip Segment Anything 2

**Examples:**
```bash
emet install submodules              # Init and update submodules
emet install sim                    # Install Robocasa, robosuite
emet install sim -d -a              # With assets and macros
emet install full                   # Full install (uv, deps, sync)
emet install full -y --sim          # Non-interactive with sim extras
emet install full --cpu             # CPU-only (no SAM2)
```

After `emet install sim`, run `emet sync -e sim` to install emet with sim extras.

---

### `emet install-completion [options]`

Print shell completion script for bash or zsh.

**Options:**
- `-s, --shell {bash,zsh}` — Shell type (auto-detected from $SHELL if omitted)

**Setup:**
```bash
# Bash: add to ~/.bashrc
eval "$(emet install-completion --shell bash)"

# Zsh: add to ~/.zshrc
eval "$(emet install-completion --shell zsh)"
```

Then restart your shell or `source ~/.bashrc` / `source ~/.zshrc`.

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
