# MolmoSpaces

[MolmoSpaces](https://github.com/allenai/molmospaces) is an open ecosystem for robot manipulation and navigation: indoor scenes (iTHOR, ProcTHOR, Holodeck), object models, and robot assets (e.g. **rby1** / Galaxea R1 family, Franka). Emet integrates MolmoSpaces so you can set up scenes, use their robots, and run or visualize results in simulation.

MolmoSpaces requires **mujoco 3.4** and **numpy>=2.2**, which conflict with the main emet environment (numpy<2, mujoco>=3.3). So the integration is split:

- **Core emet**: No dependency on molmo-spaces or mujoco 3.4. The `emet molmospaces` CLI lives in core; `list-robots` uses a static config. For `list-scenes`, `install-scene`, and `serve`, the CLI **delegates to a thin wrapper** via subprocess.
- **Wrapper (emet-molmospaces)**: A separate package that depends on emet, molmo-spaces, mujoco>=3.4, and numpy>=2.2. It provides the `emet-molmospaces` console script and implements list-scenes, install-scene, and serve. Install it in a dedicated venv (e.g. `.venv-molmospaces`) or any env where you accept those deps.

## Install the MolmoSpaces wrapper

From the project root (recommended):

```bash
./install.sh --molmospaces -y
```

This creates `.venv-molmospaces`, installs emet (no-deps) and then the wrapper from the repo (`pip install -e packages/emet_molmospaces`). The wrapper’s script `emet-molmospaces` will be at `.venv-molmospaces/bin/emet-molmospaces`. Core emet discovers it there, or via `MOLMOSPACES_PYTHON` (see below), or via `which emet-molmospaces` if on PATH.

Alternatively, install the wrapper in any venv that has molmo-spaces and mujoco 3.4:

```bash
pip install emet-molmospaces
# or from repo:
pip install -e packages/emet_molmospaces
```

Set the asset directory for scene install and serve:

```bash
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets
mkdir -p "$HOME/.cache/molmospaces/assets"
export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets"
```

Emet discovers the wrapper by (in order): the same `bin` as `MOLMOSPACES_PYTHON` (if set), then `.venv-molmospaces/bin/emet-molmospaces`, then `shutil.which("emet-molmospaces")`. To force a specific env:

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

  If the wrapper is not installed, the CLI exits with a message: install via `pip install emet-molmospaces` (in a venv with molmo-spaces) or run `install.sh --molmospaces`. With the wrapper, prints scene names (ithor, procthor-10k, etc.) and split sizes.

- **Install a scene**:

  ```bash
  emet molmospaces install-scene --scene ithor --split train --index 0 --scene-path /tmp/ithor_scene.xml
  ```

  Delegates to the wrapper; downloads and installs the scene; optionally copies the scene XML to `--scene-path`.

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

## Showing results

- **Viewer:** `emet molmospaces serve --viewer` opens the MuJoCo passive viewer.
- **Rerun:** Use `--rerun <port>` or `--rerun path.rrd`; then open the Rerun viewer (e.g. `http://localhost:9090?url=ws://localhost:9876` or load the RRD).

For a step-by-step **testing plan** (core tests, wrapper tests with mocks, optional integration), see **[docs/plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md)**.

## Troubleshooting

- **"MolmoSpaces wrapper not found" / "Install the MolmoSpaces wrapper"**
  Install the wrapper: `pip install emet-molmospaces` in a venv with molmo-spaces and mujoco 3.4, or run `./install.sh --molmospaces`. Ensure emet can find the `emet-molmospaces` executable (e.g. `.venv-molmospaces/bin/emet-molmospaces` or set `MOLMOSPACES_PYTHON` to that venv’s Python).

- **"MLSPACES_ASSETS_DIR not set"**
  Export `MLSPACES_ASSETS_DIR` to a directory where scene assets will be downloaded (e.g. `~/.cache/molmospaces/assets`).

- **"molmo_spaces not found"** (from the wrapper)
  The wrapper is running in an env that doesn’t have molmo-spaces. Use the venv where you installed emet-molmospaces (e.g. `.venv-molmospaces`) or set `MOLMOSPACES_PYTHON` so the core invokes the wrapper from that env.
