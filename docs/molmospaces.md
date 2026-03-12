# MolmoSpaces

[MolmoSpaces](https://github.com/allenai/molmospaces) is an open ecosystem for robot manipulation and navigation: indoor scenes (iTHOR, ProcTHOR, Holodeck), object models, and robot assets (e.g. **rby1** / Galaxea R1 family, Franka). Emet integrates MolmoSpaces so you can set up scenes, use their robots, and run or visualize results in simulation.

MolmoSpaces requires **mujoco 3.4** and **numpy>=2.2**, which conflict with the main emet environment (numpy<2, mujoco>=3.3). So we use a **separate venv** (`.venv-molmospaces`) for the MolmoSpaces runner. The emet CLI invokes that runner via subprocess.

## Install the MolmoSpaces runner env

From the project root:

```bash
./install.sh --molmospaces -y
```

This creates `.venv-molmospaces`, installs emet (no-deps) plus `molmo-spaces` and `mujoco>=3.4`, and sets a default asset directory. You can also set up the venv manually:

```bash
uv venv .venv-molmospaces
.venv-molmospaces/bin/pip install --no-deps -e .
.venv-molmospaces/bin/pip install molmo-spaces "mujoco>=3.4" "numpy>=2.2"
```

Set the asset directory (required for scene install and serve):

```bash
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets
# or a path under the project
mkdir -p "$HOME/.cache/molmospaces/assets"
export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets"
```

Emet looks for the runner Python at `MOLMOSPACES_PYTHON` or `.venv-molmospaces/bin/python`. To use a different venv:

```bash
export MOLMOSPACES_PYTHON=/path/to/your/molmospaces/venv/bin/python
```

## Commands

- **List robots** (no runner needed; uses static list):

  ```bash
  emet molmospaces list-robots
  ```

  Prints supported robot IDs (rby1, rby1m, franka_droid, franka_cap, etc.). Default is **rby1** (Galaxea R1 family).

- **List scenes** (uses runner venv):

  ```bash
  emet molmospaces list-scenes
  ```

  Prints scene names (ithor, procthor-10k, procthor-objaverse, holodeck-objaverse) and split sizes (train/val/test).

- **Install a scene**:

  ```bash
  emet molmospaces install-scene --scene ithor --split train --index 0 --scene-path /tmp/ithor_scene.xml
  ```

  Downloads and installs the scene; optionally copies the scene XML to `--scene-path`.

- **Run simulation (serve)**:

  ```bash
  emet molmospaces serve --scene ithor --robot rby1 --viewer
  ```

  Installs the scene (if needed), loads the MJCF, and runs MuJoCo. Use `--viewer` to open the native MuJoCo viewer. Use `--headless` for no GUI. Optional `--rerun PORT` or `--rerun path.rrd` logs step data to Rerun.

  Examples:

  ```bash
  emet molmospaces serve --scene ithor --split train --index 1 --robot rby1 --viewer
  emet molmospaces serve --scene procthor-10k --headless --rerun 9876
  ```

## Showing results

- **Viewer:** `emet molmospaces serve --viewer` opens the MuJoCo passive viewer so you can see the scene and robot.
- **Rerun:** Use `--rerun <port>` to connect to a Rerun server, or `--rerun path.rrd` to save an RRD file. Then open the Rerun viewer (e.g. `http://localhost:9090?url=ws://localhost:9876` or load the RRD) to replay.

For a step-by-step **testing plan** (automated tests, manual checks, and regression), see **[Testing MolmoSpaces](testing_molmospaces.md)**.

## Troubleshooting

- **"MolmoSpaces runner venv not found"**
  Run `./install.sh --molmospaces -y` or set `MOLMOSPACES_PYTHON` to the Python of a venv that has `molmo-spaces` and `emet` (editable, no-deps) installed.

- **"MLSPACES_ASSETS_DIR not set"**
  Export `MLSPACES_ASSETS_DIR` to a directory where scene assets will be downloaded (e.g. `~/.cache/molmospaces/assets`).

- **"molmo_spaces not found"**
  The runner is using the main emet env instead of the MolmoSpaces venv. Ensure `.venv-molmospaces` exists and emet is finding it (or set `MOLMOSPACES_PYTHON`).
