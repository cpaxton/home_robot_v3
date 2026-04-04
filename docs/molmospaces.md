# MolmoSpaces

[MolmoSpaces](https://github.com/allenai/molmospaces) is an open ecosystem for robot manipulation and navigation: indoor scenes (iTHOR, ProcTHOR, Holodeck), object models, and robot assets (e.g. **rby1** / Galaxea R1 family, Franka). Emet integrates MolmoSpaces so you can set up scenes, use their robots, and run or visualize results in simulation.

MolmoSpaces requires **mujoco 3.4** and **numpy>=2.2**, which conflict with the main emet environment (numpy<2, mujoco>=3.3). So the integration is split:

- **Core emet**: No dependency on molmo-spaces or mujoco 3.4. The `emet molmospaces` CLI lives in core; `list-robots` uses a static config. For `list-scenes`, `install-scene`, `merge-scene`, and `serve`, the CLI **delegates to a thin wrapper** via subprocess.
- **Wrapper (emet-molmospaces)**: A separate package that depends on emet, molmo-spaces, mujoco>=3.4, and numpy>=2.2. It provides the `emet-molmospaces` console script and implements list-scenes, install-scene, merge-scene, and serve. Install it in a dedicated venv (e.g. `.venv-molmospaces`) or any env where you accept those deps.

## Install the MolmoSpaces wrapper

From the project root (recommended):

```bash
./install.sh --molmospaces -y
```

`-y` only makes the script non-interactive (apt, link `emet`); it does **not** turn on MolmoSpaces unless you also pass **`--molmospaces`** (or **`--all`**). Same for `emet install full -y`: add **`--molmospaces`** or **`--all`**.

```bash
emet install full -y --molmospaces
# or
./install.sh --molmospaces -y
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
  emet molmospaces merge-scene --scene ithor --split train --index 0 --robot rby1 -o /tmp/molmospaces_rby1.xml
  ```

  Delegates to the wrapper: installs the scene if needed, merges the **rby1** (Galaxea R1) MJCF from core emet assets into the scene, and writes a **persistent** merged MJCF. Use that path with **`emet serve mujoco`** in your **main** project environment (see below), not only with passive `emet molmospaces serve`.

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

Passive `emet molmospaces serve` only steps physics in the wrapper’s MuJoCo. To drive the robot and use **`emet run agent`** (LLM + tools), run the **ZMQ MuJoCo server** from the same environment you use for normal simulation (`uv sync --extra sim`).

**Recommended (one command):** `emet serve mujoco` can merge a MolmoSpaces scene with **rby1** (Galaxea R1) via the wrapper, then start the ZMQ server. You do **not** need to run `merge-scene` first.

1. Install assets and (if needed) the wrapper venv: `./install.sh --molmospaces -y`, and set `MLSPACES_ASSETS_DIR`.
2. From the project root with the **main** `.venv` (where `emet` and sim extras live):

   ```bash
   emet serve mujoco --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 --headless
   ```

   This calls the wrapper’s `merge-scene` into a temporary MJCF, then starts `emet.simulation.mujoco_server` with `--robot rby1` (unless you pass `--robot galaxea_r1` / `rby1`). Default `--robot stretch` is upgraded to **rby1** when using `--molmospaces-scene`.

3. Run the agent:

   ```bash
   emet run agent --robot-ip 127.0.0.1 --robot rby1
   ```

**Optional: fixed path** — use `emet molmospaces merge-scene ... -o /path/to/merged.xml` if you want a stable file, then `emet serve mujoco --robot rby1 --scene-path /path/to/merged.xml`.

Use `--port-offset` on both server and agent if default ZMQ ports are busy. The agent uses **`GenericZmqClient`** for `rby1`, matching `emet run dynamem --robot rby1`.

### MuJoCo version note

MolmoSpaces assets are built with **MuJoCo 3.4**; core emet typically uses **mujoco>=3.3**. If `emet serve mujoco` fails to load a merged MJCF from the wrapper, try upgrading MuJoCo in the project env or report an asset compatibility issue. The wrapper venv is still required for **download/install/merge**; the server should run where **`emet` and sim extras** are installed.

## Showing results

- **Viewer:** `emet molmospaces serve --viewer` opens the MuJoCo passive viewer.
- **Rerun:** Use `--rerun <port>` or `--rerun path.rrd`; then open the Rerun viewer (e.g. `http://localhost:9090?url=ws://localhost:9876` or load the RRD).

For a step-by-step **testing plan** (core tests, wrapper tests with mocks, optional integration), see **[docs/plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md)**.

## Troubleshooting

- **"MolmoSpaces wrapper not found" / `pip install emet-molmospaces` fails**
  The wrapper is **not on PyPI**. From the repo root run `./install.sh --molmospaces -y`, or install editable: `uv pip install --no-deps -e .` and `uv pip install -e packages/emet_molmospaces` into `.venv-molmospaces` (see **Install the MolmoSpaces wrapper** above). Core emet discovers `.venv-molmospaces/bin/emet-molmospaces` or runs `python -m emet_molmospaces` from that venv. You can also set `MOLMOSPACES_PYTHON` to that venv’s `python` binary.

- **"MLSPACES_ASSETS_DIR not set"**
  Export `MLSPACES_ASSETS_DIR` to a directory where scene assets will be downloaded (e.g. `~/.cache/molmospaces/assets`).

- **"molmo_spaces not found"** (from the wrapper)
  The wrapper is running in an env that doesn’t have molmo-spaces. Use the venv where you installed emet-molmospaces (e.g. `.venv-molmospaces`) or set `MOLMOSPACES_PYTHON` so the core invokes the wrapper from that env.

- **`emet serve mujoco` fails to parse merged MJCF**
  See **MuJoCo version note** above (3.3 vs 3.4). Confirm `--scene_path` points to the file written by `merge-scene` and that `--robot rby1` matches the merged robot.

- **`uv pip install -e packages/emet_molmospaces` fails (Python 3.10 / emet not found)**
  Install the wrapper **only** into **`.venv-molmospaces`** (Python **≥3.11**), not the main `.venv` / conda env. Use `./install.sh --molmospaces -y` or the commands in **Install the MolmoSpaces wrapper** — do not run `uv pip install -e packages/emet_molmospaces` in the 3.10 environment.
