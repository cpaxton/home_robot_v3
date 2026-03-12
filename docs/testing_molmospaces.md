# MolmoSpaces integration – testing plan

This document describes how to test the MolmoSpaces integration (CLI, runner venv, scene install, and serve with viewer/rerun).

## Scope

- **CLI** (`emet molmospaces`): list-robots, list-scenes, install-scene, serve.
- **Runner**: Runs in `.venv-molmospaces`; installs scenes via MolmoSpaces API, loads MJCF, runs MuJoCo with optional viewer or rerun.
- **Install**: `install.sh --molmospaces` creates the runner venv and sets `MLSPACES_ASSETS_DIR`.

## 1. Automated tests (no MolmoSpaces venv)

From project root:

```bash
uv sync --extra dev
uv run python -m pytest src/test/cli/test_molmospaces_cli.py -v
```

**Expected:** 5 passed, 1 skipped. The skipped test is `test_molmospaces_list_scenes`, which requires the runner venv.

**What is tested:**

- `emet molmospaces --help` shows list-robots, list-scenes, install-scene, serve.
- `emet molmospaces list-robots` prints rby1, franka_*, and "Default: rby1".
- `emet molmospaces install-scene --help` and `emet molmospaces serve --help` run.
- Config constants: `MOLMOSPACES_ROBOT_IDS`, `DEFAULT_MOLMOSPACES_ROBOT`, `MOLMOSPACES_SCENE_NAMES`.

## 2. Runner venv not present (graceful failure)

Without creating `.venv-molmospaces`:

```bash
emet molmospaces list-scenes
```

**Expected:** Exit code 1, message that the MolmoSpaces runner venv was not found and to run `install.sh --molmospaces` or set `MOLMOSPACES_PYTHON`.

```bash
uv run python -m emet.simulation.molmospaces_runner list-scenes
```

**Expected:** Exit code 1, message that `molmo_spaces` was not found (main env does not have molmo-spaces).

## 3. Install the MolmoSpaces runner venv

```bash
./install.sh --molmospaces -y
```

**Expected:** Creates `.venv-molmospaces`, installs emet (no-deps) and molmo-spaces + mujoco>=3.4. Prints `MLSPACES_ASSETS_DIR=...`.

Set the asset dir for the next steps:

```bash
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
mkdir -p "$MLSPACES_ASSETS_DIR"
```

## 4. CLI with runner venv (manual)

- **List robots** (no network, no runner process):

  ```bash
  emet molmospaces list-robots
  ```

  **Expected:** Robots: rby1, rby1m, franka_droid, ... Default: rby1.

- **List scenes** (runner runs; may hit MolmoSpaces/HuggingFace API):

  ```bash
  emet molmospaces list-scenes
  ```

  **Expected:** Table of scene names (ithor, procthor-10k, …) and split counts. If the API or network fails, the runner may exit non-zero; that is an environment/upstream issue, not an emet bug.

- **Install scene** (downloads assets; can be slow):

  ```bash
  emet molmospaces install-scene --scene ithor --split train --index 0 --scene-path /tmp/ithor_scene.xml
  ```

  **Expected:** Scene installs; if a scene XML is found under `MLSPACES_ASSETS_DIR`, it is copied to `/tmp/ithor_scene.xml`.

- **Serve with viewer** (opens MuJoCo window):

  ```bash
  emet molmospaces serve --scene ithor --robot rby1 --viewer
  ```

  **Expected:** Scene loads, MuJoCo viewer opens and steps. Ctrl+C stops.

- **Serve headless** (no GUI):

  ```bash
  emet molmospaces serve --scene ithor --headless
  ```

  **Expected:** Sim runs until Ctrl+C. No window.

- **Serve with rerun** (optional):

  ```bash
  emet molmospaces serve --scene ithor --viewer --rerun 9876
  ```

  **Expected:** Sim runs and logs to Rerun on port 9876. Open Rerun viewer at `http://localhost:9090?url=ws://localhost:9876` to see step data.

## 5. Automated test with runner venv (optional)

If `.venv-molmospaces` exists and MolmoSpaces is working:

```bash
RUN_MOLMOSPACES_TESTS=1 uv run python -m pytest src/test/cli/test_molmospaces_cli.py -v
```

**Expected:** All 6 tests run; `test_molmospaces_list_scenes` is no longer skipped and should pass (or fail with a clear network/API error).

## 6. Regression: rest of CLI and sim

Ensure the new code does not break existing behavior:

```bash
uv run python -m pytest src/test/cli/ -v
emet serve mujoco --help
emet robocasa list
```

**Expected:** All CLI tests pass; serve and robocasa help work as before.

## Summary table

| Test | Command / action | Expectation |
|------|------------------|-------------|
| Unit (no venv) | `pytest src/test/cli/test_molmospaces_cli.py` | 5 pass, 1 skip |
| No venv | `emet molmospaces list-scenes` | Exit 1, helpful message |
| Install | `./install.sh --molmospaces -y` | .venv-molmospaces created |
| list-robots | `emet molmospaces list-robots` | Prints robot IDs |
| list-scenes | `emet molmospaces list-scenes` | Prints scene table (with venv) |
| install-scene | `emet molmospaces install-scene ... --scene-path /tmp/out.xml` | Scene installed, optional file written |
| serve viewer | `emet molmospaces serve --viewer` | Viewer opens, sim steps |
| serve headless | `emet molmospaces serve --headless` | No window, Ctrl+C stops |
| With venv | `RUN_MOLMOSPACES_TESTS=1 pytest ... test_molmospaces_cli.py` | 6 tests run |
| Regression | `pytest src/test/cli/` | All CLI tests pass |
