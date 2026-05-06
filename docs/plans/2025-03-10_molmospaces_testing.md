# MolmoSpaces integration – testing plan

**Date:** 2025-03-10

This document describes how to test the MolmoSpaces integration. The integration is split into **core emet** (CLI + config + wrapper discovery) and the **emet-molmospaces** wrapper package (list-scenes, install-scene, serve). The core never imports molmo-spaces; it invokes the wrapper via subprocess when the user runs `emet molmospaces list-scenes` etc.

## Scope

- **Core**: `emet molmospaces` Click group; `list-robots` (static config); discovery of `emet-molmospaces` executable; delegation of list-scenes, install-scene, serve to the wrapper subprocess. If the wrapper is not found, the CLI exits with a clear “install wrapper” message.
- **Wrapper (emet-molmospaces)**: Separate package; provides the `emet-molmospaces` console script; implements list-scenes, install-scene, serve using molmo_spaces and mujoco. Depends on emet (for config constants), molmo-spaces, mujoco>=3.4, numpy>=2.2.
- **Install**: `install.sh --molmospaces` (or `emet install` → MolmoSpaces) creates `.venv-molmospaces`, installs emet (no-deps) and then the wrapper (`pip install -e packages/emet_molmospaces`). The venv’s `bin/emet-molmospaces` is what the core discovers.

## 1. Core tests (no wrapper required)

From project root:

```bash
uv sync
uv run python -m pytest src/test/cli/test_molmospaces_cli.py -v
```

**Expected:** All tests run; none skipped. Without the wrapper installed, `test_molmospaces_list_scenes_without_wrapper` asserts that `emet molmospaces list-scenes` exits non-zero and stderr/stdout contains “Install” or “wrapper” or “emet-molmospaces”.

**What is tested:**

- `emet molmospaces --help` shows list-robots, list-scenes, install-scene, serve.
- `emet molmospaces list-robots` prints rby1, franka_*, and “Default: rby1”.
- `emet molmospaces install-scene --help` and `emet molmospaces serve --help` run (they invoke the wrapper; if wrapper missing, same “install wrapper” behavior).
- Config constants: `MOLMOSPACES_ROBOT_IDS`, `DEFAULT_MOLMOSPACES_ROBOT`, `MOLMOSPACES_SCENE_NAMES`.
- Without wrapper: `emet molmospaces list-scenes` exits non-zero with message to install the wrapper.

## 2. Wrapper package tests (mocked molmo_spaces)

From repo root, run the wrapper’s tests. The wrapper must be importable (e.g. install editable from repo or set `PYTHONPATH`):

```bash
# From repo root (with wrapper on path or installed editable):
uv run python -m pytest packages/emet_molmospaces/tests -v

# Or from wrapper package dir (dev deps include pytest):
cd packages/emet_molmospaces && uv sync && uv run pytest tests -v
```

**Expected:** Tests pass. They mock `_get_molmo_api` so no real molmo-spaces or mujoco is required. Tests cover: list-scenes --help (SystemExit 0), list-scenes with mocked API returns 0, install-scene --help, serve --help. Optional test for console script `emet-molmospaces list-scenes --help` is skipped if the script is not in the current env.

## 3. Wrapper not present (graceful failure)

Without installing the wrapper (no `.venv-molmospaces/bin/emet-molmospaces` and no `MOLMOSPACES_PYTHON` pointing to a env that has it):

```bash
emet molmospaces list-scenes
```

**Expected:** Exit code 1, message that the MolmoSpaces wrapper was not found and to run `pip install emet-molmospaces` (in a venv with molmo-spaces) or `install.sh --molmospaces`.

## 4. Install the wrapper

```bash
./install.sh --molmospaces -y
```

**Expected:** Creates `.venv-molmospaces`, installs emet (no-deps) and then `pip install -e packages/emet_molmospaces` when `packages/emet_molmospaces` exists. The venv will have the `emet-molmospaces` script in `bin/`. Set `MLSPACES_ASSETS_DIR` for scene data (e.g. `export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets`).

## 5. CLI with wrapper installed (manual)

- **List robots** (no network, no wrapper process):

  ```bash
  emet molmospaces list-robots
  ```

  **Expected:** Robots: rby1, rby1m, franka_droid, … Default: rby1.

- **List scenes** (core invokes wrapper; wrapper may hit MolmoSpaces/HuggingFace API):

  ```bash
  emet molmospaces list-scenes
  ```

  **Expected:** Table of scene names (ithor, procthor-10k, …) and split counts. If the API or network fails, the wrapper may exit non-zero; that is an environment/upstream issue.

- **Install scene** (downloads assets; can be slow):

  ```bash
  emet molmospaces install-scene --scene ithor --split train --index 0 --scene-path /tmp/ithor_scene.xml
  ```

  **Expected:** Scene installs; if a scene XML is found under `MLSPACES_ASSETS_DIR`, it is copied to the given path.

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

  **Expected:** Sim runs and logs to Rerun on port 9876.

## 6. Integration test with wrapper (optional)

If `.venv-molmospaces` exists and the wrapper is installed there, set `MOLMOSPACES_PYTHON` to that venv’s Python (or rely on discovery of `.venv-molmospaces/bin/emet-molmospaces` from project root). Then:

```bash
RUN_MOLMOSPACES_TESTS=1 uv run python -m pytest src/test/cli/test_molmospaces_cli.py -v
```

**Expected:** `test_molmospaces_list_scenes_with_wrapper` is run (no longer skipped) and should pass when the wrapper and network/API are available, or fail with a clear network/API error.

## 7. Regression: rest of CLI and sim

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
| Core (no wrapper) | `pytest src/test/cli/test_molmospaces_cli.py` | All pass; list-scenes without wrapper exits 1 with “install wrapper” message |
| Wrapper (mocked) | `pytest packages/emet_molmospaces/tests` | All pass with mocked molmo_spaces |
| No wrapper | `emet molmospaces list-scenes` | Exit 1, helpful “install wrapper” message |
| Install | `./install.sh --molmospaces -y` | .venv-molmospaces created, emet-molmospaces script in bin |
| list-robots | `emet molmospaces list-robots` | Prints robot IDs |
| list-scenes | `emet molmospaces list-scenes` (with wrapper) | Prints scene table |
| install-scene | `emet molmospaces install-scene ... --scene-path /tmp/out.xml` | Scene installed, optional file written |
| serve viewer | `emet molmospaces serve --viewer` | Viewer opens, sim steps |
| serve headless | `emet molmospaces serve --headless` | No window, Ctrl+C stops |
| With wrapper | `RUN_MOLMOSPACES_TESTS=1 pytest ... test_molmospaces_cli.py` | list-scenes-with-wrapper test runs |
| Regression | `pytest src/test/cli/` | All CLI tests pass |
