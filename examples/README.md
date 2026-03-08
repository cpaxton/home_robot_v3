# Stretch Examples

Simple examples for loading and running Stretch.

## Prerequisites

```bash
# From project root - install simulation support (mujoco, etc.)
uv sync --extra sim
# or: pip install -e ".[sim]"
```

## Examples

### 1. Test asset paths (no heavy dependencies)

```bash
python examples/load_stretch_assets.py
```

Verifies that robot model assets (MuJoCo XML, meshes) are found correctly.

### 2. Load MuJoCo model

```bash
python examples/load_stretch_mujoco.py
```

Loads the MuJoCo stretch model without starting the simulation.

### 3. Run MuJoCo simulation

```bash
python examples/load_stretch_mujoco.py --run
python examples/load_stretch_mujoco.py --run --headless  # No display (SSH, servers)
```

Or use the dedicated launcher:

```bash
python examples/run_stretch_simulation.py
python examples/run_stretch_simulation.py --headless
python examples/run_stretch_simulation.py --cameras  # Show camera feeds (requires DISPLAY)
```

**No display (SSH, headless server)?** Use `--headless`. Camera rendering requires OpenGL/DISPLAY, so cameras are disabled in headless mode.

### 4. Load URDF model

```bash
python examples/load_stretch_urdf.py
```

Loads the Stretch URDF from `hello-robot-stretch-urdf` for visualization/kinematics.
Requires: `pip install hello-robot-stretch-urdf`

## Running with uv

```bash
uv run python examples/load_stretch_assets.py
uv run python examples/run_stretch_simulation.py
```
