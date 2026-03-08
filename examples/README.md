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

---

## AI Demos in Simulation

Run AI apps (DynaMem, grasp, mapping) in simulation. See [Simulation docs](../docs/simulation.md) for full details.

**Terminal 1** – Start MuJoCo server:

```bash
python -m stretch.simulation.mujoco_server
# Or with Robocasa (richer scenes): python -m stretch.simulation.mujoco_server --use-robocasa
```

**Terminal 2** – Run an app:

```bash
# Visual servoing grasp (default scene has red/blue cylinders)
python -m stretch.app.grasp_object --robot_ip 127.0.0.1 --target_object "red cylinder" --parameter_file sim_planner.yaml --show_gui

# DynaMem (use --use-robocasa for server for kitchen scenes)
python -m stretch.app.run_dynamem --robot_ip 127.0.0.1 --server_ip 127.0.0.1 -S --visual-servo --match-method class

# Mapping
python -m stretch.app.mapping --robot_ip 127.0.0.1
```

Or use the demo script:

```bash
./scripts/demo_simulation.sh grasp
./scripts/demo_simulation.sh dynamem
./scripts/demo_simulation.sh mapping
```

---

## Running with uv

```bash
uv run python examples/load_stretch_assets.py
uv run python examples/run_stretch_simulation.py
```
