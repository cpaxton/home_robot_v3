# Simulation

Stretch AI includes a MuJoCo-based simulation that lets you run AI apps without a physical robot. The simulation uses the same ZMQ interface as the real robot, so apps like DynaMem, grasp_object, and mapping work with `--robot_ip 127.0.0.1`.

**Tip:** Use the [emet CLI](cli.md) for simpler commands: `emet serve mujoco`, `emet run dynamem`, etc.

## Quick Start

### 1. Install simulation support

```bash
# From project root
uv sync --extra sim
# or: pip install -e ".[sim]"
```

### 2. Test the setup

```bash
# Verify assets and model load
python examples/load_stretch_assets.py
python examples/load_stretch_mujoco.py
```

### 3. Run simulation

**Terminal 1** – Start the MuJoCo server:

```bash
emet serve mujoco
# or: python -m emet.simulation.mujoco_server
```

**Terminal 2** – Run an app (use `127.0.0.1` for local simulation):

```bash
emet run grasp --robot-ip 127.0.0.1 --target-object "red cylinder" --parameter-file sim_planner.yaml
# or: python -m emet.app.grasp_object --robot_ip 127.0.0.1 --target_object "red cylinder" --parameter_file sim_planner.yaml --show_gui
```

**No display (SSH, headless server)?** Use `--headless`:

```bash
emet serve mujoco --headless
# or: python -m emet.simulation.mujoco_server --headless
```

On **Linux**, headless mode automatically uses EGL for GPU-accelerated camera rendering (no display needed). AI apps like grasp and DynaMem work with cameras.

On **Mac/Windows** without a display, cameras are disabled. Use **Xvfb** for a virtual display:

```bash
# Terminal 1: start virtual display
Xvfb :99 -screen 0 1024x768x24 &

# Terminal 2: run with that display
DISPLAY=:99 emet serve mujoco --headless
```

---

## Running AI Apps in Simulation

All Stretch AI apps that use the ZMQ robot interface work in simulation by setting `--robot_ip 127.0.0.1`.

### Visual servoing grasp

```bash
# Terminal 1
emet serve mujoco

# Terminal 2
emet run grasp --robot-ip 127.0.0.1 --target-object "red cylinder" --parameter-file sim_planner.yaml
```

The default scene has a red and blue cylinder. Use `sim_planner.yaml` for simulation (lower thresholds, tuned detection).

### DynaMem (open-vocabulary mobile manipulation)

DynaMem supports exploration, pick-and-place, and semantic memory. Use visual servoing in simulation (AnyGrasp needs GPU + real robot calibration):

```bash
# Terminal 1 – MuJoCo server (Robocasa recommended for richer scenes)
emet serve mujoco --use-robocasa

# Terminal 2 – DynaMem with visual servoing
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class
```

- `-S` / `--skip`: skip confirmations for autonomous runs
- `--visual-servo` / `-V`: use visual servoing (required in sim; AnyGrasp needs real robot)
- `--match-method class`: class-based matching (works well in sim)

For CPU-only:

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --cpu --match-method class --visual-servo
```

See [DynaMem docs](dynamem.md) for full options.

### Mapping

```bash
# Terminal 1
emet serve mujoco

# Terminal 2
emet run mapping --robot-ip 127.0.0.1
```

### Other apps

Any app that takes `--robot_ip` can run in simulation. Use `emet run <app>` or `python -m emet.app.<app>`:

```bash
emet run timing --robot-ip 127.0.0.1
python -m emet.app.view_images --robot_ip 127.0.0.1
python -m emet.app.show_point_cloud --robot_ip 127.0.0.1
python -m emet.app.keyboard_teleop --robot_ip 127.0.0.1
```

---

## Robocasa (rich kitchen scenes)

The default scene has a robot and docking station. [Robocasa](https://robocasa.ai/) adds kitchen scenes with objects for pick-and-place.

### Install Robocasa

```bash
./scripts/install_simulation.sh
```

Or manually:

```bash
cd third_party
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
cd robosuite && pip install -e . && cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa && pip install -e .
python robocasa/scripts/setup_macros.py
python robocasa/scripts/download_kitchen_assets.py  # optional, for assets
```

### Run with Robocasa

```bash
emet serve mujoco --use-robocasa
```

Options:

- `--robocasa-task`: task name (default: PnPCounterToCab)
- `--robocasa-style`: style index
- `--robocasa-layout`: layout index

Then run DynaMem or other apps as above.

---

## Demo scripts

Convenience scripts:

| Script | Description |
|--------|-------------|
| `scripts/demo_simulation.sh` | Run grasp, DynaMem, or mapping demo |
| `emet serve`, `emet run` | [CLI](cli.md) for simpler commands |
| `examples/load_stretch_assets.py` | Check asset paths |
| `examples/load_stretch_mujoco.py` | Load MuJoCo model; `--run` to start sim |
| `examples/load_stretch_urdf.py` | Load URDF model |
| `examples/run_stretch_simulation.py` | Start simulation |

```bash
# AI demos (start MuJoCo server in another terminal first)
./scripts/demo_simulation.sh grasp
./scripts/demo_simulation.sh dynamem
./scripts/demo_simulation.sh mapping

# Basic sim
python examples/load_stretch_mujoco.py --run          # With viewer
python examples/load_stretch_mujoco.py --run --headless
python examples/run_stretch_simulation.py --headless
```

---

## Scene files

Scenes are MuJoCo XML files under `src/emet/assets/robot/`:

- `scene.xml` – default (robot + docking station)
- `stretch.xml` – robot only
- `docking_station.xml` – docking station

Custom scene:

```bash
emet serve mujoco --scene-path /path/to/your/scene.xml
```

---

## Troubleshooting

**"DISPLAY environment variable is missing"**
On Linux, `--headless` uses EGL automatically (no display needed). On Mac/Windows, use Xvfb: `Xvfb :99 &` then `DISPLAY=:99 emet serve mujoco --headless`.

**"gladLoadGL error" in headless**
Ensure EGL libraries are installed (Linux): `sudo apt install libegl1-mesa libgles2-mesa`. Or use Xvfb: `Xvfb :99 &` then `DISPLAY=:99`.

**"mesh volume is too small"**
MuJoCo is pinned to 3.2.6 for compatibility. Ensure `uv sync --extra sim` or `pip install -e ".[sim]"` is used.

**Grasp/detection fails in sim**
Use `sim_planner.yaml` and `--parameter_file sim_planner.yaml` for grasp_object.

**Robocasa import errors**
Install Robocasa and dependencies with `./scripts/install_simulation.sh`.

**DynaMem / Rerun headless**
When running DynaMem without a display, the Rerun web server starts automatically. Connect from a laptop at `http://<server-ip>:9090?url=ws://<server-ip>:9877`. See [Debug: Headless and Rerun](debug.md#headless-and-rerun).
