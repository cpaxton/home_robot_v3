# Simulation

Stretch AI includes a MuJoCo-based simulation that lets you run AI apps without a physical robot. The simulation uses the same ZMQ interface as the real robot, so apps like DynaMem, grasp_object, and mapping work with `--robot_ip 127.0.0.1`.

**Tip:** Use the [emet CLI](cli.md) for simpler commands: `uv run emet serve mujoco`, `uv run emet run dynamem`, etc. From the project root, **`uv run emet`** ensures you use this checkout’s code (see [TESTING.md](TESTING.md#run-from-this-repo)).

## Quick Start

### 1. Install simulation support

```bash
# From project root (base sim: MuJoCo, no Robocasa)
emet sync -e sim
# or: uv sync
```

For **Robocasa** kitchen scenes, install it first then sync: `emet install robocasa` then `emet sync -e sim`. See [Robocasa](#robocasa-rich-kitchen-scenes) below.

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

If this fails on scipy/numpy ABI (`undefined symbol`) while `uv run python -c "import scipy"` works, see [pythonpath.md](pythonpath.md).

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

**"Still waiting to connect to the MuJoCo Simulator" for a long time?** Without `--headless`, the server opens a viewer window. If there is no display (SSH, WSL, or headless machine), the child process can block there and never signal ready. **Use `emet serve mujoco --headless`** so it skips the viewer and uses EGL for cameras only; startup should then finish within a few seconds.

**WSL: still hanging with `--headless`?** On WSL, EGL camera rendering can hang when creating the first offscreen renderer. You have two options:

1. **Cameras needed (Graph EQA, DynaMem, etc.):** Use a virtual display and GLX so the renderer uses the display instead of EGL. In one terminal start Xvfb, then in another run the server with `DISPLAY` set and **`--use-glx`**:
   ```bash
   # Terminal 1
   Xvfb :99 -screen 0 1024x768x24 &
   # Terminal 2
   DISPLAY=:99 emet serve mujoco --headless --use-glx
   ```
   Install Xvfb if needed: `sudo apt-get install xvfb`. The server will use GLX against the virtual display and should produce camera images.

2. **No cameras needed:** Use **`--no-cameras`** so the server runs without camera rendering (physics only). Example: `emet serve mujoco --headless --no-cameras`. Vision-based apps will get no camera images.

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

The **default MuJoCo scene** has a **red cylinder** and **blue cube** on the table (see `src/emet/assets/robot/scene.xml`). Use `sim_planner.yaml` for simulation (lower thresholds, tuned detection).

### DynaMem (open-vocabulary mobile manipulation)

DynaMem supports exploration, pick-and-place, and semantic memory. Use visual servoing in simulation (AnyGrasp needs GPU + real robot calibration):

```bash
# Terminal 1 – MuJoCo server (Robocasa recommended for richer scenes)
emet serve mujoco --scene robocasa

# Terminal 2 – DynaMem with visual servoing
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class
```

Graph-based EQA on the same sim (no pick-and-place): `uv run emet run graph-eqa --robot-ip 127.0.0.1` or `uv run emet run dynagraph --robot-ip 127.0.0.1` ([graph_eqa.md](graph_eqa.md), [dynagraph.md](dynagraph.md)).

**Dynagraph testing:** unit tests (`test_dynagraph_explore.py`, `test_graph_eqa_memory.py`), multi-robot Robocasa floor E2E (`run_dynagraph_multi_robot_e2e.py`), and manual `--explore-loop` / `--export` runs — see [TESTING.md](TESTING.md) and [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md).

- `-S` / `--skip`: skip confirmations for autonomous runs
- `--visual-servo` / `-V`: use visual servoing (required in sim; AnyGrasp needs real robot)
- `--match-method class`: class-based matching (works well in sim)

**Instance display in Rerun:** With default config (`use_instance_memory: true`, `use_scene_graph: true` in `dynav_config.yaml`), DynaMem runs YoloE to segment objects and shows 3D instance boxes/icons in the Rerun UI. The default scene’s red cylinder and blue cube should appear as detected objects once the robot has looked at the table.

For CPU-only:

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --cpu --match-method class --visual-servo
```

See [DynaMem docs](dynamem.md) for full options.

**Verifying red cylinder detection in sim:** An integration test starts the MuJoCo server (headless), runs Dynamem’s rotate-in-place to build the map, then asserts that `localize_text("red cylinder")` returns a point near the default scene’s red cylinder. Run it with full env (e.g. after `emet sync -e sim`):

```bash
uv run emet test -v src/test/mapping/test_red_cylinder_in_sim.py
```

(Sim tests run by default; use `emet test --no-sim` to skip.)

This confirms the full stack (sim → camera → encoder/detection → semantic memory → localization) works for the default scene.

**First-run model downloads (DynaMem):** The first time you run DynaMem with instance memory (default), it will download:

- **yoloe-v8l-seg.pt** (~102 MB) – Ultralytics YOLOE segmentation model for object/instance detection.
- **mobileclip_blt.ts** (~572 MB) – Text-encoder used by YOLOE to embed the class names (e.g. ScanNet 200). Downloaded by the Ultralytics package to the current working directory (or their cache) on first use.
- **CLIP ViT-B-16** – Dynamem’s encoder (from Hugging Face cache) when using `--cpu` or CPU-only mode.

The message *"Ultralytics requirement ... not found, attempting AutoUpdate"* and *"Restart runtime for updates to take effect"* are from Ultralytics and can be ignored (or restart the process if you want the updated dependency). *"Using default SimpleTokenizer"* is from the text tokenizer used by CLIP/MobileCLIP. After the first run, these assets are cached and startup is faster.

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

## MolmoSpaces (scenes + rby1 / Galaxea R1)

For [MolmoSpaces](https://github.com/allenai/molmospaces) scenes (iTHOR, ProcTHOR, Holodeck) and their robots (e.g. **rby1** / Galaxea R1), use a separate runner venv and the `emet molmospaces` commands. See **[MolmoSpaces](molmospaces.md)** for install, `MLSPACES_ASSETS_DIR`, and usage (install-scene, serve, viewer, rerun). Optional **`EMET_MOLMOSPACES_*`** toggles: [molmospaces_environment_variables.md](molmospaces_environment_variables.md).

---

## Robocasa (rich kitchen scenes)

The default scene has a robot and docking station. [Robocasa](https://github.com/robocasa/robocasa) adds kitchen scenes with objects for pick-and-place. We use a fork ([cpaxton/robocasa](https://github.com/cpaxton/robocasa)) with numpy 1.24+ compatibility so the same env can run robocasa and dynamem.

We **pin Robocasa to v0.2** so its dependencies match our stack (MuJoCo 3.3+, numpy &lt; 2). Newer Robocasa (main / v1.0) may use numpy 2.x and can pull in conflicting or heavy dependencies (e.g. a different torch). Use the install script or the v0.2 clone steps below.

### Install Robocasa

From the project root:

```bash
emet install robocasa
# or: emet install sim
```

This clones robosuite and robocasa, **downloads kitchen assets (~10–15GB)** via `scripts/download_robocasa_assets.py` (textures, base **fixtures**, **fixtures_lw** LightWheel registry required for `emet serve robocasa`, objaverse, generative_textures), and runs macro setup via `python -m robocasa.scripts.setup_macros` when missing. Then sync the sim extra:

```bash
emet sync -e sim
```

Macro setup (robosuite + robocasa `macros_private.py`) runs automatically during install when missing; use `emet install robocasa -a` to force overwrite. To skip the asset download (e.g. CI): `emet install sim --no-download-assets`. If kitchen assets (textures, fixtures, etc.) are already present, the install script prompts **Re-download? (y/N)** (default N); with `./install.sh -y` we skip the prompt and do not re-download when assets exist.

**Version pairing:** Robocasa (and our fork) require **RoboSuite v1.5.0**. The install script clones robosuite v1.5.0, [cpaxton/robosuite_models](https://github.com/cpaxton/robosuite_models) (optional extra robot models), and the [cpaxton/robocasa](https://github.com/cpaxton/robocasa) fork (numpy 1.24+ compat). All live in `third_party/` and are gitignored.

**Manual install** (same as the script, for reference). From project root after cloning and `pip install -e .` for robosuite, robosuite_models, and robocasa:

```bash
cd third_party
git clone https://github.com/ARISE-Initiative/robosuite --branch v1.5.0 --single-branch
cd robosuite && pip install -e . && cd ..
git clone git@github.com:cpaxton/robosuite_models.git
cd robosuite_models && pip install -e . && cd ..
git clone git@github.com:cpaxton/robocasa.git
cd robocasa && pip install -e . && cd ..
cd ../..
# Set up system variables (creates robocasa/macros_private.py)
python -m robocasa.scripts.setup_macros
# Download kitchen assets (~10-15GB; includes fixtures_lw for LightWheel style IDs)
uv run python scripts/download_robocasa_assets.py --yes
```

### Run with Robocasa

```bash
emet serve robocasa
# or: emet serve mujoco --scene robocasa
```

If startup fails with **`Did not find style that matches "Sink025" for fixture type "sink"`**, the **LightWheel fixture pack** (`fixtures_lw`) is missing. The base fixtures zip alone is not enough for generated kitchen styles. From the project root:

```bash
uv run python scripts/download_robocasa_assets.py --yes
# or: ./scripts/install_simulation.sh -y
```

Verify registry (after ``fixtures_lw``, restore vendored YAML then sync — the lw zip adds meshes but often **replaces** ``fixture_registry/`` with a slim subset, deleting per-type files such as ``fridge_bottom_freezer.yaml`` that kitchen layouts require). Sync also registers ``objects/lightwheel/`` accessories (e.g. ``UtensilRack009``) and cabinet door/handle mesh ids:

```bash
git -C third_party/robocasa checkout -- robocasa/models/assets/fixtures/fixture_registry/
uv run python scripts/sync_robocasa_lightwheel_registry.py
grep Sink025 third_party/robocasa/robocasa/models/assets/fixtures/fixture_registry/sink.yaml
grep UtensilRack009 third_party/robocasa/robocasa/models/assets/fixtures/fixture_registry/utensil_rack.yaml
ls third_party/robocasa/robocasa/models/assets/fixtures/fixture_registry/fridge_bottom_freezer.yaml
```

If startup fails with **``fridge_bottom_freezer.yaml``** (or similar registry path), run the restore + sync commands above; do not re-download ``fixtures_lw`` unless meshes are missing.

``emet serve robocasa`` restores registry YAML from git when needed, syncs LightWheel entries, then preflights before building the scene.

If startup fails with **`AttributeError: 'NoneType' object has no attribute 'get'`** on ``reg_bbox``, the **objaverse** zip was extracted but not post-processed (914 raw ``model.xml`` files lack ``reg_bbox`` geoms). One-time fix:

```bash
uv run python scripts/process_robocasa_objaverse_reg_bbox.py
```

This runs Robocasa's ``calc_object_bb_reg`` over ``third_party/robocasa/robocasa/models/assets/objects/objaverse`` (several minutes). ``download_robocasa_assets.py`` runs it automatically after objaverse download.

List all supported env names:

```bash
emet serve mujoco --list-robocasa-tasks
```

Options when serving with `--scene robocasa`:

- `--robocasa-task`: task name (default: PickPlaceCounterToCabinet). Good for “find an object” tests: **PickPlaceCounterToCabinet**, **PickPlaceCabinetToCounter**, **OpenCabinet**, **CloseCabinet**, and other `PickPlace*` envs (e.g. PickPlaceCounterToDrawer, PickPlaceCounterToSink).
- `--robocasa-style`: style index
- `--robocasa-layout`: layout index

Then run DynaMem or other apps as above.

---

## Maintainer modules (sim internals)

Contributor reference for new simulation Python modules (home-pose tuning, stationary `ctrl`, Robocasa asset preflight, spawn metadata loaders): **[simulation_modules.md](simulation_modules.md)**.

Quick examples:

- Tune Galaxea home keyframe: `uv run python -m emet.simulation.mujoco_home_tune src/emet/assets/robot/galaxea_r1/galaxea_r1.xml`
- Molmo spawn JSON: [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md) (`emet molmospaces write-spawn-metadata`)

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
emet serve mujoco --scene /path/to/your/scene.xml
```

---

## Troubleshooting

**"DISPLAY environment variable is missing"**
On Linux, `--headless` uses EGL automatically (no display needed). On Mac/Windows, use Xvfb: `Xvfb :99 &` then `DISPLAY=:99 emet serve mujoco --headless`.

**"GLX: Failed to create context" / "gladLoadGL error"**
On Linux, the sim now sets `MUJOCO_GL=egl` when using cameras so MuJoCo uses EGL instead of GLX for rendering (avoids failures after GPU/driver issues). If you still see this, run headless: `emet serve mujoco --headless`. Ensure EGL is installed: `sudo apt install libegl1-mesa libgles2-mesa`. On Mac/Windows without a display, use Xvfb then `DISPLAY=:99 emet serve mujoco --headless`.

**"mesh volume is too small"**
MuJoCo 3.4+ is required for sim. Ensure `uv sync` (default groups include sim) or `pip install -e ".[sim]"` is used.

**Grasp/detection fails in sim**
Use `sim_planner.yaml` and `--parameter_file sim_planner.yaml` for grasp_object.

**Robocasa import errors or pip conflicts**
Run `emet install robocasa` (or `emet install sim`) to clone the [cpaxton/robocasa](https://github.com/cpaxton/robocasa) fork and robosuite v1.5.0 into `third_party/`. If you see `ImportError: cannot import name 'load_composite_controller_config'` (or `PandaOmron`), you likely have the wrong robosuite version—ensure **robosuite v1.5.0**: `cd third_party/robosuite && git fetch origin --tags && git checkout v1.5.0 && pip install -e .`. If you see numpy version errors, ensure you're using the fork (numpy 1.24+), not upstream robocasa.

**Using sim and dynamem together (uv)**
We use the [cpaxton/robocasa](https://github.com/cpaxton/robocasa) fork, which supports numpy 1.24+. The project's **uv override** (`numpy>=1.24.4,<2`, `numba>=0.58.1`) lets `emet sync -e sim -e dynamem` use one env for both robocasa and dynamem.

**DynaMem / Rerun headless**
When running DynaMem without a display, the Rerun web server starts automatically. Connect from a laptop at `http://<server-ip>:9090?url=ws://<server-ip>:9877`. See [Debug: Headless and Rerun](debug.md#headless-and-rerun).
