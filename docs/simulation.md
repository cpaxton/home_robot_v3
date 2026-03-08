# Robocasa Installation

You can install Robocasa by following the instructions below. The MuJoCo simulation stack (stretch_mujoco) is now merged into this repository under `src/stretch/simulation/`. For Robocasa support, you also need:
  - [Robosuite](https://github.com/ARISE-Initiative/robosuite)
  - [Robocasa](https://github.com/robocasa/robocasa) - [project page](https://robocasa.ai/)

## Installation

First, you need to make sure you have already installed [Stretch AI](./install_details.md) and activated the virtual environment (`source .venv/bin/activate`).

Install the stretch package with simulation extras:

```bash
pip install -e ".[sim]"
```

For Robocasa support (scene generation, etc.), use the install script from the repo root:

```bash
./scripts/install_simulation.sh
```

Or manually install the remaining dependencies:

```bash
cd third_party
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
cd robosuite && pip install -e . && cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa && pip install -e .
python robocasa/scripts/setup_macros.py
python robocasa/scripts/download_kitchen_assets.py  # if needed
```

As of 2024-12-04, you may need to update Google protobuf because of an issue with Google text-to-speech:
```bash
pip install --upgrade protobuf
```

You may see a compatibility error in pip, but it should not make a difference.

## Test Grasping in Simulation

![Grasping in simulation](images/rerun_mujoco.png)

In one terminal start the server:

```bash
python -m stretch.simulation.mujoco_server
```

Then run the grasping app:

```bash
python -m stretch.app.grasp_object  --robot_ip 127.0.0.1 --target_object "red cylinder" --parameter_file=sim_planner.yaml --show_gui
```

A few notes:
  - `--robot_ip` is the IP address of the machine hosting the simulator (does not need to be the same as running the app)
  - `--target_object` is the object to grasp; the default environment has a red and a blue object.
  - `--parameter_file` is the file that contains the parameters for the planner. For the simulator, it's best to use the `sim_planner.yaml` file.

The simulation planner config file is mostly the same, but decreases some thresholds and tweaks the object detection model, as the default real-world parameters don't work so well in simulation.

![Visual Servoing in Simulation](images/visual_servo_in_sim.png)

You should be able to see the visual servoing UI in sim, just like you would in real life. The red cylinder will be highlighted.

## Run Robocasa

In one terminal start the server:

```bash
python -m stretch.simulation.mujoco_server --use-robocasa
```

In another run an app, like mapping:

```bash
# Just point the app to the local IP address instead of to your robot.
python -m stretch.app.mapping --robot_ip 127.0.0.1
```

Using the `--robot_ip` option will update your default IP address; you will need to reset it or provide it again to connect to your physical robot from the same machine.

## Creating your own scenes

Mujoco scenes are stored as XML files. You can see an example at `src/stretch/assets/robot/scene.xml`. You can create your own scenes by modifying this file or creating a new one.
