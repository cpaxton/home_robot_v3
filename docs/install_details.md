# Stretch AI Installation

Stretch AI supports Python 3.10–3.12. We recommend using [uv](https://docs.astral.sh/uv/) for fast, reproducible installs (no conda required), or [starting with Docker](./start_with_docker.md).

**Try simulation first?** You can run AI apps without a robot using [Simulation](simulation.md). Install with `emet sync` or `uv sync` (default groups include sim), then run `emet serve mujoco` and `emet run dynamem` (or other apps). See [CLI](cli.md).

### System Dependencies

You need git-lfs:

```bash
sudo apt-get install git-lfs
git lfs install
```

You also need some system audio dependencies. These are necessary for [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/), which is used for audio recording and playback. On Ubuntu, you can install them with:

```bash
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg
```

### Install Stretch AI

On both your PC and your robot, clone and install the package:

```bash
# Set up SSH keys if you want to develop and push
git clone git@github.com:hello-robot/stretch_ai.git --recursive

# Use HTTPS if you do not have SSH keys set up
git clone https://github.com/hello-robot/stretch_ai.git
```

#### Install On PC

The installation script uses [uv](https://docs.astral.sh/uv/) to create a virtual environment and install dependencies (no conda required):

```bash
cd stretch_ai
./install.sh
```

Options:
- `-y` / `--yes`: Non-interactive (skip confirmation prompts; link emet to ~/.local/bin when offered).
- `--all`: Install everything: sim (Robocasa + robosuite), MolmoSpaces runner venv, dynamem (SAM-2), and the paper toolchain. Overridable by `--no-sim`, `--no-sam2`, `--no-molmospaces`, or `--no-paper`.
- `--sim` / `--no-sim`: Include or skip simulation (third_party/robocasa, robosuite). Sim is on by default.
- `--molmospaces`: Create `.venv-molmospaces` for [MolmoSpaces](molmospaces.md) (scenes + rby1 robot). Requires separate venv due to numpy/mujoco version mismatch with main env.
- `--paper` / `--no-paper`: Install, or explicitly skip, `latexmk`, `texlive-latex-extra`, and `texlive-bibtex-extra`. Paper tooling is optional in the normal full profile because TeX Live is large.
- `--no-sam2`: Skip Segment Anything 2 (omit the **dynamem** default group: `uv sync --no-group dynamem`). Use `--cpu` for CPU-only (also skips SAM-2).
- `--clean`: Remove and re-clone third_party/robosuite, robosuite_models, robocasa before installing. Only use if repos are in a bad state; **by default we update in place** (git fetch/pull).

When sim is installed, `scripts/install_simulation.sh` updates existing robosuite/robocasa clones (fetch + pull) instead of deleting them. Use `./install.sh --clean` only when you need a fresh clone.

**Dependencies:** Main env uses `numpy<2` and `mujoco>=3.4.0` (pyproject.toml sim extra and override-dependencies). Robocasa/robosuite are installed editable from third_party. MolmoSpaces uses a separate venv (`.venv-molmospaces`) with `molmo-spaces` and `mujoco>=3.4`, `numpy>=2.2`, because it cannot share the main lockfile.

To add SAM2 after install: `emet sync -e dynamem` or ensure **`third_party/segment-anything-2`** is present and run `uv sync` (dynamem is a default group). Or `./install.sh` (includes SAM2 by default; use `--no-sam2` to skip).

**Paper build:** Run `emet install paper -y` for only the local LaTeX
toolchain, then `./paper/build.sh`. If local TeX is absent, the build script can
instead use `texlive/texlive:latest` when Docker is available. See
[`paper/README.md`](../paper/README.md).

**Interactive menu:** Run `emet install menu` for a UI that shows the status
of each sub-asset and the paper toolchain, and lets you install or update them
one by one, or run all with prompts.

#### Install On the Robot

Even if you want to use the source install, we recommend [starting from Docker](docs/start_with_docker.md) on the robot at first.

Robot installation can be tricky, because we use some features from [ROS2](https://docs.ros.org/en/humble/index.html), specifically the [Nav2](https://github.com/ros-navigation/navigation2) package for LIDAR slam.

You will need to link Stretch AI into your ROS workspace. There are two ways to do this; either install stretch AI in your base python environment, or link the conda environment into ROS (advanced). Either way, you will then need to [set up the ROS2 bridge](#set-up-ament-workspace) in your Ament workspace.

*Why all this complexity?* We run a set of ROS2 nodes based on the [HomeRobot](https://github.com/facebookresearch/home-robot) and [OK-Robot](https://ok-robot.github.io/) codebases for mobile manipulation and localization. In particular, this allows us to use [Nav2](https://docs.nav2.org/), a very well-tested ROS2 navigation stack, for localization, which makes it easier to build complex applications. You do not need to understand ROS2 to use this stack.

##### Option 1: Install Stretch AI in Base Python Environment

To install in the base python environment, you need to make sure build tools are up to date:

```bash
deactivate  # only if you are in a virtual environment
pip install --upgrade pip setuptools packaging build meson ninja
```

This is particularly an issue for scikit-fmm, which is used for motion planning. After this is done, you can install the package as normal:

```bash
pip install .
```

Then, [set up the ROS2 bridge](#set-up-ament-workspace-on-the-robot).

##### Option 2: Link Virtual Environment into ROS (Advanced)

If you are using the uv-created virtual environment, you can link it into ROS. This is a bit more advanced, but can be useful if you want to keep your ROS and Python environments separate.

Install using the installation script with the `--cpu` flag for a CPU-only installation:

```bash
./install.sh --cpu
```

Then, activate the virtual environment:

```bash
source .venv/bin/activate
```

Then, [link the package into your ament workspace](#set-up-ament-workspace-on-the-robot) and install the package:

```bash
colcon build --cmake-args -DPYTHON_EXECUTABLE=$(which python)
```

Some ROS python repositories might be missing - specifically `empy` and `catkin_pkg`. You can install these with:

```bash
python -m pip install empy catkin_pkg
```

#### Set Up Ament Workspace on the Robot

**Preferred (from the workstation):** use `emet deploy` to rsync `emet_core` + `stretch_ros2_bridge` and run `colcon build` on the robot — see [deploy.md](deploy.md).

```bash
uv run emet connect save STRETCH_IP --user hello-robot --name stretch \
  --robot stretch --workspace ~/ament_ws
uv run emet deploy --robot stretch --start-bridge
```

**Manual (on the robot):** symlink the bridge into your ament workspace and build:

```bash
cd stretch_ai   # or this repo checkout on the robot
ln -s `pwd`/src/stretch_ros2_bridge $HOME/ament_ws/src/stretch_ros2_bridge
cd ~/ament_ws
colcon build --packages-select stretch_ros2_bridge
```

Rebuild after codebase updates (`emet deploy` does this remotely, or on-robot):

```bash
cd ~/ament_ws
colcon build --packages-select stretch_ros2_bridge
```

### Using LLMs

We use many open-source LLMs from [Huggingface](https://huggingface.co/). TO use them, you will need to make sure `transformers` is installed and up to date. You can install it with:

```bash
pip install transformers --upgrade
```

You will need to go to the associated websites and accept their license agreements.

- [Gemma 2](https://huggingface.co/google/gemma-2b)
- [Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

Then you need to login to the huggingface CLI:

```bash
huggingface-cli login
```

This will require a personal access token created on the Huggingface website. After this, you can test LLM chat APIs via:

```bash
# Start a local chat with Gamma 2-2B -- requires ~5gb GPU memory
python -m emet.llms.gemma_client

# Start a local chat with Llama 3.1 8B -- requires a bigger GPU
python -m emet.llms.llama_client
```

## Installing CUDA 11.8

Make sure you have CUDA installed on your computer, preferably 11.8. It's possible to install multiple versions of CUDA on your computer, so make sure you have the correct version installed. You do not need to and should not install new versions of your NVIDIA drivers, but you may want to [install CUDA 11.8](https://developer.nvidia.com/cuda-11.8-download-archive) if you don't have it already, following the instructions in [Installing CUDA 11.8](#installing-cuda-11.8).

Download the runfile version of CUDA 11.8. E.g. for Ubuntu 22.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

![CUDA Installation](images/cuda_install.png)

Follow the prompts to install CUDA 11.8. When you get to the prompt to install the NVIDIA driver, select "No" to avoid installing a new driver. Also make sure you deselect the prompt for setting the system CUDA version!

Before running the install script, set the `$CUDA_HOME` environment variable to point to the new CUDA installation. For example, on Ubuntu 22.04:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
./install.sh
```

This should help avoid issues.

## Debugging Common Issues (Old)

First, verify that your installation is successful. One of the most common issues with the advanced installation is a CUDA version conflict, which means that Torch cannot run on your GPU.

### Verifying Torch GPU Installation

The most common issue is with `torch_cluster`, or that cuda is set up wrong. Make sure it runs by starting `python` and running:

```python
import torch_cluster
import torch
torch.cuda.is_available()
torch.rand(3, 3).to("cuda")
```

You should see:

- `torch_cluster` imports successfully
- `True` for `torch.cuda.is_available()`
- No errors for `torch.rand(3, 3).to("cuda")`

If instead you get an error, run the following to check your CUDA version:

```bash
nvcc --version
```

Note: if `nvcc --version` fails, try `/usr/local/cuda/bin/nvcc --version` instead.

Make sure you have CUDA installed on your computer, preferably 11.8. It's possible to install multiple versions of CUDA on your computer, so make sure you have the correct version installed. You do not need to and should not install new versions of your NVIDIA drivers, but you may want to [install CUDA 11.8](https://developer.nvidia.com/cuda-11.8-download-archive) if you don't have it already, following the instructions in [Installing CUDA 11.8](#installing-cuda-11.8).
