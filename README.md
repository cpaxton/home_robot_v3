# Home Robot v3

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

![PickPlaceFullTask](https://github.com/user-attachments/assets/a1db635c-03b5-48e8-9167-45f09bc8a9b2)

**Stretch AI** is designed to help researchers and developers build intelligent behaviors for the [Stretch 3](https://hello-robot.com/stretch-3-product) mobile manipulator from [Hello Robot](https://hello-robot.com/). It contains code for:

- grasping
- manipulation
- mapping
- navigation
- LLM agents
- text to speech and speech to text
- visualization and debugging
- embodied question answering

Much of the code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information. Parts of it are derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project and are licensed under the [MIT license](META_LICENSE).

**Docs:** see the [Documentation map](#documentation-map) below for an outline of `docs/` (what to edit when updating agentic EQA, eval, sim, paper tracks, etc.).

## Installation (Astral uv)

This project installs with [Astral uv](https://docs.astral.sh/uv/) (no conda required).

**1. Install uv** (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure ~/.local/bin is on your PATH
```

**2. Clone the repo and install the project:**

```bash
git clone <this-repo>
cd home_robot_v2   # or your clone directory
```

**Option A — one-shot script (recommended):**

```bash
./install.sh              # dev (+ dynamem if SAM-2 submodule); simulation is opt-in
./install.sh -y           # non-interactive apt + link; does not add sim unless --sim / --all / profile=full
./install.sh --sim        # add MuJoCo sim extra + clone robosuite/robocasa (large)
./install.sh --profile=full   # legacy: enable sim without --sim (same as EMET_INSTALL_PROFILE=full)
./install.sh --no-sim     # skip simulation (no third_party clone)
# With sim on, MolmoSpaces (``.venv-molmospaces``) installs when ``packages/emet_molmospaces`` exists unless you pass:
./install.sh --no-molmospaces   # skip MolmoSpaces venv (lighter / CI)
./install.sh --no-sam2    # skip DynaMem/SAM2
uv run emet install menu  # Rich plan wizard (after: uv sync)
```

**Option B — uv directly:**

```bash
uv sync                                    # default groups: dev, sim, hand_tracker, dynamem, da3
uv sync --no-default-groups                # base package only (no pytest/MuJoCo/SAM-2/…)
uv sync --no-group dynamem                 # skip SAM-2 (e.g. submodule not checked out)
uv sync --group flash-attn                 # add Flash-Attn 2 (compiles CUDA kernels; SDPA otherwise)
```

The script installs system deps (e.g. `portaudio`, `espeak`, `ffmpeg`) and runs `uv sync` for you. For reproducible installs, a `uv.lock` is included; `uv sync` uses it automatically.

**Robot or simulator only (lightweight):** To run only the ZMQ bridge on the robot (or a simulator backend) without torch/LLM/simulation stacks, install **emet-core** and the bridge for your robot: **Stretch** → `stretch_ros2_bridge`, **Innate Mars** → `innate_mars_bridge`. From the repo: `pip install -e src/emet_core` then build/install the bridge with colcon. See `src/emet_core/README.md` and `src/emet_core/BRIDGE_CONTRACT.md`.

**3. Use the environment:**

From the **project root**, prefer **`uv run emet …`** so commands use this checkout’s `.venv` and current code. After `uv sync`, you can instead `source .venv/bin/activate` and run bare `emet …`.

```bash
uv run emet --help
uv run python -m emet.app.ai_pickup
# or: source .venv/bin/activate && emet --help
```

If a command fails with **`No such option: --explore-loop`** (or other flags missing from `--help`), your shell’s `emet` likely points at another install. Run `which emet` and use **`uv run emet`** from this repo (see [Testing](docs/TESTING.md#run-from-this-repo)).

## Your first test (interactive agent)

**Start here** if you want to *talk* to the robot in simulation (describe the scene, `find the Z`, `is an X on a Y?`):

→ **[Your first test](docs/first_test.md)** — default table (red cylinder) or MolmoSpaces iTHOR, with copy-paste prompts.

```bash
# Terminal 1
uv run emet serve mujoco --robot stretch --headless

# Terminal 2
uv run emet run agent --robot stretch --no-discord --rerun
# then try: describe the scene / find the red cylinder / is there a blue cube on the table?
```

MolmoSpaces home scenes (after `./install.sh --molmospaces -y` if needed):

```bash
uv run emet serve mujoco --scene ithor --split train --index 0 --robot stretch --headless --install-scene-if-missing
uv run emet run agent --robot stretch --no-discord --rerun
# then try: explore / find the sofa / is there a remote on the coffee table?
```

## Deploy to a robot (Stretch or Mars)

Same `emet connect` + `emet deploy` flow for both robots — pick `--robot`.
Full recipes: **[docs/deploy.md](docs/deploy.md)**. Replace `STRETCH_IP` / `MARS_IP` / `ORIN_HOST` with your LAN hosts.

```bash
# Stretch (native ament_ws; enable when the robot is on the LAN)
uv run emet connect save STRETCH_IP --user hello-robot --name stretch \
  --robot stretch --workspace ~/ament_ws
uv run emet deploy --robot stretch --start-bridge
uv run emet capture --robot stretch --connection stretch

# Innate Mars (innate-os)
uv run emet connect save MARS_IP --user jetson1 --name mars \
  --robot innate_mars --workspace ~/innate-os/ros2_ws \
  --config configs/agent_innate_mars.yaml
uv run emet mars start --connection mars --deploy
uv run emet mars status --connection mars   # head/wrist camera line
uv run emet capture --connection mars

# Switch active profile: emet connect use stretch|mars
# Wrist black on Mars? Arducam USB — see docs/deploy.md
```

## Remote inference + try an LLM

Run the **text tool-router** and **caption VLM** on a LAN Jetson (e.g. AGX Orin) so the workstation keeps VRAM for voxels / mapping. Full detail: **[docs/llm_serve.md](docs/llm_serve.md)**. Robot bridge deploy is separate: **[docs/deploy.md](docs/deploy.md)**.

```bash
# Deploy / restart unified Qwen2-VL-7B on the Orin (:8000 = text + VL)
uv run emet deploy llm --host ORIN_HOST --profile unified-7b

uv run emet llm health --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST

# One-shot chat (text) and VL caption
uv run emet run chat --host ORIN_HOST --once "Reply with exactly: pong"
uv run emet run chat --host ORIN_HOST --vl --once "Describe briefly"
# Interactive: omit --once
```

Optional env (no hardcoded hostname in the CLI):

```bash
export EMET_LLM_HOST=ORIN_HOST
export EMET_OPENAI_BASE_URL=http://ORIN_HOST:8000/v1
export EMET_VL_ENDPOINT=openai@http://ORIN_HOST:8000/v1
```

Discord / Mars agent: point `agent.llm` and `mapping.eqa.vl_endpoint` at the same URL (see [`configs/agent_innate_mars.yaml`](configs/agent_innate_mars.yaml) and [innate_mars_hardware.md](docs/robots/innate_mars_hardware.md)).

## Quick Start (Simulation)

Run DynaMem in MuJoCo simulation in a few steps:

```bash
# 1. Install core + sim (Robocasa) — add --sim or use: ./install.sh -y --profile full
./install.sh -y --sim

# 2. Terminal 1 — start simulation server
uv run emet serve mujoco --scene robocasa
# or: uv run emet serve mujoco --use-robocasa
# or: uv run python -m emet.simulation.mujoco_server --use-robocasa

# 3. Terminal 2 — run DynaMem
uv run emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --headless
# or: uv run python -m emet.app.run_dynamem ...
```

**4. View in browser:** Open `http://localhost:9090?url=ws://localhost:9877` to see the Rerun visualization (cameras, 3D scene, robot).

**At the prompt:** `E` = explore, `M` = pick and place, `Q` = quit.

For headless/SSH: use `--headless` and open the URL from your laptop (or use SSH port forwarding). See [Debug: Headless and Rerun](docs/debug.md#headless-and-rerun).

## Hardware Requirements

We recommend the following hardware to run Stretch AI. Other GPUs and other versions of Stretch may support some of the capabilities found in this repository, but our development and testing have focused on the following hardware.

- **[Stretch 3](https://hello-robot.com/stretch-3-product) from [Hello Robot](https://hello-robot.com/)**
  - When *Checking Hardware*, `stretch_system_check.py` should report that all hardware passes.
- **Computer with an NVIDIA GPU**
  - The computer should be running Ubuntu 22.04. Later versions might work, but have not been tested.
  - Most of our testing has used a high-end CPU with an NVIDIA GeForce RTX 4090.
- **Dedicated WiFi access point**
  - Performance depends on high-bandwidth, low-latency wireless communication between the robot and the GPU computer.
  - The official [Stretch WiFi Access Point](https://hello-robot.com/stretch-access-point) provides a tested example.
- (Optional) [Stretch Dexterous Teleop Kit](https://hello-robot.com/stretch-dex-teleop-kit).
  - To use the learning-from-demonstration (LfD) code you'll need the Stretch Dexterous Teleop Kit.

## Quick-start Guide

Artificial intelligence (AI) for robots often has complex dependencies, including the need for trained models. Consequently, installing *stretch-ai* from source can be challenging.

First, you will need to install software on your Stretch robot and another computer with a GPU (*GPU computer*). Use the following link to go to the installation instructions: [Instructions for Installing Stretch AI](./docs/start_with_docker_plus_virtenv.md)

Once you've completed this installation, you can start the server on your Stretch robot.  Prior to running the script, you need to have homed your robot with `stretch_robot_home.py`. Then, run the following command:

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

After this, we recommend trying the [Language-Directed Pick and Place](#language-directed-pick-and-place) demo.

### Simulation (try without a robot)

You can run AI apps in MuJoCo simulation before connecting a physical robot:

```bash
# Dependencies (MuJoCo, dev tools, etc.) — plain uv sync uses default groups in pyproject.toml
uv sync

# Terminal 1: start simulation server
python -m emet.simulation.mujoco_server

# Terminal 2: run an app (use 127.0.0.1 for local sim)
python -m emet.app.grasp_object --robot_ip 127.0.0.1 --target_object "red cylinder" --parameter_file sim_planner.yaml --show_gui
```

Use `--headless` when running without a display (e.g. SSH); on Linux, EGL enables cameras. See [Simulation](docs/simulation.md) for DynaMem, mapping, Robocasa, and demo scripts. The [emet CLI](docs/cli.md) (`emet serve`, `emet run`, `emet test`, etc.) provides a unified interface.

#### Experimental support for Older Robots

The older model of Stretch, the Stretch RE2, did not have an camera on the gripper. If you want to use this codebase with an older robot, you can purchase a [Stretch 2 Upgrade Kit](https://hello-robot.com/stretch-2-upgrade) to give your Stretch 2 the capabilities of a Stretch 3. Alternatively, you can run a version of the server with no d405 camera support on your robot.

Note that many demos will not work with this script (including the [Language-Directed Pick and Place](#language-directed-pick-and-place) demo) and [learning from demonstration](docs/learning_from_demonstration.md). However, you can still run the [simple motions demo](examples/simple_motions.py) and [view images](#visualization-and-streaming-video) with this script.

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh --no-d405
```

#### Optional: Docker Quickstart

To help you get started more quickly, we provide two pre-built [Docker](<https://en.wikipedia.org/wiki/Docker_(software)>) images that you can download and use with two shell scripts.

On your remote machine, you can install docker as normal, then, you can start the client on your GPU computer:

```bash
./scripts/run_stretch_ai_gpu_client.sh
```

This script will download the Docker image and start the container. You will be able to run Stretch AI applications from within the container.

### Language-Directed Pick and Place

![orangecupinbox](https://github.com/user-attachments/assets/f6659e40-8ed2-410a-889e-84f8bf8d38ad)

Now that you have the server running on Stretch, we recommend you try a demonstration of language-directed pick and place.

For this application, Stretch will attempt to pick up an object from the floor and place it inside a nearby receptacle on the floor. You will use words to describe the object and the receptacle that you'd like Stretch to use.

While attempting to perform this task, Stretch will speak to tell you what it is doing. So, it is a good idea to make sure that you have the speaker volume up on your robot. Both the physical knob on Stretch's head and the volume settings on Stretch's computer should be set so that you can hear what Stretch says.

Now, on your GPU computer, run the following commands in the Docker container that you started with the script above.

You need to let the GPU computer know the IP address (#.#.#.#) for your Stretch robot.

```bash
./scripts/set_robot_ip.sh #.#.#.#
```

*Please note that it's important that your GPU computer and your Stretch robot be able to communicate via the following ports 4401, 4402, 4403, and 4404. If you're using a firewall, you'll need to open these ports.*

Next, run the application on your GPU computer:

```bash
python -m emet.app.ai_pickup
```

It will first spend time downloading various models that it depends on. Once the program starts, you will be able to bring up a [Rerun-based GUI](https://rerun.io/) in your web browser.

![Rerun-based GUI for the ai_pickup app.](docs/images/rerun_example.png)

Then, in the terminal, it will ask you to specify an object and a receptacle. For example, in the example pictured below, the user provided the following descriptions for the object and the receptacle.

```
Enter the target object: plush rabbit toy
Enter the target receptacle: yellow chair
```

![RabbitChair](https://github.com/user-attachments/assets/dc7c19d2-49bd-45af-95ef-42abf22be5aa)

At Hello Robot, people have successfully commanded the robot to pick up a variety of objects from the floor and place them in nearby containers, such as baskets and boxes.

Find out more about the LLM-based AI agent in its [documentation](docs/llm_agent.md). And once you're ready to learn more about Stretch AI, you can try out the [variety of applications (apps)](docs/apps.md) that demonstrate various capabilities.

## Documentation map

Use this outline when updating docs or finding the right page. Prefer editing the **canonical** doc in each row; satellite pages should link back rather than duplicate policy.

### Start here

| Doc | When to use |
|-----|-------------|
| [first_test.md](docs/first_test.md) | Interactive agent in sim (table or MolmoSpaces); copy-paste prompts |
| [cli.md](docs/cli.md) | `emet serve` / `run` / `test` / `eval` / `jobs` / `hmeqa` flags (verify with `--help`) |
| [deploy.md](docs/deploy.md) | Stretch + Mars bridge deploy (`emet deploy --robot …` / `emet mars`) + LAN Orin LLM; wrist Arducam |
| [emet_config.md](docs/emet_config.md) | Nested YAML (`configs/emet/default.yaml`), `--set` / `-O`, robot overlays |
| [TESTING.md](docs/TESTING.md) | `uv run emet test`, memory-backend smokes, Dynagraph harnesses |
| [known_issues.md](docs/known_issues.md) | Open bugs, Habitat EGL / agent segfault notes |

### Memory, EQA, and agents

| Doc | When to use |
|-----|-------------|
| [dynamem.md](docs/dynamem.md) | Voxel map / open-vocab manipulation (`emet run dynamem`) |
| [graph_eqa.md](docs/graph_eqa.md) | GraphEQA how-to (`emet run graph-eqa`) |
| [graph_memory.md](docs/graph_memory.md) | Graph memory **code structure** (`emet.memory.graph_eqa`, `GraphStore`, two EQA loops) |
| [dynagraph.md](docs/dynagraph.md) | Dynagraph (GraphEQA + voxels + merge/staleness); Robocasa explore |
| [attempt_ledger.md](docs/attempt_ledger.md) | Opt-in action-outcome ledger (nav/verify/manip attempts in graph memory) |
| [eqa.md](docs/eqa.md) | Embodied QA overview (older Stretch path + pointers) |
| [AGENT_RUN.md](docs/AGENT_RUN.md) | Discord/CHAT vs Habitat `EQA_EPISODE`; skill packs |
| [llm_serve.md](docs/llm_serve.md) | **Remote inference** (OpenAI LAN) + **testing LLMs** (`emet llm` / `emet run chat --host`) |
| [llm_agent.md](docs/llm_agent.md) | Historical Stretch LLM pickup walkthrough → points at modern agent docs |
| [adding_a_new_task.md](docs/adding_a_new_task.md) | Register a new LLM-callable task |
| [discord_bot.md](docs/discord_bot.md) | Discord bridge for the interactive agent |

**Agentic HM-EQA (paper / Habitat):** approach and failure notes live under experiments —

- [agentic_qwen_context.md](docs/experiments/agentic_qwen_context.md) — **current approach** (evidence-card recall, VLM-first assess, frontier retirement)
- [agentic_scale.md](docs/experiments/agentic_scale.md) — holdout / bal-32 scale ladder
- [evaluation.md](docs/evaluation.md) — agentic tool loop + overnight / GPU runbook

### Simulation and visualization

| Doc | When to use |
|-----|-------------|
| [simulation.md](docs/simulation.md) | MuJoCo / Robocasa overview |
| [simulation_modules.md](docs/simulation_modules.md) | Module map under `src/emet/simulation/` |
| [molmospaces.md](docs/molmospaces.md) | iTHOR merge + `emet serve molmospaces` |
| [sim_configs.md](docs/sim_configs.md) | Sim-oriented YAML / planner params |
| [rerun.md](docs/rerun.md) | Rerun world frame, flags per subcommand |
| [zmq_obs.md](docs/zmq_obs.md) / [zmq_session_metadata.md](docs/zmq_session_metadata.md) | ZMQ observation contract + session GT metadata |
| [dynagraph_robocasa_e2e.md](docs/dynagraph_robocasa_e2e.md) | Dynagraph Robocasa end-to-end testing |

Env toggles: [environment_variables.md](docs/environment_variables.md), [molmospaces_environment_variables.md](docs/molmospaces_environment_variables.md).

### Evaluation and paper

| Doc | When to use |
|-----|-------------|
| [experiments/README.md](docs/experiments/README.md) | **Master index** — all paper tracks, smokes, output dirs |
| [paper_benchmarks.md](docs/paper_benchmarks.md) | Operator runbook ↔ LaTeX table mapping |
| [evaluation.md](docs/evaluation.md) | Cross-track overnight, `emet eval` / `emet jobs`, agentic verify |
| [habitat_eqa.md](docs/habitat_eqa.md) + [habitat/](docs/habitat/README.md) | HM-EQA harness install / data / troubleshooting |
| [simulation_testing_plan.md](docs/simulation_testing_plan.md) | Seven-track paper sim smoke battery |
| [ovmm.md](docs/ovmm.md) / [ovmm_find_phase_benchmark.md](docs/ovmm_find_phase_benchmark.md) | OVMM find-phase / full |
| [sqa3d.md](docs/sqa3d.md) / [sqa3d_compute.md](docs/sqa3d_compute.md) | SQA3D replay + compute notes |
| [dynagraph_benchmarks.md](docs/dynagraph_benchmarks.md) | Dynagraph-specific bench helpers |
| [dynamic_exploration_benchmark.md](docs/dynamic_exploration_benchmark.md) | Dynamic exploration phases |

Paper LaTeX: `paper/main.tex` → `paper/sections/` (method § EQA loops; appendix agentic tools).

### Robots and hardware

| Doc | When to use |
|-----|-------------|
| [robots/supported_robots.md](docs/robots/supported_robots.md) | Registry of embodiments |
| [robots/nori.md](docs/robots/nori.md) | Nori A3 bimanual backend (vendored MJCF, ArmChains, nori-sdk follow-up) |
| [robots/innate_mars.md](docs/robots/innate_mars.md) | Innate Mars bridge / sim / DA3 |
| [robots/sourccey.md](docs/robots/sourccey.md) | Sourccey sim support (Vulcan Robotics) |
| [start_with_docker_plus_virtenv.md](docs/start_with_docker_plus_virtenv.md) | Stretch GPU client + robot install |
| [update.md](docs/update.md) | Updating code on the robot |
| [jetson.md](docs/jetson.md) | Jetson notes |
| [llm_serve.md](docs/llm_serve.md) | Jetson / LAN OpenAI serve, remote VL, smoke/chat |
| [simple_api.md](docs/simple_api.md) | Lightweight wireless control API |

### Apps, LfD, install extras

| Doc | When to use |
|-----|-------------|
| [apps.md](docs/apps.md) | Catalog of `emet.app.*` demos |
| [data_collection.md](docs/data_collection.md) / [learning_from_demonstration.md](docs/learning_from_demonstration.md) | LfD collect → train |
| [install_details.md](docs/install_details.md) / [docker.md](docs/docker.md) | Deeper install / image builds |
| [debug.md](docs/debug.md) | Headless, Rerun, common failures |
| [plans/](docs/plans/README.md) | Design / investigation notes (not user runbooks) |

## Development

Clone the repo, then install with [Astral uv](https://docs.astral.sh/uv/) as in [Installation (Astral uv)](#installation-astral-uv):

```bash
./install.sh
# or: uv sync
pre-commit install
```

A plain `uv sync` installs **default dependency groups** (dev, sim, hand_tracker, dynamem, da3) and works without cloning simulation repos; **robosuite/robocasa** still come from `./install.sh --sim` / `emet install sim`. `./install.sh` defaults to **no** Robocasa clone (use `--sim` or `EMET_INSTALL_PROFILE=full` for the old behavior). With sim enabled, MolmoSpaces (``.venv-molmospaces``) installs automatically when ``packages/emet_molmospaces`` is present unless you pass ``--no-molmospaces``. Use `./install.sh --no-sam2` or `uv sync --no-group dynamem` if the SAM-2 submodule is absent.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information, and [debug](docs/debug.md) / [update](docs/update.md) for troubleshooting. You can test most code in [simulation](docs/simulation.md) without a physical robot. Simulation and MolmoSpaces optional env toggles: [docs/environment_variables.md](docs/environment_variables.md) ([MolmoSpaces-specific](docs/molmospaces_environment_variables.md)).

### Testing

From the repo root (uses project `.venv` via uv):

```bash
uv sync
uv run emet test                    # full suite (sim on by default)
uv run emet test --no-sim           # faster; skip sim integration tests
uv run emet test -v src/test/memory/test_memory_backends_smoke.py
```

**Dynagraph (this branch):** unit tests for explore-loop and graph memory, plus an optional multi-robot Robocasa floor-metrics harness (~20 min). See [docs/TESTING.md](docs/TESTING.md) and [docs/dynagraph_robocasa_e2e.md](docs/dynagraph_robocasa_e2e.md).

### Updating Code on the Robot

See the [update guide](docs/update.md) for more information. Code installed from git must be updated manually, including code from this repository.

You can also pull the latest docker image on the robot with the following command:

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh --update
```

### Building Docker Images

Docker build and other instructions are in the [docker guide](docs/docker.md). From the project root (ensure submodules are inited, e.g. `git submodule update --init --recursive third_party/segment-anything-2`):

```
docker build -t stretch-ai_cuda-11.8:latest . -f docker/Dockerfile.cuda-11.8
```

Or use the helper: `./docker/build-docker.sh`. See the [docker guide](docs/docker.md) for more.

## Acknowledgements

Parts of this codebase were derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project, and is licensed under the [MIT license](META_LICENSE). We thank the Meta team for their contributions.

The [stretch_ros2_bridge](src/stretch_ros2_bridge) package is based on the [OK robot](https://github.com/ok-robot/ok-robot) project's [Robot Controller](https://github.com/NYU-robot-learning/robot-controller/), and is licensed under the [Apache 2.0 license](src/stretch_ros2_bridge/LICENSE).

We use [LeRobot from HuggingFace](https://github.com/huggingface/lerobot) for imitation learning, though we use [our own fork](https://github.com/hello-robot/lerobot).

## License

This code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information.
