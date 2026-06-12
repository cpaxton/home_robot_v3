# Dynamem

[![arXiv](https://img.shields.io/badge/arXiv-2401.12202-163144.svg?style=for-the-badge)](https://arxiv.org/abs/2411.04999)
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-262626?style=for-the-badge)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/Videos-Website-db6a4b.svg?style=for-the-badge&logo=airplayvideo)](https://dynamem.github.io)

Dynamem is an open vocabulary mobile manipulation system that works life long in any unseen environments. It can continuously process text queries in the form of "pick up A and place it on B" (e.g. "pick up apple and place it on the plate").

Compared to Stretch AI Agent mentioned [here](llm_agent.md), Dynamem can
* continuously update its semantic memory when they observe the environment changes, which allows the system to work life long in homes without rescanning the environments;
* pick up more objects, especially bowl like objects.

However, there is some reason why sometimes we should still use AI agent:
* Dynamem does an open loop pick up, which requires the robot urdf to be very well calibrated as the robot does not take new observation to correct itself once the action plan is generated.
* Dynamem uses Anygrasp, a closed source gripper pose prediction model. Some researchers or companies might not be allowed or be able to use it.

_Click to follow the link to YouTube:_

[![Example of Dynamem in the wild](images/dynamem.png)](https://youtu.be/oBHzOfUdRnE)

[Above](https://youtu.be/oBHzOfUdRnE) shows Dynamem running in NYU kitchen.

# Understanding Dynamem code structure
Dynamem consists of three components, navigation, picking, and placing.
To complete "Pick up A and Place it on B", it will call 4 commands sequentially:
- `navigate(A)`
- `pick(A)`
- `navigate(B)`
- `place(B)`

Besides these commands, Dynamem also provides exploration module
- `explore()`

## Navigation and exploration
Dynamem stores (two) voxelized pointcloud for navigation and exploration. The first pointcloud is used to generate obstacle map for running A* path planning while another is used to store vision language features for visual grounding and generate value map for exploration.

In [Dynamem paper](https://arxiv.org/pdf/2411.04999), three ways to query semantic memory for visual grounding are introduced, in this stack we only set up querying with vision language feature similarity and querying with the hybrid of mLLMs and vision language feature similarity. The first strategy is faster while the second has better performance. By default the stack will chose VL feature similarity to do visual grounding.

In terms of exploration, we discovered that commonly used frontier based exploration (FBE) is not suitable for dynamic environments because obtacles might be moved around, creating new frontier, and already scanned portions of the room might also be changed. Therefore, we introduced a value based exploration that assigns any point in the 2D map a heuristic value evaluating how valuable it is to explore to this point. The detailed analysis is described in [Dynamem paper](https://arxiv.org/pdf/2411.04999).

For **graph-based EQA** on the same voxel stack (optional graph node merge and staleness), see [Dynagraph](dynagraph.md).

### Configuration and Rerun `map_topdown`

Navigation and mapping hyperparameters live in **`dynav_config.yaml`** (or `--dynav-config`). See **[Dynav configuration](dynav_config.md)** for a section-by-section reference, including:

- **`map_boundary`** — optional grid-edge obstacle barrier (default **off**). To restore the legacy red frame at the map rim used for exploration safety:

  ```yaml
  map_boundary:
    obstacle_barrier_cells: 30
    history_penalty_cells: 35
  ```

- **Top-down map in Rerun** — the `map_topdown` panel and `send_map_snapshot` crop to the explored region (same framing as Discord), without large unexplored margins.

## Picking and placing
Dynamem has two manipulation systems, one is Stretch AI Visual Servoing code, as described in the [LLM agent](llm_agent.md) while another is [OK-Robot manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation).

Instructions for AnyGrasp manipulation is put [here](#manipulation-with-anygrasp) and instructions for visualn servoing manipulation is put [here](#manipulation-with-stretch-ai-visual-servoing-manipulation).

The high level idea for AnyGrasp picking is
- Transform RGBD image from Stretch head camera into a RGB pointcloud.
- [AnyGrasp](https://arxiv.org/abs/2212.08333) proposes a set of collision free gripper poses given a RGB pointcloud.
- [OWLv2](https://arxiv.org/abs/2306.09683) and [SAMv2](https://ai.meta.com/blog/segment-anything-2/) to select only gripper poses that actually manipulates the target object.
- Transform the selected 6-DoF pose into gripper actions using URDF.

Placing is relatively simpler as all you need to do is to segment the target receptacle in the image and select a middle point to drop on.

The advantages of AnyGrasp manipulation system, compared to visual servoing manipulation in [LLM agent](llm_agent.md) includes:
- More general purpose, dealing with objects with different shapes, such as bowls, bananas.
The disadvantages includes:
- Open loop so unable to recover from controller errors.
- Reliance on accurate robot calibration and urdf.

# Running Dynamem

## Running DynaMem in Simulation

You can run DynaMem in MuJoCo simulation without a physical robot. See [Simulation docs](simulation.md) for setup.

**Install SAM2** (required for OWL+SAM segmentation): `emet sync -e dynamem` or `uv sync` (dynamem is a default group when `third_party/segment-anything-2` exists) or run `./install.sh` (includes SAM2 by default; use `--no-sam2` to skip).

**Headless (no DISPLAY)**: The native Rerun viewer is disabled, but the web server starts automatically. To view from a laptop over Tailscale or VPN, use SSH port forwarding: `ssh -L 9090:localhost:9090 -L 9877:localhost:9877 user@<robot-ip>`, then open `http://localhost:9090?url=ws://localhost:9877`. Direct connection at `http://<robot-ip>:9090?url=ws://<robot-ip>:9877` may fail because Rerun binds to localhost by default. See [Debug: Headless and Rerun](debug.md#headless-and-rerun) for more.

**Explicit headless (`--headless`)**: Use when you have a display but want to view Rerun from another machine (e.g. SSH). Disables the native viewer and serves only the web UI at `:9090`.

**Terminal 1** – Start the MuJoCo server (Robocasa recommended for richer scenes):

```bash
emet serve mujoco --scene robocasa
# or: python -m emet.simulation.mujoco_server --use-robocasa  # internal server flag; CLI: --scene robocasa
```

**Terminal 2** – Run DynaMem with visual servoing (AnyGrasp requires real robot):

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class
# or: python -m emet.app.run_dynamem --robot_ip 127.0.0.1 --server_ip 127.0.0.1 -S --visual-servo --match-method class
```

For headless (view Rerun from another machine at `http://<server-ip>:9090?url=ws://<server-ip>:9877`):

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class --headless
```

For CPU-only: add `--cpu`. The `-S` flag skips confirmations for autonomous runs. At the mode prompt, enter **Q** (or **quit**) to exit cleanly.

### Debugging map drift (rotate in place, floor height)

Spinning **in place** should leave **logged world geometry** (voxel map, fused points) in the navigation world frame while `gps`/`compass` yaw changes. The **default Rerun 3D view** is robot-centered (`origin=world/robot`), so the map **appears to co-rotate** with the base during in-place spins — that is expected for the viewer, not proof that fusion used the wrong frame. If you need to verify world-lock visually, use a fixed-world blueprint (`spatial3d_view_world()`; see [rerun.md](rerun.md)). If the cloud still looks wrong **in a world-fixed view**, split **depth** vs **pose** vs **nav frame**:

**Rerun head points vs voxel cloud:** When DynaMem is running, **`world/head_camera/points`** is built from the **same resolved depth map** as voxel fusion (whatever `depth_source` / DA3 / `--perfect-depth` produced), not a separate raw-ZMQ-only buffer. **`world/point_cloud`** is the fused voxel representation and often looks smoother than the head channel, which shows **per-pixel** depth unprojected into the world (holes, flying pixels, ceiling strips). With **`--perfect-depth`** in sim, a **clean voxel layout** plus a **noisy-looking head PCD** usually means the fusion path matches ground-truth depth—the voxels are the clearer signal for “did we get the room right?”

**When to use DA3 in sim:** Prefer **sensor** depth whenever the simulator publishes it (default `dynav_config.yaml`). Use **DA3** only when reproducing a **model or deployment** whose depth path is DA3 (see [Innate Mars / sim depth](robots/innate_mars.md)). Turning on DA3 in sim while the rest of the stack expects sensor depth makes debugging harder.

1. **Isolate depth** — `emet run dynamem ... --perfect-depth` or `EMET_DYNAMEM_PERFECT_DEPTH=1` forces observation **sensor** depth when available. If the map stabilizes, focus on DA3 / intrinsics; if not, focus on `camera_pose`, `gps`/`compass`, and `emet_session.navigation_origin_xyt`.

2. **Rerun** — Scrub the **frame** timeline: compare `world/point_cloud`, `world/robot`, `world/head_camera`. In the default robot-centered 3D view, static room features co-rotate with the base; switch to a world-fixed view (see [rerun.md](rerun.md)) before concluding the map data is rotating in world coordinates.

3. **Per-step logging** — `export EMET_DYNAMEM_MAP_DEBUG=1` prints camera translation, world `base_xyt`, `depth_source`, whether depth came from DA3 inference, and whether `navigation_origin_xyt` was on the observation.

4. **Saved tensors** — `logs/memory_*/.../debug/` (see `SparseVoxelMap` `DEBUG_SUBDIR`) stores `rgb*.npy`, `depth*.npy`, `intrinsics*.npy`, `pose*.npy` per observation for offline replay.

5. **Narrow the stack** — `emet debug-da3-depth` (live ZMQ depth + Rerun) tests projection without the full voxel pipeline.

6. **Sensor vs DA3 sky mask** — Raw **sensor** depth from `auto` or perfect-depth mode does **not** get `da3_ignore_sky_fraction_top` (that mask applies only to DA3-produced maps). Ceiling rows can add bogus vertical structure; adjust dynav height / depth clamps or use DA3 when you need that mask on inferred depth.

See `docs/logs/pointcloud-alignment-circle-test.md` for nav vs world notes on ZMQ sim.

---

## Running DynaMem on Physical Robot

You should follow the these instructions to run DynaMem. SLAM and control codes are supposed to be run on the robot while perception models are supposed to be run on the workstation (e.g. a laptop, a lambda machine; might also be run on the robot but not recommended).

So you should clone stretch ai repo with this command
```
git clone https://github.com/hello-robot/stretch_ai.git --recursive
cd stretch_ai
```
On **BOTH** your robot and workstation.

## On the robot
### Startup
Once you turn on Stretch robot, you should first calibrate it
```
stretch_free_robot_process.py
stretch_robot_home.py
```
If you have already run these codes to start up the robot, you may move to the next step.

### Launch SLAM on robots

To run navigation system of [Dynamem](https://dynamem.github.io), you first need to install environment with these commands:
```
bash ./install.sh
```

Next you are going to set up your robot launch files, please follow instructions in [Stretch AI startup guide](start_with_docker_plus_virtenv.md) to set up either [Docker](start_with_docker_plus_virtenv.md#run-the-robots-script) or [ROS2](start_with_docker_plus_virtenv.md#installing-ros2-packages-without-docker).

Then we launch SLAM on the robot.

If you choose to install with ROS2, run
```
ros2 launch stretch_ros2_bridge server.launch.py
```
Or if you choose to use docker, run
```
bash ./scripts/run_stretch_ai_ros2_bridge_server.sh --update
```

For more information on how to launch your robot, see the [Stretch AI startup guide](start_with_docker_plus_virtenv.md).

## On the workstation

Most of AI codes (e.g. VLMs, mLLMs) should be run on the workstation.

You need to first install the conda environments on the workstation, we recommend you run
```
./install.sh --no-version
mamba activate stretch_ai
```

If you use AnyGrasp manipulation, please refer to [these instructions](#prepare-manipulation-with-anygrasp) for the installation,
you would need to create a new conda environment on your worstation.

### Copying URDF from robot to workstation
No matter whether you choose to run which manipulation, having a well calibrated robot URDF is important, you should follow these steps to set up robot URDF (while visual servo picking does not require accurate robot URDF, placing heuristic is shared between these two systems):
- On your robot, follow instructions described in [Stretch Ros2](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_calibration) to calibrate your robot.
- Once you have a well calibrated urdf (in `~/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf` on your stretch robot), copy it to your workstation `src/emet/config/urdf/stretch.urdf`. It is recommended to run following commands on your workstation:
```
scp hello-robot@[ROBOT IP]:~/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf stretch_ai/src/emet/config/urdf/
```
- Run the following python scripts to replace urdf modification described in [OK Robot calibration docs](https://github.com/ok-robot/ok-robot/blob/main/docs/robot-calibration.md)
```
python src/emet/config/dynamem_urdf.py --urdf-path src/emet/config/urdf/stretch.urdf
```

Note that while URDF calibration is important for both manipulation systems, AnyGrasp manipulation has much higher requirement on robot calibration. On the other hand, even though the calibration is not perfect in visual servo manipulation, in most cases the robot is still going to complete the task.

You might want to check your calibration if the following things happen:
- Floor in the navigation pointcloud does not fall on `z=0` plane.
- Manipulation does not follow AnyGrasp predictions.

### Specifying IPs in Dynamem scripts

Firstly you should know the ip address of your robot and workstation by running `ifconfig` on these two machines. Continuously tracking ips of different machines is an annoying task. We recommend using [Tailscale](https://tailscale.com) to manage a series of virtual ip addresses. Run following command on the workstation to run dynamem
```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S
```
`robot_ip` is used to communicate robot and `server_ip` is used to communicate the server where AnyGrasp runs. If you don't run anygrasp (e.g. navigation only or running Stretch AI visual servoing manipulation instead), then set `server_ip` to `127.0.0.1` or just leave it blank.
If you plan to run AnyGrasp on the same workstation, we highly recommend you find the ip of this workstation instead of naivly setting `server_ip` to `127.0.0.1`.

Once the robot starts doing OVMM, a rerun window will be popped up to visualize robot's thoughts.
![Example of Dynamem in the wild](images/dynamem_rerun.png)

### Manipulation with AnyGrasp
The very first thing is to make sure OK-Robot repo is a submodule in your Stretch AI repo in `third_party/`!!!
If not, run `git submodule update --init --recursive` to update all submodules.

Next, please strictly follow [aforementioned steps](#copying-urdf-from-robot-to-workstation) to prepare accurate robot URDF!!!

Few steps are needed to be done before you can try AnyGrasp:
- Since AnyGrasp is a closed source model, you should first request for AnyGrasp license following [These instructions](https://github.com/graspnet/anygrasp_sdk?tab=readme-ov-file#license-registration)
- Install a new conda environment for running anygrasp following [OK Robot environment installation instructions](https://github.com/ok-robot/ok-robot/blob/main/docs/workspace-installation.md). **NOTE** that `stretch_ai` environment does not support AnyGrasp because the AnyGrasp packages conflict with `stretch_ai`'s python version.
- Run AnyGrasp with following commands in a new terminal window
```
# If you have not yet activated anygrasp conda environment, do so.
conda activate ok-robot-env

# Assume you are in stretch_ai folder in the new window.
cd third_party/ok-robot/ok-robot-manipulation/src/
python demo.py --open_communication --port 5557
```
To understand more options in running AnyGrasp, please read [OK Robot Manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation).

After AnyGrasp is launched, you can run default Dynamem commands as described above.
```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP
```

### Two DynaMem modes: exploration and manipulation
Dynamem support both exploration & mapping and OVMM tasks. So before each task it will ask you whether you want to run E (denoted for exploration) and M (denoted for OVMM).

One exploration iteration includes
* Looking around and scanning new RGBD images;
* Moving towards the point of interest in the map.
To specify how many exploration iterations you want the robot to run after selecting exploration, set up `explore-iter`. For example, if you want the robot to explore for 5 iterations, use the command.
```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S --explore-iter 5
```

### Visual grounding with GPT4o
[As mentioned previously](#navigation-and-exploration), by default we run visual grounding by doing object detection on the robot observataion with the highest cosine similarity. While this strategy is fast, another querying strategy, prompting GPT-4o to process top-k robot observations has better accuracy.

To try this querying strategy that uses GPT-4o boost your navigation accuracy, you first need to follow [OPENAI's instructions](https://platform.openai.com/docs/overview) to create API keys. After that you can try this version by turning on mllm(`-M`) in your scripts:
```
OPNEAI_API_KEY=$YOUR_API_KEY python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S -M
```

### Loading from previous semantic memory
Dynamem stores the semantic memory as a pickle file after initial rotation-in-place and every time `navigate(A)` is executed. This allows Dynamem to read from saved pickle file so that it can directly load the semantic memory from previous runs without rotating in place and scanning surroundings again.

You can control memory saving and reading by specifying `input-path` and `output-path`.

By specifying `output-path`, the semantic memory will be saved to `dynamem_log/` + `specified-output-path` + `.pkl`; otherwise, the semantic memory will be saved to pickle file named by the current datetime in `dynamem_log/`.

By specifying `intput-path`, the robot will first read semantic memory from specified pickle file and will skip the rotating in place.

The command looks like this

```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --output-path $PICKLE_FILE_PATH --input-path $PICKLE_FILE_PATH -S
```

### Ask for humans' confirmations before doing each subtask
Dynamem OVMM task implementation hardcodes such API calling sequence: navigating to the target object `navigate(A)`, picking up the object `pick(A)`, navigating to the target receptacle `navigate(B)`, placing the object on the receptacle `place(B)`. However, sometimes we might want to interfere with robot task planning. For example, if first picking up fails, we humans might want the robot to try again.

So how can we steer robot actions? One functionality we provided is asking for humans' confirmations. That is to say, even though by default the system still calls `navigate(A)`, `pick(A)`, `navigate(B)`, `place(B)` in sequence, but before it implements each module, humans can explicitly tell the robot whether they want it to call this API call.

How is that functionality helpful? Sometimes when the robot is already facing the object, we might not want to waste time in navigation, by selecting `N` (no) when asked "Do you want to run navigation?", the robot can skip navigation and directly pick up objects.

The flag `-S` in previous commands, it configures Dyname to skip these human confirmantions. To enable this functionality, you need to run

```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP
```

### Manipulation with Stretch AI visual servoing manipulation

If you do not have access to AnyGrasp, you can run with the Stretch AI Visual Servoing code, as described in the [LLM Agent documentation](llm_agent.md). In this case, you can run Dynamem with the following command:
```
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --visual-servo
```

If you use this manipulation on GPU, it will use Owlv2 + SAMv2 as segmentation models. This means that you need to install SAMv2. **Be aware that Stretch AI does not install SAMvs by default**, so you need to install SAMv2 yourself. Following these commands:

```
cd third_party/segment-anything-2
pip install -e .
```

### Running with the LLM Agent

You can also run an equivalent of the [LLM agent](llm_agent.md) with Dynamem. In this case, you can run Dynamem with the following command:
```
python -m emet.app.run_dynamem --use-llm
```

All of the flags [in the agent documentation](llm_agent.md) are also available in Dynamem:
```
# Start with voice chat
python -m emet.app.run_dynamem --use-llm --use-voice
```

You can specify an LLM, e.g.:
```bash
# Run Gemma 2B from Google locally
python -m emet.app.run_dynamem --use-llm --llm gemma

# Run Openai GPT-4o-mini on the cloud, using an OpenAI API key
OPENAI_API_KEY=your_key_here
python -m emet.app.run_dynamem --use-llm --llm openai
```

### Running on CPU

In some cases a GPU may not be available, so we provide a lightweight version of DynaMem that can run entirely on CPU, including on the robot's onboard NUC. While this version has reduced performance compared to the GPU version, it can still perform useful mobile manipulation tasks.

#### CPU vs GPU: Model Differences

When running on CPU, DynaMem automatically switches to lighter-weight models:

| Component | GPU Version | CPU Version |
|-----------|-------------|-------------|
| **Encoder** | SigLIP-so400m | CLIP ViT-B/16 |
| **Object Detection** | OWLv2-L-p14-ensemble | YoloE-L |
| **Segmentation** | SAM2 | Not available |
| **Image Resolution** | 480 x 360 | 360 x 270 |

#### CPU vs GPU: Threshold Differences

The feature matching thresholds are adjusted for the different encoders:

| Parameter | GPU (SigLIP) | CPU (CLIP) |
|-----------|--------------|------------|
| Feature matching threshold | 0.14 | 0.35 |
| Detection confidence | 0.15 (OWLv2) | 0.05 (YoloE) |

These thresholds are automatically configured when using `--cpu`.

#### Installation for CPU

Install a CPU-only environment with:
```bash
./install.sh --conda --cpu
```

This creates a separate conda environment named `stretch_ai_cpu_<version>` and skips installing SAM2 (which requires GPU).

Activate the environment:
```bash
conda activate stretch_ai_cpu_<version>
```

#### Running DynaMem on CPU

Run DynaMem on CPU with:
```bash
python -m emet.app.run_dynamem --robot_ip $ROBOT_IP --cpu --match-method "class" --vs
```

**Required flags for CPU mode:**
- `--cpu`: Enables CPU-only mode (automatically detected if no GPU available)
- `--vs` or `--visual-servo`: Uses visual servoing manipulation (required since AnyGrasp needs GPU)
- `--match-method "class"`: Uses class-based matching instead of feature matching

#### Limitations on CPU

When running on CPU, be aware of the following limitations:

1. **No SAM2 segmentation**: The Segment Anything Model 2 is not available, affecting placing accuracy
2. **No AnyGrasp manipulation**: AnyGrasp requires GPU, so visual servoing (`--vs`) is required
3. **No mLLM visual grounding**: GPT-4o visual grounding (`-M`) is not recommended due to latency
4. **Slower inference**: Expect significantly slower perception compared to GPU
5. **Lower resolution**: Images are processed at 360x270 instead of 480x360

#### Automatic CPU Detection

If no GPU is available, DynaMem automatically falls back to CPU mode:
```python
if not torch.cuda.is_available():
    print("Setting up to use CPU as there is no GPU!")
    cpu_only = True
```

This means you can omit the `--cpu` flag if running on hardware without a GPU.

#### Typical CPU Workflow

A typical workflow for running DynaMem on the robot's NUC (CPU-only):

1. **On the robot** - Start the ROS2 bridge server:
```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

2. **On the robot** - Run DynaMem:
```bash
conda activate stretch_ai_cpu_<version>
python -m emet.app.run_dynamem --robot_ip 127.0.0.1 --cpu --match-method "class" --vs -S
```

The `-S` flag skips human confirmations for autonomous operation.

## Cite Dynamem

If you find Dynamem useful in your research, please consider citing:
```
@article{liu2024dynamem,
  title={DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation},
  author={Liu, Peiqi and Guo, Zhanqiu and Warke, Mohit and Chintala, Soumith and Paxton, Chris and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2411.04999},
  year={2024}
}
```
