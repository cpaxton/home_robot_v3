# GraphEQA: Graph-Based Memory for Embodied Question Answering

Stretch AI has **three memory models**: **sparse voxel map** (base voxel map in `emet.mapping.voxel`), **DynaMem** (voxel + VL features + EQA in `emet.memory.dynamem` / `emet.mapping.voxel`), and **Graph EQA** (graph-based in `emet.memory.graph_eqa`). GraphEQA is the graph-based option for Embodied Question Answering (EQA): it maintains an **object-centric semantic scene graph** (nodes = objects/regions with labels and 3D positions; edges = spatial relations like *near*, *on*) and uses the same mLLM (e.g. Gemini) to answer questions and suggest where to explore. The default EQA pipeline uses DynaMem (see [EQA](eqa.md)).

This implementation is a **re-implementation** inspired by the [GraphEQA paper](https://arxiv.org/abs/2412.14480) (Saxena et al.). The [original repository](https://github.com/SaumyaSaxena/graph_eqa) is not open source; no code was copied from it.

## The three memory models

| Model | Description | Typical use |
|-------|-------------|-------------|
| **Sparse voxel map** | Base 2D/3D voxel map + optional instance memory. | Default agent (InstanceMemoryController), mapping, navigation. |
| **DynaMem** | Voxel map + VL features + EQA, pick-and-place. | `emet run dynamem`; EQA with voxel-based memory. |
| **Graph EQA** | Scene graph (nodes + edges) + task-relevant images. | `emet run graph-eqa`; EQA with graph-based memory. |

## When to use GraphEQA vs DynaMem EQA

| | **EQA (DynaMem)** | **Graph EQA** |
|---|-------------------|----------------|
| Memory | Voxel map + VL features + task-relevant images | Scene graph (nodes + edges) + task-relevant images |
| Representation | Dense 2D/3D voxels, instance segments | Explicit objects and relations (e.g. “table near cup”) |
| Best for | Environments where dense spatial grounding helps | When you want an interpretable, graph-style memory and the same EQA workflow |

Both use the same exploration loop (rotate, navigate, look around), the same mLLM for answers and confidence, and the same Rerun visualization when the robot client enables it (Stretch and registry robots such as **rby1** / **galaxea_r1** wire `GenericZmqClient` the same way as `emet run dynamem`: live maps and scene graph under `world/…`). **`emet run dynamem`** turns Rerun on by default (use `--no-rerun` to disable). **`emet run agent`** leaves Rerun off unless you pass **`--rerun`**. **Prerequisite:** the MuJoCo ZMQ server must be publishing observations and state before the client connects; if Rerun’s web viewer is empty, confirm `emet serve mujoco --robot <name>` is running first (large scenes can take 30–90s). With `--headless` (with Rerun) or no display, open `http://<host>:9090?url=ws://<host>:9877` (or set `RERUN_BIND_ALL=1` / `--rerun-bind` for remote access). Saved runs still use `--save_rerun` / `--SR` as below. GraphEQA feeds the graph string into the prompt instead of (or in addition to) the voxel-based image descriptions.

## Running GraphEQA

### Prerequisites

- Stretch robot or [simulation](simulation.md) (e.g. `emet serve mujoco`).
- Same as [EQA](eqa.md): Gemini API key (`GOOGLE_API_KEY`), and optionally Discord token for the bot.

### Command line

From the project root:

```bash
# Simulation
emet run graph-eqa --robot-ip 127.0.0.1

# Physical robot (set ROBOT_IP)
emet run graph-eqa --robot-ip $ROBOT_IP
```

Other options (mirroring `run_eqa`):

- `--not_rotate_in_place` / `-N`: skip initial rotation-in-place scan.
- `--discord` / `-D`: use the Discord bot (task `graph_eqa`).
- `--save_rerun` / `--SR`: save Rerun logs under `graph_eqa_log/`.

Example:

```bash
export GOOGLE_API_KEY=your_key
emet run graph-eqa --robot-ip 127.0.0.1 -N
```

Then type questions at the prompt; the agent will explore and answer using the graph memory.

## Testing Graph EQA in Robocasa

Use [Robocasa](simulation.md#robocasa-rich-kitchen-scenes) kitchen scenes so the robot can explore and fill the graph with real objects (fruit, pots, sink, etc.), then answer questions about them.

### 1. Prerequisites

- Robocasa and sim extra installed:
  ```bash
  emet install robocasa
  emet sync -e sim
  ```
- Gemini API key set: `export GOOGLE_API_KEY=your_key`

### 2. Start the MuJoCo server with Robocasa

**Terminal 1** – from the project root:

```bash
emet serve mujoco --use-robocasa
```

Optional: change task, style, or layout (see [simulation.md](simulation.md#run-with-robocasa)):

```bash
emet serve mujoco --use-robocasa --robocasa-task PnPCounterToCab --robocasa-style 1 --robocasa-layout 0
```

Wait until the simulator window shows the kitchen scene and the robot is ready.

### 3. Run Graph EQA

**Terminal 2** – from the project root:

```bash
export GOOGLE_API_KEY=your_key
emet run graph-eqa --robot-ip 127.0.0.1
```

- By default, the robot does an **initial rotation-in-place** to scan the room and build the first graph nodes from what the camera sees.
- To skip the initial scan: `emet run graph-eqa --robot-ip 127.0.0.1 -N`
- To save Rerun logs: add `--save_rerun` (or `--SR`).
- CPU-only: the app uses the same encoder as DynaMem; if you need CPU, you may need to set config/CLI options that disable GPU-heavy models (see [simulation.md](simulation.md) for DynaMem CPU notes).

### 4. Explore until the robot finds a known object

1. After startup (and optional rotate-in-place), you get a prompt: **Question (Press enter to quit):**
2. Type a question about something that **exists in the kitchen scene**, for example:
   - *Where is the apple?*
   - *What is on the counter?*
   - *Is there a pot on the stove?*
3. The agent will:
   - Use the current scene graph (and task-relevant images) to try to answer.
   - If not confident, it will **navigate to a suggested frontier** and **look around** again; each observation updates the graph with new object labels from the encoder.
   - Repeat until it can answer with confidence or hits the step limit.
4. The graph is updated on every observation (from the voxel map’s image descriptions); objects like “apple”, “sink”, “pot” appear as nodes as the robot explores.

So “find some known object” means: **ask about an object that is actually in the scene** (e.g. an apple or a pot in the default Robocasa task). The robot will explore until the graph contains enough information for the mLLM to answer.

### 5. First-run model downloads

As with DynaMem/EQA, the first run may download vision/encoder models (e.g. for image descriptions). See [simulation.md](simulation.md) for typical first-run messages and caches.

### Programmatic use

- **Graph memory only**: use `emet.memory.graph_eqa.GraphEQAMemory` to build a scene graph from observations and call `query_answer(question, xyt, planner)` (same return contract as the DynaMem voxel map).
- **Full agent**: use `emet.controller.robot_agent_graph_eqa.GraphEQAController` (or `RobotAgentGraphEQA`) with your robot client; it uses the voxel map for navigation and the graph memory for EQA.
- **Executor**: use `emet.controller.task.dynamem.EQAExecuter(agent)` with a GraphEQA agent; the executor interface is the same as for EQA.

## Code layout

All three memory models:

| Memory model | Package | Notes |
|--------------|---------|--------|
| **Sparse voxel map** | `emet.mapping.voxel` | `SparseVoxelMap`; base voxel map used by default agent. |
| **DynaMem** | `emet.memory.dynamem` / `emet.mapping.voxel` | Re-exports `SparseVoxelMapDynamem`; VL + EQA voxel memory. |
| **Graph EQA** | `emet.memory.graph_eqa` | `GraphEQAMemory` in `graph_memory.py`. |

Other components:

| Component | Location |
|-----------|----------|
| GraphEQA agent | `src/emet/controller/robot_agent_graph_eqa.py` (`GraphEQAController`) |
| App entry point | `src/emet/app/run_graph_eqa.py` |
| Plan (design) | [docs/plans/GRAPH_EQA_PLAN.md](plans/GRAPH_EQA_PLAN.md) |

## Tests and contributing

- **Tests**: `pytest src/test/memory/test_graph_eqa_memory.py` for the graph memory; controller smoke tests include `GraphEQAController`; CLI tests check that `emet run graph-eqa` is available.
- **Contributing**: Follow the main [CONTRIBUTING.md](../CONTRIBUTING.md). The implementation is intentionally a clean re-implementation; do not paste code from the closed-source GraphEQA repo.

---

*Last updated: March 2025*
