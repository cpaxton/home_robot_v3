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

Both use the same exploration loop (rotate, navigate, look around), the same mLLM for answers and confidence, and the same Rerun visualization. GraphEQA feeds the graph string into the prompt instead of (or in addition to) the voxel-based image descriptions.

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
