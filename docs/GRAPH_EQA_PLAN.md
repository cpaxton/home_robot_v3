# GraphEQA: Plan for Adding a Graph-Based Memory Model

This document outlines the plan to add **GraphEQA** as a separate memory model for Embodied Question Answering (EQA), similar to DynaMem but using a **graph-based semantic memory** instead of a voxel-based one. The implementation is a **re-implementation** inspired by the [GraphEQA paper](https://arxiv.org/abs/2412.14480) and the [graph_eqa repository](https://github.com/SaumyaSaxena/graph_eqa); the original code is not open, so we do not copy it.

## Motivation

- **GraphEQA** (Saxena et al., arXiv:2412.14480) uses **3D semantic scene graphs** and task-relevant images as multi-modal memory for EQA, with hierarchical planning and semantics-guided exploration.
- The existing Stretch AI EQA pipeline uses **DynaMem** (voxel-based semantic memory + VL features). Adding a **graph-based** memory gives:
  - An alternative representation (explicit objects + relations) that can suit different question types and downstream planners.
  - A pluggable “memory backend” (DynaMem vs GraphEQA) for EQA.

## High-Level Design

1. **Graph memory module**  
   A new module that maintains an **object-centric semantic scene graph**:
   - **Nodes**: objects/regions with labels and 3D positions (and optional observation IDs).
   - **Edges**: spatial/semantic relations (e.g. `near`, `on`, `left_of`) derived from geometry and existing heuristics (inspired by `emet.mapping.scene_graph.SceneGraph` but focused on EQA and serialization for mLLM).
   - Built **incrementally** from the same exploration flow as DynaMem (robot observations → instances or image-level labels → graph nodes/edges).
   - Exposes the same **EQA query contract** as the DynaMem voxel map: `query_answer(question, xyt, planner)` returning `(reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images)`.

2. **Integration**
   - **Agent**: Reuse the same robot client and exploration/navigation; the only difference is the **memory backend** (voxel vs graph). We introduce a **GraphEQA agent** (or a parameter to select memory type) that uses the graph memory for EQA instead of the voxel map.
   - **Task executor**: A **GraphEQA executor** (like `EQAExecuter`) that calls the agent’s EQA API; the agent internally uses the graph memory’s `query_answer`.
   - **CLI**: New app `graph-eqa` and `emet run graph-eqa` (and optional Discord/run_eqa-style entry point).

3. **Re-implementation (no copy)**
   - **Scene graph**: Implement our own graph data structure and relation logic (nodes, edges, serialization to text). We can reuse existing building blocks (e.g. `Instance`, observation list, or VLM-generated labels) but the graph structure, relation rules, and prompt format are implemented from the paper’s description and our design.
   - **Question answering**: Same high-level flow as current EQA (keyword extraction → select task-relevant content → build prompt with memory + images → mLLM → parse answer/confidence/action). The “memory” part is the **graph string** + selected images instead of voxel-based image selection only.
   - **Exploration**: Reuse existing exploration/navigation; no duplication of low-level robot code.

## Components

Both memory models live under **`emet.memory`**: DynaMem (voxel) in `emet.memory.dynamem` (re-exports from `emet.mapping.voxel`), GraphEQA in `emet.memory.graph_eqa`.

| Component | Purpose |
|-----------|--------|
| `emet.memory.dynamem` | Re-exports DynaMem (voxel) memory types from `emet.mapping.voxel`. |
| `emet.memory.graph_eqa` | Graph-based EQA memory: graph construction, serialization, `query_answer`, and (if needed) target point from “explore this node”. |
| Graph data structure | Nodes (id, label, position_3d, obs_id), edges (id1, id2, relation). Build from instances/observations; update on new observations. |
| `GraphEQAExecuter` | Task executor that takes (question) and calls agent’s EQA API (which uses graph memory). Same interface as `EQAExecuter`. |
| `RobotAgentGraphEQA` (or agent with `memory_type="graph_eqa"`) | Agent that uses graph memory for `run_eqa` instead of voxel map. Same exploration/navigation as DynaMem agent. |
| `emet.app.run_graph_eqa` | Entry point: create robot client, create GraphEQA agent, run GraphEQA executor loop (questions until quit). |
| CLI | Add `graph-eqa` to `emet run` choices; wire to `run_graph_eqa`. |
| Tests | Unit tests for graph build/serialize, `query_answer` parsing; integration test for executor/import and CLI. |
| Docs | `docs/graph_eqa.md`; update `docs/eqa.md` and README to mention GraphEQA as an alternative memory model. |

## Implementation Steps

1. **Graph memory module**
   - Define graph structure (nodes, edges) and a class `GraphEQAMemory` (or similar).
   - Add logic to build/update the graph from a list of observations and optional instance list (e.g. from existing pipeline): each observation or instance becomes one or more nodes; compute pairwise relations (near, on, etc.) from positions/heuristics.
   - Implement `to_string()` (or equivalent) to serialize the graph for mLLM prompts (e.g. “Node 1: table at (x,y,z); Node 2: cup on Node 1; …”).
   - Implement `query_answer(question, xyt, planner)`:
     - Extract keywords from the question (reuse or mirror current EQA keyword extraction).
     - Select task-relevant images (by keyword overlap with node labels or by observation IDs attached to nodes).
     - Build prompt: question + graph string + image descriptions + relevant images.
     - Call mLLM (reuse existing EQA client); parse answer, confidence, and exploration action (e.g. “navigate to node X” or “image ID”); map to `target_point` using planner/grid if needed.
   - Implement `get_2d_map` / `xy_to_grid_coords` / `get_target_point_from_action` as needed so the agent can navigate when the mLLM suggests exploration (reuse planner from voxel space or a simple wrapper).

2. **Agent and executor**
   - Implement (or parameterize) agent that holds `GraphEQAMemory` and uses it in `run_eqa` / `run_eqa_one_iter` instead of the voxel map.
   - Implement `GraphEQAExecuter(__call__(question))` that delegates to this agent’s EQA API.
   - Add `run_graph_eqa` app: parse CLI, create client, create GraphEQA agent, run executor loop; support `--robot_ip`, `--discord`, etc., similar to `run_eqa`.

3. **CLI**
   - Add `graph-eqa` to the `run` app choices in `emet.cli` and dispatch to `emet.app.run_graph_eqa`.

4. **Tests**
   - Unit: graph construction from mock observations; serialization format; `query_answer` parsing (mock mLLM response).
   - Integration: import `GraphEQAExecuter`, `run_graph_eqa`; `emet run graph-eqa --help` and that `graph-eqa` appears in `emet run` choices.

5. **Documentation**
   - **graph_eqa.md**: What GraphEQA is, how it differs from DynaMem EQA, how to run `emet run graph-eqa`, options, and that it is a re-implementation inspired by the paper.
   - **eqa.md**: Mention GraphEQA as an alternative memory model and link to `graph_eqa.md`.
   - **README**: In EQA/Next Steps, add a line about GraphEQA and link to `graph_eqa.md`.

## Success Criteria

- GraphEQA can be selected as a separate app (`emet run graph-eqa`).
- EQA runs with graph-based memory (graph string + task-relevant images) and produces answers and optional exploration actions.
- Tests and documentation are in place; implementation does not copy closed-source code and is a clean re-implementation following the plan above.
