# Testing the three memory backends (SVM, DynaMem, GraphEQA)

## Refactor summary (branch vs main)

The **mapping** and **memory** refactor on this branch:

- **mapping** (`emet.mapping`): Exposes grid, instance, scene_graph, and voxel (base + Dynamem). See `docs/plans/MAPPING_REFACTOR.md`. Imports like `from emet.mapping import SparseVoxelMap, SparseVoxelMapDynamem` work; Dynamem types also live under `emet.mapping.voxel`.
- **memory** (`emet.memory`): New package. Exposes `GraphEQAMemory` and (lazy) `SparseVoxelMapDynamem` / `SparseVoxelMapNavigationSpaceDynamem` so GraphEQA can be used without importing mapping.voxel directly. DynaMem implementation remains in `emet.mapping.voxel` and is re-exported from `emet.memory.dynamem`.

The refactor is consistent: mapping = spatial representation; memory = semantic/EQA backends (sparse voxel in mapping, DynaMem and GraphEQA in memory).

## Test matrix for all three backends

| Backend   | What it is                    | Unit / smoke tests                    | Integration (sim) test                    |
|-----------|--------------------------------|---------------------------------------|--------------------------------------------|
| **SVM**   | Instance memory (RobotAgent)   | `test_svm.py`, `test_memory_backends_smoke::test_svm_backend_smoke` | — (SVM uses pkl or real robot)             |
| **DynaMem** | Voxel + VL features          | `test_semantic_memory.py`, `test_memory_backends_smoke::test_dynamem_backend_smoke` | `test_red_cylinder_in_sim.py`              |
| **GraphEQA** | Graph-based EQA memory     | `test_graph_eqa_memory.py`, `test_memory_backends_smoke::test_graph_eqa_backend_smoke` | — (same sim as DynaMem for nav; EQA is graph) |

## How to run tests

Use **uv** and the **`emet test`** CLI so the project env is used. From repo root:

```bash
uv sync --extra dev
uv run emet test -v
```

Smoke and unit tests require the main project dependencies. The integration test additionally needs the sim extra and `--sim` (or `RUN_SIM_TESTS=1`).

### All three backends (smoke, no sim)

```bash
uv run emet test -v src/test/memory/test_memory_backends_smoke.py
```

- **SVM**: Creates `RobotAgent` with `DummyStretchClient` and `src/test/mapping/planner.yaml`, checks `get_voxel_map()` and `get_navigation_space()`.
- **DynaMem**: Builds a small semantic map with a red cylinder, runs `localize_text("red cylinder")`, asserts the returned point is near the expected position.
- **GraphEQA**: Creates `GraphEQAMemory` with mock EQA/description clients, adds an observation, runs `query_answer`, checks the return tuple.

### Backend-specific tests

```bash
# SVM (requires planner.yaml; optional pkl for full eval)
uv run emet test -v src/test/mapping/test_svm.py

# DynaMem (unit)
uv run emet test -v src/test/mapping/test_semantic_memory.py

# GraphEQA (unit)
uv run emet test -v src/test/memory/test_graph_eqa_memory.py

# Controller exports (all three agents)
uv run emet test -v src/test/controller/test_controller_smoke.py
```

### Integration test: robot moves in scene and finds red cylinder (DynaMem, with timeout)

This test starts the default MuJoCo scene (red cylinder + blue cube), runs the robot’s **rotate_in_place** to build the map, then asserts **localize_text("red cylinder")** returns a point near the table. The test is skipped unless `--sim` or `RUN_SIM_TESTS=1`, and has a **120s timeout** (pytest-timeout, in dev deps).

```bash
uv run emet test --sim -v src/test/mapping/test_red_cylinder_in_sim.py
```

On Linux, MuJoCo runs headless (EGL). Requires full env (e.g. `pip install -e ".[sim]"` or `emet sync -e sim`).

## Summary

- Refactor: mapping vs memory split is clear; imports are consistent.
- All three backends have unit/smoke tests; one command runs them: `pytest src/test/memory/test_memory_backends_smoke.py -v`.
- The “robot moving around and find red cylinder” integration test is `test_red_cylinder_detected_in_sim` in `src/test/mapping/test_red_cylinder_in_sim.py`, with a 120s timeout.
