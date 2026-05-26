# Testing the three memory backends (SVM, DynaMem, GraphEQA)

**Index:** [TESTING.md](../TESTING.md) lists all testing documentation and automated suites in one place.

## Refactor summary (branch vs main)

The **mapping** and **memory** refactor on this branch:

- **mapping** (`emet.mapping`): Exposes grid, instance, scene_graph, and voxel (base + Dynamem). See `docs/plans/MAPPING_REFACTOR.md`. Imports like `from emet.mapping import SparseVoxelMap, SparseVoxelMapDynamem` work; Dynamem types also live under `emet.mapping.voxel`.
- **memory** (`emet.memory`): New package. Exposes `GraphEQAMemory` and (lazy) `SparseVoxelMapDynamem` / `SparseVoxelMapNavigationSpaceDynamem` so GraphEQA can be used without importing mapping.voxel directly. DynaMem implementation remains in `emet.mapping.voxel` and is re-exported from `emet.memory.dynamem`.

The refactor is consistent: mapping = spatial representation; memory = semantic/EQA backends (sparse voxel in mapping, DynaMem and GraphEQA in memory).

## Test matrix for all three backends

| Backend   | What it is                    | Unit / smoke tests                    | Integration (sim) test                    |
|-----------|--------------------------------|---------------------------------------|--------------------------------------------|
| **SVM**   | Instance memory (RobotAgent)   | `test_svm.py`, `test_memory_backends_smoke::test_svm_backend_smoke`, `test_unified_backend_svm_empty` | — (SVM uses pkl or real robot)             |
| **DynaMem** | Voxel + VL features          | `test_semantic_memory.py`, `test_memory_backends_smoke::test_dynamem_backend_smoke`, `test_unified_backend_dynamem` | `test_red_cylinder_in_sim.py` (parametrized: **stretch** and **innate_mars** + default table), `test_robocasa_memory_after_spin.py` (Robocasa) |
| **GraphEQA** | Graph-based EQA memory     | `test_graph_eqa_memory.py`, `test_memory_backends_smoke::test_graph_eqa_backend_smoke`, `test_unified_backend_graph_eqa` | — (same sim as DynaMem for nav; EQA is graph) |
| **Dynagraph (multi-robot Robocasa)** | Voxel nav + graph EQA + floor export | `test_floor_metrics.py`, `test_nav_xyt_session.py` | [`run_dynagraph_multi_robot_e2e.py`](../../src/test/app/run_dynagraph_multi_robot_e2e.py) — see [dynagraph_robocasa_e2e.md](../dynagraph_robocasa_e2e.md) |

Default MuJoCo scene (`scene.xml`): **red cylinder** (object2) at (0.08, -0.55, 0.6) and **blue cube** (object1) at (-0.02, -0.55, 0.6). After a single **rotate_in_place**, both should be visible and in memory for any method (DynaMem integration test asserts red cylinder; blue cube asserted when detected).

## Heavy Hugging Face VLLM load tests

Optional **CUDA** smokes that download large checkpoints are marked ``vllm_load`` and are **not** part of the default ``emet test`` run (see ``docs/plans/TESTING_VLLM_LOAD.md``). Fast registry/factory tests live under ``src/test/llms/test_vllm_registry.py``.

## How to run tests

Use **uv** and the **`emet test`** CLI so the project env is used. From repo root:

```bash
uv sync
uv run emet test -v
```

Smoke and unit tests require the main project dependencies. Sim integration tests run by default when you run `emet test`; they need the sim extra. Use `emet test --no-sim` or `RUN_SIM_TESTS=0` to skip sim tests for a faster run.

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

### Integration test: default MuJoCo scene — red cylinder and blue cube after one spin (DynaMem, 120s timeout)

This test starts the default MuJoCo scene (red cylinder + blue cube), runs the robot’s **rotate_in_place** to build the map, then: asserts **localize_text("red cylinder")** returns a point near (0.08, -0.55, 0.6); if **localize_text("blue cube")** returns a point, asserts it is near (-0.02, -0.55, 0.6); uses the **unified MemoryBackend** and asserts **check_memory_for_object("red cylinder")** has confidence > 0 (and blue cube when detected). Runs by default with `emet test`; skip with `emet test --no-sim` or `RUN_SIM_TESTS=0`. Has a 120s timeout (pytest-timeout, in dev deps).

```bash
uv run emet test -v src/test/mapping/test_red_cylinder_in_sim.py
```

Only the **innate_mars** parametrized case:

```bash
uv run emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
```

(`emet test` forwards pytest `-k`, `-m`, etc.; see `docs/cli.md`.)

On Linux, MuJoCo runs headless (EGL). Requires full env (e.g. `pip install -e ".[sim]"` or `emet sync -e sim`).

### Integration test: Robocasa scene — at least one object in memory after one spin (180s timeout)

Starts MuJoCo server with **--use-robocasa** (default kitchen task), connects, runs **rotate_in_place** once, then uses the unified backend to try common object names and asserts at least one has confidence > 0. Runs by default with `emet test`; skip with `emet test --no-sim` or `RUN_SIM_TESTS=0`. Requires robocasa assets.

```bash
uv run emet test -v src/test/simulation/test_robocasa_memory_after_spin.py
```

## Summary

- Refactor: mapping vs memory split is clear; imports are consistent.
- All three backends have unit/smoke tests; one command runs them: `pytest src/test/memory/test_memory_backends_smoke.py -v`.
- The “robot moving around and find red cylinder” integration test is `test_red_cylinder_detected_in_sim` in `src/test/mapping/test_red_cylinder_in_sim.py`, with a 120s timeout.
