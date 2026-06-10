# Testing index

Central map of **how to run tests**, **what each suite validates**, and **where detailed write-ups live**. From the repo root, use **`uv sync`** then **`uv run emet test`** (see [cli.md](cli.md#testing)).

## Run from this repo

Use **`uv run emet …`** (or `source .venv/bin/activate` then bare `emet`) from the **project root** so commands pick up **this checkout’s** code and virtualenv.

| Symptom | Fix |
|---------|-----|
| `No such option: --explore-loop` on `emet run dynagraph` | Wrong `emet` on PATH (another clone or old install). Run `which emet`; use **`uv run emet`** from this repo. Confirm with `uv run python -m emet.app.run_dynagraph --help` (should list `--explore-loop`) |
| Sim logs show `Sim navigation: driving toward` (no `[sim_nav]`) | **`emet` from another checkout** (e.g. `home_robot_v4`). From this repo: `cd ~/src/home_robot_v2 && emet serve …` (yellow “re-running: uv run emet”) or `./scripts/emet-v2 serve …` |
| rotate_in_place sends goals hundreds of meters away | Usually stale sim + wrong `nav_world`; run [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) after nav changes |
| Tests skip sim unexpectedly | Default runs include sim; set `RUN_SIM_TESTS=0` or `uv run emet test --no-sim` to skip |
| Missing MuJoCo / Robocasa | `uv sync` with default groups, or `./install.sh --sim` |

## Quick commands

| Goal | Command |
|------|---------|
| Full suite (sim on by default) | `uv run emet test` |
| Skip sim (faster CI-style) | `uv run emet test --no-sim` |
| Verbose | `uv run emet test -v` |
| Memory backend smokes | `uv run emet test src/test/memory/test_memory_backends_smoke.py -v` |
| Multi-robot Dynagraph floor E2E | `uv run python src/test/app/run_dynagraph_multi_robot_e2e.py` |
| Dynagraph unit (explore loop, graph memory) | `uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v` |
| GraphObjectFusion + GT export (fast) | `uv run emet test src/test/memory/test_graph_object_fusion.py src/test/simulation/test_mujoco_gt_objects.py -v` |
| Dynagraph benchmark smoke (unit) | `uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py -v` |
| Dynagraph staleness / disappearance | `uv run emet test src/test/memory/test_dynagraph_staleness_disappearance.py -v` |
| Unified Dynagraph eval CLI | `uv run emet eval-dynagraph --episode /tmp/export` |
| SQA3D benchmark (unit) | `uv run emet test src/test/benchmarks/sqa3d/ -v` |
| SQA3D ScanNet embodied smoke | `uv run python scripts/run_sqa3d_scannet_smoke.py` |
| SQA3D EM@1 scoring | `uv run emet eval-sqa3d -p preds.jsonl --split val` |
| Graph fusion calibration (one scene) | `emet export-sim-gt` → `emet run dynagraph --calibration-export` → `emet tune-graph-fusion` (see [dynagraph.md](dynagraph.md#object-gt-export-and-graphobjectfusion-calibration)) |
| GraphEQA human-answer formatter | `uv run emet test src/test/memory/test_graph_eqa_human_answer.py -v` |
| Manual Dynagraph EQA + export (Robocasa) | See [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#single-eqa-question-manual-per-robot) |
| Dynagraph Robocasa question CLI (CI) | `uv run emet test src/test/app/test_dynagraph_robocasa_question_cli.py -v` |

Environment: **`RUN_SIM_TESTS=0`** skips sim integration tests. Heavy VLLM download tests use marker **`vllm_load`** and are excluded from default runs — see [plans/TESTING_VLLM_LOAD.md](plans/TESTING_VLLM_LOAD.md).

---

## Documentation by topic

| Topic | Doc | What it covers |
|-------|-----|----------------|
| **Memory backends (SVM, DynaMem, GraphEQA)** | [plans/TESTING_BACKENDS.md](plans/TESTING_BACKENDS.md) | Test matrix, red-cylinder / Robocasa spin integration, backend unit tests |
| **Dynagraph multi-robot Robocasa E2E** | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) | Floor-area parity (stretch / innate_mars / galaxea_r1), spawner maps, artefact paths |
| **GraphEQA manual Robocasa** | [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Interactive questions in kitchen sim (Gemini / encoder) |
| **MolmoSpaces** | [plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md), [molmospaces.md](molmospaces.md) | Wrapper venv, CLI smoke, optional integration |
| **Multi-robot sim plumbing** | [plans/MULTI_ROBOT_TESTING.md](plans/MULTI_ROBOT_TESTING.md) | Registry, MJCF load, GenericZmqClient (partially superseded by unit tests) |
| **CLI / install smoke** | [cli.md](cli.md#testing), [simulation.md](simulation.md#2-test-the-setup) | `emet test`, serve smoke |
| **SQA3D + ScanNet EQA** | [sqa3d.md](sqa3d.md) | Situated QA loaders, EM@1 eval, Open3D mesh replay |
| **Agent / LLM** | [AGENT_RUN.md](AGENT_RUN.md#testing), [llm_agent.md](llm_agent.md#testing-the-llm-agent) | Agent loop, component tests |
| **Refactor logs (not test specs)** | [logs/README.md](logs/README.md) | Historical change notes |

There is **no other single “master” file** today; this page is the hub. Feature-specific docs link back here where useful.

---

## Automated test map (by validation goal)

### Geometry / navigation / floor coverage

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Explored floor vs spawner walkable (3 robots, Robocasa) | [run_dynagraph_multi_robot_e2e.py](../src/test/app/run_dynagraph_multi_robot_e2e.py) | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md); **does not** assert graph nodes or EQA |
| Nav world ↔ session frame | [test_nav_xyt_session.py](../src/test/utils/test_nav_xyt_session.py) | Unit |
| **rotate_in_place** world (x,y) stays at spawn (innate_mars + Robocasa) | [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) | Sim; asserts `spawn_compose` goals, not `nav_world` |
| Robosuite nav frame unit (spawn compose, jump guard) | [test_robosuite_nav_world_clamp.py](../src/test/simulation/test_robosuite_nav_world_clamp.py) | Unit |
| Floor metrics export | [test_floor_metrics.py](../src/test/memory/test_floor_metrics.py) | Unit |
| Red cylinder localize (DynaMem, default scene) | [test_red_cylinder_in_sim.py](../src/test/mapping/test_red_cylinder_in_sim.py) | Sim; stretch + innate_mars |
| Robocasa memory after spin | [test_robocasa_memory_after_spin.py](../src/test/simulation/test_robocasa_memory_after_spin.py) | Sim |

### Scene graph / object labels (model-derived)

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| **OpenVocab scene graph** in Robocasa (dedup, kitchen labels) | [test_scene_graph_robocasa.py](../src/test/scene_graph/test_scene_graph_robocasa.py) | Uses `SceneGraphProcessor` + `cpu_scene_graph`; **not** GraphEQAMemory / Dynagraph path |
| Open-vocab unit behaviour | [test_open_vocab_scene_graph.py](../src/test/scene_graph/test_scene_graph_in_sim.py) | Default scene spin |
| GraphEQA memory unit (merge, staleness, query) | [test_graph_eqa_memory.py](../src/test/memory/test_graph_eqa_memory.py) | Mock EQA clients |
| GraphEQA + default scene (red/blue) | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Sim RGB + **injected** graph labels; mocked `query_answer` |
| Per-frame detections on export | Dynagraph `--export` → `graph/frames/detections_*.json` | Manual review; see [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality) |

### EQA / answering questions

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Mocked EQA on constructed graph | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Known scene; labels not from full VL pipeline |
| Interactive Robocasa EQA | Manual — [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Requires API keys / models |
| **Dynagraph graph build + NL answer on known scene** | [`test_dynagraph_benchmark_smoke.py`](../src/test/app/test_dynagraph_benchmark_smoke.py) (unit); sim harness [`run_dynagraph_benchmark_smoke.py`](../src/test/app/run_dynagraph_benchmark_smoke.py) | Question bank + `eqa_results.json`; full sim via `RUN_DYNAGRAPH_BENCHMARK_SMOKE=1` |

### Infrastructure / CLI / controllers

| Validates | Test / harness |
|-----------|----------------|
| `emet` CLI, `run dynagraph` help | [test_cli.py](../src/test/cli/test_cli.py) |
| Controller smoke (SVM, DynaMem, GraphEQA, Dynagraph imports) | [test_controller_smoke.py](../src/test/controller/test_controller_smoke.py) |
| Multi-robot registry / MJCF | [test_multi_robot.py](../src/test/simulation/test_multi_robot.py) |
| Dynagraph explore loop helper | [test_dynagraph_explore.py](../src/test/app/test_dynagraph_explore.py) |

---

## Known gap: graph + EQA on a known scene (Dynagraph)

**Current state:** The multi-robot Dynagraph E2E proves **exploration geometry** (explored m², spawner parity). Explore-only exports often show **`Nodes (0)`** in `scene_graph_report.txt` even when **`graph/frames/detections_*.json`** contains model labels (range hood, counter, etc.). There is **no CI test** yet that:

1. Runs **Dynagraph** (or GraphEQA) on a **fixed known scene** (default red/blue table or Robocasa seed 0),
2. Asserts **`GraphEQAMemory`** (or export) has **≥ N nodes** whose **primary labels** come from the **detection / description models** (not hand-injected),
3. Runs **`--question`** (or `query_answer`) about **ground-truth objects** and checks the answer / confidence.

**Related but different paths:**

- **`test_scene_graph_robocasa`** — strong on **OpenVocabSceneGraph** labels and dedup; different stack than Dynagraph’s `GraphEQAMemory`.
- **`test_graph_eqa_default_scene_sim`** — sim + graph string checks; labels are **added manually** to memory, not wired through Dynagraph’s full perception → graph hook.

### Proposed next step (recommended)

Add **`test_dynagraph_graph_eqa_known_scene.py`** (or extend the E2E harness) that:

| Step | Detail |
|------|--------|
| Scene | Default MuJoCo table (**red cylinder**, **blue cube**) and/or Robocasa **seed 0** with documented fixture objects |
| Run | `uv run emet run dynagraph --explore-loop … --question "What colors are the objects on the table?" --export /tmp/…` (or pytest driving the same APIs) |
| Assert graph | `scene_graph_report.txt` or `GraphEQAMemory.to_string()`: **≥ 1 node**; labels match allowlist (e.g. contains *red*, *blue*, or kitchen nouns from detections JSON) |
| Assert EQA | Answer text contains expected tokens; optional mocked EQA client for CI stability (pattern from `test_graph_eqa_default_scene_sim`) |
| Docs | Link from [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) and this page |

Until that exists, use **manual** `--question` runs (documented in [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality)) and inspect **`detections_*.json`** + **`scene_graph_report.txt`**.

---

## See also

- [Dynagraph](dynagraph.md) — CLI, explore loop, export layout
- [GraphEQA](graph_eqa.md) — graph memory design and manual Robocasa testing
- [plans/README.md](plans/README.md) — design plans (architecture, GraphEQA plan, mapping refactor)
