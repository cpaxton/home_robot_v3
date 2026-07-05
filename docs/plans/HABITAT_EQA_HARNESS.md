# Habitat EQA harness plan

Engineering plan for reproducing GraphEQA-style embodied question answering evaluation in Habitat-Sim, driving **emet** `GraphEQAMemory` / `DynagraphController` instead of the original GraphEQA ROS/Hydra stack.

**Status:** **HM-EQA implemented** — phases A–G below are done (`src/emet/habitat/`, `packages/emet_habitat/`, `emet run graph-eqa-habitat`, batch JSONL, diagnostics). Preliminary paper numbers exist on held-out slices; **full 113-question sweep** and **OpenEQA** remain open.

**User docs:** [docs/habitat/README.md](../habitat/README.md) (install, data, Matterport API tokens, troubleshooting, [navigation](../habitat/usage.md#navigation-habitat-only)).

**Paper dependency:** Headline HM-EQA table in `paper/sections/05_results.tex` pending full 113-question post–July-2026 nav-fix sweep. Recorded numbers: [docs/experiments/habitat_eqa_results.md](../experiments/habitat_eqa_results.md).

**Branch:** Harness merged to `main`; active eval work on feature branches (e.g. `feature/eval-tools`).

---

## Goal

Same datasets and metrics as [GraphEQA](https://github.com/SaumyaSaxena/graph_eqa), but observations and actions flow through emet's memory and EQA executors:

- **HM-EQA:** 113 HM3D questions, scene init poses
- **OpenEQA:** Meta dataset subset per GraphEQA paper split

Methods to compare:

| Method | Config |
|--------|--------|
| GraphEQA baseline | `dynagraph_merge_xy_m=0`, `dynagraph_staleness_horizon=0` |
| Dynagraph | Default merge + staleness |
| Ablations | Per paper `04_experiments.tex` |

---

## Reference implementation

Study [SaumyaSaxena/graph_eqa](https://github.com/SaumyaSaxena/graph_eqa):

- `scripts/run_vlm_planner_hmeqa_habitat.py` / `grapheqa_hmeqa_habitat` config
- `scripts/run_vlm_planner_openeqa_habitat.py` / `grapheqa_openeqa_habitat`
- HM-EQA: 113 HM3D questions; `scene_init_poses`
- OpenEQA: Meta dataset download
- Docker split: Habitat-Sim image vs Stretch embodied image

---

## Architecture

```mermaid
flowchart LR
  subgraph habitat [Habitat stack]
    HS[habitat-sim HM3D scenes]
    DS[HM-EQA / OpenEQA loaders]
  end
  subgraph bridge [Emet bridge]
    Obs[RGB-D + pose + semantics adapter]
    RobotIface[RobotClient protocol shim]
  end
  subgraph emet [Emet memory]
    Vox[SparseVoxelMap]
    Graph[GraphEQAMemory / Dynagraph]
    EQA[EQAExecuter query_answer]
  end
  HS --> Obs
  DS --> EQA
  Obs --> Vox
  Obs --> Graph
  Graph --> EQA
```

---

## Engineering phases

| Phase | Work | Notes |
|-------|------|-------|
| **A. Environment** | `.venv-habitat` or `uv` extra `habitat`; pin `habitat-sim` / `habitat-lab` compatible with HM3D | Likely **separate venv** (like MolmoSpaces) due to numpy/CUDA conflicts with main `.venv` |
| **B. Assets** | HM3D scenes, HM-EQA JSON, OpenEQA subset, `scene_init_poses` | Document download scripts; mirror graph_eqa layout under `data/habitat_eqa/` |
| **C. Observation bridge** | Map Habitat agent sensors → emet `Observations` (rgb, depth, gps/compass or SE(3) pose, optional semantic) | [`src/emet/habitat/coordinates.py`](../../src/emet/habitat/coordinates.py): Habitat Y-up → voxel `(X, Z, Y-floor_y)`; OpenGL sensor → OpenCV cam pose |
| **D. Control bridge** | Map DynaMem nav actions (xyt, rotate_in_place, look_at) → Habitat velocity / discrete actions | Stretch agent in Habitat or locobot-style base |
| **E. Batch runner** | `emet run graph-eqa-habitat` or `scripts/run_hmeqa_batch.py` | Config flags: `--method graph_eqa\|dynagraph`, `--dataset hmeqa\|openeqa` |
| **F. Metrics** | Success rate, mean planning steps, optional LLM grader for OpenEQA | Match GraphEQA paper tables for direct comparison |
| **G. CI** | Smoke: 1 scene, 1 question, mocked mLLM | Full eval GPU-only, not default CI |
| **H. OpenEQA** | Meta OpenEQA val subset loader + grader + `dataset=openeqa` CLI | **Not started** — paper goal; parity appendix row still Missing |

### Phase H — OpenEQA (planned)

1. Download / layout OpenEQA subset per GraphEQA paper split (see upstream `grapheqa_openeqa_habitat` config).
2. Add `dataset=openeqa` to `packages/emet_habitat/emet_habitat/cli.py` and loader under `src/emet/habitat/`.
3. Grading: LLM grader or normalized open-answer protocol (not HM-EQA MCQ).
4. Smoke: 1 scene, 1 question, `--mock-llm`; then small val slice before paper table.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| numpy / mujoco / habitat version hell | Isolated `.venv-habitat`; `install.sh --habitat` profile |
| Stretch-in-Habitat vs locobot agent mismatch | Start with same agent class GraphEQA paper used; document embodiment |
| Long HM-EQA runtime | Parallelize episodes; cache scene loads |
| Semantic segmentation differences vs original Hydra graphs | Acknowledge in paper; emet uses `SensorGraphBuilder` not Hydra—compare on EQA outcome metrics |

---

## Timeline relative to paper

1. **Parallel track 1:** `paper/` LaTeX scaffold on `paper/dynagraph-corl`.
2. **Parallel track 2:** This harness (phases A–G done; H OpenEQA pending) — **blocks full Habitat results table**.
3. **Parallel track 3:** Emet-EQA question banks for Robocasa/Molmo.
4. **Integrate:** Habitat + emet sim results into `05_results.tex`.

---

## Verification (when implemented)

```bash
# Smoke (mocked LLM)
RUN_HABITAT_TESTS=1 uv run emet test src/test/habitat/ -k smoke

# Single episode (confident mock — fast grading smoke)
uv run emet run graph-eqa-habitat --dataset hmeqa --scene-id 0 --question-id 0 --mock-llm

# Movement smoke (mock returns confidence:false each planning step)
.venv-habitat/bin/emet-habitat run-episode --question-id 3 --mock-llm --mock-llm-explore --max-planning-steps 5
```
