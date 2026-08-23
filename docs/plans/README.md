# Plans

Design and refactor plan documents live here (`docs/plans/`).

## Paper / eval status (2026-07)

Operator index: [experiments/README.md](../experiments/README.md) · LaTeX: `paper/sections/04_experiments.tex`, `05_results.tex` · Runbook: [paper_benchmarks.md](../paper_benchmarks.md)

| Doc | Status | Next eval / milestone |
|-----|--------|------------------------|
| [HABITAT_EQA_HARNESS.md](HABITAT_EQA_HARNESS.md) | **HM-EQA implemented** (phases A–G); OpenEQA not wired | OpenEQA harness milestone (phase H) |
| [2026-06-03_habitat_eqa_exploration_improvements.md](2026-06-03_habitat_eqa_exploration_improvements.md) | P0–P2 done in code; P3 partial; P4 scripts ready | Frontier ablation Q0–19; full 113 @ 8B post-nav-fix |
| [fable5-dynagraph-habitat.md](fable5-dynagraph-habitat.md) | Historical (debias, bake-off, balanced-32 @ 3B) | Re-run balanced-32 @ Qwen3-VL-8B after July nav fixes |
| `paper/sections/*` | Habitat preliminary slices only; other tables placeholders | See [habitat_eqa_results.md](../experiments/habitat_eqa_results.md) priority queue |

**July 2026 nav stack** (landed on `feature/eval-tools`): Image-N viewpoint/standoff waypoints, navmesh path following, `already_at_goal` blocking, frontier distance sort, eval diagnostics (overlay maps, substep MP4). Tag JSONL before this date as **pre-nav-fix** when comparing.

---

## Plan index

- **[ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md)** – Multi-robot, multi-simulator refactor (emet rename, robots/simulators abstraction).
- **[GRAPH_EQA_PLAN.md](GRAPH_EQA_PLAN.md)** – Plan for adding GraphEQA as a graph-based EQA memory model.
- **[MAPPING_REFACTOR.md](MAPPING_REFACTOR.md)** – Mapping module layout, instance/memory split, and shared UI.
- **[2025-03-10_molmospaces_testing.md](2025-03-10_molmospaces_testing.md)** – Testing plan for MolmoSpaces integration (CLI, runner venv, scenes, serve).
- **[innate-mars-development.md](innate-mars-development.md)** – Innate Mars bridge, sim experiments, hardware bring-up; where to resume.
- **[2026-06-03_habitat_eqa_exploration_improvements.md](2026-06-03_habitat_eqa_exploration_improvements.md)** – HM-EQA exploration/prompting improvements (frontier map vs nodes, navigation fix, ablation plan).
- **[2026-07-15_dynagraph_graph_quality_dynamic_eqa.md](2026-07-15_dynagraph_graph_quality_dynamic_eqa.md)** – Graph-health metrics, prompt top-K, label dedup; dynamic EQA deferred until quality gates pass.
- **[fable5-dynagraph-habitat.md](fable5-dynagraph-habitat.md)** – Dynagraph HM-EQA results, MCQ debiasing, VLM bake-off findings (links to [vlm_bakeoff.md](../habitat/vlm_bakeoff.md)).
- **[HMEQA_STRATEGY.md](HMEQA_STRATEGY.md)** – Post-#114 HM-EQA plan: verify-gate regression check (A), joint agentic loop across tasks (B), room ladder coordination (C), accuracy levers / full-113 / 32B (D). Branch `feat/hmeqa-strategy`.
- **[2026-08-08_embodied_agent_planning.md](2026-08-08_embodied_agent_planning.md)** – World model + tool calling + motion (design + phase checklist; branch `feature/agent-world-model`). **Shipped reference:** [../attempt_ledger.md](../attempt_ledger.md).
- **[2026-08-22_tamp_agent_tools.md](2026-08-22_tamp_agent_tools.md)** – Semantic CHAT TAMP tools, simulator grounding boundary, guarded execution, and acceptance criteria.
