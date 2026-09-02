# OVMM find-phase benchmark (FindObj / FindRec)

Memory ablation benchmark inspired by [OVMM](https://ovmm.github.io/) find phases. We score **FindObj** and **FindRec** localization against MuJoCo `sim_object_placements` ground truth—not full pick/place manipulation success. For the four-phase harness (Pick/Place), see [ovmm_full_benchmark.md](ovmm_full_benchmark.md).

## Scene tier ladder

| Tier | Sim config | Scale |
|------|------------|-------|
| **S0** | `configs/sim/default_table_stretch.yaml` | Default table, ~4 GT bodies |
| **S1** | `configs/sim/robocasa_pick_place.yaml` | Robocasa kitchen, ~20–40 bodies |
| **S2** | `configs/sim/molmospaces_ithor_train_{0,1,2}.yaml` | MolmoSpaces iTHOR multi-room (GT scan may cap) |

Episodes are listed in `configs/ovmm/find_phase_episodes.yaml`.
Paths and smoke defaults: `configs/ovmm/benchmark.yaml` (outputs under `~/runs/emet/…`, caches under `~/.cache/…`).

## Assets + smoke (multi-agent friendly)

```bash
# Verify / fetch CSVs; check HM3D scenes for habitat episodes; create ~/runs dirs
uv run python scripts/download_ovmm_benchmark_assets.py

# Full smoke: unit tests + S0 sim GT + one Habitat GT episode (~5–10 min)
uv run python scripts/smoke_ovmm_benchmark.py --cpu-only
```

Override output location: `EMET_OVMM_OUTPUT_SIM=~/runs/emet/ovmm_find_phase` (or edit `benchmark.yaml`).

## Quick start (S0)

Prefer the first-class CLI: **`emet ovmm find`** (scripts remain thin wrappers).

```bash
# Unit tests (no sim)
uv run emet test src/test/memory/test_ovmm_find_phase_metrics.py -q

# S0 default table, all backends (~2–5 min with GPU)
uv run emet ovmm find \
  --episodes configs/ovmm/find_phase_episodes.yaml \
  --tier S0 \
  --backend dynamem --backend static_graph --backend dynagraph \
  --cpu-only \
  --output-dir runs/ovmm_find_phase/s0
```

Integration gate (optional CI):

```bash
RUN_OVMM_FIND_TESTS=1 uv run emet test src/test/memory/test_ovmm_find_phase_integration.py -q
```

## Batch runner

```bash
uv run emet ovmm find \
  --episodes configs/ovmm/find_phase_episodes.yaml \
  --backend dynagraph \
  --tier S1 \
  --output-dir runs/ovmm_find_phase/s1_dynagraph
```

### Multi-env paper sweep (Robocasa + MolmoSpaces)

No `default_table`. Preset: `configs/ovmm/sweeps/molmo_robocasa.yaml`.

```bash
uv run emet ovmm sweep --preset molmo-robocasa --backend dynagraph --via-jobs
# or stepwise: prepare → find → full → rates / status
uv run emet ovmm prepare --preset molmo-robocasa --out ~/runs/emet/ovmm_molmo_robocasa/DATE
uv run emet ovmm rates --out ~/runs/emet/ovmm_molmo_robocasa/DATE
```

See [cli.md](cli.md#emet-ovmm-subcommand) and [paper_benchmarks.md](paper_benchmarks.md).

### Memory backends

| Backend | Role |
|---------|------|
| `dynamem` | Voxel semantic memory only (no graph) |
| `static_graph` | Object graph, merge/staleness off (alias `graph_eqa`) |
| `dynagraph` | Instance graph + merge/staleness (`merge_xy_m=0.15` default in find-phase) |
| `ground_truth` | Oracle upper bound (graph from sim GT) |

### Benchmark modes (default vs optional)

**Default (fair ablation):** instance-graph mapping only, **no per-frame VLM**, voxel-first query.

| Setting | Default | CLI override |
|---------|---------|--------------|
| `use_sensor_perception` | `false` | `--sensor-perception` (full GraphEQA; ~10× slower on S0) |
| `prefer_voxel` | `true` | `--graph-query` (graph-first localization) |

Dynagraph/static_graph without `--sensor-perception` still build graph nodes from YOLO instance detections (same voxel frames as dynamem) but skip Qwen3-VL label extraction every `update()`. That is why older GPU runs showed ~30–40 min dynagraph vs ~6 min dynamem.

### Scaling / ablation flags

```bash
# Mapping coverage budget (episode YAML ``mapping_max_nav_steps``; ``explore_steps`` is a deprecated alias)
uv run python scripts/eval_ovmm_find_phases.py --episode-id molmo_ithor_s2_idx0_explore15

# Merge / staleness grid on S2
uv run python scripts/eval_ovmm_find_phases.py --tier S2 --backend dynagraph \
  --merge-xy-m 0 --staleness-horizon 0

uv run python scripts/eval_ovmm_find_phases.py --tier S2 --backend dynagraph \
  --merge-xy-m 0.75 --staleness-horizon 512
```

Outputs per run: `runs/ovmm_find_phase/<episode_id>_<backend>.json` plus `aggregate_<backends>.csv`.

### Teleport sim (routine agentic regression)

**Default robot for agentic find iteration is rby1** (wide FOV; no Stretch head-sweep
tax). Use `scripts/run_ovmm_find_recep_slice.sh` (`PROFILE=smoke` or `slice`).

For fast dynagraph agentic find regression, teleport is opt-in
(`EMET_SIM_NAV_TELEPORT=1`). **Do not** pass `--not-rotate` on perception
backends — table mapping uses `--mapping-rotate-steps 4`. The find-phase
harness already enables `_fast_explore_lookaround`. Head pans are off by
default (`mapping.look_around_head_sweep: false`); `PROFILE=stretch` /
`stretch-kitchen` also set `EMET_SKIP_HEAD_SWEEP=1`. Paper pans:
`PROFILE=stretch-legacy` (`EMET_FORCE_HEAD_SWEEP=1`).

Stretch episodes in `find_phase_episodes.yaml`: no-pan table/kitchen gates
(`PROFILE=stretch`, `stretch-kitchen`) or paper pans (`stretch-legacy`). See
[experiments/ovmm_agentic_find_teleport.md](experiments/ovmm_agentic_find_teleport.md).

### Metrics

- `find_object_success`, `find_recep_success` @ `success_radius_m`
- `find_partial_success` = mean of the two (OVMM-style 2-phase partial)
- `localization_err_obj_m`, `localization_err_recep_m`
- `pred_obj_xyz`, `pred_recep_xyz` — predicted world XYZ (MuJoCo world or Habitat Y-up) for audit
- `obj_max_cosine`, `recep_max_cosine`, `obj_yoloe_hit`, `recep_yoloe_hit` — oneshot voxel localize diagnostics (max SigLIP cosine; YoloE `compute_obj_coord` hit). Habitat and sim oneshot both emit these via `ovmm_find_query_row`.
- `obj_localize_source`, `recep_localize_source` — winning query path (`voxel`, `graph_near_recep`, `memory_localize_text_graph`, …; `null` on miss)
- `seed` — RNG seed when set via replicate runner or `FindPhaseRunConfig.seed`
- `use_sensor_perception`, `prefer_voxel` — mode flags (see above)
- `init_wall_s`, `mapping_wall_s`, `query_wall_s`, `episode_wall_s` — timing breakdown
- Scaling: `n_graph_nodes`, `n_voxel_explored_cells`, `n_voxel_explored_area_m2`, `n_placements`
- Optional GT diagnostics: `gt_graph_completeness`, `instance_gt_association_recall` (GT-oracle graph only)

### Multi-seed replication (variability audit)

Perception backends are non-deterministic; use multiple seeds and inspect `pred_*_xyz`:

```bash
uv run python scripts/replicate_ovmm_find_phases.py \
  --episode-id default_table_s0 \
  --backend dynagraph \
  --replicates 5 \
  --seed-base 0 \
  --cpu-only \
  --output-dir ~/runs/emet/ovmm_find_phase/s0_audit
```

Writes `seed_<n>/<episode>_<backend>.json` plus `aggregate_replicates.json` (mean/std per metric).
Each replicate uses `port_offset = port_offset_base + seed * 2` to avoid ZMQ port clashes.

## Measured results (emet sim)

Fair-default GPU replicate (`default_table_s0`, 5 seeds, no VLM, voxel-first query):

| Backend | mapping_wall_s (mean±std) | partial | obj err (m) | localize source |
|---------|---------------------------|---------|-------------|-----------------|
| dynagraph | 213 ± 4 s | 1.0 | 0.088 ± 0.001 | obj/recep: voxel (5/5) |
| dynamem | 227 ± 36 s | 1.0 | 0.079 ± 0.0005 | obj/recep: voxel (4/5; seed 1 init flake) |

Dynagraph/dynamem mapping ratio ≈ **1×** (not 10×). Full `--sensor-perception` mapping ≈ **2261 s** (~10× fair default).

Target reference (real OVMM paper): ~70% FindObj / ~30% FindRec — not comparable to this memory-localization harness.

**Mapping vs find budgets (do not conflate).** `mapping_max_nav_steps` (CLI `--mapping-max-nav-steps`; deprecated alias `explore_steps` / `--explore-steps`) is the **mapping-phase** agentic `max_nav_steps` — how many coverage journeys `run_mapping_protocol` may run before FindObj. `0` is rotate-only (S0). FindObj/FindRec use `--agentic-max-rounds` / `--agentic-max-nav-steps`. After hop-until-arrival, one mapping step is one completed path, not one leftover A* chunk.

**Mapping = same AgenticEQAExecutor as EQA, but coverage-only.** When `mapping_max_nav_steps>0` the harness runs `run_agentic_eqa_result(agent, None, goal="explore and map the environment", max_nav_steps=mapping_max_nav_steps, max_rounds=mapping_max_nav_steps+1)` — `mode=explore` (router only `explore_frontier` / `finish`). Frontier picks are **uncovered-first / VLM among frontier RGBs**, not object-biased `toward=jar`; OVMM objects are placed randomly and biasing toward a SigLIP ghost while mapping wastes steps. Arrival capture is `look_ahead` (tilt 0) facing the frontier then `update()` — not `look_front` −30° and not a 4-pan sweep before leaving. `S0` (`mapping_max_nav_steps=0`) stays rotate-only plus `_prepare_default_table_rby1_mapping_view`.

**Find = voxel-first, then AgenticEQA.** At find time `localize_text("jar")` / `"cab"` is run on the **finished** voxel map and an `investigate` at that XYZ beats any camera-pose-at-feet graph view (redirect `CAMERA_POSE_PLACE` → unused detection). SigLIP on arrival RGB is the query — YOLOE need not know `jar`. If there is no voxel hit, one `explore_frontier` extends coverage rather than chewing 150 wall nodes. Context that survives between phases is `agent.voxel_map` + `agent.graph_memory` (find starts a new executor; the map is the agent state).

**Default find path (dynagraph): same AgenticEQA loop as HM-EQA.**
Episode fields are phrased as questions (`Where is the jar on the counter?` / `Where is the cab?`) and run through [`AgenticEQAExecutor`](../src/emet/memory/graph_eqa/agentic_eqa.py). The agent may call `inspect_graph` → live `localize_text` as an investigate card; close-map stays on that XY; VLM verify is a check. **FindObj/FindRec score the loop's object-phrase voxel XYZ**, never camera pose, and never a harness pin of episode YAML. Voxel proposals are **one-shot** (a close ABSENT blocks the handle and unpins that XYZ, so the loop re-localizes from the grown map instead of re-chasing a wall point), and the harness **re-attaches SigLIP before each find phase** (`re_attach_siglip_encoder`) so FindRec can still `localize_text` the finished map after FindObj released the encoder for Qwen. `--oneshot-localize` / `agentic_find: false` is a leftover **mapping ablation**, not the product path and not the map-sanity check (that is pytest `test_red_cylinder_detected_in_sim`). Method: [dynagraph.md](dynagraph.md#method). Preset: `agentic_find: true` in `configs/ovmm/sweeps/molmo_robocasa.yaml`. See [plans/2026-08-26_ovmm_voxel_close_map.md](plans/2026-08-26_ovmm_voxel_close_map.md).

**Query / scoring notes (agent language, no GT query leakage):**
- The agent localizes with **episode task language** (`object` / `goal_recep`), not sim GT fixture paths.
- `_query_variants` is language-side only (does not inject placement `cat` strings into `localize_text`).
- `resolve_object_query` keeps usable episode labels (e.g. `jar`); GT `cat` is only a fallback when the label is a stub (`obj`).
- `object_gt_body` remains for **scoring** FindObj against the intended body.
- FindRec scores one disambiguated GT body (prefer exact / short labels), not min-distance over every substring match.
- Phases with no resolvable GT body are **unscored** (`find_*_scored=false`) and do not count as localization misses in partial success.

**Reading the results — rooms help receptacles, not objects.** Room context (graph room tags / `current_room`) is a strong disambiguator for **FindRec**: receptacles are room-typical fixtures (`table` vs `counter` vs `cab`), so the room signal can localize or reject a candidate receptacle. It is **not** a useful cue for **FindObj**: a target object (`jar`, `red cylinder`) can sit on any receptacle in any room, so a room match (or mismatch) carries almost no evidence about where the object is. When triaging a miss:
- FindRec miss + wrong-room candidate ⇒ look at room-cluster recall / room-mismatch escape floor (recep is room-correlated).
- FindObj miss ⇒ the object's **receptacle / visual presence** is the signal; room alone should not be trusted to drive the search. An agent that over-trusts rooms may strand itself in a "matching" room without the object.

Fair-default verification (GPU, one job at a time):

```bash
uv run python scripts/replicate_ovmm_find_phases.py \
  --episode-id default_table_s0 \
  --backend dynamem --backend dynagraph \
  --replicates 5 --seed-base 0 \
  --output-dir ~/runs/emet/ovmm_find_phase/fair_default
```

Optional ablations:

```bash
# Graph-first query (no voxel shortcut)
uv run python scripts/eval_ovmm_find_phases.py --episode-id default_table_s0 \
  --backend dynagraph --graph-query \
  --output-dir ~/runs/emet/ovmm_find_phase/graph_query

# Full GraphEQA perception (per-frame VLM; legacy slow column)
uv run python scripts/eval_ovmm_find_phases.py --episode-id default_table_s0 \
  --backend dynagraph --sensor-perception \
  --output-dir ~/runs/emet/ovmm_find_phase/full_perception
```

**Do not** pass `--not-rotate` for perception backends; mapping requires rotate-in-place.

GT oracle smoke:

```bash
uv run python scripts/eval_ovmm_find_phases.py --tier S0 \
  --backend ground_truth --not-rotate --cpu-only \
  --output-dir ~/runs/emet/ovmm_find_phase/s0_gt
```

## S1 / S2 notes

- **S1** requires sim install (`emet install sim`) and Robocasa kitchen assets.
- **S2** requires MolmoSpaces wrapper (`.venv-molmospaces`; see `docs/molmospaces.md`). Large scenes may return capped `sim_object_placements`; compare `n_placements` across Molmo indices.
- Use distinct `--port-offset` values when running parallel jobs.

## Phase 2: Habitat find-phase (HM3D proxy)

Episodes: `configs/ovmm/habitat_find_phase_episodes.yaml` (HM3D train scenes + semantic GT).
Same FindObj/FindRec metrics with **XZ** horizontal scoring (`frame: habitat_yup`).

```bash
./scripts/install_habitat.sh
uv run python scripts/download_habitat_eqa_data.py --fetch-csv --fetch-hm3d train
# Optional: HM3D semantic meshes (if scenes lack semantics)
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d-semantics train

# Batch GT smoke (3 HM3D scenes, ~7 min CPU; agentic stays off for ground_truth)
uv run python scripts/eval_habitat_ovmm_find_phases.py \
  --backend ground_truth --not-rotate --device cpu \
  --output-dir runs/ovmm_habitat/gt_batch

# Single dynagraph episode (GPU SigLIP + agentic VLM loop; default --device cuda)
.venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 --backend dynagraph --device cuda
```

Verified GT batch: `find_partial_success=1.0`, `localization_err_*_m=0.0` on
`hm3d_lamp_bed_00006`, `00025`, `00057` (June 2026).

### Agentic find on Habitat (large scenes)

The default dynagraph/static_graph path routes FindObj / FindRec through the
**shared query dispatcher** (`emet.eval.ovmm_agentic_find.run_ovmm_find_queries`)
used by sim OVMM find. Agentic backends call
`run_ovmm_agentic_localize` (same AgenticEQA loop as HM-EQA); `--no-agentic-find`
is the one-shot memory-localize ablation. Habitat **does not** use the sim
MuJoCo placement oracle (`run_ovmm_gt_oracle_find_pair`) — Habitat `ground_truth`
localizes from the GT graph after `refresh_ground_truth`. Scoring and JSON keys
(`ovmm_find_query_row`, `score_ovmm_find_query`) are shared; Habitat passes
`frame="habitat_xz"` / `planar_frame="habitat_xz"`.

The loop phrases the episode as two open questions (`Where is the lamp on the bed?` /
`Where is the table?`), then navigates to navmesh frontiers, investigates place
cards, and verifies before the loop's object-phrase voxel XYZ is scored. Mapping
only rotates in place, so exploration happens inside the loop.

```bash
# Bounded GPU smoke on the large scene 00006 (VLM + SigLIP + YoloE required)
bash scripts/smoke_habitat_ovmm_agentic_find.sh

# Full-budget agentic find (default eqa.agentic_max_tool_rounds / max_nav_steps = 8)
uv run emet jobs run --name habitat-ovmm-agentic-find --need-mib 12000 --gpu-exclusive -- \
  .venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 --backend dynagraph --device cuda --agentic-find

# Batch
uv run python scripts/eval_habitat_ovmm_find_phases.py \
  --backend dynagraph --device cuda \
  --output-dir ~/runs/emet/ovmm_habitat/agentic
```

Flags: `--device cuda|cpu` (default **cuda**), `--agentic-find/--no-agentic-find`
(default on for dynagraph/static_graph, off for dynamem/ground_truth),
`--agentic-max-rounds`, `--agentic-max-nav-steps`. `--cpu-only` is an alias for
`--device cpu` and does **not** turn off the agentic loop. `--no-agentic-find`
restores the one-shot memory-localize ablation.

Full OVMM-HSSD minival (official leaderboard) is not wired yet; HM3D proxy validates the Habitat
memory → find-phase metric path before HSSD scene download.

## Paper

- Unified runbook: [paper_benchmarks.md](paper_benchmarks.md)
- Experiments plan: `paper/sections/04_experiments.tex` (`sec:ovmm_find_phase`, Table `tab:benchmark_configs`)
- Results tables: `paper/sections/05_results.tex` (`tab:ovmm_find_backend_tier`, scaling figures)
