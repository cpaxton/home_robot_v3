# Navigation and exploration rework

## Contract and scope

One harness for exploration, find, EQA and manipulation: select a useful viewing
pose, validate and execute its route, verify measured arrival, then score the
fresh observation separately. Keep A* and the existing server ArrivalMonitor.
Acceptance requires Stretch and a separately identified second robot in simulation.
The current `rby1` backend is a Galaxea R1 proxy, including MolmoSpaces scenes;
native RB-Y1 acceptance needs a separate adapter and is not claimed here.
No hardware or full sweeps;
heavy jobs run serially with CPU-safe/GPU-exclusive scheduling.

Baseline: `faebea0f`, including September 27 navigation-only pilot evidence in
[floor coverage recovery](../experiments/floor_coverage_recovery.md). Find verified
a fresh tomato view after an approximately 27 cm approach; exploration produced
a turn-only route and correctly failed for insufficient translation. Neither
establishes manipulation acceptance. Freeze model, perception, physics and task
budgets for paired comparisons. The sibling physical-TAMP work owns grasp/carry
repairs; selectively integrate reviewed navigation fixes, never edit its worktree.

## Reviewable implementation sequence

1. **Geometry.** Use planning-frame poses for ranking. Match physical-map cell
   indexing to nearest-center insertion/footprint geometry, preserve continuous
   endpoints including same-cell short moves, and refuse off-map snapping.
   Keep legacy map conventions unchanged pending separate migration. Record
   requested/resolved endpoints and snapping reasons. Tests must reproduce the
   recorded start (-1.0015765,-0.2790600) and goal (-1.1000000,-0.3000000).
2. **Contract.** Extend existing plan/attempt results with requested and effective
   XY/yaw, optional head pose, purpose, map revision, measured arrival errors,
   settling and terminal reason. Replace NaN/object-XYZ trajectory tails with
   explicit target metadata; temporary compatibility parsing belongs at one
   boundary. Preserve server-owned exploration (0.07 m/0.15 rad) and precision
   (0.02 m/0.03 rad) policies, freshness and bounded corrections. Distinguish
   unknown clearance from the 10 m fallback sentinel.
3. **Viewing-pose selection.** Retain the bounded eight-candidate frontier set.
   Resolve feasible poses before deduplication/ranking. Use shared search for
   all reachable costs, rather than first-goal selection. Estimate unknown area
   visible within camera FOV with map occlusion, then rank by gain divided by
   path length + footprint radius * absolute yaw change + one grid cell.
   Break ties by travel cost then stable order. Suppress previously unhelpful
   views until relevant evidence changes. Already-satisfied poses are explicit
   observation-only actions, never fictitious translation progress.
4. **Execution and task progress.** Validate yaw and translation, retaining the
   conservative sweep unless an adapter guarantees a specific motion profile.
   Revalidate after material map updates/deviations. Confirm stopping before
   correction; require fresh observation at measured arrival. Report motion,
   observation and task outcomes separately. Count sensor-supported newly
   observed area, not padded map growth. A useful stationary look is progress;
   motion without information is not automatically exploration success.

Keep behavior changes in the named physical navigation pilot until acceptance.
Do not add robot/task-specific recovery branches or lower safety thresholds.
An SE(2)/state-lattice planner is deferred: first collect independently feasible
routes that the repaired XY planner genuinely cannot represent.

## Acceptance

- Deterministic tests: world/local transforms, negative coordinates, boundaries,
  same-cell moves, snapping, resolved deduplication, zero-cost goals, heading wrap,
  blocked rotations, stale feedback, cancellations, duplicate commands and
  false-success prevention. Replay the recorded failure.
- Controller sim: each of Stretch and RBY1, three repetitions each of short
  translation, turn-only, translation plus final heading, feasible narrow
  passage, obstructed rotation, and unknown route followed by stationary sensing
  and guarded replanning. Positive cases require declared arrival/settling;
  negatives require correct safe stops. Unexpected contact, false arrival or
  motion after unconfirmed stop blocks promotion.
- Learned paired pilots: existing small Molmo exploration/find; Molmo/RoboCasa
  OVMM subsets; existing 12-question EQA dev set with two matched seeds per
  version. Exercise TAMP navigation/approach separately from carry/grasp scoring.
  Investigate each newly failing paired case; small totals are not statistical
  proof of non-regression. Simulator truth scores outcomes, not learned actions.
- Retain candidate scores/rejections, requested/planned/executed maps, pose/error
  timelines, command receipts, before/after camera views and independent safety
  measurements. Document results, figures and outstanding TODOs before promotion.

## Implementation checkpoints

- Frontier planning-frame fix: `ca09d8d8`. Geometry and continuous endpoint
  preservation: `6c97f8c3`. Requested/resolved endpoint provenance and honest
  clearance hints: `7c607d5e`. Combined regression suite: 496 passed, two existing
  SWIG warnings. Legacy padded-map indexing is unchanged.
- Geometry-only live Stretch recheck `20260928_215530_fb91dd`, frozen at
  `6c97f8c3`, is complete. It planned two waypoints instead of the old one-point
  turn-only route, but the full sweep rejected one unknown cell during rotation.
  Qwen chose stationary observe_floor (+20 observed cells; blocker unchanged),
  then diagnostics and a safe stop. No navigation/manipulation acceptance.
  Evidence: `~/runs/emet/navigation-geometry-pilot-20260928/explore/`.
- This checkpoint preceded the selection/execution changes below. The two-robot
  controller battery and paired task pilots remain acceptance gates.

### September 29 implementation checkpoint

- `571b20f4`: shared search retains paths to every reached candidate. The
  physical pilot checks executable full routes before ranking calibrated
  horizontal camera sectors by unknown area / travel-and-turn cost. Duplicate
  and already-satisfied views are rejected. The 2D view estimate is a heuristic,
  not proof against vertical occlusion. Known head geometry comes from the
  current observation; this version holds the forward head configuration rather
  than searching additional head poses.
- `af28c77e`: new routes carry semantic targets and completion separately from
  base waypoints. Legacy input tails remain accepted at a compatibility boundary.
  Sensor coverage is measured before visited-disk/morphology expansion, with the
  before snapshot taken after startup scans and immediately before movement.
  Physical exploration reports motion and arrival-view gain separately.
- `f72777f7`: physical segments are checked again from measured poses, with map
  updates between segments and server-owned arrival/settling unchanged. Reject
  newly unsafe segments and ambiguous execution; verify payload between steps.
- `96fbdf44`: stationary views toward existing frontiers plus bounded local-map
  revision history. `fe8e7379` retains suppression across measured drift within
  existing arrival tolerances. Head-pose search remains unimplemented; camera
  estimates hold the measured configuration and use guarded floor recovery.
- `4cc110c4`: existing camera route probe checks fresh measured endpoint dwell,
  including oscillation, independently of success receipts. It explicitly does
  not score continuous collision safety.
- `202ede0e`, `9e7f0864`: publish frame sequences only after decoded observations
  become available, require fresh mapped observations after physical segments,
  and prevent old inferred depth being reprojected at a new camera pose.
- `7bf047de`: preserve every physical exploration step's candidates, map,
  sensor coverage, matching command receipt and post-attempt camera view.
- `af46be53`: measured look-joint feedback replaces generic placeholder zeros.
  No new native robot model or changed actuation. The proxy uses upper torso
  look joints; it is not a head-only mechanism.
- Latest expanded controller/floor-recovery/view/acceptance suite: **618 passed**,
  two existing SWIG warnings. Fixed stale string-result test assertions and
  isolated native control configuration from perception's Hydra singleton.

### Completed diagnostic runs (not promotion gates)

| Frozen source / job | Observed result |
| --- | --- |
| `571b20f4` / `20260929_033047_e29a8f` | All 8 translational frontier routes rejected at the same unknown cell during clockwise turn. Floor view added 16 padded-map cells; blocker remained. Subsequent `find_objects("the unknown footprint area")` is a semantic diversion, not exploration acceptance. |
| `96fbdf44` / `20260929_034837_ce938a` | Three guarded stationary viewing turns completed. Final step added 7 raw sensor cells (0.07 m²); final candidate set included feasible translations. The run stayed near its start: not room-coverage acceptance. |
| `96fbdf44` / `20260929_034840_2f314c` | Tomato find initially rejected unknown clearance; floor recovery added 96 padded-map cells but did not clear that exact footprint. Replan approached roughly 0.30 m and verified tomato from a fresh RGB-D view. Saved RGB manually inspected. Find pass only, no manipulation. |
| `5a1e6279` / `20260929_035052_731d84` | **Galaxea proxy labeled rby1**, not native RB-Y1. Initial footprint had 9 unknown cells; no motion authorized. Floor recovery failed because generic measured look feedback was hardcoded zero. No acceptance. |

Roots: `~/runs/emet/navigation-views-pilot-20260929/`,
`~/runs/emet/navigation-integrated-pilot-20260929/`, and
`~/runs/emet/navigation-multirobot-pilot-20260929/`. The older queued route job
`20260929_033720_b08fa1` was canceled before launch in favor of the integrated
build; it is not a failed scored episode. Some completed Stretch runs still
emit shutdown-manager BrokenPipe errors; process exit alone is not a pass.

Offline replay of the saved selection-only map permits a CCW stationary turn
and a 10 cm forward-then-turn alternative, while direct clockwise rotation
crosses an unknown cell. This replay uses the **padded** saved map conservatively,
not a ground-truth collision proof or live controller pass.

![Conservative saved-map turn replay](../experiments/figures/navigation-turn-replay-20260929.png)

Replay input: `navigation-views-pilot-20260929/explore/evidence/navigation/` +
`floor_observation_1790667836952017845.npz`. Footprint width/length/offset
0.34/0.33/−0.10 m; grid 0.10 m, origin [512,512]. Validate each route using
`validate_navigation_sweep` on the archived padded obstacle/explored arrays.
Do not erode archived obstacles to invent missing physical-map evidence.

![Manually inspected tomato arrival view](../experiments/figures/navigation-tomato-arrival-20260929.png)

Arrival RGB: `navigation-integrated-pilot-20260929/find/evidence/grounding/` +
`grounding-d7c6d420cda244db9df1ad5387e1a145.png`; paired JSON and NPZ retain
VLM selection and calibrated depth. This establishes visibility, not graspability.

### Latest frozen checks

Source `2ad84057`, worktree `/tmp/emet-navigation-acceptance-2ad84057`:

- `20260929_041312_eee432`: Stretch route, one repetition as initial diagnostic,
  existing 14-waypoint precision contract and independent endpoint dwell.
- `20260929_041316_5a73db`: same route on labeled Galaxea/RBY1 proxy in iTHOR.
- `20260929_041321_0ec970`: Stretch exploration with fresh-map guards and
  per-step evidence.
- `20260929_041324_924219`: proxy exploration with measured look feedback.

All scheduled CPU-safe/GPU-exclusive. These are diagnostics, not the full
three-repetition six-case contact/unknown-clearance controller acceptance matrix.

Remaining gates: rerun latest freshness/look-feedback changes, complete the
specified controller matrix with independent contact/stop evidence, then paired
task pilots. Stationary coverage gain must not be mistaken for successful room
traversal. Keep native RB-Y1 support separate from this navigation repair. No
latest EQA/OVMM/complete-TAMP non-regression claim is made here.

### Contact and head-recovery investigation (September 29, continued)

- The `2ad84057` route diagnostics finished: Galaxea proxy **14/14** measured
  endpoint/dwell checks; Stretch **11/14**, then a confirmed stop on command 12
  at 0.058 m XY error. These are not collision-safety passes.
- Fresh-map Stretch exploration passed all three guarded turns with **70, 70,
  11** newly sensor-observed cells (0.70, 0.70, 0.11 m²). It still did not traverse
  the room. This distinction is explicit in compact LLM tool feedback (`4dfc992b`).
- Short coupled-route reproduction `20260929_042327_4f78b0`, same frozen source,
  reproduced the stall. Full qpos/qvel/ctrl reconstruction reveals contact
  between `link_SG3_gripper_body` and the kitchen island in **37 of 233** sampled
  states, from sim time 13.322 to 23.468 s, with reconstructed normal force up to
  **45.56 N**. Wheels remain commanded before the stop. This invalidates the
  nominal positive route; it is not evidence that the controller merely needs
  more time. The earlier base-only static reconstruction missed this contact.
  A proposed simulation-clock stall extension was discarded. Existing wall-clock
  stall/freshness limits and arrival tolerances remain unchanged, with added tests.
- Reproduce the diagnostic with `scripts/replay_navigation_contacts.py --scene
  <resolved scene.xml> --trace <physics.jsonl> --robot-root base_link --allow-geom
  floor --output <new report.json>`. The trace lives under
  `~/runs/emet/navigation-acceptance-20260929/stretch-coupled-trace/`; resolved
  scene is `~/runs/emet/molmo-wheel-contact-20260915/fixture/scene.xml` (SHA256
  `c41295d5f87110363b3ab9c0ce727cbbe55365ac7d8a5d7205643e49c063befc`).
  The tool includes all robot descendants and explicit support-geometry exclusions.
  Other floor-mesh contacts remain listed. This is **sampled reconstructed
  evidence**, not continuous measured collision acceptance or a policy oracle.
- Proxy floor recovery was not just a fixed-angle feedback issue: generic head
  commands acknowledge before measured arrival. `86a05c13` waits for the existing
  pose tolerance and another fresh frame. Retest `20260929_043746_fb5e72` at
  `4f423ec2` reached the requested view and updated the map, then exposed a
  Stretch-only pose call in artifact writing. `4eb1e7f6` fixes that shared-frame
  contract, tests evidence capture, and removes Stretch-index assumptions from
  optional head sweeps. No robot-specific recovery rule added.

Next gate: retain the failed route, establish whole-robot/posture clearance for
positive controller cases and capture continuous contact/stop evidence. Do not
extend timeouts, remove obstacles, or enlarge/shrink padding until the particular
geometry is understood. Only then count controller acceptance and proceed to the
paired task promotion gates. A base-footprint map check is not whole-robot safety.

### Latest frozen regression jobs (`669d74fe`)

- **834 scoped tests passed**, with two existing SWIG warnings.
- Proxy exploration `20260929_044404_00efe3` completed recovery capture, map
  update and artifact writing successfully. Floor observation added 28 padded
  cells, but the nine blocking footprint cells remained unknown. Calibrated
  saved depth shows nearest floor within ±0.12 m of world Z=0 about **0.61 m**
  from the base. That view does not cover the missing near-base region. This
  fixes the adapter bugs, not the coverage deadlock; no navigation pass.
  Evidence: `~/runs/emet/navigation-shared-pose-20260929/rby1/explore/`.
- Existing simple-table fixture: `20260929_044426_fb159d` passed **14/14**
  independent endpoint/dwell checks with unchanged precision limits, including
  coupled position/heading. This is a different diagnostic fixture, not a paired
  replacement for the kitchen failure. Summary correctly says
  `incomplete_telemetry`: continuous contacts and base posture are not certified.
  Sampled reconstruction found two fingertip/table contacts at startup sim time
  0.102 s and no later non-floor contacts in 1,172 saved states. Startup and
  inter-sample safety remain unscored. Root: `~/runs/emet/navigation-table-route-20260929/`.

Proceeding with **independent EQA regression evidence** while physical safety
remains open, not promotion. Four GPU-exclusive/CPU-safe jobs serialize the same
12 questions (`2 6 12 14 15 16 25 28 31 56 65 68`), seeds 0 and 1, local
Qwen3-VL-8B-Instruct int4, lazy_graph + query-driven memory, 20 planning steps and
10 movement steps; no GT semantics/enriched labels. Identical existing
`scripts/run_dev_loop.sh` on baseline `faebea0f` and candidate `669d74fe`.
`EMET_VL_ENDPOINT`, `EMET_CONFIG` and simulator trace overrides are unset.

| Version / seed | Job ID |
| --- | --- |
| baseline / 0 | `20260929_045029_264778` |
| candidate / 0 | `20260929_045048_8c0fe0` |
| baseline / 1 | `20260929_045051_0a1604` |
| candidate / 1 | `20260929_045055_bf35c2` |

Output root: `~/runs/emet/navigation-paired-eqa-20260929/{baseline,candidate}/seed{0,1}/`.
Check manifests, metrics and `failed.txt` before paired scoring; crashes and
missing metrics are not scored answer misses. These are queued/running, not
results. After EQA, retain per-question transitions and inspect newly failing
views. Small-room OVMM and TAMP navigation pilots, three-repetition controller
matrix, whole-robot clearance and useful proxy floor coverage remain open.
