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
Completed: all 48 metrics present, no `failed.txt`. Seed 0 is 6/12 for both
versions; seed 1 is 7/12 for both. **13/24 each, no paired correctness changes.**
This is a small Habitat regression check, not evidence that physical-map
navigation or answer accuracy improved.

### September 29 acceptance and recovery follow-up

- User narrowed TAMP to regression only; another agent owns its repairs.
  48 tests passed across MCTS/kinematic helpers, agent bridge, clutter config,
  floor/clutter script smokes and agent-tool gate. No new physical TAMP success
  is claimed.
- Small-room OVMM baseline `faebea0f` and candidate `669d74fe` both scored
  **0/2 object find and 0/2 receptacle find**. Episodes `robocasa_pp_s1` and
  `molmo_ithor_s2_idx0`, seed 0, four mapping views, six agentic rounds,
  three nav steps, physical-map pilot, lazy_graph/query-driven, fresh maps,
  local Qwen, no manipulation. Jobs `20260929_083209_180f86` and
  `20260929_083212_4d3021`; root
  `~/runs/emet/navigation-paired-ovmm-20260929/{baseline,candidate}/`.
  Navigation was safely rejected; these are failures, not capability acceptance.
- Trace inspection found an integration gap: internal find could not call the
  existing `observe_floor` tool, and target navigation discarded swept-footprint
  details. `da5d2a80` adds capability-gated binding, shared recovery feedback,
  exact rejection status and distinct bounded-view action signatures. Habitat
  tools stay unchanged; safety checks and budgets stay unchanged. 219 targeted
  tests pass. Fixed-budget retest `20260929_085020_0968a4` writes `recovery/`.
- Opt-in full-physics-step contact/posture/actuator evidence in `aabf4644`
  replaces sampled-state reconstruction for new route gates. Native Stretch
  three-repeat route (`20260929_084039_ac2b91`) passed **28/42 expected endpoint
  dwells**, then failed command 29 (third repetition's +10 degree turn):
  measured yaw error 0.2104 rad, confirmed stop, navigation-stalled receipt.
  The executed command window had 107,166 consecutive physics steps, no
  unexpected contacts and minimum upright dot 0.9999918. Contact-clear is not
  route success. Root `~/runs/emet/navigation-contact-gate-20260929/stretch/`.
  A second robot run (`20260929_084101_2d1c46`) uses the explicitly identified
  Galaxea proxy, not native RB-Y1. Narrow/obstructed/unknown-route cases remain
  unscored physically; positive tabletop routes do not replace them.
- Socket-dependent tests cannot run under the restricted sandbox (ZMQ bind
  denied); serialized job `20260929_085040_ab3963` runs them with the frozen
  recovery revision. Do not report the aborted sandbox suite as passing.
- `5e092282` repairs new-goal XY/heading phase initialization. The saved pose
  before command 29 was already within the 20 mm acceptance radius, but outside
  the 10 mm approach hysteresis band. Resetting to approach made a requested
  heading turn chase the small XY residual instead. New goals initialize from
  measured XY acceptance; approaches from outside still acquire the inner band.
  Recorded-pose regression plus wheel/sweep tests: 54 passed. Identical live
  route retest `20260929_085341_7063a8`, output `stretch-heading-fix/` under the
  contact-gate root; do not assume the live stall is resolved until this scores.

Promotion remains blocked on useful room exploration/find, controller repetition
reliability and remaining physical safety cases. No timeout or clearance limits
were relaxed to pass these gates.

### Follow-up scores and review boundaries

- Galaxea proxy tabletop route: **42/42 endpoint dwells**, 76,873 consecutive
  physics steps, zero unexpected contacts, minimum upright dot 0.9999963.
  This is the declared route only, not native RB-Y1 or narrow-passage acceptance.
- Native Stretch heading-repair retest `20260929_085341_7063a8`: **42/42 endpoint
  dwells**, 156,986 consecutive physics steps, zero unexpected contacts, minimum
  upright dot 0.9999944. Previously failing command 29 passes with XY error
  0.012275 m and yaw error 0.006515 rad. Tolerances remain 0.02 m / 0.03 rad.
  `stretch-heading-fix/summary.json` records the measured command window.
  This closes the repeated tabletop-route failure, not the separate kitchen
  gripper collision or the rest of the physical acceptance matrix.
- Serialized socket/navigation regression `20260929_085040_ab3963`: **45 passed**.
  Broad controller/agent/navigation/parity suite: **766 passed**, excluding the
  two socket-dependent agent simulation files. Those sandbox attempts failed
  to bind ZMQ; no learned/task result can be inferred from them.
  Final expanded suite at `de7aae86`, including agentic routing/evidence tests:
  **841 passed**. The separate minimal TAMP suite was rerun: **48 passed**.
- Recovery-only OVMM `da5d2a80`: still **0/2 object, 0/2 receptacle**. In RoboCasa,
  a VLM-selected floor observation removed the checked one-cell blocker and
  added 16 observed cells. Molmo gained 22 then 3 cells but its checked blocker
  remained. Repeated unhelpful floor views consumed rounds. Tool availability
  and sensing are demonstrated; successful search is not.
- `2d92eeda` and `de7aae86` stop permanently blacklisting a destination solely
  for missing floor coverage, both in route-abort and attempt-reporting paths.
  Failed continuations are still invalidated; new routes must pass the full
  safety filter. Unrelated failures and retry budgets are preserved. The
  real abort/report-chain regression passes. `51a9656b` synchronizes both shared
  motion controllers into standalone emet-core and adds them to parity checks;
  14 parity/heading/translation tests pass. No hardware deployment.
- Retest `20260929_090518_81c52e`, frozen `de7aae86`, keeps the original two-scene
  budgets and writes `~/runs/emet/navigation-paired-ovmm-20260929/replan-v2/`.
  Superseded job `20260929_090141_595975` was canceled **before execution**; it
  has no result. No benchmark source worktree was edited during a run.
  **Completed: 0/2 objects, 0/2 receptacles.** RoboCasa selected a floor look,
  cleared its checked blocker, then retried investigation; target sampling
  failed. A later frontier hit another unknown cell. Molmo target attempts
  remained unknown-footprint rejections despite floor views. Recorded
  investigate distances were all zero: this is not useful room navigation.
  Repeated views/graph inspections also consumed the six-round budget. Keep
  model limits separate from missing/incorrect tool feedback; do not relax
  geometry or silently increase budgets to make the row pass.
- One independent S0 control (not a paired comparison), job
  `20260929_091423_f6ff37`, uses `default_table_s0_distinct_recep` (red cylinder,
  blue cube), the same settings, and frozen `de7aae86`; artifacts under
  `~/runs/emet/navigation-paired-ovmm-20260929/table-control/`. Its result is
  **0/1 object and 0/1 receptacle**, not a room-scale acceptance substitute.
  Red-cylinder investigation stayed blocked on unknown footprint coverage.
  Blue-cube search reached two sampled navigation goals, but verification
  rejected the views. The first voxel proposal had score 0.1108 and XYZ
  `[-2.1968, 1.5080, 0.0217]` (near floor height); manual inspection of
  `images/rgb_1099511627787.png` confirms no visible cube. Do not treat a
  low-confidence search proposal as a localized object or an arrival as task
  success. Physical translation is working; basic useful find is still unproven
  under this pilot configuration. All scheduled jobs in this battery finished.

Keep review split by responsibility: controller phase/standalone-package parity;
physical-map planner/route contract; shared recovery-tool integration; opt-in
evaluation instrumentation and evidence. Do not present the accumulated branch
diff as one merge-ready PR or change defaults based on these diagnostic pilots.
The controller slice needs the live repetition result; the exploration slice
still needs useful room traversal and the unscored narrow/obstructed/unknown
physical cases. TAMP repairs remain with the sibling agent.

## Resume from this checkpoint

Worktree `/tmp/emet-eqa-progress`, branch `experiment/eqa-inspection-progress`;
tested runtime `de7aae86`. Main and other agents' source worktrees were not
modified. No defaults or hardware deployment changed. Do not launch another
sweep to measure the same blocked attempts.

Next bounded diagnostic: replay the saved unknown-cell/approach-sampling cases,
and establish whether a stationary calibrated view can actually observe each
blocking region before asking the VLM to choose it. Preserve unobservable
regions as unknown; never clear them just because they block motion. Record
why approach candidates fail (occupancy, unknown footprint, reach, connectivity)
and carry that existing sampler evidence into tool feedback. For search
proposals, retain the source view and uncertainty rather than interpreting a
voxel peak as object truth. Then retry the same S0 and two-room gates with
unchanged model/budgets. Keep TAMP at regression scope and keep the already
passing controller slice separate from unfinished search/coverage changes.

### September 29 inspection continuation

`928b29dd` repairs a remaining inconsistency: ordinary inspection still required
a clear ground-plane line of sight to the target, while manipulation approaches
already used camera verification for raised objects above supporting surfaces.
Inspection now uses that same existing distinction. Footprint, connectivity,
clearance and swept-route checks remain; exploration frontiers retain planar
visibility. This is a reproduced contract inconsistency, **not yet proven to be
the cause of the saved RoboCasa failure** because its sampler counts were lost.

Sampler counts now survive in tool feedback and agentic state. Failed inspection
attempts save the same replay inputs as manipulation, plus the candidate grid/XY
mapping, visibility flag and JSON rejection summary. Physical `occupied_footprint`
is classified as obstruction rather than an unspecified workspace failure.
Validation: 799 controller/agent/navigation tests pass. No arrival or safety
thresholds changed; Habitat's early navmesh path is unchanged.

Frozen pilot `20260929_131536_9b7b78` (`nav-inspection-pilot-0929`) uses
`/tmp/emet-inspection-928b29dd`, the same model/config/seed and 4 mapping views,
6 agentic rounds, 3 nav steps on `default_table_s0_distinct_recep`,
`robocasa_pp_s1`, and `molmo_ithor_s2_idx0`. Output:
`~/runs/emet/navigation-inspection-20260929/`. At launch it waits behind the
sibling TAMP job `20260929_131223_7489e3`; do not interfere with that run or
launch a parallel simulator. Completed results follow below.

Offline Molmo floor replay identifies a separate viewing problem. The rejected
cell is XY `(1.7, 1.3)`, derived from the trace's checked pose and world-axis
offset. Projecting `[1.7, 1.3, 0]` through each saved optical pose/intrinsics via
`agentic.views.target_in_view` puts it at rows **527–591**, outside a **424-row**
image. Measured tilts stayed approximately -0.98 to -1.00; pans were 0 or -0.5.
The floor height of zero is an explicit diagnostic assumption, supported by
the first saved depth's world-Z median 0.002 m (1st percentile -0.0115 m for
valid depth below 3 m). It is not an object localization or free-space inference.
Root: `navigation-paired-ovmm-20260929/replan-v2/molmo_ithor_s2_idx0_lazy_graph/`;
first archive `navigation/floor_observation_1790687649758707921.npz`.
Thus repeated captures could not observe the checked region. A steeper bounded
head view is the next controlled diagnostic; actual fresh depth must resolve
the footprint before any route can execute. No pan/tilt limits were expanded.

### Completed inspection pilot and remaining gates

Job `20260929_131536_9b7b78` finished normally. All three cases
(S0, RoboCasa, Molmo) failed object and receptacle localization: **0/3 + 0/3**.
This is not acceptance and must not be described as a successful find battery.

RoboCasa now records two reached object-inspection poses, with navigation
distances 0.455 m and 0.500 m. Saved assessment images at rounds 2 and 4
show a paper-towel roll rather than the requested jar; rejecting these views
is appropriate. The two voxel proposals differ by about 5 mm but have different
source observations. This merits retained negative/source-view evidence, not
an arbitrary spatial blacklist: an ambiguous view is not proof that an object
is absent from a region. These are single-seed diagnostics, not a measured
accuracy improvement attributable solely to the visibility repair.

Molmo's object trace captures four floor views at approximately -1.0 rad tilt;
the rejected footprint remains at one unknown cell throughout. The last two
views add zero explored cells. Existing tool results report capture success,
so recovery feedback now separately labels checked-pose validity, reduced
unknown footprint, unchanged/increased unknown footprint, or unavailable
evaluation. Capture success still does not authorize motion. Targeted recovery
tests: **285 passed**, including partial/no progress, valid checked pose,
and newly identified obstruction. Safety limits and recovery dispatch are unchanged.

Next: controlled stationary depth capture with a steeper **existing bounded**
tilt, followed by rechecking the same rejected footprint and replanning.
Do not spend another learned sweep repeating the current view or change
collision padding to conceal missing observations. No new simulator or hardware
job was launched for this feedback-only change.

### Bounded floor-view intervention queued

Job `20260929_152136_d2985b` (`nav-floor-steeper-diagnostic-0929`) is queued
behind the sibling TAMP job, with CPU-safe/GPU-exclusive serialization.
Frozen runtime remains `928b29dd`, identical Molmo seed/model/mapping/search
budgets to the failed pilot. Diagnostic driver `/tmp/emet_floor_view_probe.py`
(SHA256 `d55499e627a82f4bfbed452ffab5c67a22d88d8c5099ca84552f6ef264b80537`)
intercepts only the first requested floor capture: capture the original view,
then capture at -1.35 rad tilt with the same requested pan. It returns that
second observation and leaves subsequent policy actions unchanged, allowing
normal checked replanning. No base-motion command, collision bypass, semantic
GT, or new head limits are introduced by the driver.

Output root: `~/runs/emet/navigation-floor-steeper-20260929/`. The episode's
`floor_view_intervention.json` records both tool results and measured base drift;
the existing navigation directory retains RGB/depth/calibration/map snapshots.
This is explicitly an intervention diagnostic, **not** a learned-policy score
or a paired accuracy comparison. Check capture success, unchanged base pose,
same-footprint validity and actual subsequent motion separately. If it fails
to clear the blocker, inspect calibration/depth coverage instead of assuming
steeper tilt solves it. No result available at queue time.

### September 30: recovery and completion contracts

The September 29 intervention completed, but did **not** isolate a steeper-tilt
benefit: the original -1.0 rad capture already cleared its checked blocker at
a different rejected pose from the earlier failure. Additional -1.35 rad sensing
kept it valid, with measured base translation 0.0000039 m. The bowl policy then
wasted two more floor captures; the microwave policy wasted three after its
own blocker cleared. Both localization results failed. The bowl assessment
claimed presence despite region grounding returning zero detections; the loop
stopped with viewpoint-coordinate prose. The final localization evaluator
correctly rejected that as missing object geometry.

Runtime `be38c83f` addresses two specific contracts:

- Recovery output is associated with the exact navigation-plan object. For
  that attempt, a newly valid checked footprint supersedes the old warning
  and requests ordinary checked replanning, not more floor captures. A new
  attempt cannot inherit the previous attempt's recovery result.
- Query-driven OVMM explicitly requests `require_grounded_object`. Existing
  grounding must admit an object before visual presence can confirm completion.
  No new verifier, detector dependency, model call or automatic recovery policy
  was added. Ordinary EQA retains visual answerability; other localization
  backends retain their existing contracts. Per-run grounded identity resets.

Validation: 469 agentic/recovery tests, 62 OVMM routing/localization tests,
and 9 minimal TAMP regression tests pass; pre-commit checks pass. A separate
71-test focused run includes the new completion truth table and stale-attempt
recovery test (overlaps the broader suites; do not add counts). No new live EQA
accuracy run was performed, so earlier paired EQA numbers are not evidence
for this revision.

Frozen pilot `20260930_232606_c35512` (`nav-recovery-contract-0930`) at
`/tmp/emet-recovery-be38c83f` completed serially with the previous model, seed,
config, four mapping views, six rounds and three navigation steps. No scripted
head intervention. Output: `~/runs/emet/navigation-recovery-contract-20260930/`.
Configured order was S0, RoboCasa, Molmo. **0/3 objects, 0/3 receptacles**:

- RoboCasa object round 0 rejects unknown floor; round 1 clears the checked
  footprint; round 2 immediately retries investigate and reaches (0.644 m
  recorded navigation distance). Further views do not verify the jar/cab target.
- S0 and Molmo retain an unknown footprint cell despite repeated captures.
  Neither exercises the resolved-blocker branch. Receptacle search repeats
  floor recovery on the remaining blocker without a new target approach.
- No live `object_localization_required` deferral occurred: the relevant
  presence/grounding conflict remains unit-tested, not live-reproduced here.

No task-accuracy gain or physical-find acceptance is claimed. Next isolate the
missing-floor visibility failure using calibrated views of the actual blocked
region; do not relax clearance or launch another identical learned sweep.

### October 1: targeted stationary floor observation

`26245fd7` adds `target_blocker=true` to the existing shared `observe_floor`
tool (chat and internal find). The opt-in mode overrides manually specified
head angles, rechecks the rejected pose, selects a currently unknown cell,
and uses calibrated optical projection for at most two head-only corrections.
Existing absolute pan [-1,1] and tilt [-1.4,-0.7] limits and fresh-frame/measured
head checks remain unchanged. Every capture uses normal map integration and
footprint rechecking. Already-valid footprints request replanning without moving.

The aim point uses the navigation grid's explicit zero-height floor reference,
not inferred free space. Results retain projected target coordinates and, when
in frame, measured optical depth versus reference depth. In-frame is not proof
of unoccluded floor, and valid footprint is not a certified route. Missing
geometry, behind-camera targets, unsupported adapters or exhausted head bounds
do not authorize base motion. The loop exposes failure to the agent; automatic
selection of a different safe sensing pose is still a separate, unimplemented
extension. No changes to A*, collision padding or motion limits in this slice.

292 focused recovery/schema/camera tests pass. `720d4239` retains the measured
Molmo calibration regression: target cell projects below all earlier captures;
an ideal re-aim is geometry only, not proof of physical head reachability.

Frozen runtime `/tmp/emet-target-floor-26245fd7`, job
`20261001_083133_a9a622` (`nav-targeted-floor-1001`), output
`~/runs/emet/navigation-targeted-floor-20261001/`. One Molmo seed-0 episode,
unchanged model/config, four mapping views, six rounds, three navigation steps;
CPU-safe and GPU-exclusive. No scripted forced tool call. At launch no live
result is claimed: verify that the model selects the mode, measured projection
changes, fresh depth reaches the blocked region, and replanning actually moves.

Review organization: #167 is the grounding parent; #169 remains its stacked
navigation/pregrasp PR; #168 is the integration PR. #176 owns shared VLM dialogue
and #177 owns the separate TAMP work. Do not fold the experiment branch wholesale
into #169: the local comparison reports 353 commits beyond cached origin/main,
so review should extract the targeted recovery/completion commits with their
tests after reconciling current main. No existing PR base/head was changed.

### October 5: targeted-floor result reviewed

Job `20261001_083133_a9a622` completed. The model chose
`observe_floor(target_blocker=true)` in object round 3. Its current blocked
cell was world XY (1.0, 2.2), not the earlier (1.7, 1.3) replay case. One bounded
capture at requested pan -0.4564 / tilt -1.4 (measured -0.4561 / -1.3986)
moved the target projection from (268, 1439), outside the 240x424 image, to
(192, 328), inside it. Measured optical depth was 1.2060 m versus 1.2040 m
reference depth; the checked footprint changed from one unknown cell to valid.
This is sensor/map evidence for this region, not blanket route certification.

The next object action selected exploration, but the trace does not establish
a completed post-recovery object approach. Receptacle search first rechecked
the already-valid footprint without another capture, then reached three
inspection poses (recorded navigation distances 0.634, 0.640, 0.365 m).
Both object and receptacle localization still failed. No task-accuracy gain,
physical-find acceptance, or controlled replication of the earlier blocked
cell is claimed. Investigate the resulting views/proposals next; do not relax
clearance or expand head limits based on this result.

Saved receptacle views identify the next concrete issue. Round 1's proposed
microwave XYZ is (2.0669, 0.5890, -0.0121); its projection is in frame, but manual
RGB inspection shows floor and cabinet, consistent with grounding abstention.
Rounds 3/4 use a new proposal at (1.9689, -0.2324, 2.5376). Both project behind
the optical camera (camera Z -1.204/-1.161 m), and `aim_arrival_view` immediately
returns without a head correction for behind-camera targets. This does not
establish that either proposal is a microwave. Next isolate head-state handoff
from downward floor inspection to object viewing, using known camera/target
geometry and bounded measured head control. Do not blacklist heights or change
the detector solely to fit these two hypotheses.

PR status checked October 5: #167/#168/#169/#176/#177 remain open; #178 is
additional TAMP work owned separately. Experiment source is published on
`experiment/eqa-inspection-progress`; it is not added wholesale to those PRs.

### October 5: measured aiming loop and acceptance jobs

`b1840598` extends existing arrival aiming: at most two head commands total,
including one horizontal-forward reset for a behind-camera proposal. Existing
adapter clipping remains authoritative; measured arrival must match the
requested pose within the existing floor-tool 0.12 rad tolerance. Clipped or
stalled requests fail rather than falsely succeeding. After measured arrival,
require a newer received frame before retaining/reprojecting the capture.
Missing physical calibration, unsupported control, stale camera frames/captures,
or an exhausted aim budget return `TARGET_OUTSIDE_VIEW` with a specific reason.
Nonphysical missing-geometry compatibility remains unchanged. No base motion,
new verifier, detector dependency, expanded head limits or collision relaxation.

Tests: 297 focused camera/floor tests and a separate 288-test agentic/OVMM/
grounding/minimal-TAMP suite passed. Two additional tests bring the view-quality
file to 12 passing tests (overlap, not additive): forward reset plus correction
share the command budget; physical missing geometry cannot pass inspection.

Stationary diagnostic `20261005_182928_024a2e` passed on frozen runtime
`/tmp/emet-aim-b1840598`. Driver `/tmp/emet_stationary_aim_probe.py` takes the
initial horizontal camera's center-pixel sensor XYZ, commands downward tilt,
and runs ordinary arrival aiming with no semantic GT. Target starts in front
of the optical camera but outside frame (pixel 123,-1026), ends at 120,221;
measured base translation 0.00000145 m. This exercises fresh measured aiming,
not the behind-camera reset or semantic localization. Artifact:
`~/runs/emet/navigation-stationary-aim-20261005/molmo_ithor_s2_idx0_lazy_graph/stationary_aim.json`.

Follow-up jobs, all CPU-safe/GPU-exclusive and fresh processes:

- `20261005_183231_5c54a0`: S0, RoboCasa and Molmo, unchanged seed/model/config
  and 4 mapping views / 6 rounds / 3 nav steps. Launch is gated on the passing
  stationary artifact. Root `~/runs/emet/navigation-aim-pilot-20261005/`.
- `20261005_183235_4313af`: paired EQA q2/15/25, seed 0, baseline `26245fd7`
  versus candidate `b1840598`, existing dev-loop budgets. Root
  `~/runs/emet/navigation-aim-eqa-20261005/`. Small smoke, not general accuracy.

At this checkpoint the room pilot has started and EQA is queued. Score saved
results/failed.txt before claiming either gate passed. Preserve failed views.

Disk: confirmed pip cache path `/home/cpaxton/.cache/pip`; ran the approved
pip cache purge, which reported zero package files. Free space subsequently
measured about 23 GB versus 435 MB initially; do not attribute all reclaimed
space to that command. No experiment bundles, datasets, model weights, uv
cache or worktrees were deleted by this agent. New simulation cleared 10 GB gate.

### October 6: pilot results and shared head capability

The October 5 room pilot finished with **0/3 object and 0/3 receptacle
localizations**. S0 reached two object inspection poses; Molmo reached two
object and three receptacle poses; RoboCasa reached three object poses.
Arrival alone is not useful-view or task acceptance. The paired EQA smoke
finished **1/3 for both revisions**, with q15 correct and q2/q25 wrong on each.
All six EQA records are present. This is unchanged correctness on a small
smoke, not evidence of general non-regression or an accuracy improvement.

Molmo exposed an interface inconsistency: an elevated voxel hypothesis at
world Z 2.538 m requested tilt +0.756 rad, but the client silently clipped it
to zero despite the active simulator joint allowing upward tilt. Measured
tilt stayed near zero. The hypothesis is not a verified microwave; repairing
control does not certify its semantic identity. The next round repeated the
same request, while router state omitted the limit/pose failure details.

Runtime `eb1e0890` introduces one effective `HeadCapability` consumed by head
commands and arrival-inspection preflight. Simulation advertises the
intersection of active joint and direct position-actuator limits; unsupported
actuator mappings abstain. Stretch simulation retains its established pan
envelope, but can use advertised model tilt bounds. Real Stretch and old
bridges retain the conservative legacy limits. No hardware range expansion,
base reversal, collision relaxation, or new semantic verifier is introduced.

An infeasible look now returns `TARGET_OUTSIDE_VIEW` / `HEAD_LIMIT` before
sending the head command, including requested angles, effective limits and
guidance to select another collision-checked viewpoint. The failure reaches
agent state explicitly as an inspection limitation, not evidence of object
absence. Measured head arrival and fresh-frame checks remain required after
feasible commands. Automatic selection of an alternate reachable viewing
pose is still open; this change supplies the contract for it.

Validation: 239 broader agentic/OVMM/session/head-spec tests and 25 focused
head-capability/inspection-feedback tests pass (overlapping suites). Frozen
runtime `/tmp/emet-head-eb1e0890`, stationary diagnostic
`20261006_160307_45da38`, requests the exact +0.756-rad tilt and measures head
arrival/base drift. Output root:
`~/runs/emet/navigation-head-capability-20261006/`. It passed: advertised tilt
[-1.53, +0.79], measured tilt +0.686 rad versus +0.756 requested, within the
existing 0.12-rad tolerance; base drift 0.00000294 m. This diagnostic is not
semantic localization or full collision acceptance. Driver is preserved as
`~/runs/emet/jobs_runs/nav-head-capability-1006/driver.py`.

Fresh learned Molmo search `20261006_160636_2837e8` uses the same frozen runtime,
model/config/seed and 4 mapping views / 6 rounds / 3 navigation steps, with
teleports disabled. It is CPU-safe/GPU-exclusive and starts after the diagnostic
finished. Results pending at `~/runs/emet/navigation-head-molmo-20261006/`.
Inspect requested/measured head state, failure feedback and resulting RGB
before claiming useful inspection or semantic success.

### October 9: preserve the waypoint through floor recovery

The October 6 learned Molmo retest finished 0/1 object and 0/1 receptacle,
389 seconds episode time, with six graph nodes. Three targeted floor captures
each cleared their checked unknown footprint. The object loop nevertheless
spent its five actionable turns on graph query, investigate, floor sensing,
investigate, floor sensing; the sixth round forced an answer. Both phases
ended with two navigation attempts and zero exploration actions. A receptacle
approach reached its candidate, but the saved grounding RGB shows a nearby
cabinet/wall surface and floor, not a microwave. Control recovery is not
semantic task success.

New shared navigation recovery keeps the chosen waypoint and arrival heading
for both investigate and frontier navigation. It permits at most two targeted
floor observations and three ordinary planner calls within one high-level
navigation decision. Each retry starts at the current measured pose. Only a
fresh unknown-footprint rejection is eligible; sensing must return a valid
checked footprint before replanning. Unchanged/invalid footprints, occupancy,
unsupported sensing, stale attempt records, and the retry cap stop recovery.
Approach retirement and high-level navigation counters occur after this loop,
not between its internal retries. The full route is always checked again.

This deliberately changes work per decision: unchanged round/nav settings
are NOT equal-compute comparisons. Preserve wall time and internal recovery
traces alongside task scores. The round budget still governs other small
actions; this is a bounded repair of navigation continuation, not a wholesale
weighted-budget redesign. Unit and serial simulator acceptance must be
recorded before claiming a task improvement.

Runtime commit `17d96365`: 178 agentic tests and 62 OVMM find/routing tests
pass. Serial CPU-safe/GPU-exclusive Molmo pilot `20261009_073120_0cb088`
uses frozen `/tmp/emet-floor-resume-17d96365`, seed 0, four mapping views,
six rounds, three high-level navigation decisions, unchanged Qwen/config and
teleports disabled. Result pending at
`~/runs/emet/navigation-floor-resume-20261009/`. Inspect
`navigation_floor_recovery` trace events and downstream arrival evidence;
internal retries are bounded separately and must not be hidden in runtime
comparisons with the earlier single-attempt pilot.

### October 9: completed recovery pilot and PR integration review

The recovery pilot completed 0/1 object and 0/1 receptacle. The object phase
cleared a blocker, resumed the same waypoint and reached it (0.382 m recorded
navigation distance); a later approach also reached (0.482 m). Both phases
used two investigations and one exploration action. This establishes one live
recovery continuation, not semantic localization or manipulation acceptance.
Episode time was 603.5 s versus 389.2 s previously; query time 519.2 s versus
305.7 s. Initialization/mapping took 18.1/35.5 s. Most elapsed time is in the
search phase, but this does not yet isolate navigation versus model latency.
The single stochastic comparison cannot attribute the increase to floor
recovery alone. Ten minutes without locating either target is not acceptable
task performance; retain runtime as a gate alongside success.

Queued frozen comparisons (all CPU-safe/GPU-exclusive; no hardware):

- `20261009_074725_cf590d`: baseline `eb1e0890` versus candidate `17d96365`,
  S0 and RoboCasa, seed 0, four mapping views/six rounds/three navigation
  decisions. Four serial find episodes, no manipulation. Artifacts:
  `~/runs/emet/navigation-resume-paired-room-20261009/`.
- `20261009_074727_8892fd`: the same revisions, paired 12-question EQA dev
  set, seed 0, existing 20-planning/10-movement budgets. Artifacts:
  `~/runs/emet/navigation-resume-paired-eqa-20261009/`. A dev comparison,
  not held-out paper evaluation or a main-versus-stack acceptance claim.

Open PR review against fetched main `a9a5f1d1` (October 9):

| PR | Finding | Integration recommendation |
| --- | --- | --- |
| #176 | Three-file synchronous dialogue isolation; applies cleanly to current main in disposable worktree | First standalone landing candidate after focused tests/review; does not require an OVMM accuracy claim |
| #167 | Draft, conflicts with main; 231 files and unmet learned grounding/EQA gates | Extract independently justified repairs rather than merge accumulated branch wholesale |
| #169 | Mergeable onto #167, not independently main-ready; 31 files mixing wheel/pregrasp/EQA work | Split mechanical fixes from semantic changes, reconcile shared motion changes with TAMP, validate extracted main-based revisions |
| #168 | Conflicting 171-file integration branch; PR description still template-only | Reconcile provenance/scope and replace or refresh as an explicit integration sandbox; not a landing candidate |
| #177 | Large physical-TAMP measurement/control change; admitted fixtures and assisted-control results, known latch limitations | Review reusable motion/evaluation contracts separately from task-success claims; coordinate with TAMP owner |
| #178 | Contains #177 head as an ancestor plus 23 commits although both target main | Treat as a stack, not independent PRs; review delta after #177 or retarget with owner approval; assisted GT results are not learned physical acceptance |
| #179 | Draft stacked on #178; collision-aware placement smoke pending | Keep isolated until owner finishes acceptance; no merge based only on unit results |

GitHub reports no status checks or review decisions on these seven PRs.
For #176, cherry-pick onto current main in `/tmp/emet-review-176-1009`
passed 35 conversation/dispatch/prompt tests; this is a local integration
check, not a change to main or to the PR branch.
Mergeability alone is not approval. No PR was merged or retargeted. The
experiment branch is not a substitute for testing extracted changes on main.
Next integration order: narrow dialogue fix; independently reviewed motion/
grounding repairs; explicit sandbox combining accepted revisions; then paired
EQA/small-room gates. Do not pull the pending placement stack into these pilots.
