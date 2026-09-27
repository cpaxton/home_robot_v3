# Physical TAMP and motion acceptance: new-agent handoff

Status: requirements and proposed execution plan, 2026-09-26. No new experiment
was launched to prepare this handoff. This is not a claim of task acceptance.

Implementation and measured results are tracked in the
[acceptance report](../experiments/physical_tamp_acceptance_20260926_results.md).
The original handoff and navigation-agent notes below are retained.

### Navigation-agent update, September 26, 09:08

Please inspect start-pose rasterization in the motion work. A fresh learned run
`20260926_090242_0c1603` at `4a143651` successfully performed the bounded sequence
pick_place failure -> observe_floor -> retry. The observation added 89 cells,
but the same four unknown rear footprint cells still rejected the measured
start. No approach or pickup executed. This is not a VLM recovery-selection issue.

Saved initial checkpoint audit: base XY (-1.00237, -0.27980), yaw 0.01856;
grid resolution 0.1, origin (512,512); is_valid truncates the continuous base
grid coordinate to (501,509). Unknown cells: (498,507), (498,508), (498,510),
(498,511). Thus their X centers are -1.4, about 0.398 m behind the measured
base, although diagnostics relative to the truncated grid center report -0.3 m.
The configured footprint is length .33, width .34, length_offset -.1. Check
whether fractional-cell placement and conservative rasterization incorrectly
extend the rear footprint. Do not solve this by declaring arbitrary unseen
cells free or disabling measured-obstacle checks. This agent is NOT changing
the collision geometry or start-state policy; that belongs to your motion work.

Artifacts: `~/runs/emet/navigation-floor-recovery-20260926/`, including
`offline_initial_footprint/` and the live `molmo/` log/depth/map evidence. The
four cell centers project below the default floor image (rows ~690 vs height
424). A separate explicitly assisted steeper-head diagnostic is queued as
`20260926_090710_f40f4b`; do not treat it as an autonomous success. Observation
tool changes are commits `4a143651` (failure detail/recovery permission) and
`169af836` (optional bounded tilt, default unchanged) on the experiment branch.

Follow-up result, 09:12: the steeper head command (-1.4 rad, measured -1.3095)
observed all four missing cells; the physical-map route passed the unchanged
sweep and executed with `nav_teleport=False`. Start XY was approximately
(-1.00270,-0.27959), planned endpoint (-0.7,-0.3), and fresh arrival observation
XY (-0.740715,-0.296769), yaw -0.07820. The newly observed tomato surface was
(0.125654,-0.279428): ~0.8665 m away, still outside the .85 m grasp limit.
Please check task-space arrival tolerances in the motion work; do not count
controller completion as manipulation reach acceptance. The run then stopped
at query-candidate ambiguity before pickup. The learned-harness agent is
investigating duplicate references separately (new commit `d3a85724`).

Repeatability caution, 09:16: matched assisted recheck
`20260926_091258_08c640` did NOT repeat the successful approach. Steeper floor
views reduced four unknown rear cells to one, but (498,507) remained unknown
and the start sweep correctly rejected motion. That run did not exercise the
identity repair. Do not treat the one successful observation/approach as a
robust navigation gate. Full report and three trial artifact roots are in
`/tmp/emet-eqa-progress/docs/experiments/floor_coverage_recovery.md`.

## Objective

Answer: with ground-truth scene knowledge, can our robot physically execute the
small-room pick-and-place tasks that the learned harness is trying to solve?
Then test whether partial-observation exploration can discover a usable route
and manipulation approach. Improve shared motion tools, not scene-specific
VLM workarounds. The product remains one agent harness across EQA, OVMM, and
multi-step TAMP, with robot geometry/capabilities supplied through profiles.

User concerns: stacked padding may exclude valid approaches; movement tools
must plan and execute reliably, expose useful failure reasons, and permit
safe observation/replanning. Do not merely increase grasp reach or lower all
clearance thresholds until an episode passes.

## Branch and ownership

- Work in `/home/cpaxton/src/home_robot_v2`. At handoff this checkout was clean
  on `feat/sourccey-tamp-side-approach` before this document was added. Do not
  assume its implementation matches the current experiment branch.
- Create a NEW branch, suggested `feat/physical-tamp-motion-acceptance`.
  Inspect status, local/remote main, and their diff before selecting the base.
  Prefer current main plus narrowly required repairs; do not inherit unrelated
  Sourccey work or merge an entire experiment branch just to obtain one helper.
  Preserve this handoff and any intervening user edits when changing branches.
- The parallel navigation work is in `/tmp/emet-eqa-progress`, branch
  `experiment/eqa-inspection-progress`, last observed HEAD `a2af43e9`.
  Coordinate shared-file ownership before duplicating its repairs. Do not
  modify that checkout or its running experiments from this assignment.
- Relevant commits to inspect, not blindly cherry-pick:
  `52d90088` physical-map navigation pilot; `9452caa3` optional MCTS approach
  route validator; `7fbe4377` navigation rejection propagation;
  `a2af43e9` evidence and remaining gates.
- Never push to main. Split changes into reviewable planning, execution,
  evaluation, and documentation commits as appropriate. No real-robot tests.

## Known evidence and limitations

Molmo learned pickup failed because the nearest reachable center was 0.927 m
away, beyond the existing 0.85 m grasp limit. The map used two cells of obstacle
padding at 0.1 m resolution, followed by 0.22 m planner clearance and footprint
checks against the padded geometry.

The opt-in physical-map replay recovers an approach approximately at
(-0.7, -0.3), 0.827 m from the observed target surface, without reducing the
clearance or increasing reach. A private MuJoCo static contact audit passed 64
samples on that route and rejected a route through the island. Existing MCTS
selected the accepted alternative. This does NOT establish wheel tracking,
arm IK, grasp success, retention, placement, or complete task feasibility.
The GT object center differs from the observed surface by about 4 cm; neither
distance alone nor the surface-to-base reach threshold certifies an IK solution.

Live learned Molmo trial `20260926_084004_726c49` at `9452caa3` still failed
before pickup: target and fallback routes were rejected with
`rejected_swept_footprint:unobserved_footprint`. The generic pickup failure hid
this reason; the reporting follow-up fixes that without bypassing unknown-space
checks. The physical-map configuration remains opt-in.

Artifact roots:

- `~/runs/emet/navigation-contract-replay-20260926/`
- `~/runs/emet/navigation-tamp-witness-20260926/`
- `~/runs/emet/navigation-physical-molmo-20260926/`
- `~/runs/emet/placement-runtime-overnight-20260926/`

The earlier overnight run stopped after 14/36 scheduled trials. The tiny EQA
slice was 2/4 per arm; both Molmo runs failed approach; three completed RoboCasa
runs picked but failed placement; one RoboCasa control timed out. Do not cite
this as a completed sweep, broad non-regression, or a validated placement system.

## Existing implementation to inspect and reuse

Read repository instructions, then these files where present on the chosen base:

- `docs/experiments/tamp_clutter_testing.md`
- `docs/experiments/tamp_clutter.md`
- `docs/plans/2026-08-13_agent_mcts_tamp.md`
- `docs/experiments/floor_coverage_recovery.md` (latest on experiment branch)
- `scripts/eval_tamp_clutter.py`, `scripts/scripted_mcts_pick_place.py`
- `src/emet/controller/task/tamp/task_search.py`, `clutter_chain.py`
- `src/emet/controller/task/tamp/agent_bridge.py`
- `src/emet/motion/agent_mcts.py`, `src/emet/eval/tamp_clutter.py`
- `src/emet/controller/dynamem/navigation.py`, `mapping.py`
- `src/emet/motion/navigation_sweep.py`, `src/emet/robots/footprint.py`
- `configs/emet/query_navigation_physical_pilot.yaml`

The existing clutter battery defaults to `sim` teleport manipulation; `latch`
is also not contact-based grasp execution. Its historical 24/24 result predates
later navigation guards. Preserve those useful symbolic/oracle tests, but label
them separately. Do not count them as new physical execution acceptance.

The added MCTS callback validates only an approach route during grounding;
the private diagnostic did not supply an IK executor. Extend existing interfaces
only where necessary. Avoid building a second general planning framework.

## Required contracts

1. Separate observed/GT obstacle geometry, uncertainty margin, and robot
   footprint. Document which layer owns each inflation. Preserve unknown-space
   rejection in learned/partial-map mode and measured geometry in GT mode.
2. Candidate base poses must satisfy collision-free approach, robot-specific
   arm IK/reach and orientation, and safe manipulation transitions. A center
   point, Euclidean distance, or static base-only sweep is insufficient.
3. Account for carried-object and arm geometry when validating transport and
   placement. Explicitly report unsupported collision checks, rather than
   silently claiming a complete certificate.
4. Plan to alternate approach/grasp/place candidates with the existing bounded
   search. Fix candidate ordering, seed, and budget. A found validated plan is
   a witness; exhausted search means `no_plan_within_budget`, not impossibility.
5. Execute via normal wheel/base and arm/gripper controllers, measuring actual
   pose and progress. Recheck/replan after meaningful divergence. Bounded retries
   must terminate with structured reasons, not endless nudges or success by timeout.
6. GT supplies state for the oracle planner and independent scorer only. It
   must not leak into learned grounding or partial-observation planning.
7. Simulator resets may establish the initial fixture. During scored execution,
   prohibit base teleport, object pose-setting, kinematic attachment/latching,
   and hidden oracle manipulation. Instrument or assert these restrictions.
8. Failed movement must expose phase, rejection reason, current/goal pose,
   relevant clearance/coverage, and execution residual where available. Do not
   reuse stale success, empty-payload claims, or recovery permission.

## Phased implementation and acceptance gates

### A. Freeze fixtures and reproduce the gap

Start with the existing Stretch Molmo tomato-to-bowl and RoboCasa can-to-counter
fixtures used by the learned experiments. First verify exact scene, robot,
physics, object/support identities, initial qpos, and scorer. Missing fixture
metadata is a blocker to a matched claim, not a reason to invent equivalents.

Molmo references:

- `~/runs/emet/molmo-wheel-contact-20260915/fixture/sim.yaml`
- `~/runs/emet/molmo-wheel-contact-20260915/fixture/scene.xml`
- `~/runs/emet/workspace-handoff-20260915/molmo/physical_eval_config.json`
- `/tmp/run_arrival_small_rooms_20260923.sh` (inspect; do not assume portable)

Resolve RoboCasa's exact fixture/config from the overnight manifest and driver.
Record immutable hashes and the explicit GT receptacle identity/frame. In the
GT test a semantic relation such as right-of-stove must not be guessed by a VLM.

Run a baseline with current settings and save stage failures before repairing.
Gate: reproducible manifest, independently scored results, and clear identification
of the execution backend (physical versus oracle/latch).

### B. GT navigation and manipulation feasibility

Use the existing candidate/MCTS pipeline with GT geometry. Establish, in order:

- a valid approach pose/path, including final orientation and arm IK;
- a collision-checked grasp and lift;
- a feasible carried-object route and placement approach;
- a collision-checked release pose on the named support and retreat.

Start with a stationary reachable pickup, then a short straight approach, then
turning/alternative-approach cases. Include a blocked-route negative and a scene
where the first candidate is blocked but another is feasible. Keep unknown-space
negatives for the partial-map stage. Do not use GT knowledge to mark an unobserved
learned map free.

Gate: accepted candidates pass the relevant geometry/IK checks, blocked candidates
are refused, and the search can choose a feasible alternative within its budget.
Document any remaining gap between a static witness and executable trajectory.

### C. Execute the same plans physically

Run through the shared controller path, not a special GT motion executor.
Log commanded and measured motion, contacts, payload state, and stage boundaries.
Independent success requires pickup/lift, retained payload throughout transport,
release on the specified support, stable support after a predeclared settling
interval, and collision/pose criteria. Controller return True is not the scorer.
Freeze tolerances, dwell times, forbidden contacts, and timeout before candidate
runs; distinguish expected wheel/floor and grasp/support contacts from collisions.

Proposed small pilot: three matched initial-state seeds per task per baseline/
candidate, run serially. Claim the controlled gate only when all candidate trials
pass independently, no forbidden actuation occurs, and the negatives remain safe.
If baseline also passes, report parity; if failures remain, report stage counts
and repair the demonstrated fault. This pilot is not a reliability estimate.

### D. Partial-observation exploration and shared harness

Full GT cannot validate exploration. After C, hide unseen geometry from the
planner and reveal it through normal observations. Keep GT only in the scorer.
Test stationary floor inspection, previously unseen approach space, an alternate
route, and a blocked case. Reuse the same planner/controller APIs, with bounded
observe/replan behavior; do not force the robot through unknown footprint cells.

Measure object/approach discovery, usable floor coverage, actual base displacement,
time to feasible approach, and final task outcome. Do not equate camera sightings
with arrival or reachable manipulation. Compare with the GT witness to distinguish
search/coverage failure from an execution failure.

Gate: a feasible partially observed task is discovered and physically executed;
blocked/unknown cases stop safely with useful diagnostics. Then run small paired
learned Molmo/RoboCasa checks and the established EQA regression slice with model,
physics, budget, prompts, and seeds frozen. No full Habitat-OVMM sweep is required.

## Failure response rules

| Symptom | Next diagnostic; do not silently change the benchmark |
| --- | --- |
| No GT route | Compare raw obstacles, footprint, margins, connectivity, frames and candidate diversity. |
| Route but no IK | Inspect approach yaw, arm profile, limits, grasp frame and alternate base poses. |
| Static witness but motion fails | Check odometry/frame agreement, wheel tracking, swept contacts and controller termination. |
| Pickup succeeds, carry drops | Inspect contact/retention trace; retain original physics results when testing numerical changes. |
| Placement fails | Separate route/IK, payload loss, support geometry and release/scoring failures. |
| GT passes, partial-map fails | Inspect observation coverage, frontier selection and recovery contract, not GT-assisted motion. |
| GT passes, learned fails | Separate grounding/model choice from planning/control; no scene-specific language patches. |

## Evidence, tooling, and resource requirements

- Heavy simulations serial only, using the existing job runner with `--cpu-safe`
  and `--gpu-exclusive`; do not compete with the other agent's running jobs.
  CPU cores 8/9 have been unstable (existing affinity policy excludes 8–11).
  Set OMP/OpenBLAS/MKL threads to 1. Inspect runner semantics and dry-run before
  queueing; use the proper environment for the chosen simulator.
- Record source SHA/dirty state, resolved config, scene/model hashes, seed,
  initial state, action budget, wall timeout, scorer version and execution mode.
  Use frozen worktrees or snapshots for long runs, not a checkout being edited.
- Persist per-stage structured JSON/JSONL, sampled robot/object states, search
  candidates/rejection reasons, controller residuals, and forbidden-call audit.
- Save legible top-down maps with route, footprint and rejected candidates;
  head RGB/depth and chase/side-camera frames at approach, grasp, lift, transport,
  release, and failure. A reviewer must be able to manually verify the scorer.
- Timeouts, crashes and skipped fixtures are distinct results, not omitted rows.
  Verify simulator/job cleanup before starting another heavy trial.
- Add targeted unit tests for contracts, plus live positives and negatives.
  Historical unit counts and old oracle battery scores are not new acceptance.

## Deliverables / definition of done

1. A reviewable branch with minimal shared motion/TAMP repairs, not robot-name
   special cases or default safety relaxation.
2. Reproducible fixture/config manifests and a documented serial command to run
   each tier: oracle symbolic, static feasibility, physical GT execution, and
   partial-map exploration. Clearly label existing versus new entrypoints.
3. Results table per task/seed/stage with evidence links, paired baseline where
   feasible, negative controls, and honest unresolved failures.
4. Updated environment/testing docs and TODO for deferred robots, Habitat-OVMM,
   long-horizon tasks and paper claims. Sourccey/Galaxea/hardware are not required
   for this initial gate; avoid side quests.
5. PR summary specifying what is proven, what remains untested, and whether the
   shared learned harness can adopt the repair. Coordinate promotion with the
   navigation agent; do not independently change its live experiment defaults.

Suggested first message for the new agent:

> Read docs/plans/2026-09-26_physical_tamp_motion_acceptance.md and repository
> instructions. Create a new branch after inspecting base differences. Implement
> the phased physical TAMP/motion acceptance plan, starting with reproducible
> GT approach and pick/place on the existing Molmo/RoboCasa fixtures. Reuse the
> shared harness and existing MCTS, run heavy tests serially, and report evidence
> rather than oracle/latch success as physical acceptance. Coordinate overlapping
> navigation changes with the other agent. Do not push main or use real robots.
