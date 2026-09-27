# Physical TAMP acceptance and full experiment accounting

The physical and symbolic tracks report different evidence. `sim` is an object/base
teleport oracle; `latch` uses attachment. Neither is contact-based task acceptance.
The full clutter registry remains 200 templates (rby1 110; Stretch, Innate Mars,
and Nori 30 each). Learned-agent comparisons and hardware tests are deferred.

## Shared motion repairs

The opt-in `motion_planner.obstacle_map_mode: physical` uses observed obstacles
before legacy dilation, the robot's configured footprint, and the existing planner
clearance gate. Unobserved cells touched by the actual footprint remain invalid.
Dynamem rounds points into cells centered at integer grid coordinates. Physical
footprints now rasterize at the continuous robot pose instead of truncating the
base to a grid cell first. On the archived September 26 start-pose map, the four
reported unknown rear cells are outside the measured footprint; the corrected
24-cell footprint has no occupied or unknown cells. This is a start-state replay,
not proof of an executable approach or pickup.

Base execution checks controller results and measured residuals; physical runs
require a swept-route executor. Arm planning validates short paths and the exact
endpoint. Pose IK enforces orientation and supports coupled telescoping joints.
The physical executor uses the normal base, arm and gripper client methods; it
never attaches objects or writes live object poses. Coupled-joint arm paths use
validated linear interpolation; alternate base approaches provide search diversity.

The physical GT base planner searches controller-compatible turns and forward/reverse
translations, bounded to 0.2 m per command. Grasp candidate screening tests 16
samples at the precision policy’s declared 20 mm / 0.03 rad arrival bounds and
reserves 5 mm at each telescoping joint limit. Sampling does not certify every
possible arrival: measured arrival still triggers fresh IK and arm collision checks.
The GT driver declares three grasp yaw choices (0 and ±0.08 rad). At arrival the
executor validates a complete grasp/lift sequence for each choice before any
gripper command; it records the selected world-frame targets. Other callers
retain a single fixed frame unless they explicitly provide alternatives.

Carried-object route validation additionally screens 26 fixed base-pose offsets
at the same 20 mm / 0.03 rad arrival bounds. This catches nominal paths that lose
clearance when the extended payload moves with small base yaw errors. It is
sampled screening, not a continuous tracking or payload-slip certificate; fresh
measured-state checks and the independent contact scorer remain mandatory.
Precision command startup now acquires XY from fresh measured feedback when it
already meets the declared bound, allowing a following yaw command to turn.
Drift outside that bound still requires translation to the tighter inner target.
After lift, the GT driver reruns the same bounded placement search from measured
arm/base/payload state, trying the original placement first. A replacement must
pass the same carried route and preplace/release/retreat checks before transport.
Both original and executed placement witnesses are retained. This changes only
private planning state; live actuation remains through the shared controllers.

## Commands

Use a clean frozen checkout. Set OMP/OpenBLAS/MKL threads to 1 and run heavy trials
serially through `emet jobs run --cpu-safe --gpu-exclusive`. The runner can dry-run;
`emet jobs run` itself does not have a dry-run flag. Use the simulator's installed
Python environment and `EMET_UV_RUN=1` to avoid unplanned dependency synchronization.

Physical inputs must be the original fixture YAML and scorer identities from the
handoff; do not substitute an equivalent-looking scene. For each fixture:

```bash
python scripts/eval_physical_tamp.py --sim FIXTURE_SIM_YAML \
  --scorer PHYSICAL_EVAL_JSON --tier physical --seed 0 \
  --output-dir FRESH_OUTPUT --dry-run

# Submit the same command without --dry-run through emet jobs.
# Repeat with matched seeds 0,1,2, retaining baseline and candidate artifacts.
# --tier static and --initial-state measured_initial_state.npz replay a measured
# configuration without executing it. Default static state is pre-startup only.
```

New serial orchestration (existing episode definitions and executor/scorer paths):

```bash
python scripts/run_tamp_experiments.py --suite protocol --output-dir OUT/protocol --dry-run
python scripts/run_tamp_experiments.py --suite small --output-dir OUT/small --dry-run
python scripts/run_tamp_experiments.py --suite floor --output-dir OUT/floor --dry-run
python scripts/run_tamp_experiments.py --suite full --output-dir OUT/full --dry-run
```

Submit one suite at a time through the managed job runner after inspecting its
preceding gate. `--resume` preserves terminal rows; it does not overwrite failures.
After a timeout the runner stops, leaving unrun cases explicitly pending. Inspect
simulator/process cleanup before resuming. Repairs and reattempts use fresh output
roots, so original failures remain evidence.

Use `--robot stretch` for an explicit robot subset, or repeat `--case-id ID` to
revalidate exact cases after a fixture or implementation repair. Unknown IDs fail
before execution. These filters preserve registry definitions and ordering; the
manifest and separate ledger record the selected cases. Subset results never
replace the original full-suite denominator. The generated registry explicitly
sets `scene_split: val` for FloorPlans 13–22; their original `train` split did
not resolve in MolmoSpaces. The house numbers and task definitions are retained.

`eval_tamp_floor.py --gt-only` runs the declared manipulation controls with GT task
inputs. The find-only exploration row is explicitly deferred, not scored as a GT
success. Existing scripted CHAT tool-contract controls remain available through
`run_tamp_agent_tools_gate.sh` with `ITEMS="chat kinematic stretch"`; these are
oracle/latch integration tests, not a learned-agent comparison.

## Evidence and limits

Each physical trial archives scene and compiled-model hashes, initial model state,
measured startup state, task identities, source SHA/dirty state, budgets, candidate
rejections, witness trajectories, measured residuals, sampled state/contact traces,
and an actuation audit. The execution marker separates fixture startup from scored
contact auditing. Simulator dispatch and low-level state-write seams reject
teleports, body/joint pose-setting, attachments, and kinematic base holds in physical
mode. Robosuite models requiring base pose holds are unsupported in this track.

The scorer requires sustained lift, payload retention, stable release on the named
support, complete execution, and clean actuation/contact evidence. It does not
promote controller returns to task success. Collision validation covers the model's
declared collision geometry and exclusions; sampled witnesses are not continuous
or dynamics certificates. Visual-only robot links and unsupported physical adapters
must be reported, not counted as passed robots.

The initial static Molmo check rejects the pre-startup wrist/base self-collision.
This is not a claim that the task is impossible: acceptance requires replaying a
measured settled state and then passing the physical gates. Broader environment
variants, partial-observation execution, and completed full-suite result tables
must be backed by their own artifacts before being claimed.

Dated submissions and measured outcomes are tracked in
[the September 26 ledger](physical_tamp_acceptance_20260926_results.md). The live
server now archives its compiled model and physics settings; physical planning
uses that exact model. A profile navigation posture is itself collision-checked
and measured before base motion. This does not bypass failed posture transitions.

The physical CLI exposes `--route-timeout` (default 10 seconds) separately from
`--timeout`; both must be positive and finite and are recorded in the manifest.
Carried-route envelope checks can exceed ten seconds even for a valid route;
Molmo r27 explicitly uses `--timeout 1800 --route-timeout 60`. The nominal and
measured base center retain 22 cm clearance. Tracking offsets test full geometry
without adding a second center-clearance margin. Placement checks arm feasibility
before searching a carried route; both checks are required for acceptance.

Detailed Python profiling is opt-in with `--profile-planning`; the manifest records
whether it is enabled. Profiling overhead consumes the declared wall budget.
Per-stage and per-route timing/rejection events remain enabled without it.
Scene collision checks filter contact body IDs and penetration depths in bulk
before resolving allowed pairs; full MuJoCo geometry and contact exclusions remain
unchanged, including explicit contact pairs.

Precision final turns apply a bounded forward-position correction while turning:
one-second proportional response, capped by both the outer XY tolerance per
second (2 cm/s for precision) and configured linear speed. Reverse correction
respects the controller's reverse-distance setting. This addresses measured
wheel/contact creep before it leaves the existing XY bound; outer-bound escape
still triggers reacquisition, and measured arrival criteria are unchanged.

The robot profile declares the fully open gripper coordinates, including dependent
finger joints. Static search checks opening at the approach pose and plans the
nominal grasp/lift with this geometry. Runtime checks opening before actuation,
checks the grasp with open geometry, then rebuilds paths from the measured open
state after the normal gripper command. Closed-finger paths cannot certify an
open-finger descent. Actual grasp width and carried geometry still require the
measured lift/placement replan. Missing opening metadata is unsupported in the
physical GT driver. Review cameras now use a nearby oblique view relative to the
robot and target, rather than a fixed distant room-exterior viewpoint.
