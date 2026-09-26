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
