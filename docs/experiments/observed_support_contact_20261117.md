# Local observed support-contact mechanics

Continuation of `953e7ab5` on `fix/tamp-observed-admission`. No complete observed
Stretch transfer or hardware readiness is claimed. These are calibrated-geometry,
synthetic-interface and offline planning/audit tests. All jobs use CPU MuJoCo with
rendering disabled; no local GPU or VLM is used.

The new opt-in support policy is connected to `ObservedNativeSnapshot`, the
stored-plan validation/refresh path and the observed collision checker. It is
payload-only, finite, capture-bound and keeps the 1 mm penetration limit. A source
must explicitly supply the payload's encoder/FK/base/retention displacement bound
and any backing-material prior; neither is inferred from a table label or two
successful controls. Live source qualification remains open.

| Check | Result | Scope |
| --- | --- | --- |
| Initial focused batch | 135 pass, 3 fail | Retained; see failures below |
| Corrected observed support/background/contact/API/capture batch | 161 pass, zero failures/errors/skips | Includes mixed excluded owners, exact historical intervals and no-command stale support |
| Support/path audit batch | 31 pass, zero failures/errors/skips | Real ray and joint-path fixtures; independent dense payload collision replay |
| Full motion/TAMP/observed capture regression | 988 pass, zero failures/errors/skips | Before the final same-attachment replan guard |
| Independent audit measurements | 2 pass; 110 samples per lift/place path; peak 0.3 mm each | Actual support contact occurs in both paths; 1 mm bound unchanged |
| Final held-payload replan/physical/interface checks | 188 pass, zero failures/errors/skips | After the final floor/uncertainty preservation guard |

The suites overlap and are not independent task trials. Raw logs/XML/source
hashes remain in `.tamp-runs/observed-support-20261117/`; compact hashes and job IDs
are in the accompanying JSON. The runtime source is dirty during these development
checks. Hash snapshots bind the broad and final source; the first failed attempt
is an earlier unfrozen development source. Do not pool these checks with old
native/agent task populations or rescore historical tracking runs.

## What is checked

- Occupied history unions owners and keeps its oldest age. Legacy unattributed
  evidence stays background. Excluding a foreground label from a mixed cell
  cannot promote it to support-only. Ownership changes alter the scene revision.
- Support refinement requires exclusive occupied support provenance, a finite
  measured patch and complete projected depth/mask coverage. Unrelated occupied
  cells win first; unknown cells never consume the support exception.
- A hidden backside band needs a named bounded solid-layer prior. The physical
  uncertain payload keeps its pose and motion errors. Ordinary planning padding
  cannot erase another obstacle or extend contact beyond the patch/backing.
- Unit-plane worst-corner distance rejects deep/sloped contact. A single measured
  lift floor survives shuffled planner queries and hypothetical-state restoration.
- Current observed geometry is checked after release alongside the frozen release
  hypothesis; displacement cannot inherit a stale supported pose.
- Wrong actor/phases, holes, expired/calibration-changed evidence and same-ID
  retiming reject. Snapshot profile changes stop before checker mutation/commands;
  expired support with fresh object evidence returns a no-actuation tool failure.
- MuJoCo depth-ray fixtures distinguish shallow table contact from a small visible
  peg. The independent scorer's geom labels are test-only attribution, not runtime
  grounding. Separate lift/place paths are densely replayed in an independent
  MuJoCo payload/table collision model and stay within 1 mm penetration. Scorer
  pose writes are offline audit state, never physical execution or runtime inputs.

## Retained failures and review corrections

The first 138-check batch has three failures. The sloped-plane case exposes actual
extra quantization: `capture_arrays` used the generic camera's float16 XYZ default.
It now retains input depth precision using float64 unprojection, with a direct
sub-millimetre regression. This is a numeric precision fix, not new sensor accuracy
or a fix for the old wrist-camera RGB/depth ownership mismatch.

The other two initial failures are fixture defects: NumPy treats a bare string
in `r_` as a directive, and the 4 mm peg initially falls between all sampled depth
rays. The corrected peg remains 4 mm and is positioned on a sampled ray, making
its occupied evidence visible. No component is discarded or threshold relaxed.

Read-only review identifies a real mixed-owner bug in the first implementation:
excluded contributors were omitted even when their cell remained occupied. The
fix preserves every contributor of retained occupied cells and adds a direct
old-support/new-excluded-foreground counterexample. Review also prompts exact
historical capture binding, an explicit motion-error budget and fresh released
geometry checks. Follow-up review finds no remaining concrete blocker in those
changes. Mask correctness, backing priors and calibration remain producer contracts,
not qualifications established by the ideal zero-motion-error fixtures.

Final self-review identifies a measured-pose replan edge case: hypothetical
reattachment of the same held object must not reset its original departure floor
or shrink its uncertainty. The final guard preserves both until explicit detach;
the regression lowers the newly measured pose by 0.4 mm and requires the original
lift floor to reject it. Read-only review confirms this matches the executor's
payload capture/restore semantics. The post-guard focused batch is recorded
separately; production bytes outside this guard are unchanged from the 988-check
broad run. Audit measurement recording and non-vacuous-contact assertions are
test-only additions after that broad collection.

## Remaining live gate

Assemble and qualify actual RGBD/motor source/refinement, initial support evidence
under the already resting target, target self-material and robot ownership. Resolve
or explicitly reacquire after registration/ownership ambiguity. Then execute the
stationary public-tool transfer with independent physical scoring, stale-precommand
and loss-after-dispatch controls. Held-out clutter and multistep Caliban orchestration
follow that gate. Local mechanics alone do not enable default observed execution.
