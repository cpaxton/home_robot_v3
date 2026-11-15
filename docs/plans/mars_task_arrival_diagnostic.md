# Bounded Mars task-arrival experiment

The retained actor reaches the matched actual TAMP pregrasp with about
3.01 mm / 0.02966 rad final error. The current 3 mm / 0.02 rad endpoint contract
stops execution before closure. Test whether accepting a nearby pregrasp pose
helps the actual task, without relabeling the original precision result.

Four fixed full public-tool attempts, all on `development:right:50g:1`:

1. Analytic teacher, original precision contract.
2. Retained frozen actor, original precision contract.
3. Same teacher, explicit pregrasp arrival region 4 mm / 0.04 rad.
4. Same actor, same pregrasp region.

No checkpoint selection, new reset seed, retries, training or hardware. Current
wheel feedback is enabled from construction, with the previously validated
gain, physical wheel slew and turn guard. Each attempt owns one isolated native
worker on CPU14, with one numerical thread and CUDA hidden. CPU15 handles
focused checks. Freeze source and scripts before native attempts.

The region changes **endpoint acceptance only**. The existing IK/witness
tolerance, reference geometry, speed limits, phase retiming scales, execution
time budget and independent 10 mm / 0.1 rad path tube remain unchanged. The
experiment is pregrasp-only: grasp, lift and placement keep 3 mm / 0.02 rad.
Completion still requires measured quiet hold. Per-cycle collision, whole-base
drift, static support, contact-force, retention and wheel-unloading checks stay
active. Invalid regions fail closed before actor inference.

The planned pregrasp is 10 mm above the grasp pose. This makes the nearby arrival
region worth testing, not a geometric robustness certificate. Record signed
finger/source-support and finger/object distances at the initial and each
executed stage endpoint, using copied private simulator state. Benign object
contact is permitted. Runtime configuration checks and actual measured grasp
replanning remain authoritative; refuse motion if they reject it. Do not claim
the whole uncertainty region is certified by a few sampled distances.

Run `scripts/probe_mars_task_arrival.py --output logs/eval/mars-task-arrival-v1`
from a frozen checkout. Preserve every public tool response, stage result,
raw per-tick trace, final task predicate and measured-stop receipt. A successful
pick/place requires completed grasp/place operations, independent placement,
support, release, retention/safety and stillness predicates, plus two further
seconds of physical stability. Later-stage failure remains failure; pregrasp
arrival alone is not a successful grasp or task.

Publish compact outcomes, endpoint errors/geometry, closure/lift progress,
retention/contact/support metrics and provenance. Raw `.json.gz` files stay
ignored. The prior full regression receipt does not qualify changed source.
These four development attempts cannot qualify hardware or broad task success.

## Predeclared follow-up after the four outcomes

The actor's region trial completes pregrasp but fails the tighter grasp pose.
Both teacher trials reach lift and stop at the static support margin. Offline
inspection finds the admitted lift witness only about 0.035 mm above the 5 mm
runtime margin in a preliminary fixed-payload audit. The later attached-payload
audit refines that reserve to 0.0041 mm. Do not lower that guard or repeat the
original four attempts.

Add exactly two explicitly different reset cases, teacher and retained actor,
using `--study maximum_margin` and a fresh output directory. Their case ID is
`development:right:50g:1@maximum_margin`. Enable the new opt-in
`grasp_setup_prefer_nominal=False` before fixture initialization; its default
remains True. Select the existing collision-checked setup candidate with the
largest sampled loaded lift margin, preserving the initial world hand pose.
This is procedural simulator reset generation, not an executed hardware stance
change. Keep mass, seed, physical model, task logic and all guards. Both use the
4 mm / 0.04 rad pregrasp-only region; all later stages remain strict.

Record the actual setup transform, selected sampled margin, model/initial-state
identities and any changed destination placement. These are new initial
conditions and cannot be pooled with the original matched-case evidence.
No automatic retries or training. Record numeric support evidence on rejection.
Private forward kinematics refresh geometry before signed-distance queries;
geometry measurements remain sampled evidence, not robust-region certificates.

## Grasp-region follow-up

After those two outcomes, run exactly two more original-reset attempts, teacher
and retained actor, using `--study grasp_region` with fresh output. The actor's
strict grasp timeout is 3.31 mm / 0.03491 rad; its measured finger/table
separation is about 8.55 mm. Test 4 mm / 0.04 rad grasp tolerances, keeping the
same pregrasp-only arrival region and unchanged lift/place contracts.

Unlike the first experiment, this changes grasp IK/witness admission, retiming
scale and endpoint acceptance together. It is a separate task contract, not
proof of an isolated endpoint-gate effect. All actual configuration, collision,
support, force, retention, wheel and stop checks remain active. Restore the
original grasp tolerances in `finally`, including on exceptions. No retries,
physics changes or policy updates, and no relabeling the original failed runs.

## Bounded hold reacquisition replay

The grasp-region actor reaches hold, then drifts just outside the region:
terminal error is 3.28 mm / 0.04052 rad while the actor remains disabled in hold.
The existing loop waits until timeout without resuming correction. Fix this
nonfault handoff, then run exactly two matched original-reset grasp-region
attempts again (teacher/actor) into a new output directory. Keep all old runs.

The executor may resume policy tracking after hold drifts outside its contract,
at most twice and inside the original deadline. Re-entry requires 90% of the
arrival bounds to prevent boundary chatter. Preserve trajectory progress/history,
grip and release authority; never clear a fault, reset physics or extend time.
Repeated loss of hold produces `cartesian_hold_unsettled`, publicly reported as
`tracking_timeout`. Test a physical wrist disturbance, repeated disturbance
exhaustion, unchanged grip/history/clock and refusal to clear a stop before
freezing the updated source. No further tolerance or physics change.
