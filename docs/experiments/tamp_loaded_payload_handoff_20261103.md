# Checked loaded-payload handoff — November 3

The follow-up does **not establish a benefit from rebinding**. Both policies
complete the familiar control and stop before the shifted cylinder's release.
The intended cube comparison is masked by this earlier opening-path rejection.
Retain all four outcomes and the preceding failed experiments; the actual-agent
result remains 2/4.

| Layout | Payload reference after identical checked hold | Complete task | Completed transfers | Stop |
| --- | --- | --- | --- | --- |
| Familiar control | Preserve | pass | 2/2 | none |
| Familiar control | Bind measured loaded pose once | pass | 2/2 | none |
| Initial X +15 mm | Preserve | fail | 0/2 | cylinder release preflight |
| Initial X +15 mm | Bind measured loaded pose once | fail | 0/2 | cylinder release preflight |

The fixed population completes **4/8 planned transfers**, with **1,436 audited
commands, zero forbidden commands and zero infrastructure failures**. The
shifted cylinder reaches the requested XY region, but remains held; neither its
placement nor the whole task passes. The safe stop occurs before opening.

## What failed

Private replays of the recorded `place_complete` states reproduce
`invalid_goal:collision` for the gripper-opening path. At the measured arm pose,
the predicted opened left fingertip penetrates the tabletop by **1.370 mm** in
the preserve condition and **1.431 mm** in the rebind condition, beyond the
unchanged 1 mm collision tolerance. These are hypothetical opening-goal contacts,
not contacts produced by executing an unsafe opening. A separate counterfactual
that sets the passive rubber joints to neutral still collides (1.629/1.670 mm).
Passive-joint relaxation alone does not explain away the rejection.

The public receipt in these captured runs says `task_execution_failed`: the
gripper-opening wrapper discarded the planner's collision detail. A post-capture
fix preserves that detail so `failure_code()` returns the existing public
`collision` code. Real MuJoCo wall/bounds regressions distinguish collision from
joint-limit failure and verify no motion/state writes on rejection. The same
captured release states now reproduce
`gripper_open_path_failed:invalid_goal:collision` through the updated wrapper.
This improves attribution; it changes neither geometry nor acceptance and does
not rescore these failed tasks.

## Handoff and controls

Source `dbc708ab`; [frozen protocol](../../configs/eval/tamp_loaded_payload_handoff.json).
Both policies use the same finer-yaw candidate set, native controllers, goal
regions and additional 0.3 simulated-second stability check. Both require the
original 20 mm / 0.1 rad bound throughout that interval, contact with exactly the
two fingertips, and at most 2 mm / 0.01 rad drift from a fixed interval anchor.
The preserve condition keeps the original transform; the rebind condition
updates it once, then uses the existing complete placement replan. That is an
explicit reference change between primitives, not a larger numerical threshold.

Every reached handoff check passes its stricter stability interval. Successful
intervals contain 30–32 samples spanning at least 0.302 simulated seconds.
The largest recorded interval drift is 0.223 mm / 0.000944 rad, comfortably
inside the declared bounds. No reference changes at individual waypoints.
The cube handoff occurs only in the familiar controls; the shifted cylinder
opening failure prevents any conclusion about the shifted cube's later rotation.

These are scripted public task-tool episodes in privileged, stationary, live
MuJoCo simulation. There are zero VLM calls, cameras or hardware trials. Nominal
scene hashes match within each pair; live initial states are archived but not
bit-identical. The recorded model/source/initial-state differences remain
available for comparison rather than being treated as an exact reset certificate.
Neither candidate yaw search nor the loaded-binding profile is promoted to a
production default or an observed-input capability.

The [compact manifest](tamp_loaded_payload_handoff_20261103.json) records all
four outcomes, stable intervals, reference changes, receipts, final goal checks,
job IDs and source/trace hashes. Raw data and private replay diagnostics remain
under `.tamp-runs/loaded-payload-handoff-20261103/`. The earlier raw receipts and
all diagnostic attempts are preserved.

## Next experiment

Fix release planning before repeating the handoff comparison. Candidate release
poses must include the arm, held payload, opening-finger sweep and retreat at the
measured state. Use bounded alternative poses with explicit support/drop-height
constraints; do not waive fingertip/table collision or change the scored goals.
Revalidate against fresh geometry before opening and preserve the independent
post-release support/settling check. A nominally clear opening at the planned
pose does not certify the delivered pose under the controller's tracking error.

After release works across prospective initial placements, repeat the isolated
handoff comparison and actual `run agent` cases using Caliban. Keep the prior
denominators and qualify observed robot/object/background/contact sources
separately. Hardware Stretch remains paused.

## Validation

The checked-handoff/native diagnostic set passed **46 tests**, with a final
overlapping six-case guard rerun after anchoring the stability interval to its
first sample. The attribution fix then passed **102 physical-motion/contact/
payload tests** and **80 native-adapter/API contract tests**, all with zero
failures/errors/skips. The first attribution test run had one overly narrow
string assertion for joint-limit context; the corrected test checks the reason
prefix and exact public code. Both JUnit attempts remain in raw artifacts.
Ruff/whitespace checks pass. The paper builds at 55 pages with no undefined
references or citations. Live captures retain source `dbc708ab`; the later
attribution change is verified by private recorded-state replay and contract
tests, not a rescore or a new successful physical episode.
