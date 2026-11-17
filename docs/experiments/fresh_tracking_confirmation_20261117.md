# Fresh nearby tracking confirmation — complete

This predeclared set tests whether longer fine-tuning improves beyond the twelve
training and repeatedly inspected audit cases. It is frozen while the longer run
is still training and never enters PPO or selects intermediate checkpoints.
The sole learned checkpoint evaluated is its fixed final 65,536-additional-decision
export. The teacher and completed ordinary checkpoint provide matched baselines.

Generator seed 2026111707 produces 24 new identities and seeds 410300000–410300023,
disjoint from the preceding training/development pool. Eight cases constrain
base and hand, eight leave base free for the hand, four are base-only, and four
are coupled two-stage commands. Half vary starting heading, including two broad
±π heading draws. Existing native seed-dependent nearby arm-posture resets remain;
this does not establish global workspace/posture coverage. Motion targets are
FK-witness paths with arm deltas within ±0.04 rad/m and signed base travel
0.12–0.18 m / turn 0.18–0.42 rad. Geometry, joint-margin and support checks retain
all rejections; neither teacher failures nor inconvenient cases are discarded.

The four second commands arrive 0.25 s after the first nominal trajectory ends,
before its required two-second quiet hold can complete. They test a goal change
while the robot may still be settling, not a mid-reference-velocity interruption.
The second path is constructed in scratch state at the first nominal endpoint,
so its targets are independent of controller tracking error. Both references
have zero endpoint velocity/acceleration and matching poses. Four prechecks
verify C2 joins and unchanged live qpos; all four second paths pass sampled
geometry/support admission. Runtime never teleports the robot, clears a fault,
or re-arms stopped control. Existing trajectory APIs install the next command.

This distinction matters: the older procedural `retarget` helper builds a path
from the current measured pose. Such reactive replans can legitimately differ
between controllers. This set uses nominally anchored paths, and its final
comparison explicitly matches initial-state, model/configuration and both
reference hashes across teacher/source/final controller.

The original 25 s horizon and precise hand/coarse base contracts remain fixed.
Reports distinguish all 24 attempts, geometry-admitted executions, ordinary
tracking success, safety flags and confirmed stops. All actual contact/force,
jerk, wheel support and retention guards stay active; no perturbation or loaded
object is added here. Kinematic admission is not a dynamic/hardware certificate.

Job `20261117_085617_9ac2ab` uses CPU15 with one numerical thread and hidden CUDA,
while the single CPU14 fine-tuning worker continues. It evaluates teacher and
ordinary first, waits for successful completion of fine-tuning job
`20261117_083406_6c3285`, then evaluates only its fixed final actor. All inputs
and runner hashes bind before readout. Source `9d0620f9` retains its verified
1,585-test receipt. Raw readouts/progress are under ignored
`logs/eval/fresh-tracking-confirmation-v1/`; exact generator/protocol/runners
under `logs/recovery/fresh-tracking-confirmation-v1/`. No promotion or hardware.

## Completed results

All 24 cases execute under every controller. All initial-state, model/config and
both-stage reference hashes match; all **72/72 stops confirm**.

| Controller | Success | Contact stops | Wheel-lift episodes | All safety-flagged episodes |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 17/24 | 1 | 1 | 2 |
| Ordinary source | 12/24 | 2 | 2 | 4 |
| Longer-trained | 16/24 | 1 | 0 | 1 |

No invalid physics or retention loss occurs. Free-base hand cases improve 7/8 →
8/8, base-only 2/4 → 4/4, ordinary coupled 2/8 → 4/8; goal changes regress
1/4 → 0/4. Teacher scores those groups 8/8, 1/4, 6/8 and 2/4. The teacher's
failures are preserved; FK admission is not dynamic success or proof of optimality.

Among active-hand cases, median error improves **2.255 → 1.826 mm**, 90th
percentile **19.575 → 17.595 mm**, worst **27.726 → 25.318 mm**. Teacher median
is 0.000679 mm in this idealized simulator. Peak base linear/yaw jerk worsens
**14.88/65.57 → 28.68/78.54** in respective SI units, so reduced safety-event
counts do not establish globally smoother/safer behavior. The continued policy
is not promoted to hardware. Complete identities, targets and all outcomes are
in `paper/data/whole_body_control/fresh_tracking_confirmation_v1/results.json`.

These cases are fresh confirmation for the completed longer run. Once their
results motivate the next curriculum they become inspected development evidence;
do not describe their reuse as a new independent final qualification.

The [target gallery](../../paper/data/whole_body_control/fresh_tracking_confirmation_v1/target-gallery.svg)
shows active base paths/headings and hand endpoint orientation axes for all cases.
Its five samples per stage are a visual aid, not execution interpolation. The
exact generator and reference hashes remain authoritative.
