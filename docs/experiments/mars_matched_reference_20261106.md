# Matched Cartesian reference exposes loaded retention and fault-stop gaps

Neither the analytic Cartesian reference nor the rejected learned final completes
these three transfers. The earlier joint/legacy reference's 3/3 result does not
establish the Cartesian execution path's loaded readiness. Keep the original
joint baseline and every failed Cartesian case.

`TaskPoseTampReference` loads the same admitted checkpoint only to configure and
verify execution settings; it never calls the learned actor. It explicitly uses
the measured Jacobian teacher. Both comparison arms use Cartesian execution,
jerk-limited reference compilation, identical checkpoint actuator settings and
identical initial state. This is a privileged simulator baseline, not a fallback
or a learned controller. Call counts and admissions distinguish it explicitly.

| Frozen first transfer | Matched reference | Rejected learned final |
|---|---|---|
| left,20g,seed0 | Preplace retention limit;352 reference calls | Pregrasp path envelope;57 policy calls |
| left,50g,seed0 | Lift retention limit;237 reference calls | Pregrasp path envelope;73 policy calls |
| right,20g,seed0 | Preplace retention limit;368 reference calls | Pregrasp path envelope;57 policy calls |

At the first fault, object-relative rotations are 0.1059/0.1044/0.1034 rad,
above the unchanged 0.1 rad retention limit, while hand-position errors are
2.04/1.73/2.68 mm. Episode peaks of 0.147/0.241/0.103 rad include fault stopping. A retention-contract failure does not
by itself prove the object has dropped. Peak contact force stays below 6 N and no
wheel lift is recorded; these facts do not qualify the failed transfers.

All three reference fault stops remain unconfirmed: final arm velocity components
reach 0.040/0.119/0.089 rad/s, above the unchanged 0.03 rad/s measured-stop threshold.
The recorded fault-stop resets are preserved for a separate stopping regression.
Inspection finds repeated `stop()` calls rebase the arm target to its latest
position, allowing the target to follow sag. A fix must latch the first stop
position, retain grip authority, and prove measured stopping without clearing the
fault or widening thresholds. Loaded retention remains an independent gap.

The learned runs still confirm their stops and reproduce the previous three
runs exactly on recorded actions, poses, velocities, forces and trajectory
progress. New reporting/controller selection did not change their physical
behavior. The analytic reference reaches grasp/lift, so its later failures do
not explain away the policy's earlier tracking deficit.

All 64 focused tests pass. Tests cover strict reference selection, unchanged
checkpoint admission, no learned inference in the analytic wrapper, correct
trace/public counters, native Cartesian execution, and CLI mode pairing. The
CLI also fixes `--arm-reference jerk_limited` being incorrectly forwarded to
the joint-reference comparison; joint baseline remains joint/legacy, while
`--reference-controller task_pose --arm-execution cartesian` selects the matched
reference. A fresh full branch gate follows the fault-stop correction.

[Complete compact outcomes and phase metrics](../../paper/data/whole_body_control/mars_matched_reference_v1/results.json)
and [three diagnostic stop reset states](../../paper/data/whole_body_control/mars_matched_reference_v1/stop-cases.json)
retain source hashes and raw artifact checksums. The reset states contain
qpos/qvel/ctrl, not a complete resumable integration save. No goals, thresholds,
physics or speed limits change in this comparison; no hardware is used.
