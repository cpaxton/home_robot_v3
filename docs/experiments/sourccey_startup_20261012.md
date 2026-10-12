# Planar startup divergence — October 12

The Sourccey kitchen-0 startup failure reproduced on main before #179 and on the
placement branch: slide values reached roughly 1e22 m, followed by MuJoCo's
`FactorizeHessian: rank-deficient sparse Hessian` during home initialization.

The planar correction used the yawed anchor body's rotation as the slide
Jacobian. Slide axes do not rotate with a later yaw joint on that body. The
result could diverge across candidates while still returning success. A minimal
rig with a rotated ancestor and offset base reproduced position errors exceeding
5 km while the old writer reported success.

Fix #200 (`2808c547`) solves using the measured joint Jacobians on private MjData,
validates finite FK and joint limits, and commits only accepted poses. Failed
writes and exhausted searches leave caller state unchanged. Collision thresholds
and robot models are unchanged; no Sourccey-specific rule is introduced.

Validation:

- Ten new MuJoCo regressions pass, covering repeated poses/yaws, offsets, singular
  axes, limits, nonfinite targets, missing joints and search rollback.
- Four existing relevant planar-spawn tests passed; one skipped because that
  random layout placed the hint too close to the centroid. Two unrelated
  `wait_for_obs` tests have stale client fixtures and were excluded from this
  focused rerun. The first sandboxed run also aborted at a disallowed ZMQ bind;
  socket tests were rerun outside that sandbox.
- The exact regenerated kitchen-0 CPU reproducer now completes load/autoplace/home
  with finite state. Its bounded search finds no accepted pose, so it preserves
  the initial pose. This is not evidence of collision-free initial placement.
- Managed GPU job `20261012_153010_4c32aa`, case `sourccey_startup`, passed on
  `2808c547`: full simulator initialization, telemetry, public scene discovery,
  and assisted open/close with both states verified. Exit 0 in 54.3 seconds.

Artifacts: `~/runs/emet/articulation-startup-fixes-20261012/sourccey_startup/` and
`cpu_evidence/`; original comparison provenance is in
`~/runs/emet/placement-merge-triage-20261012/`. Live checks used simulator assistance
and did not test arm placement, native-contact execution or real-robot readiness.
