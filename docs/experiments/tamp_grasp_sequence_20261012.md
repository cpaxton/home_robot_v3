# Grasp endpoint admission, October 12

The integrated Sourccey kitchen-0 case reached its selected base approach, then
failed `pregrasp_ik_failed`. Inspection found a generic admission/execution
mismatch: approach ranking tested only grasp XYZ at 5 cm tolerance with 80 IK
iterations; execution first tested pregrasp at 8.75 mm with 150 iterations.
Grasp reachability does not imply pregrasp or lift reachability.

Ranking now checks pregrasp, grasp and lift in sequence using the executor's
standoff, lift distance, solve tolerance and iteration limit. Each candidate
starts from a private copy of the approach state so one trial cannot seed another
or alter caller state. The search can continue to alternate base poses when any
endpoint fails. This is endpoint admission, not a collision-free trajectory or
successful execution certificate. Measured execution still replans and verifies.

Validation: 86 TAMP tests pass, including real MuJoCo regressions for unreachable
pregrasp, unreachable lift, and a 3 cm residual that the old tolerance accepted.
The API test verifies custom executor settings reach approach ranking. Ruff passes.
Fresh live validation is required before attributing the Sourccey failure entirely
to this mismatch or claiming it is fixed. Extra endpoint solves may increase
planning time; observe live duration and retain the existing bounded candidate set.
