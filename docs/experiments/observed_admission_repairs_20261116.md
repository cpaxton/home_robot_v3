# Observed admission development checks — November 16

Branch `fix/tamp-observed-admission`, based on main `6386e9d3`.
These are CPU regression checks with rendering disabled. No observed task success,
hardware qualification, force bound or new agent benchmark score is claimed.

The branch adds captured-material queries that preserve unknown and occupied
space, geometry-qualified self endpoints, capture/calibration pinning, target
contact at exact fingertip geoms, and finite collision-checked motor closing.
It also repairs symmetry/error propagation and pre-command failure receipts.
Release checks retain the release hypothesis and the fresh visual target, so
object motion cannot disappear from retreat collision checks.

| Check | Passed | Failures / errors / skips |
| --- | ---: | --- |
| Updated full motion suite | 506 | 0 / 0 / 0 |
| Updated full TAMP and selected CHAT suite | 437 | 0 / 0 / 0 |
| Final affected contact/integration suite | 210 | 0 / 0 / 0 |

The suites overlap; do not sum them. The broad suites precede two final review
corrections: target convergence throughout closing settlement and a timeout
check immediately before dispatch. The final focused suite verifies both
corrections and the existing affected execution/API contracts. Ruff passes on
all 22 changed Python files. Earlier failed attempts are retained, including the
public error-code regression and incomplete custom-close fixture.

The [small JSON record](observed_admission_repairs_20261116.json) lists raw local
XML paths, failed attempts, the base revision and final source/test digest. Raw
RGBD, simulator state and verbose test artifacts remain in `.tamp-runs/`.

The independent contact review found and prompted fixes for stale released
geometry, lost nonfinger padding, vanishing target cores, uncertified full-close
commands, post-arrival drift and timeout-before-dispatch. Closing now requires
an explicit finite motor target/mapping and measured feedback; acknowledgement
alone does not establish closure or a grasp. Environment and unknown-space
checks remain independent of allowed target contact.

The archived 1/81 tracking failure remains unresolved. Inspection found two
mixed-colour target-silhouette pixels with registered depth from a different
surface. The archive has no capture-bound motor feedback or source-depth
reprojection provenance, and its final privileged state cannot supply either.
Do not drop the pixels by size/fit quality or reinterpret the historical result.

Remaining gates are local support contact, complete Stretch geometry over
unmeasured joints, a live source/refiner, depth ownership/explicit reacquisition,
and a public-tool observed transfer with independent physical scoring. See the
[observed admission development history](observed_admission_development_20261116.md).
