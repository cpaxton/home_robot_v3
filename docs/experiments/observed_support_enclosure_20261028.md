# Observed support and object enclosure — October 28

Implemented measured support footprints and a conservative free-space-enclosure
constructor. Support was exercised on fresh simulator RGBD using Caliban and CPU
OSMesa. **Observed pick/place and multi-step execution remain unqualified.** No
manipulation or new LLM agent task was attempted.

## Observed support producer

`observed_support.py` fits a selected near-horizontal plane and searches for a
rectangle whose complete cell projections lie in measured support pixels. It does
not fill holes or infer that the whole visible-component AABB supports an object.
Refresh remeasures only the reference footprint; missing/occluding pixels shrink
it or cause rejection. Original receive times are retained; old reference geometry
cannot become a fresh measurement by updating a timestamp.

The first live attempt (`20261028_084417_623bba`) failed support grounding with
`candidate_overflow` in the broad surface-proposal path. Object refresh still
passed 5/5; process completion did not imply support success. This attempt remains
in the record. The harness now fails its requested support diagnostic if no
reference/fresh support candidate exists.

A separate, explicit partial-support path grows a planar component from the
VLM-selected support pixel, within its verified region. It cannot jump to another
larger disconnected surface. The object proposal path and its eight-candidate
budget remain unchanged; this is not an automatic overflow fallback.

The repeat (`20261028_085139_b4f42d`) used three Caliban calls in total: two for
object initialization and one support seed. Object refresh passed 5/5. The fresh
support candidate measured **83 × 10 cm**, with maximum plane residual **0.64 mm**.
The requested 5 mm measurement/admission margin is provisional, not independently
calibrated hardware accuracy. Support geometry is deliberately separate from the
object-only discovery provider: these measurements do not enable task execution
or constitute successful agent discovery/placement.

## Conservative complete-extent constructor

`observed_object_bound.py` retains every occupied and unknown cell as possible
extent of an explicitly connected rigid object. A 26-connected component seeded
by target observations must be enclosed by measured free space and must not reach
any scan boundary. Background endpoint labels are not treated as fully solid
negative cells. Accepted bounds enclose the entire possible component rather
than just the visible points; original evidence times and scan revisions remain
bound to the result. No object orientation or tracked pose is fabricated.

Tests cover hidden extent, diagonal unknown connections, background contacts,
view/boundary rejection, and an additional measured free barrier closing an
otherwise open volume. This strict constructor is useful only when such closure
is observed. Occlusion and unseparated support/robot contacts can keep a component
open; more views alone need not resolve contacting bodies. Further support/contact
constraints or a separately declared model prior are needed in those cases.

Offline spatial reanalysis of all five recorded live captures returned
`object_geometry_unbounded`. It did not renew capture ages or count as five new
live episodes. The single-view visible box therefore was not promoted to a
complete payload. Full pose, robot/contact occupancy policy, encoder alignment,
independent calibration and concrete execution-source assembly remain open.

## Validation

Managed CPU regression `20261028_085742_b934e2` covers all motion and TAMP tests,
CHAT/source binding, surface proposals/reassociation, camera timing/registration,
and the new support/enclosure cases. Exact counts and source hashes are in the
[JSON evidence](observed_support_enclosure_20261028.json). All checks passed with
zero failures/errors/skips. Ruff and `git diff --check` passed.

Raw results: `.tamp-runs/observed-support-20261028/`; the original failed support
attempt, successful support candidate, and offline geometry reanalysis are kept
separate. No paper task-success number changes.
