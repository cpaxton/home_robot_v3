# Measured coverage scan disposition, 2026-10-25

PR #198 is a deferred experiment, not a missing dependency of the shared-agent
stack. Its implementation is recoverable at branch `feat/measured-coverage-scan`,
head `cb481970`. Keep the branch when closing the PR.

Its distinct contribution is an opt-in fixed-XY scan over eight headings and
up to two supported head elevations, with measured arrival, fresh captures,
validated turns and explicit partial outcomes. It is not wholly superseded by
the landed navigation, head inspection or view-selection contracts.

There is no completed corrected local-versus-coverage comparison on the exact
owner head. Historical integration scans and tests do not establish task value;
the old learned run used `local` due to a repaired configuration dispatch bug.
The branch also includes pre-squash ancestors of now-landed changes. Merging it
as-is would add stale integration work and a larger fixed scan without evidence
that it is a good default or worth its time/image cost.

Reopen as a small current-main experiment only if diagnostics show a repeatable
coverage gap. Reuse the measured navigation/head contracts rather than fork
them. Require a same-provider local/coverage pair, four initial headings and an
opposite-side fixture, offline visibility/depth controls, real capture timestamps,
and task-quality plus wall/sim-time costs. Keep oracle visibility scorer-side.
Do not change the default scan or infer known free floor from angular coverage.
