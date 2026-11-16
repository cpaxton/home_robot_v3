# Coarse general base benchmark and fixed-weight feedback ablation

Runtime `57e8e73b`; see [the complete report](../../../../docs/experiments/coupled_tracking_20261116.md).
The suite keeps 3 mm / 0.02 rad hand accuracy, starts general base acceptance at
5 cm / 5°, and requires 15 cm base paths except separately labeled nullspace
hand-hold diagnostics. Frozen task contracts remain unchanged.

Native teacher outcomes: Mars 8/10, Stretch 9/10. Fixed-weight Mars candidate:
v1 9/10, v2 5/10, original six open-jaw TAMP grasp-pose cases 6/6. No new PPO
or supervised updates occur. The opt-in actor flag removes only the base
head's direct measured-rate path; the arm retains it. Both source and candidate
weight hashes are recorded. Base-only remains a known failure.

The original readout job completes teacher and 20 general episodes, then fails
on a runner API typo before any TAMP decision. A first completion attempt also
executes zero actor decisions because it lacks a multiprocessing main guard.
Both are preserved, excluded as setup failures, and checksum-bound. The guarded
completion executes exactly the six missing TAMP cases; general cases are not
repeated. Text runners are archived alongside compact results. Raw traces,
source archive, XML/receipt, actor weights and W&B files remain ignored.

These small deterministic cohorts, including reused case/seed identities, do
not establish statistical reliability, contact robustness, loaded retention,
complete task execution or hardware readiness. No actor is promoted.
