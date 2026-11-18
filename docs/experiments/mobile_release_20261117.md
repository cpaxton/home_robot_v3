# Mobile release investigation and broader regression — November 17

Starting checkpoint: `36f8d482` on `fix/tamp-observed-admission`.
The prior [mobile map control](local_map_mobile_20261117.md) passes the full cube
sequence but rejects cylinder release. These are assisted MuJoCo development
controls with provided RGBD maps and native state, not observed-only or hardware
qualification. Physics advances during controller sleeps, not asynchronously
through expensive planner wall time.

## Reproduction and cause

The instrumented mobile cylinder (`20261117_195949_d760e1`) reproduces the exact
failure. The translated stationary cylinder (`20261117_195950_f1d700`) succeeds
at the same world destination without a release adjustment. Its nominal base is
x=0.14 m, compared with measured mobile arrival x=0.136148 m; this is a nominal
arrival control, not an exact final-configuration match. Both already use the
native 0.3 s endpoint-stillness gate.

The stationary control has a 18.909 mm command/FK height shortfall and succeeds;
the mobile control starts at 18.899 mm. Height shortfall alone does not explain
the failure. Candidate-level events show the mobile current-pose opening fails
at the 3 mm lowered probe, then the planner takes the first feasible raised pose.
After two measured corrections the payload bottom is 1.985507 mm above the table.
The unchanged 3 mm lowered probe puts it 1.014493 mm into the support: **14.493 µm
beyond the existing 1 mm penetration limit**. The planner correctly rejects it.
Actual geometry is not asserted to penetrate by that amount; this is the
conservative lowered probe.

An offline private-state audit (`20261117_200856_530718`) distinguishes measured
geometry from those probes. Initial actual gaps are 2.484147 mm stationary and
2.099825 mm mobile. Actual arm/payload contacts and force/actuation state are
archived for debugging. The mobile first correction leaves a -0.021882 mm payload
gap; its final actual gap recovers to 1.985507 mm. No contact tolerance is relaxed.
Raw snapshots and candidate events remain under
`.tamp-runs/release-matrix-20261117/`.

## Candidate selection change

An already certified measured pose still returns immediately. When an adjustment
is needed, evaluate the same finite candidate set and choose maximum bottleneck
support slack across the nominal and lowered delivery endpoints, including the
existing maximum drop constraint. This avoids spending both corrections on the
first feasible candidate with almost zero lower-support margin. Raise/drop,
collision, retention, contact and adjustment limits remain unchanged. Measured
tracking reserve remains a diagnostic estimate, not a future error guarantee.

Before this change, the instrumented release/physical/native regression batch
passes 186 checks (`20261117_200042_82c616`). The final focused ranking/config batch
passes 57 checks, zero failures/errors/skips (`20261117_200952_0f23cf`). An initial
ranking run exposed an old test's first-feasible/monotonic-height assumption; the
updated test checks both delivered support limits, and a new FK/collision case
requires selecting the largest certified support slack without dispatching.

## Declared expanded matrix

`configs/eval/mobile_map_regression_20261117.json` declares seven task cases
(two object controls, nominal-arrival stationary control, two shifted starts/goals,
a 10% larger cylinder at constant density, and additional clutter), plus two
negative controls: occupied-map preflight and evidence loss after grasp. These
are development regressions; new positions have not been qualified as held-out
benchmark fixtures with independent replay certificates. No construction or
planning failure is silently excluded. Lane manifests record task success,
negative-control expectation, process exit, timeout and runtime hashes separately.
Native-state/map assistance and slow sampled sweeps remain explicit.

Full post-change outcomes are pending at this checkpoint. Source is frozen before
the parallel lanes; runtime hashes are retained per launched case.
