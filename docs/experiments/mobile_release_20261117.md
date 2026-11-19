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

Post-change outcomes are recorded below. Source is frozen before the parallel
lanes; runtime hashes are retained per launched case.

## Post-change execution evidence

Frozen runtime `d330c53f` runs four CPU-only lanes:
`20261117_201133_9157d1`, `20261117_201134_a12e68`,
`20261117_201134_eaff9c`, `20261117_201134_b657ef`.
The expanded affected suite passes **261 checks**, zero failures/errors/skips
(`20261117_201135_5312ac`, 220.380 s). The report accounting checks also pass
(`20261117_201830_cdaa8e`, two checks), preserving declared unfinished cases and
separating negative controls, native crashes and task successes.

The original cube and cylinder both complete the full mobile sequence through
release/retreat and departure. The stationary cylinder remains successful without
an adjustment. At the cylinder's identical initial release state, certified
candidate support slacks are **0.364, 8.296 and 1.430 mm**. Ranking selects the
middle candidate; one correction then permits opening and measured support.
The final actual payload gap is 2.195365 mm and its lowered probe is -0.804635 mm,
inside the unchanged -1 mm support limit. This recovers the reproduced failure
without changing controller physics or any rejection threshold. The small actual
margin still does not qualify real sensor/controller uncertainty.

[Release-margin comparison](figures/mobile_release_margin_20261117.pdf) shows the
same candidate set, the chosen bottleneck margin, and the final measured/lowered
support checks. Private simulator state supplies these diagnostics; it is not a
claimed robot sensor.

Two expanded cases expose source-evidence lifetime limits rather than successful
manipulation. The shifted cube on CPUs 20–21 takes 79.487 s to build its source
map and 219.576 s planning; the original 299 s window expires and planning stops.
The clutter case also exhausts source evidence during early execution. Both remain
failed cases in the primary matrix. A distinct shifted-cube repeat on CPUs 0–7
(`20261117_201829_899f05`) retains the slow-CPU failure; it does not overwrite it or
renew its captured timestamps. Planning/refresh scheduling is a separate remaining
integration issue, and successful controls do not close that issue.

## Next source-lifecycle gate

Independent review confirms that the release ranking evaluates only previously
certified candidates and preserves measured-pose acceptance, private checker
ownership and the original bounds. The remaining long-search solution should be
an explicit **post-search reacquisition boundary**:

1. Consume the old public plan handle, retaining its geometry only as an untrusted
   proposal, and acquire a genuinely new source scan while stopped.
2. Verify task/model/policy and measured start/object state, then revalidate the
   chosen dense edges, endpoint FK, gripper phases, payload hypotheses and loaded
   base sweep under the new lease, without restarting candidate/IK enumeration.
3. Publish a new handle only after success. Failure leaves both handles invalid;
   an earlier partial pregrasp still requires inspection rather than automatic
   replay. Remaining lifetime and validation budgets must also pass.

A coarse `geometry_revision`/grid hash cannot transfer a certificate: support
rays, original depth/K/pose/mask, support plane/footprint, backing prior, error and
contact budgets also affect admission. Required controls include changed rays or
new obstacles rejecting before commands, fresh identical geometry getting a new
handle, old handles staying invalid, and expired revalidation producing no handle.
This is a planned follow-up, not an implemented capability or a retroactive repair
of the current cohort's failures.


## Completed matrix and partial feedback

The [compact JSON index](mobile_release_20261117.json) records the full cohort:
**5/7 task cases pass; 2/2 negative controls pass; zero cases remain unfinished**.
There are no native crashes or timeouts. This is a development regression matrix,
not an agent or held-out benchmark rate.

| Task case | Outcome |
| --- | --- |
| Original mobile cube | Complete through departure |
| Original mobile cylinder | Complete through departure; one release correction |
| Nominal-arrival stationary cylinder | Complete; no release correction |
| Shifted mobile cube, slow CPU | Source evidence expires during planning |
| Shifted mobile cylinder | Complete through departure |
| 10% larger cylinder, constant density | Complete through departure |
| Cluttered mobile cube | Source evidence expires during pregrasp |

The larger cylinder is 0.6655 kg versus 0.5 kg at baseline. Its geometry and mass
are modified together only in initial-condition XML; no physics/contact parameter
is changed during execution. The faster-CPU shifted-cube repeat also completes
through departure with matching runtime hashes and remains separate from the
primary matrix. All four maps retire and counted scene storage returns to zero
in completed mobile cases.

The occupied-map negative control rejects before any command. After-grasp
evidence loss returns `partial`, `code=collision`, `completed_ops=["grasp"]`,
`failed_stage=preplace`, `execution_started=true`, `recovery=inspect`; command
count remains **54 → 54**. No extra opening/place command follows the fault.
The real clutter expiry also reports partial pregrasp execution with no completed
operation and inspection recovery. The slow-CPU planning expiry reports no
primitive execution started; its earlier separately checked unloaded approach
is retained in the trial command trace.

The paper appendix now includes the labelled provided-map control, release-margin
figure and the primary failures. `paper/build.sh` passes under managed CPU job
`20261117_203134_2edde4` (71 pages); the new figure reference resolves on the final
LaTeX pass. The final reproducible figure also builds successfully in
`20261117_204027_34ecd9`. Existing unrelated font/float warnings remain. No headline native or
agent cohort is rescored by these development controls.

The release figure is reproducible from the versioned compact evidence alone:
`python scripts/plot_mobile_release.py --output OUT/mobile_release.pdf`
(optionally `--preview OUT/mobile_release.png`). It defaults to the compact JSON
index linked above; `--index PATH` selects another compatible record. Raw dense
traces and original captures remain local rather than a public evidence archive.
