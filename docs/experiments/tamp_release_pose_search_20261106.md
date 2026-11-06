# Native release-pose search — November 6

The first bounded policy preserves safe stops but does **not improve task
completion**. Both the existing policy and bounded search pass their familiar
control and fail their +15 mm shifted task: **1/2 complete tasks per policy**.
This is a privileged, stationary, asynchronous MuJoCo tool diagnostic, with zero
LLM calls or hardware trials. The actual-agent baseline remains **2/4**.

| Layout | Release policy at `d5e1df78` | Complete task | Transfers | Failure |
| --- | --- | --- | --- | --- |
| Familiar control | Existing | pass | 2/2 | none |
| Familiar control | Bounded | pass | 2/2 | none |
| Initial X +15 mm | Existing | fail | 1/2 | second-object opening collision |
| Initial X +15 mm | Bounded | fail | 0/2 | first-object correction budget exhausted; no opening |

All four episodes are retained: **5/8 planned transfers**, 1,767 audited commands,
zero forbidden/malformed commands and zero infrastructure failures. Nominal scene
hashes match within each pair; asynchronous initial states are archived and are
not bit-identical. These are development fixtures, not a held-out benchmark.
The [manifest](tamp_release_pose_search_20261106.json) records source, inputs,
receipts, release events, final scores, jobs and raw artifact hashes.

## What the search checks

The opt-in profile uses `emet.motion.release` through the existing shared
`PhysicalPickPlaceExecutor.place_only`. It checks closed-arm motion with both
original retained and measured held-object geometry, finger opening, and retreat
with the released object watched. A sampled, fixed-orientation vertical descent
checks payload/support/obstacle overlap. This is static validation, not a
certificate of falling-object dynamics. The original retention bound is not
renewed during corrections; contact and collision tolerances remain unchanged.

Limits are declared before launch: maximum 40 mm EE raise from the measured
release-refinement start, maximum 40 mm object-bottom gap above the support,
3 mm additional lowered opening probe, and at most two correction commands.
Full-object goal-region, support and settling checks are unchanged. The native
profile waits for measured opening joints and no fingertip/payload contact before
retreat. Those simulator measurements are privileged diagnostics, not claims of
hardware tactile sensing. Production and observed-input defaults stay unchanged.

## Why the first live correction fails

Both saved November 3 opening failures have private collision-free alternatives
with predicted gaps 20.878/20.895 mm; no recorded state or controller commands
change. Live execution exposes the missing distinction between a nominal target
and the delivered pose. In the shifted bounded trial, EE z is 0.52060 m at place
completion, 0.52047 m at the first correction acknowledgment, and 0.51915 m at the
second. Both joint residuals are inside the unchanged 0.03 tolerance. Small
nominal increments cannot reliably produce opening clearance under this loaded
tracking offset. The existing pose-IK tolerance is 3 mm, so a 3 mm nominal
increment need not increase the achieved command-FK height either. Geometry
must be checked at the solved configuration and the delivered pose, rather than
assuming that a target-height increment equals physical clearance. The policy exhausts its two-command budget and withholds opening.
The failed task remains failed even though a later hypothetical candidate exists.

## Separate measured-error follow-up

Source `816f1f4e` freezes a separate control/shift pair in
[`tamp_release_tracking_reserve.json`](../../configs/eval/tamp_release_tracking_reserve.json).
For new correction targets, the lower opening probe includes the positive
command-FK minus measured EE-height error as well as the 3 mm reserve. Alternatives
are +3/+10/+20 mm nominal height, under the same 40 mm raise/drop bounds and
two-command budget. Both the nominal opening and lowered opening are checked.
The extra nominal probe has an overhang regression: a lowered probe alone can
miss an opening-arc collision. The existing shared actual-opening guard also
checks the full sweep before every physical opening.

Measured error is a search reserve, not an upper bound on future tracking error;
fresh measured geometry must still qualify release. Saved-state replays with
18.282/18.811 mm error reserves select +10 mm alternatives with predicted gaps
28.723/28.773 mm. They issue no commands and preserve all recorded state buffers.
These replays are not successful physical tasks. All earlier outcomes remain
unchanged; the follow-up is not an exact reset-paired comparison.

## Validation and remaining gate

At the first frozen source, 125 focused motion/native-adapter/diagnostic tests
pass with zero failures or skips. The offset-aware revision passes 96 focused
motion/diagnostic checks, overlapping the earlier set, including two additional
reserve/overhang regressions. Earlier test-development failures and final JUnit
records remain under `.tamp-runs/release-search-20261106/`.

Raw first-batch inputs/traces/replays are under
`.tamp-runs/release-search-20261106/`; the independently frozen follow-up is under
`.tamp-runs/release-tracking-reserve-20261106/`. Expand prospective initial
placements and explicit recovery cases before promoting this diagnostic to an
agent default. Observed full-arm/contact qualification, household integration
and physical Stretch remain separate gates.

The measured-error follow-up finishes **1/2 tasks and 2/4 planned transfers**,
with 675 audited commands, no forbidden/malformed commands and no infrastructure
failures. The shifted cylinder opens and settles inside its goal, but the task
stops before retreat, so the transfer remains incomplete. Its rotated local
bounding box predicts bottom z = 0.478946 m, while the cylinder's analytic extent
and MuJoCo contacts agree on z = 0.479113 m against a 0.480 m tabletop. The first
value crosses the 1 mm rejection threshold; actual penetration is 0.887 mm.
This is a conservative-box false rejection, not permission for deeper contact.
The familiar control passes. See the
[separate follow-up manifest](tamp_release_tracking_reserve_20261106.json).

A third, separately frozen pair uses analytic primitive extents (and mesh vertex
bounds) for the released payload, retaining conservative bounds for other shapes.
It removes the released object/support collision exemption, preserving the
ordinary 1 mm penetration check. Opening verification now also waits for 0.1 s of
measured supported stillness, clear geometry, arrived open joints and no fingertip
contact, within the existing three-second wait. This prevents retreat planning on
a bouncing object. Final scoring is unchanged, including its original conservative
XY footprint calculation. The protocol is
[`tamp_release_shape_extents.json`](../../configs/eval/tamp_release_shape_extents.json).

## Shape-aware native outcome

The fresh pair at `6a1d063c` passes **2/2 complete two-object tasks and 4/4
planned transfers**, including the +15 mm shifted layout. Both objects meet the
unchanged final full-footprint, support and settling goals. The pair contains
**988 audited commands, zero forbidden/malformed commands and zero infrastructure
failures**. Each release also passes the new measured opening/separation,
geometry and supported-stillness gate. The [shape-aware manifest](tamp_release_shape_extents_20261106.json)
retains both cases, source/input/trace hashes and job IDs. Raw artifacts are under
`.tamp-runs/release-shape-extents-20261106/`.

![First-cylinder release refinement, measured heights and command FK](media/tamp_release_pose_search_20261106.png)

The figure compares first-cylinder traces from two separately initialized
shifted development trials. Blue is measured EE height; orange is FK with the
commanded arm joints and other joints measured. Green marks measured opening
and supported settling. It is not a same-state paired benchmark or a full-task
trajectory plot. Reproduce with `scripts/plot_tamp_release_search.py`; plotted
CSV and PDF/PNG exports are retained in the shape-aware raw root's `figures/`.

The shape-aware revision passes **97 focused tests**, zero failures/skips,
including the tilted-cylinder/excess-penetration regression and measured
opening/support gate. The recorded offset-aware release state's old box depth
is rejected, while exact geometry passes at the same 1 mm tolerance without
changing qpos/qvel/qacc/contact buffers or sending commands. The captured task
remains failed. An initial test setup put a box exactly at the zero-margin contact
boundary, producing no contact row; the corrected test uses shallow physical
penetration within tolerance. Both attempts are retained.

This is a successful native primitive/tool reference on two familiar development
layouts. It does **not** update the actual-agent 2/4 baseline, establish observed
execution, or qualify household environments. The release profile stays opt-in.
Next: freeze additional initial placements and failure/recovery cases, repeat the
loaded-binding versus equal-hold comparison after removing the early-release
confound, then run the production agent using Caliban with the qualified policy.

The paper builds at **56 pages**, with no undefined references or citations in
the final TeX pass. Initial-pass reference warnings and existing font warnings
remain in the build output; the final log and figure provenance/CSV hashes are
recorded in the shape-aware manifest. Ruff and whitespace checks pass.
