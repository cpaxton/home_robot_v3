# Grasp height does not repair Cartesian carry; inspect full pad fit

Reject a height-only planning change. All90 diagnostic episodes use the admitted
initialized retained-source settings and the explicit Cartesian/jerk-limited
analytic reference, with learned calls zero. Existing object geometry, poses,
planner budgets, force/retention/wheel/stop thresholds, torque and friction stay
unchanged. Each episode runs in a fresh serial subprocess. This in-memory
planning intervention is explicitly ineligible for canonical qualification.

| Priority, all30 cases | Transfers | Retention | Gripper bounds | Object moved | Support margin | No plan |
|---|---:|---:|---:|---:|---:|---:|
| Existing lower pinch first |0/30|20|6|2|0|2|
| Center first |0/30|20|4|2|2|2|
| Horizontal,0.25 object-height above center first |0/30|24|2|2|0|2|

No wheel-lift flag or unconfirmed fault stop is recorded. Planning failures with
no execution remain explicit; they are not successful stops or qualified cases.
The public `operation_failed` code masks several distinct semantic causes;
raw diagnostic logs preserve the actual reason. Future public diagnostics
should allowlist those reasons, while continuing to suppress arbitrary private
exception text.

The baseline's six bound failures include four at release planning and two
before grasp. Its measured-jaw guard already permits3mrad of compliant-stop
deflection. In the two20g cases failing at release, stopped driver angles are
about0.3556/0.3545rad while the grip target is zero and the closing actuator is
at its existing−0.08Nm cap. These are post-failure stopped values, not exact
private check states. Do not classify them as small open-servo rounding errors
or relax the guard. The source nominal open limit is0.3491rad.

A separately verified static audit finds a geometric reason to investigate the
axial grasp point. Both pads are the pinned48×16×38.9mm boxes; the fixture box
is16×16×60mm. At the planned measured-open grasp goal and the saved first
pregrasp payload pose, all24 cases with an available frame overlap at least one
fully open pad. The6 unavailable frames remain in the report. This is a complete
oriented-box separating-axis test; negative separation is not a penetration-depth
measurement, nor does it by itself prove the cause of a dynamic failure.

Moving the declared pad-axis point from30mm to36mm or38mm, while leaving the
pads and opening fixed, gives positive separating axes in all24 static cases:
minimum0.296mm and0.939mm respectively. The exact symmetric open-jaw projection
is used, not an assumed equal hand displacement. Native FK checks of all56
available recorded endpoint states match the analytic pad transforms within
2×10⁻¹⁵, and verify the actual admitted box dimensions. Two planning-only cases
have no executed state. The first native audit stopped on a missing optional
physics snapshot in terminal-stop ticks; the corrected reader checks the first
and last available snapshots without dropping failed cases.

[Complete outcomes, per-case loads and artifact identities](../../paper/data/whole_body_control/mars_grasp_priority_v1/results.json)
and the [aperture audit](../../paper/data/whole_body_control/mars_grasp_priority_v1/aperture.json)
are retained. Raw models/traces stay ignored. There is no successful loaded
controller or hardware qualification.

The next declared diagnostic changes only the private axial pinch point: all30
cases at36mm and all30 at38mm, retaining the original height priorities and all
physical limits. First require three exact30mm identity replays against the
saved baseline, including actions and physical snapshots. Reuse the full
baseline only after that check. Keep every failure/admission and a separate
script/source digest. Geometry clearance does not establish reachability,
retention or placement. A promising candidate still needs an ordinary committed
implementation, fresh regressions and integrated replay before any qualification.

All three identity replays now pass exactly; changed-depth episodes are running.
A subsequent three-case read-only contact diagnostic will record solved
pad/object contact forces and geometry, requiring exact physical baseline
replay. This prepares a force/fit diagnosis if changed depth also fails; it is
not another physical parameter sweep.
