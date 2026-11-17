# Mars learned-controller hardware readiness

Updated 2026-11-17 from source, simulation and a new passive robot audit.
Herman is reachable after power-on; source revision is `0d4c350d`, ROS Humble with
Zenoh/domain 0. Existing arm/odom publications arrive at about 200/30 Hz. No
motion commands, driver/service starts, rebuilds or deployments were performed.
[Current audit and offline patch validation](../experiments/herman_feedback_audit_20261117.md).

## Current position

Connectivity and passive telemetry checks now pass. Shadow-policy evaluation
still requires the unit/validity and joint-mapping audit. Learned
whole-body motion and hand-contact tests are not ready to enable yet. There
are two independent tracks: verified robot-side execution/feedback, and acceptable
policy behavior in the specific initial test envelope. Perfect full TAMP success
is not required before a constrained free-space test, but correct stop behavior
and bounded motion in that selected envelope are required.

| Capability | What is implemented/observed | What remains before physical use |
| --- | --- | --- |
| Simulator policy | Fresh cases improve 12/24 → 16/24, safety flags 4 → 1, no longer-trained wheel lift in this set. | Coupled/goal-transition misses, a contact stop and large jerk peaks remain; checkpoint is not promoted. |
| Robot-side execution | `GuardedControlSession` defines SI base/arm velocity callbacks and command slew. | No attached verified hardware driver. Existing bridge is not this whole-body executor. Validate actual servo modes and supported command semantics. |
| Measured feedback | Bridge reads arm q/dq and odometry pose. | Current source drops odometry twist, packs placeholder base velocities, and lacks verified source-capture freshness for stop checks. Typed current/load diagnostics are also needed for later contact work. |
| Units / read validity | Current robot source hashes retain the known velocity conversion and zero-on-failure defects. Updated patch passes offline C++ checks. | Review/build/deploy the fix and verify live units, read validity, signs and capture ages. Source inspection is not proof of compiled-binary behavior. |
| Fault/stop boundary | Loopback-tested latched origin, stale-state/overspeed/stream-gap checks and stop-confirmation interface. | Demonstrate real arm+base stop and independent robot-side watchdog under dropped commands, stale telemetry and client loss. |
| Wheel support / tilt | Simulator checks wheel gaps/loads at physics cadence. | Identify and verify real tilt/support sensing and its stop response; simulator support measurements are not hardware sensors. |
| Contact compliance | Simulator force/wheel monitors and bounded disturbances exist elsewhere. | Calibrate current/load signals and force response. Tracking error alone is not a force limit; current/load proxies are not automatically Nm or external N. |

The current bridge source still stores only pose in `_odom_callback` and calls
`pack_innate_mars_joint_velocities(arm_dq)` without measured base twist. Cached
state republished with a new timestamp cannot establish freshness. These are
integration blockers independent of PPO success.

The published current arm pose is outside the latest narrow training reset
neighborhood. Validate joint/model mapping and select a tested initial envelope;
do not stream the current policy from an arbitrary parked posture.

## First physical sequence

1. **Passive audit and shadow inference:** verify installed driver/profile,
   q/dq/base twist units and timestamps; record actual feedback cadence and
   policy runtime while sending no policy commands.
2. **Driver and stop verification:** connect the real executor to the guarded
   interface; test robot-side timeout, command loss, invalid/stale feedback,
   low limits and measured whole-body stop in a supervised bring-up procedure.
3. **Small free-space test:** selected nearby reachable arm targets, held base,
   conservative speed/acceleration, clear workspace, operator stop available.
   Log measured versus commanded state and settle/drift/jerk. This can precede
   full manipulation qualification once the preceding checks pass.
4. **Bounded whole-body hand tracking:** then exercise coordinated motion within
   the agreed fixed-origin envelope. Existing target limits hold XY and allow
   yaw ±45°; measured hard boundary is 5 cm translation / ±47°, odometry age
   ≤200 ms. The origin never recenters. A violation stops, reports confirmation
   and exits. Verify the installed path rather than relying on the class existing.
5. **Gentle contact and objects:** only after calibrated load response and
   stopping are demonstrated. Hand pushing, grasp/lift/carry and full TAMP are
   separate later tests, not consequences of a free-space success.

The guarded callback contract requires an independent robot watchdog ≤200 ms and
bounded stop confirmation (currently requested with a 0.75 s timeout). Those are
software interface requirements, not measured stopping performance on Herman.
No reliable motion-test ETA is justified until the deployed driver audit and
stop/feedback work are complete. Continuing PPO alone does not finish that work.

References: [guarded session](../../src/emet/robots/innate_mars/control_session.py),
[boundary](../../src/emet/robots/innate_mars/motion_boundary.py),
[passive feedback audit](../experiments/herman_feedback_audit_20261023.md),
[bring-up runbook](../robots/innate_mars_hardware.md),
[current training mixture](TRAINING_MIXTURE.md).
