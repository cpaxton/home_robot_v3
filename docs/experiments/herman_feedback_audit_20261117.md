# Herman passive feedback audit, 2026-11-17

Herman became reachable at the saved `jetson1@192.168.1.43` address after the
operator powered it on. Hostname is `herman`; ROS Humble and installed arm
artifacts are present. Checked-out innate-os revision is
`0d4c350d83ff1838be5db291dded4b412af9dff5`. This identifies the source tree,
not proof that every installed binary corresponds to that exact source.

**No motor commands, actuator command publishers, driver/service starts or active
software deployments were performed on Herman.** Existing topics were read.
The follow-up compilation below uses a temporary copy, not the active installation.
An earlier same-day connection attempt had failed before the robot was available.

## Middleware and available feedback

The running publisher uses domain 0 and **`rmw_zenoh_cpp`**. Default CLI discovery
using a different RMW saw only logging topics; that was not evidence that drivers
were absent. Repeating with the running process's ROS/Zenoh settings exposes
`/maurice_arm`, Nav2/controller/velocity-smoother nodes and arm/odometry topics.
Only names of copied settings are recorded; secret environment/config values are
not dumped. A Zenoh shared-memory scheduling/watchdog warning does not establish
an actuator command-loss watchdog.

A six-second passive subscription records:

| Topic | Samples | Median arrival interval | Maximum arrival gap | Median publication age |
| --- | ---: | ---: | ---: | ---: |
| `/mars/arm/state` | 1,203 | 4.97 ms (~200 Hz) | 21.48 ms | 0.65 ms |
| `/odom` | 180 | 33.37 ms (~30 Hz) | 38.28 ms | 2.06 ms |
| `/mars/arm/command_state` | 0 | — | — | — |

No arm/odom zero timestamps occur. The command-state channel may publish only
when commands change; absence in this observation window is not proof of a
broken channel. Odometry is zero in this window. Some joint position/velocity
quantization is visible. Publication freshness/rate does **not** prove successful
motor-bus acquisition or correct SI units. `effort` remains an uncalibrated mix
of current/load proxies, not verified Nm or external force.

## Confirmed source issues

The active `arm_control.cpp` and `robot.cpp` hashes exactly match the previously
audited affected files. The unused legacy `arm.cpp` has been removed in this
revision; CMake still builds `arm_control.cpp` into `arm`.

- Velocity publication still applies `2*pi/4096`, the position-encoder conversion,
  to raw velocity counts. The manufacturer-unit correction from the previous
  [audit](herman_feedback_audit_20261023.md) is still relevant to this source.
- Exhausted state/position reads still return zero vectors; velocity reads still
  lack transaction-success validation. Freshly published fabricated stationary
  data can therefore defeat speed/stop checks.
- The copied configuration uses position/extended-position modes and profiles.
  Configuration files do not establish actual live register values or a streamed
  whole-body velocity executor. No motor-register access was attempted.

The live first five joint positions are approximately
`[1.594, -1.746, 1.563, 1.091, -0.020]` rad as published. These are far from the
small simulated reset neighborhood. Joint naming, signs, offsets and model FK
must be validated before treating that observation as a policy-ready state or
choosing a first test posture. No repositioning was requested.

## Updated offline patch

[The current-source patch](../patches/innate_os_feedback_units_herman_20261117.patch)
is pinned by its [metadata](../patches/innate_os_feedback_units_herman_20261117.meta.json).
It targets the active publisher/reader and adds the conversion helper; it omits
the removed legacy file. It changes no gains, limits, motor modes or targets.

Four source files were copied read-only into ignored local storage. On temporary
copies, the established validator applies the patch, compiles the real reader
against an SDK test double, and passes signed conversion, valid-state,
communication-failure and incomplete-servo-data checks. The active timer already
catches read exceptions; skipping invalid publication is not a substitute for
a robot-side command watchdog and verified stop path.

At that initial audit stage no upstream or active robot file was patched and
only SDK-double tests had run. Hardware unit/stop validation has not occurred. Exact source
hashes, passive timing and offline validation are in
`paper/data/whole_body_control/herman_passive_20261117/`; raw source copies remain
ignored under `logs/recovery/herman-passive-20261117/source/`.

Next: review and validate the source fix as a deployment change, forward real
odom twist/source ages through the bridge, implement the guarded executor and
independent command watchdog, then perform separately scoped supervised stop
and low-speed free-space tests. Contact/load calibration and full manipulation
remain later milestones. See [hardware readiness](../rl/MARS_HARDWARE_READINESS.md).

## Follow-up native build and live console

The [deployment division and console](../rl/CONTROLLER_DEPLOYMENT.md) now document
planner, reference, policy, servo and operator responsibilities. A subscriber-only
SSH console receives both head cameras, arm/odom and hand TF; wrist produced no
frames. Its target edits are local previews with no motor/enable/stop endpoint.

The exact same patch also passes a full native `colcon build` of `maurice_arm`
in `/tmp/emet-feedback-build-20261117-g2_cxgr6`, using one compiler thread and the
existing ROS underlay. Build time is 106.83 s; the 26 original package files
remain unchanged. The resulting ARM64 binary is 1,628,288 bytes, SHA-256
`badf57caf493ab685698ff862515e10b0ebbfe38ee7259c9542dcad23675b79d`.
No active installation was changed, no package overlay was sourced into a running
driver, and the new binary was never executed. Source and native-build evidence
is in `paper/data/whole_body_control/herman_operator_console_v1/`.
