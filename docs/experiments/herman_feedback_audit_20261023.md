# Herman feedback audit, 2026-10-23

Herman responded to passive SSH at `192.168.1.43`. ROS Humble is installed; the
default ROS domain showed `/parameter_events` and `/rosout`, and two serial
devices were enumerated. No drivers were started, no software was deployed, and
no motion, motor-bus command or contact test was performed. The installed driver
revision and any other ROS domain remain unverified.

The local source audit covers `innate-os` revision
`9e10601a124992068762528806f02c56ebc76f55`, under
`ros2_ws/src/maurice_bot/maurice_arm`. Exact file hashes are pinned in
[`innate_os_feedback_units.meta.json`](../patches/innate_os_feedback_units.meta.json).
This identifies local source, not the software actually running on Herman.

## Findings

1. `Robot::readState()` returns raw Present Velocity register values. Both
   `arm_control.cpp` (the active CMake target) and the retained `arm.cpp` convert
   those using `2*pi/4096`, which is the position-encoder scale. The motor manuals
   specify 0.229 revolutions/minute per velocity count: the SI conversion is
   `raw * 0.229 * 2*pi/60`, about 15.63 times the existing value. This can invalidate
   measured-stop and speed-limit checks. See the primary
   [XC330-M288](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#present-velocity128),
   [XL430-W250](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#present-velocity128)
   and [XC430-W150](https://emanual.robotis.com/docs/en/dxl/x/xc430-w150/#present-velocity128)
   register tables.
2. Exhausted state/position reads return zeros, while velocity reads do not check
   transaction success. The publisher can subsequently stamp fabricated stationary
   data as fresh. Missing per-servo data also needs to invalidate the transaction.
3. `JointState.effort` contains current/load proxies rather than calibrated Nm.
   The publisher chooses its scaling from `current_limit`, so a current-capable
   joint without that configured limit can also be mislabeled. XC330 reports
   current, while XL430 reports a load estimate; the latter is explicitly not a
   torque sensor. See the [XC330 current table](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#present-current126)
   and [XL430 load description](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#present-load126).
4. Emet's current bridge forwards arm positions/velocities, drops effort, and
   calls `pack_innate_mars_joint_velocities` without odometry velocity. Its first
   three velocity entries are therefore placeholder zeros, not measured base
   motion. The odometry callback stores pose only. Cached values without source
   capture ages cannot establish a measured whole-body stop.

## Prepared patch and verification

[`innate_os_feedback_units.patch`](../patches/innate_os_feedback_units.patch)
corrects velocity units in both publishers and makes the three reader methods
reject failed transactions or missing servo data. It changes no motor modes,
gains, current limits, position targets or watchdog settings. The patch was
**not applied to the upstream checkout or Herman**.

Validation copies the audited sources into a temporary directory, applies the
patch there, and compiles the actual patched `Robot` reader against an SDK test
double. It tests signed velocity conversion, valid reads, communication failures
and missing servo data. The upstream files remain unchanged:

```bash
taskset -c 14 .venv/bin/python scripts/validate_mars_feedback_patch.py \
  --upstream /home/cpaxton/src/innate-os
```

The [validation receipt](../../paper/data/whole_body_control/herman_feedback_patch_validation.json)
records the patch and validator hashes. These checks passed; the full ROS package
and hardware have not been tested with the patch.

Before a physical driver can satisfy `GuardedControlSession`, the remaining work
includes source-capture freshness and invalid-read propagation through the bridge,
actual odometry twist, typed current/load diagnostics, verified units/signs/modes,
an independent robot-side arm/base watchdog, and measured stopping under command
loss. Force limiting requires calibration; current/load proxies alone do not
establish safe external forces. The existing driver boundary stays disabled.
