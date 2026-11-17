# Herman operator console and deployment boundary

The live console receives existing head stereo, arm/odom and hand TF through a
subscriber-only SSH process. Wrist camera produced no frames. Keyboard and touch
controls modify bounded local masked base/hand target previews only. There is no
motor command, enable or robot-stop endpoint; the UI explicitly displays that.

116 targeted HTTP/feedback/session/boundary/CLI tests pass, with separate live
Chromium checks for stereo video, keyboard base/hand previews, touch orientation,
focus-loss input clearing, base-free masks and desktop/mobile layout. No robot
motion commands were sent. Browser photos, browser profiles, raw XML, model
weights and traces remain ignored; no real camera imagery is committed here.

Local bridge changes preserve capture/arrival ages and odometry twist, distinguish
structural freshness from verified units/bus reads, and map child-frame twist
into world planar joint velocity. Guarded sessions require separate arm/odom
captures, verified SI/read validity, full six-component base twist and measured
velocity checks. The motor-reader/unit patch passes a full native ROS build in an isolated
temporary workspace on Herman; its binary is not executed or installed into the
active system. Local bridge changes are not ROS-build or hardware validated. Real robot watchdog/stop and policy transfer are pending.

`tests.json` fingerprints the local source tested. `training-gate.json` instead
records the completed 1,604-test gate of frozen simulator source 2a2ce5c1 and
16-case teacher/source admission (13/16 vs 9/16, all stops confirmed). That receipt
does not qualify later hardware/UI changes. `training-progress.json` is a single
in-progress snapshot, not a final checkpoint comparison or qualification.

See docs/rl/CONTROLLER_DEPLOYMENT.md, TRAINING_MIXTURE.md and
MARS_HARDWARE_READINESS.md for responsibilities, usage and remaining work.

The final preview reset copies measured targets while preserving its original
preview envelope, with no robot state/fault reset. The current console job is
`20261117_120035_064abd`; prior launches were stopped to load reset and snapshot-copy protection.
