# Herman wave preflight — 2026-11-17

The requested next test is a small learned whole-body arm wave with the base
held at its initial pose. A head nod is a separate head driver test: the current
actor outputs base forward/yaw, five arm channels and a gated gripper channel;
it has no head output. Neither a head nod nor the analytic teacher is presented
as evidence that the learned WBC works on hardware.

## Current robot, checked again

An eight-second subscriber-only capture at about 18:19 EST receives 1,600 arm
states, 241 odometry states, 64 left RGB/depth frames, 115 right images and
32 lidar scans. Head, hand and optical TF are available; the wrist image and wrist
optical TF are still missing. The reported joints match the afternoon capture,
including the same joint1/joint2 declared-limit discrepancies. FK/TF consistency
passes, but it is not independent physical joint calibration.

Read-only inspection still finds innate-os source `0d4c350d`, with the affected
`arm_control.cpp` and `robot.cpp` hashes unchanged. The known velocity-unit and
failed-read-validity issues therefore remain present in the source; this does
not prove which binary is installed. No 4400–4409 bridge listener is present.
There is still no attached verified WBC driver, robot command-loss watchdog or
measured whole-body stop path. No motor commands, driver starts, active installs
or robot repositioning were performed.

## Tiny offline wave

`scripts/rehearse_herman.py --case-set wave` uses the normal masked task-pose
preview and measured history. A private FK witness supplies a wrist rotation of
**±0.06 rad (±3.44°)**, via neutral → positive → neutral → negative → neutral.
A two-second initial hold precedes the motion; the reference ends at about
17 seconds, leaving eight seconds for settling in the unchanged 25-second
rehearsal. All base masks stay active at the initial base pose, and the gripper
target stays unchanged. Quintic segments bound reference speed/acceleration/jerk
at **0.03 rad/s, 0.06 rad/s², 0.24 rad/s³**, respectively, or the profile limits
if lower. Those are nominal test-reference limits, not guarantees on actual
policy motion or calibrated robot ratings.

The captured pose remains blocked. Every closed-loop run starts from the same
explicitly synthetic geometry-admitted staging pose used by the prior rehearsal;
no physical staging trajectory is supplied. Full-path joint-margin, collision
and support admission checks run before execution. Teacher and actors receive
identical state/reference/model hashes. The independent measured 5 cm / ±47°
boundary and physical simulator stop check remain active.

A wave must track its moving reference, not just return to its starting point:
the wave interval must complete and all sampled active-interval errors must stay
within the existing 3 mm / 0.02 rad contract, in addition to native end/quiet/safety
success. The checker rejects an otherwise stationary trajectory whose final pose
matches but which misses the intermediate ±0.06 rad targets.

The matched run completes: **teacher 1/1; both actors 0/1; all 3/3 simulator
stops confirmed**. The teacher stays within 0.041 mm / 0.00085 rad peak active
tracking error, with peak measured arm speed 0.0250 rad/s. Both learned actors
cross the independent 5 cm translation boundary by about 4.5 seconds, before
completing the wave; measured peak arm speeds reach 0.276 / 0.289 rad/s. These
are different from the slow nominal reference limits and prevent physical use
of this actor/posture combination. The captured start is still blocked; these
results are from the separate synthetic staged pose.

[Compact results and source delta](../../paper/data/whole_body_control/herman_wave_v1/results.json)
retain failures and bind the previous full source manifest plus changed files.
Fifty focused geometry/gesture/reference/preview/viewer tests pass; runtime parity
and Ruff pass. No full training qualification is claimed. The first local wave
attempt exposed tuple repetition in a new jerk-limit expression; it failed before
rollout, was corrected to numeric multiplication, and the failed run is retained.
Private camera frames, observed home geometry, model binaries and replays remain
under ignored `logs/eval/herman-wave-20261117-v3`; do not commit them.

## Viewer availability

The earlier HTTP processes were no longer running, although their legacy job
records still said running. That caused SSH's `channel ... Connection refused`:
the SSH connection itself succeeded, but Olympia had no localhost listener.
The restored viewers run as independent **transient user services** with
`Restart=on-failure`. This survives the chat tool session, not a workstation
reboot; these are not boot-installed services. The live ROS relay remains bounded
to six hours and can be restarted when disconnected.

```bash
systemctl --user status emet-herman-wave-replay.service emet-herman-console.service
journalctl --user -u emet-herman-wave-replay.service -n 30 --no-pager
```

Port **8771** is recorded simulation; **8770** is the live passive camera/telemetry
console with local previews only. Neither page can drive or stop a robot. A
browser's Stop/Pause replay action is not a hardware stop.

From PowerShell, forward both workstation-local ports:

```powershell
ssh -N -L 8771:127.0.0.1:8771 -L 8770:127.0.0.1:8770 cpaxton@olympia
```

Then open `http://localhost:8771` or `http://localhost:8770`. If the existing SSH
process is still running with the 8771 forward, refreshing that page is enough;
stop that SSH process before opening another tunnel on the same local port.

## Before physical execution

Reconcile the parked-state/model mismatch and validate a staging path; demonstrate
stable tracking of this selected wave/hold envelope; then attach and test the
corrected source-timed SI/read-validity feedback, physical command-loss watchdog
and measured stop interface. Do not stream a learned actor directly to the
position queue or add an ad hoc ROS command publisher to bypass those gaps.
