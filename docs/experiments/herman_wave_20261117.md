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

## Rendered comparison and playback fix

A **25-second, 1536×768, 20 fps** comparison is available at
`http://localhost:8771/video.mp4`, with WebM at `/video.webm`. The viewer's
**Watch / download** link selects a supported format. These files stay in ignored
`logs/eval/`; only their checksums and [validation record](../../paper/data/whole_body_control/herman_wave_v1/video-validation.json)
are committed. This is recorded simulation from the synthetic start, not hardware
video. The three panels show teacher/older actor/newer actor, target and measured
hand axes, close-ups and measured-versus-target rotation plots. Failed recordings
freeze the last active frame with an explicit label; post-stop motion was not
recorded and is not fabricated. Playback is at recorded speed, without magnifying
the robot's motion.

The page originally selected a static captured frame and a head camera that hid
the small wrist motion. It now defaults to the teacher's moving recording and an
external camera. It identifies the initial two-second hold, labels the static
capture, adds hand close-up and target/actual axes, restarts an ended recording on
Play, decodes images before swapping them, and skips obsolete frames to maintain
recorded time over a slow link.

A separate HTTP stall was reproduced: an idle browser preconnection could block
the single-threaded server. Concurrent HTTP handling now sends every graphics
operation and context teardown to one dedicated render worker. Regression tests
cover an idle connection plus active requests, concurrent render calls, graphics
thread identity, and byte-range seeking for both media formats. **19 tests pass.**
Chromium checks verify decoded image pixels change, 400 ms simulated latency
preserves playback timing, rewind works, static capture is explicit, mobile layout
fits, and native video plays/seeks. The test Chromium lacks H.264 support and uses
the WebM fallback; ffmpeg successfully decodes all 500 MP4 frames. No policy,
physics, admission or hardware-motion behavior changed.

To reproduce the MP4 with one rendering worker (choose a fresh output filename):

```bash
PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 15 .venv/bin/python -m emet.visualization.herman_video \
  --rehearsal logs/eval/herman-wave-20261117-v3 \
  --output logs/eval/herman-wave-fresh-comparison.mp4
```

For the optional browser fallback, transcode the generated MP4 to the same basename
with a `.webm` extension using ffmpeg/libvpx-vp9. By default the viewer looks beside
its rehearsal directory for `<directory-name>-comparison.mp4` and the matching
WebM. `--video PATH.mp4` selects a different generated video. The same 8771 SSH
tunnel serves replay frames and the downloadable files.
