# Checked elbow-and-wrist wave — 2026-11-17

The original "wave" was a wrist-only reference. The user correctly pointed out
that this did not visibly exercise the arm. Its recorded teacher elbow travel
was only about 0.033°, versus 6.9° peak-to-peak wrist roll. Playback fixes did not
repair that scenario definition. The wrist-only v1 artifacts remain preserved;
`--case-set wrist-only` retains its geometry for historical use.

The replacement **herman-wave-v2:elbow_wrist_wave** holds the base/gripper targets,
raises the forearm, waves the elbow and wrist, and returns. Private FK witnesses
produce the ordinary Cartesian SE(3)/SE(2) input and preview; no joint witness is
fed to the actor. Joint limits and full-path collision/support checks pass without
clamping, reset injection during execution, or relaxed hand tolerances. Reference
speed/acceleration/jerk limits are 0.25 rad/s, 0.5 rad/s², 2 rad/s³, bounded also
by the robot profile. These are simulation reference settings, not a newly
verified physical hardware envelope.

The reference ends at 22.25 s after a 2 s initial hold; a declared 35 s horizon
allows the full return and quiet hold. The exact captured hardware posture is
still blocked. All three runs use the same separate synthetic staged pose,
scene, initial-state/reference hashes and unchanged actor weights.

## Measured outputs

Pre-step traces now include actual joint positions, active reference-phase time,
FK witness joints, last accepted actuator position targets, requested actions
and measured/commanded velocities. NPZ replays retain these arrays. The viewer
checks measured joints agree with the qpos used to render the robot and displays
measured/reference/actuator target joint travel. A frozen reference phase cannot
claim completion just because physical time reaches the deadline.

| Joint | Full reference travel | Measured teacher travel |
| --- | ---: | ---: |
| Elbow (joint3) | 37.24° | **37.21°** |
| Wrist pitch (joint4) | 25.78° | **29.65°** |
| Wrist roll (joint5) | 34.38° | **35.04°** |

Teacher peak hand tracking error is **1.61 mm / 0.01164 rad** and it passes the
unchanged moving-path 3 mm / 0.02 rad gate plus terminal quiet success. Its peak
measured arm speed is 0.212 rad/s. Joint reference and measured values need not
match exactly: this is Cartesian tracking with physical base/arm compensation,
not joint-space target imitation.

Both learned actors visibly move the arm but fail: the older checkpoint crosses
the independent translation boundary at about 8.1 s; the newer checkpoint stops
for unsafe contact at about 6.75 s. Neither completes the wave or is promoted.
All three simulator stops confirm. No hardware commands or driver changes occur.
[Source-bound result record](../../paper/data/whole_body_control/herman_elbow_wave_v2/results.json).

## Video was rendered and checked

The new comparison is **35 seconds, 1536×768, 10 fps**. It shows teacher and both
learned policies with close-ups and actual elbow/pitch/roll plots: cyan FK
reference, orange measured joints, grey actuator position targets. Ended learned
recordings are explicitly frozen at the last active pose; stopping motion was
not recorded or fabricated. Robot motion is not magnified.

Decoded frames at **0, 6, 10, 15, 22 and 30 seconds** were inspected visually:
the forearm raises, elbow/wrist wave, then returns. Excluding the inset, the
teacher panel's orange robot mesh changes by **5,200–5,576 pixels** at 6–10 s
relative to the start, rather than only moving a target marker. Browser checks
play/scrub the actual clip, read a measured elbow change of −25.5° at 5.75 s,
verify replay progress over 400 ms network latency, and check mobile layout.
The MP4 fully decodes with ffmpeg; Chromium plays/seeks the matching WebM.

Open **http://localhost:8771/video.mp4?recording=elbow-wave-v2** or use
`/video.webm?recording=elbow-wave-v2`. The viewer on 8771 now points to this run
and adds a recording identifier to video links to avoid showing the old clip.
Its transient user service is `emet-herman-arm-wave.service`; the passive live
console remains separate on 8770. The same existing SSH forward works.

Raw video, scene geometry, screenshots and traces remain ignored under
`logs/eval/herman-elbow-wave-v1*` and `logs/recovery/herman-elbow-wave-v1/`.
Only numeric/checksum evidence is committed. This demonstrates visible motion
and exposes learned-controller failures; it does not qualify real robot motion.

Validation: 24 focused gesture/replay/video tests pass, along with browser
playback, image-pixel change, joint readout and seek checks. Ruff and runtime
package parity pass. No new full training qualification is claimed.
