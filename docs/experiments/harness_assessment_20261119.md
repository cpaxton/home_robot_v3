# First general-purpose harness assessment

The [assessment protocol](../testing/agent_harness_assessment.md) inventories
44 registry entries and freezes 57 case assignments by adapter and evidence
tier. Its frozen runtime is `c9a8cf413aa181be61d2538359a68e11805aec6f`.
The terminal-feedback evaluator recheck uses
`91304072380277f00115c16d3705dfbe983da2a1`.

| Check | Result |
| --- | --- |
| Focused CPU contracts | 490 passed. |
| TAMP workflow's CPU test list | 421 passed. |
| Recorded geometry | 12/12 controls passed and 24 views rendered. |
| Remote model replay | Full history 3/6, current only 1/6. Zero inference errors. Both arms guessed on the absent and ambiguous controls. |
| Privileged stationary Stretch agent | 0/1 task. The model called discovery, skipped planning and supplied an invented plan handle. The executor rejected it with `unknown_plan` and stopped. The cylinder stayed supported outside the goal. Seven audit entries, zero forbidden commands. |
| Live Stretch seen-recall case | Three turns completed and five HTTP requests were captured. The answer was green, and the box remained visible. |
| Live Stretch unseen-recall case | Three turns completed. The model correctly declined to name a color for the absent box. |
| Measured turn in both active cases | Both 90-degree requests were rejected on an unobserved footprint. Floor sensing reported `floor_target_unobservable`. Measured yaw stayed within 0.001 degrees of the earlier sample in both cases. |

The first privileged run took 72.43 wall seconds and 36.73 measured simulator
seconds. The replay uses remote Qwen3-VL-8B-Instruct with unverified server
precision and decoding. The replay corpus is development-only and so is the
active Stretch fixture. The state-only trial exercises the CLI and guarded
physics boundary with privileged object inputs. It gives no observation or
manipulation success credit.

The original stop-only run also reached a typed `unknown_plan` refusal with a
forced final response. Its evaluator was still waiting for the expected partial
result, so the job was cancelled after retaining the logs. The evaluator now
ends on unexpected terminal feedback, and a regression checks this path before
any retry authorization. The recovery recheck is a separate cohort. Both trials finished with
`unexpected_terminal_feedback:execute_task_plan:unknown_plan`. Neither
reached the specified partial-transfer boundary or received a recovery turn.

See the [coverage and per-case figure inputs](../../paper/data/harness_assessment_20261119/README.md).
Managed job receipts preserve the commands and timestamps. Discord awaits an
explicit private test channel. Stationary hardware and observed execution
remain gated by the simulator review and #245 respectively.

The active cases give evidence for camera transport, absent-target handling,
and no-dispatch motion refusal. The changed-view recall gate remains open.
The next isolated navigation comparison is a scripted feasible rotation
with qualified footprint coverage versus the fixed unknown-footprint
refusal, followed by the same agent requests. Keep broad downward
inspection and targeted floor projection separately measurable.

## Recovered cross-simulator assessment

Job `20261119_170502_21b235` recorded five completed verdicts before the host
reboot at 17:50 EST. The available journal does not establish the cause.

| Case | Recovered result | Evidence |
| --- | --- | --- |
| Stretch S0 find | Object and destination gate passed | Supervisor log |
| Molmo rby1 find | Finished and missed the joint object and destination gate | Supervisor log |
| Habitat HM-EQA q2/q6/q12 | 2/3 correct. q6 and q12 passed. | Original episode bundles and model-final metrics |
| Habitat nearest-v2 find | Interrupted, no task verdict | Missing final metric after reboot |

The reboot cleared the original scratch checkout and temporary run directory.
The surviving job logs and all three EQA bundles are archived under
`logs/harness-assessment-resume-20261119/recovered` in the persistent workspace.
The two room verdicts have log-level evidence. Their episode timings, packets,
and phase details remain unavailable. The revised figure includes the
recovered rows, and the [EQA map panel](../../paper/figures/harness_assessment_habitat_eqa_20261119.pdf)
uses the original bundles.

Only the pending Habitat proxy scene was probed after reboot. CUDA was hidden
and the probes were limited to two CPU cores, 16 GiB address space, and 55–75
seconds. The installed Habitat 0.3.3 build first rejected the software EGL
context for lack of a CUDA device. Selecting the default EGL device with
`gpu_device_id=-1` reached Mesa but aborted at LLVM option registration. The
system-loader probe reproduced that abort. The host boot ID stayed stable and
the agent trial remained unscored. The renderer failures are recorded
separately from the unexplained reboot. A compatible software Habitat build
or an isolated simulator host is the next path for the remaining case.

## Stretch floor follow-up

The frozen Stretch source passed 424 CPU motion, floor, and camera contracts.
Mesa llvmpipe rendered the exact fixture with CUDA hidden. The first two-core
clear-floor attempt added 362 observed cells and dispatched navigation before
its 240-second wall limit. The matched repeat ran the occupied and clear cases
serially on four physical CPU cores, with a 360-second wall limit per case.

| Case | Measured result |
| --- | --- |
| Occupied floor, `cap-dev-06` | Passed. Floor capture added 238 cells. The planner returned `rejected_swept_footprint:occupied_footprint` for the drive. The audit recorded zero navigation dispatches and the base stayed within 0.001 m of its start. |
| Clear floor, `cap-dev-05` | Timed out. Floor capture added 364 cells and the bridge accepted one navigation command. A passive state subscriber measured 0.702 m of forward progress at its last sample, with the navigation receipt still running. Final arrival remained unverified. |

The tool bounded the requested 1.6 m drive to 1.5 m. During the passive sampling
window, the clear-floor simulator advanced at 0.038 simulated seconds per wall
second. This records useful motion with a wall-budget failure. Both matched
attempts recorded zero forbidden commands. Job `20261119_213124_7e9921` and the
[tool and pose audit](../../paper/data/harness_assessment_20261119/stretch_floor_pair.json)
preserve the evidence. The raw bundle is under
`logs/harness-assessment-resume-20261119/stretch-floor-pair`.

## Simulator and planning performance

On the clear-floor fixture with Mesa llvmpipe and four physical cores, the
new headless default advanced at **0.540×** real time with the environment
variable unset. The same bridge ran at **0.041×** with
`EMET_SIM_SYNCHRONIZED_RGBD=0`. The default published head RGB and depth from
one capture interval. These were idle bridge probes. Loaded navigation and
frame age still need measurement.

The assisted cylinder-right replay used identical saved RGB-D, grid and
pre-planning state. Reusing validated convex geometry and computing world
bounds once per placed solid cut its first complete loaded sweep from
**133.4 s to 66.1 s**, with 7,500 configuration checks in each run. The
search completed three loaded sweeps instead of one before returning
`support_geometry_unknown`. Both searches took about 275 s and produced no
executable plan. A paired prefix of 256 sampled poses matched all 10,752
admission decisions, subdivisions and collision outcomes. That prefix took
10.0 s before caching and 4.4 s after. Observed planning and hardware latency
still need separate measurements.

The assisted mobile Stretch path now checks the transport endpoint and plans
placement from that endpoint before certifying travel. For a straight segment,
it uses a conservative swept bound for each rigidly carried robot link with the
existing XY and yaw tracking envelope. Only links whose closed bounds lie in
fresh, known-free space use the batch proof. Exact native contact checks,
payload checks, uncovered map queries, and map-lease checks still run at the
original samples.

On the frozen cylinder-right source, the combined changes returned a plan
after **76.2 s** of planning plus **24.2 s** of map construction. The earlier
cached-only replay returned `support_geometry_unknown` after 275 s of
planning. On the same selected carry and frozen map, detailed and batched
checks both returned clear. Batching certified 12 links across the segment,
reduced detailed map queries from **124,200 to 34,200**, and reduced sweep
time from **61.4 s to 43.5 s**.

With fresh-source revalidation enabled, the hard cylinder and far cube
returned plans in 192.9 s and 165.7 s respectively, including recertification.
Both completed the carry in simulation. The cylinder stopped at a gripper
opening collision and the cube returned an internal error during release.
A private release-planning replay from the saved cube state found a feasible
adjustment. The live error did not reproduce in that isolated replay.
Placement remains unverified in both cases.

The raw clock, planning profiles and replay scripts are under
`logs/harness-speed-profile-20261119/`.

### Reused link bounds

The assisted mobile map guard now computes one bound per rigidly carried link.
When that bound proves the link crosses a map face at every sampled pose,
the guard uses the same native-only route as the detailed query. The other
links still use the known-free proof or detailed map checks. Both attachment
hypotheses retain the original native collision checks. Fresh-source
revalidation uses detailed checks. A trial that forwarded the batch proof into
fresh-source revalidation hit an internal error before issuing a plan handle.

The matched cylinder sweep took **44.0 s before reuse and 16.1 s after**.
Both runs visited 3,750 poses with identical recorded collision outcomes,
34,200 detailed map queries and 168,300 logical native-only queries. Reuse
handled 19 links at the segment level, eliminating 142,500 per-pose solid
builds across the two attachment checks. The raw comparison and its inputs
are under `logs/harness-speed-profile-20261120/`.

## Observation-only Stretch controls

Two live loopback simulations ran production CHAT with scripted tool decisions.
The bridge withheld object placements, full scene state, articulation metadata,
the spawn map and simulator clearance queries. It supplied onboard RGB-D and
ideal simulated robot telemetry, including camera poses and odometry. The
wire audit found no withheld fields in the sampled state or the 14 captured
CHAT requests.

| Floor case | Result |
| --- | --- |
| Occupied | Downward RGB-D added 268 observed cells. The footprint check refused the drive in 0.3 s with no navigation dispatch or measured translation. |
| Clear | Downward RGB-D added 367 observed cells. One navigation command reached the distance tolerance with 1.470 m of travel for the requested 1.5 m. The drive tool took 41.3 s and the measured heading changed by about 9°. |

In both cases, discovery and pick/place planning returned
`observed_scene_unavailable`, with no plan handle or task actuation.
The live observed source and motion refiner remain open. The assisted carry
benchmarks use native MuJoCo state and an external simulated depth scan.

CHAT's top-level defaults previously overwrote the evaluator's requested
manipulation mode. The evaluator now passes physical mode and observed inputs
through the CHAT config merge explicitly. The focused suite passed 597 checks,
and a separate regression exercises this config merge. The raw captures and
action audit are under `logs/harness-observed-20261120/`.

### Stationary mapping from head encoders

`scripts/eval_stretch_encoder_floor.py` exercises the calibrated robot-only
head transform with live, registered onboard RGB-D in two stationary Stretch
simulator cases. The floor classifier receives the camera intrinsics, depth,
and the base-relative camera transform computed from same-capture motor
readouts. The evaluator discards the renderer's camera pose and full robot
configuration. The fixed base frame and nominal floor height are declared
calibration. Wheel readouts stayed within the 2 mm stationary check.

At two head tilts, the clear case produced **330 observed floor cells and zero
raised cells** in the requested corridor. The occupied case produced **297
observed floor cells and 33 raised cells**. This qualifies the stationary floor
evidence check in the development fixture. Navigation from a moving base
still requires a separately estimated, capture-aligned base pose.

The 93 focused checks include a base-relative FK regression with a translated
and rotated simulator base and no localized camera pose supplied to the
consumer. RGB-D captures, motor packets and floor grids are under
`logs/harness-observed-20261120/encoder-floor/`.

### Moving mapping from wheel and head encoders

The same evaluator with `--moving` captures unwrapped left and right wheel
angles alongside the head encoders and registered RGB-D. It integrates the wheel
angles in the initial base frame using the nominal wheel radius and track
width. Each depth projection composes that estimate with the head transform for
the same capture. The simulator's chassis pose is scored after mapping.

The prescribed drive and turn finished **0.487 m** from the start with a
**29.31°** heading change. Maximum planar error across both trajectories was
**12.3 mm** and maximum heading error was **0.92°**. The clear corridor had 289
observed floor cells and zero raised cells. The occupied corridor had 256
observed floor cells and 33 raised cells. The stationary recheck reproduced its
original counts. Logs and frame data are under
`logs/harness-observed-20261120/encoder-floor-moving-long/`.

An offline replay of the 53 frames at the driving head tilt tested the existing
RGB-D tracker with and without a wheel-motion initializer. Both variants
reported tracking throughout both cases. Maximum position error with images
alone was 1.52 m in the clear case and 0.15 m in the occupied case. With the
wheel initializer, those errors were 0.43 m and 0.09 m. Registration took a
median 42–43 ms per frame on the CPU. The tracker reported successful
registration even where accumulated drift exceeded the commanded travel.
Wheel dead reckoning gave the smaller error on this fixture. The replay
excluded the separate final head tilt and is saved as `rgbd-replay.json`.

The moving trial uses short, scripted motion on level simulated ground. It
leaves production navigation and grasp execution, hardware alignment, and
detection of wheel slip and accumulated drift as separate gates.

### Live CHAT and waypoint control from encoder feedback

The Stretch encoder evaluation bridge uses one capture-aligned pose estimate
for the agent's RGB-D map, the regular Stretch waypoint controller, and its
settling receipt. Its local origin comes from the first wheel sample. The
bridge publishes no simulator chassis or camera poses, scene state, start
map or clearance query. The scripted local model drove production CHAT
through discovery, planning, floor inspection and a 1.5 m forward request.
The paired run used the standard simulator head-camera self mask.

The clear and occupied cases added 353 and 215 observed cells respectively.
Both requests returned `rejected_swept_footprint:unobserved_footprint` at the
start, identifying the same four rear cells. There were zero CHAT navigation
dispatches, and the held-out chassis remained stationary. Discovery and
pick-and-place planning returned `observed_scene_unavailable`. Fourteen model
requests and the observation streams passed the audit for withheld fields.
A stationary visibility survey with the robot rendered tested 21 pan/tilt
combinations. Some rear cells had measured returns. The positive-side rear
corner at approximately (−0.3 m, +0.2 m) had none in those views.

An independent 0.5 m waypoint on the declared-clear fixture exercised
controller feedback without the agent admission step. It settled in 8.9 s
with an encoder estimate of 43.7 cm and a held-out displacement of 42.7 cm.
The controller accepted the estimate under its 7 cm exploration tolerance.
The held-out endpoint fell approximately 3 mm outside that band. The
exploration receipt therefore does not qualify physical endpoint accuracy.
Cold-start clearance and an arrival margin for odometry error are open.

Evidence and the action audit are under
`logs/harness-observed-20261120/encoder-live-final/`. The focused suite passed
89 tests, with seven RoboCasa-dependent tests skipped.

### Measured launch-area survey

`scripts/eval_stretch_launch_survey.py` captures a simulated depth survey before
Stretch is placed in the fixture. It registers the survey to the encoder-derived
starting frame using four floor markers measured in both the survey and the
onboard RGB-D. The largest fitted marker residual was 5.4 mm in both fixtures.
Only the measured points within 55 cm of the start enter the floor classifier.
The survey expires and is bound to one map scope. Measured raised surfaces and
drops still take precedence over surveyed floor returns.

Without a survey, both starting footprints failed. A 12 cm error in one
anchor was rejected and left both footprints blocked. With the aligned
survey, the production sweep admitted the nominal 1.5 m clear route and
rejected the occupied route with `occupied_footprint`. This check submitted no
drive command. The two rendered cases took 11.5 s with CPU-only rendering, a
four-core affinity and about 2.4 GiB peak resident memory. The focused mapping
suite passed 26 tests, including expiry of the cached survey.

RGB-D, measured markers, sweep results and resource measurements are under
`logs/harness-observed-20261120/launch-survey-verified/` and the adjacent logs.

The live encoder-fed CHAT pair used a fresh survey and onboard marker capture
in each case. The largest marker residual was 4.7 mm. The occupied request was
rejected on a measured raised surface with zero drive commands. The clear
request dispatched one wheel-driven waypoint and reported arrival after
33.3 s. For the 1.50 m request, the encoder estimate was 1.435 m and the
held-out chassis traveled 1.409 m. The surveyed start resolves the earlier
clearance blocker in these fixtures. Arrival accuracy remains open.

The two live trials ran sequentially with CPU-only rendering, four allowed
cores and roughly 2 GiB peak resident memory per trial. The action, pose and
resource audit is in `logs/harness-observed-20261120/launch-survey-live/audit.json`.

### Arrival allowance and feedback loss

The encoder trial accepts an explicit position and heading error allowance.
The waypoint controller targets the policy tolerance minus that allowance.
The arrival monitor reports both the measured residual and its upper limit,
formed by adding the allowance. Success still requires fresh, settled
samples. A command with no room for its allowance fails before drive dispatch.

For these flat-floor runs, the trial allowance is 4 cm and 0.025 rad. The
previous long straight run had peak errors of about 3.75 cm and 0.019 rad
against its held-out chassis trace. The allowance is provisional, and each
new run is scored against separate held-out measurements. Qualification on
other surfaces, against sustained slip, and on hardware remains open.

In the live dropout control, the harness withheld encoder pose updates after
the base started moving. The bridge cancelled the velocity drive in 0.92 s,
confirmed wheel stop in 0.97 s and published a failed receipt with
`stale navigation pose` during the dropout. Held-out motion after confirmed
stop was 1.4 mm, with zero additional drive dispatches.

The 1.50 m clear drive settled in 30.2 s with a held-out endpoint error of
5.0 cm, within the 7 cm exploration tolerance. The requested 30° turn
produced a held-out 22.5° turn, within the 8.6° exploration tolerance. The
single-view occupied repeat stopped on one unknown floor cell. A repeat
with a second measured head view stopped on the raised obstacle. Each
occupied trial dispatched zero drive commands.

The five runs were sequential with four allowed CPU cores and CUDA
disabled. The focused suite passed 35 tests and skipped two that require
RoboCasa. The scored receipts, held-out endpoints, stop trace and resource
counts are in `logs/harness-observed-20261120/arrival-bounds/audit.json`.

### Independent RGB-D slip check

`scripts/eval_stretch_slip.py` compares measured wheel travel with a fixed
RGB-D camera looking at two calibrated chassis markers. The guard compares
position and heading in the two sources' relative starting frames. It
latches disagreements, missing views and stale or unaligned samples for a
motion. Fresh encoder readings confirm wheel rest. Held-out chassis pose
is used only to score residual motion after the stop.

On normal traction, the camera measured 32.04 cm and the held-out chassis
moved 32.05 cm. On the low-traction fixture, wheel odometry reported 4.20 cm
when the guard stopped the drive. The camera saw effectively zero movement
and the held-out chassis confirmed just 0.19 mm of travel. On normal
traction with depth disabled mid-drive, the guard stopped after 7.22 cm of
held-out chassis travel because the independent pose was unavailable. The
three runs applied no further drive after the terminal guard decision.

The bench initially stopped during a normal drive when part of the robot
obscured one marker. The final fixture raises the camera for an unobstructed
view, and the detector rejects partially visible markers based on their
projected area. This confines the result to the instrumented bench. The
live agent adapter uses this detector in the surveyed simulator controls below.

The three cases took 24.8 s total with four allowed CPU cores, CUDA
disabled and about 2.3 GiB peak resident memory. RGB-D, wheel and reference
traces, and held-out scoring are under
`logs/harness-observed-20261120/slip-bench-final/`.

### Tracked Stretch runner and measured floor recovery

`scripts/eval_stretch_navigation_chat.py` builds the surveyed development
fixture, runs production CHAT with local scripted replies, and invokes the
tracked scorer. It saves the exact request bodies, both survey captures,
command audit and independent pose trace. The first clear and occupied
packaged controls both stopped on unknown floor after one extra head view
and a second sweep. These outcomes are preserved in
`logs/harness-observed-20261120/stretch-tracked-recovery/`.

Measured-floor mode now retains valid depth returns until voxel pooling.
The clear rerun dispatched one waypoint and finished 4.8 cm short of the
1.5 m target by held-out measurement, within the 7 cm tolerance. The
occupied rerun reported a raised surface and dispatched no drive. Both
passed their tracked scorers. The 396 focused tests cover rechecking the
same goal after one targeted floor view, retaining no-drive on a
still-unknown cell or occupied footprint, and preserving valid depth
before pooling. The scored reruns are under
`logs/harness-observed-20261120/stretch-tracked-observed-depth/`.

### Live independent motion guard

The surveyed Stretch runner now checks wheel-derived motion against the
calibrated fixed RGB-D camera at waypoint dispatch and during execution.
Each comparison uses a shared clock and capture interval from the same
snapshot. The fixed camera observes two chassis markers. MuJoCo chassis
pose is recorded separately for scoring. The three cases use the production
CHAT tool path with scripted local replies.

| Case | Scored outcome |
| --- | --- |
| Clear drive | Reached the 45 cm waypoint within 3.4 cm by held-out measurement, inside the 7 cm limit. Logged 128 independent motion comparisons. |
| Wheel slip | Aborted with `wheel_motion_disagreement`. The wheels reported 4.46 cm of travel and the held-out chassis moved 0.15 mm. |
| Reference loss | Aborted with `independent_pose_unavailable` after 7.16 cm of held-out travel. Fresh encoder feedback confirmed rest within 43 ms of the injected loss. |

Each fault run confirmed wheel rest after a single dispatch with no retry.
The loss case withheld the independent pose mid-drive while wheel feedback
and head-camera mapping remained available. The results qualify the surveyed
simulation fixture with its fixed camera and marker calibration. Each case
took 32–40 s wall time with four allowed CPU cores and CUDA disabled, with
about 2.1 GiB peak resident memory. The clear motion itself took 9.2 s.
The focused command, encoder and motion-guard suite passed all 58 tests.
The scorer verdicts, per-frame comparisons and stop traces are under
`logs/harness-observed-20261120/stretch-live-witness/`.

### Real-model Stretch terminal controls

The configured Caliban hostname did not resolve from the workstation. The
same Qwen3-VL-8B-Instruct checkpoint ran from the local cache on a loopback
endpoint in BF16 with greedy decoding. The robot used the surveyed Stretch
adapter and software rendering. A private loopback recorder saved the exact
camera packets, tool requests and responses. The scorer found no private
simulator fields in the robot or model packets.

| Case | Independent result | Turn wall | Model wall |
| --- | --- | --- | --- |
| Read-only scene and photo | The model called `send_image` and identified the blue, square-shaped block. The actuation trace was empty during the turn. | 3.8 s | 3.4 s |
| Guarded forward drive | The model called `move_forward` once for 0.45 m. The guard logged 126 matching camera and wheel comparisons. Held-out travel was 41.75 cm and endpoint error was 3.25 cm, within the 7 cm tolerance. | 12.5 s | 3.0 s |

The model saw the drive result and reported the 42.6 cm encoder estimate.
Motion took 9.2 s of the terminal turn. Including fixture startup and
cleanup, the two cases took 34.1 and 42.5 s. The simulator processes used
four allowed CPU cores with CUDA disabled and peaked at about 2.1 GiB RSS.
The GPU reached 19.4 GiB in the exclusive model job. The raw runs and
checkpoint revision are under
`logs/harness-observed-20261120/stretch-real-model-local-v1/`. These are
development controls with explicitly scoped tools, on the assessment
worktree. The interactive Discord session and hardware qualification are
still pending. The focused gateway, CHAT and Discord channel suite passed
39 tests. The Discord checks exercised an absent channel, reconnect and
cross-channel mentions without contacting Discord.

### Discord operator trial startup

The managed read-only trial connected Stretch3005 to the exact test channel
and Discord confirmed delivery of its startup greeting. The bridge identified
itself as Stretch MuJoCo with encoder pose, and its session advertised no
teleport capability. The CHAT tool list contained `send_image`. The local
BF16 Qwen endpoint served this run while Caliban's model ports were down.
The launch survey registered 30,929 points with a 4.6 mm maximum anchor
residual. The trial artifacts are under
`logs/harness-observed-20261120/stretch-discord-operator-v1/`.
Operator-turn evidence is scored separately from startup.

The first operator question arrived after the initial session ended. An
explicit replay through the shared input queue exposed stale routing examples
that suggested tools outside the read-only tool list. The model tried an
unavailable tool and returned malformed JSON on its next round. With the
camera-only prompt, the same question got a correct scene description in
1.6 s from one image-bearing request and zero navigation dispatches. Discord
confirmed delivery of the text reply. This replay verifies the camera and
outbound path. Live inbound, photo delivery and follow-up still have their
own trial checks. The retained attempts are in
`logs/harness-observed-20261120/stretch-discord-operator-v2/` and
`logs/harness-observed-20261120/stretch-discord-operator-v3/`.

The operator's live photo request reached CHAT. The model called
`send_image`; Discord delivered a 240 × 424 PNG 3.1 s after the request,
and the operator confirmed it looked correct. The agent turn itself took
2.4 s. No base waypoint was dispatched in the read-only session.

### Operator stop control

The bounded simulator profile binds an exact `stop` message ahead of the
conversation queue. The active navigation wait cancels the original command
and obtains the terminal stop receipt. The session then keeps drive disabled.
The separate scripted control triggered the same cancellation path after the
encoder estimate passed 7 cm on a requested 45 cm drive. Each of two runs
sent one waypoint and confirmed wheel rest 51–76 ms after the local stop
request. Held-out travel was 7.3–7.4 cm, with about 1.3 mm of motion after
confirmed stop. The focused command, Discord and CHAT suite passed 60 tests.
The two runs are under
`logs/harness-observed-20261120/stretch-operator-stop-control/` and
`logs/harness-observed-20261120/stretch-operator-stop-control-verified/`.
They measure the simulator stop response. The Discord transport for a stop
while driving still needs its live operator trial.

### Active camera views for Discord

In a reopened read-only session, the operator asked a relation follow-up
about a blue square and asked about an absent red mug. The agent answered
from the image and declined to invent a location for the mug. Its responses
to looking around and exploring exposed the practical limit of that profile.

The active fixture adds a green cylinder on the left and red box on the
right, outside the forward route. The terminal real-model look-around
control selected pans of +0.60 and -0.60 radians, produced three distinct
decision images and correctly described both landmarks in 8.0 s. The scorer
recorded two measured head arrivals and zero base waypoints. The two head
views are published individually in Discord. The active profile also accepts
one guarded forward step for an explicit operator exploration request. The
control and operator artifacts are under
`logs/harness-observed-20261120/stretch-discord-active-observation-v1/`.

### Full Stretch CHAT on the default table

The full operator session uses the stock table, two free-body objects and
the normal CHAT registry on the observed simulator bridge. DynaMem and
visual servo perception loaded cached CLIP ViT-B/32 and YOLOE on CPU.
The prior operator map request was replayed through the shared input queue.
Qwen selected `send_map_snapshot`, which rendered a map with 292 explored
cells at 10 cm resolution from measured depth. Discord delivered the 398 ×
512 PNG. The map turn took 4.4 s and dispatched zero base waypoints. The
artifacts are under
`logs/harness-observed-20261120/stretch-discord-full-table-v1/`.

The live scene question exposed a second malformed model reply. CHAT now
offers one reply-only retry and relays a visible fallback if the retry also
fails. Two regression cases check that this path dispatches no tools. The
focused prompt and loop suite passed 40 tests. The first replay answered
from the floor view because controller startup reset the camera angle. The
next run measured the table-facing head pose after startup and recorded a
new RGB-D frame. Qwen described the red cylinder and blue cube on the brown
table in 2.1 s, and Discord confirmed delivery. The exact model image and
reply are under `logs/harness-observed-20261120/stretch-discord-full-table-v3/`.

### Automated Stretch CHAT and motion regression

On 2026-11-21 the focused suite passed 168 tests. A full Qwen3-VL-8B terminal
session correctly described the table scene, identified the object on the
right and reported that the red mug was absent. The production map tool
produced a 455 × 512 occupancy PNG with 405 explored cells. All four turns
received measured camera input and dispatched zero base waypoints. The initial
scorer rejected the valid phrase “brown surface.” The original audit was
retained, the same packets passed the corrected scorer, and a fresh full run
exited zero. In the fresh run, the model overstated the map's detail by saying
it displayed the red cylinder and blue cube individually. The PNG shows
occupied and explored space.

| Scripted control | Scored outcome |
| --- | --- |
| Clear 45 cm drive | One guarded waypoint succeeded with 3.2 cm held-out endpoint error. |
| Occupied route | The floor recheck found a raised surface and issued zero drive commands. |
| Operator cancellation | One waypoint, wheel rest confirmed 43 ms after the local request and less than 1 mm travel after confirmed stop. |

Runs used software rendering on four allowed CPU cores. The loopback model
ran under the exclusive GPU lease and peaked at 21.2 GiB. The JUnit report,
motion audits and first run are under
`logs/harness-observed-20261120/stretch-automated-20261121/`. The clean full
CHAT rerun is under
`logs/harness-observed-20261120/stretch-automated-20261121-verified/`.

### Object recovery after a head-view change

The `model-object-recovery` case tests retrieval from measured semantic
memory and confirmation from a new RGB-D view. On the first trial, the
agent reported the hidden block's remembered location 3.7 cm from the
fixture center. It then tried to turn the base for the head-only request.
The independent motion guard stopped the waypoint on wheel disagreement,
and the block stayed out of the camera image.

After adding routing for earlier head views, a repeat used `describe_scene`
to return to the prior table view. The matching RGB-D mask went from 580
pixels on the block to zero while looking away and back to 581 on return.
The agent identified the recovered block on the table and dispatched zero
base waypoints. The initial and corrected trials are under
`logs/harness-observed-20261120/stretch-object-recovery-20261121/` and
`logs/harness-observed-20261120/stretch-object-recovery-20261121-head-routing/`.
The final prompt preserved ordinary look-at routing and passed the same
scenario. It recovered the location to 3.7 cm and the RGB-D mask went from
571 pixels to zero and back to 571. All 55 focused prompt, CHAT and gaze
tests passed. The final simulator run is under
`logs/harness-observed-20261120/stretch-object-recovery-20261121-final/`.

### Household task baseline on the full CHAT agent

The first `model-table-transfer` run asked the same Qwen3-VL-8B CHAT agent
to place the red cylinder on a green tabletop region and protect the blue
cube. RGB-D and encoder timing came from the surveyed Stretch bridge. Two
free-body pose traces were recorded on the scorer side. The model chose
`pick_place`, which returned `observed_scene_unavailable` before actuation.
The independent scorer confirmed that both objects remained in place and
the base dispatched zero waypoints. The task scored incomplete and the safe
stop passed. The live `ObservedNativeAdapter` still needs a bound measured
scene source and refiner for this path.

The focused observed-adapter and CHAT contracts passed 141 tests. The
terminal receipts, raw model packets and independent body traces are under
`logs/harness-observed-20261120/stretch-table-transfer-20261121/`.

### Sourccey Mk.V simulator bridge

`scripts/smoke_sourccey_bridge.py` verified advancing 640×480 front-left
RGB-D, front-right RGB and all 16 joint channels through the normal
loopback simulator bridge. The default spawn has little color contrast.
An absolute table-facing command turned the planar base 81.4° clockwise
using the simulated velocity drive and brought the table into view. All
five configured cameras rendered separately. The test used the pinned
meshes from the SHA-verified local LFS cache. The model and trial use
simulator localization and planar kinematics.

The bridge receipt and sensor captures are in
`logs/harness-observed-20261120/sourccey-velocity-bridge/`. The individual
model-camera renders are in
`logs/harness-observed-20261120/sourccey-sensors/rendered-cameras/`.

## Expanded simulator assessment

The `eval/agent-harness-expanded-testing` branch freezes the previous
assessment at `c4988b6d`. The expanded run artifacts are in
`logs/harness-expanded-testing-20261121/` in the workspace. The simulator
controls ran serially on four allowed CPU cores. The model trials used the
local Qwen3-VL-8B-Instruct checkpoint `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
with BF16, greedy decoding, and the exclusive GPU lease. The final focused
suite passed all 179 tests.
The `figures/` subdirectory contains the capability matrix, case and
milestone CSVs, latency panels, and the changed-object camera comparison.

| Case | Result | Independent evidence |
| --- | --- | --- |
| Stretch tabletop capture and expired packet | Sensing control passed | Measured green patch, 2.3 mm peak plane residual, synchronized arm estimate, stale packet rejected |
| Stretch tabletop sweep | Three captures measured | 16.8 mm maximum camera separation. Two side views lost target or destination association. The whole cylinder remained unqualified |
| Stretch tabletop transfer repeat | Incomplete with safe refusal | The agent selected `pick_place`. No lift, release, or base dispatch occurred. The blue cube stayed in place |
| Free-form changed-object recovery | Failed | The block moved 20.4 cm. The agent recalled the previous position within 3.8 cm, sent a base turn, and had no current-view grounding provider |
| Explicit head revisit after the same fixture change | Passed | The visible blue mask went from 575 pixels to zero to 409 at the new location. The agent identified it on the table and dispatched zero base waypoints |
| Sourccey read-only CHAT | Passed | The model received a camera frame, the occupancy tool delivered a map with 76 explored cells, and the turn audit recorded zero commands |
| RoboCasa365 Stretch fixture validation | 12 of 12 passed | Metric RGB-D renders and untouched upstream predicate checks on initial and constructed positive states |

The explicit head revisit took 75.5 s from CHAT setup through the final
turn. Reported model calls consumed 24.0 s and the independent turn trace
advanced 20.2 simulated seconds. The longer free-form run reached 23,812 MiB
of GPU memory. No physical plan ran in either tabletop transfer trial.

The kitchen preflights took 82.1 s in total on the pinned RoboCasa365
v1.0.1 source and MuJoCo 3.3.1. Each output keeps the Stretch include and
official fixture-joint initialization with the scene XML. The predicate
controls are scorer-only. Kitchen execution remains gated on measured
reachability and the corresponding observed action. The held-out placements
are frozen for the model baseline and later training comparisons.

The core manipulation blocker is a sensor view that can bound the full
payload while tracking its pose and keeping collision and support evidence
qualified. The stale-packet result covers capture admission. The public
preflight and post-grasp stop still require a live observed executor.
Private Discord motion and stationary Stretch and Innate Mars recordings
remain pending.

### Tabletop manipulation and RoboCasa reachability follow-up

The observed CHAT transfer stops at the adapter gate because the session has
no measured manipulation provider bound. The matched tabletop control used
simulator state for geometry and ran the native contact controller on the
same scene and green placement region. It grasped and placed the cylinder
in 17.6 s of wall time, with a stable release at `[0.049, -0.709, 0.520]` m.
The live observed plan and command path still need qualification.

| Offline RGB-D control | Associated views | Rigid reference | Whole-object bound |
| --- | ---: | --- | --- |
| Plain cylinder resting on table | 5 | Insufficient landmarks | Unbounded |
| High-contrast textured cylinder resting on table | 7 | 15 and 24 landmarks in two views | Unbounded |
| Plain cylinder suspended 12 cm above the table | 5 | Insufficient distinct landmarks | Candidate envelope |

The external camera array provided up to 84 cm of separation. Reusing the
destination plane across a static scan let close views contribute when their
images cropped out the green patch. A 5 mm depth-margin sensitivity check
also left the supported object unbounded. The conservative enclosure algorithm
admits the connected occupied and unknown component. Contact with the table
leaves unresolved possible volume at the support surface. Texture supplies a
tracking reference and a separate support-aware envelope is still required.
The suspended control addresses enclosure. The high-contrast control reaches
reference-feature admission. Both precede tracking through grasp.

The reach screen loaded the original archived RoboCasa XML, Stretch includes
and fixture-joint states. A fresh generation with the same manifest seeds
changed objects and poses in the sink cases, so retain the exact archived
scene files for paired comparisons. All 12 original fixtures were screened.
The frozen start and the seven sampled in-place yaw angles yielded zero
qualifying poses across 35 reference points under the tested orientations.
Translated stances yielded 27 pose solutions. These probes use evaluator-only
coordinates and measure endpoint contacts under the sampled geometry.

The development sink center hit a distractor. Seven of eight off-center
probe points had a pose solution with no detected scene contact, and the
80 mm raised pickup pregrasp also cleared the endpoint check. The sink
pilot needs a head scan, guarded translated stances, fresh evidence at a
stopped handoff, and separate source-support and destination-containment
checks. The drawer pilot needs handle and slide-axis perception, a
guarded pull and measured opening. The condiment pilot still needs feasible
cabinet access and multi-object carry and placement.

The run used four CPU cores and passed 100 focused tests. The native control
receipt, RGB-D captures, original-fixture reach screen and sink offset
probe are in `logs/harness-robocasa-followup-20261121-Vd4Y/`.

### Development sink handoff controls

The frozen RoboCasa development start had 216.3 mm of modeled base-center
clearance against the 220 mm guard. It refused at route admission. A separate
development reset shifted the base 50 mm into the aisle and produced a sampled
eight-waypoint route to the pickup.

The first unloaded physics control timed out on the opening turn and confirmed
wheel rest. The second control applied the archived wheel actuators' gear scale
and friction feed-forward. It reached the pickup pose in 28.45 simulated
seconds and 56.36 s wall time. The guard checked 568 measured poses and the
minimum modeled base-center clearance was 266.5 mm. Arrival was within
14.2 mm and 0.0204 rad, with wheel and base rest confirmed.

The calibrated camera sweeps exposed the tongs from the frozen start and at
the pickup stance. The stopped pickup render contained 550 target depth
pixels. The best stopped landing view contained 1,703 basin depth pixels
within 80 mm of its candidate placement center. The scorer used simulator
segmentation to count pixels. Saved camera packets contain RGB, metric depth,
intrinsics and camera pose.

Five landing stances cleared the unloaded route. The original arm trajectories
collided with the counter or sink, and the higher target alone also failed.
Raising the stowed lift to 1.04 m before the wrist and extension sequence
cleared the sampled above-rim empty-arm paths at all five stances. The pickup
hover cleared an empty-arm path with the gripper open.

Three pickup controls at the declared pickup reset dispatched zero arm or
gripper commands. The legacy world-frame pregrasp returned 189 IK failures.
Aligning the withdrawal direction to the base yielded 147 IK failures and
42 collision rejections involving the counter or coffee machine. Raising the
pregrasp offset to 100 mm produced 189 IK failures. The tongs still require a
grasp and approach that fit the local fixture geometry.

The sink artifacts, measured route, geometry receipts and stopped camera
comparison are in `logs/harness-robocasa-sink-20261121-Pv3S/`. The controls
ran serially on four CPU cores.

### PandaOmron counter-to-sink control

The RoboCasa365 development sink pilot uses its native PandaOmron embodiment.
The paired control restored the archived tongs, counter bottle, sink jug,
62 fixture joint positions and the archived offsets of the coffee machine,
paper towel holder and toaster. It starts with the base stopped at
`[0.92, -2.43, π]` and the torso at 0.16 m. The controller uses simulator
state to choose the grasp, check sampled paths and select a landing point.
The rollout sends seven arm joint targets, a gripper command and a torso
target through native RoboSuite physics. The object moves through contact and
gravity after the reset.

On the verified run, PandaOmron lifted the tongs 19.8 cm and maintained both
pad contacts throughout the 132 sampled commands from lift through descent.
The tongs came to rest with their horizontal extent inside the basin. The
released object passed the native `PickPlaceCounterToSink` predicate after
the gripper withdrew. The control recorded zero unintended robot contacts
at its 20 Hz sampling rate. The run took 17.0 seconds wall time for 23.55
simulated seconds including the stopped settling intervals. A fresh process
repeated the pass in 20.3 seconds wall time. Both runs averaged about one
CPU core with rendering disabled.

The initial pinch-height sweep also showed that reaching the handle does
not guarantee retention. At the same handle offset, a 7 mm approach lifted
the tongs 4.5 cm before losing them. A 4 mm approach retained them through
release. The final verification uses a pinch near the projected center of
mass and a landing farther into the basin for full footprint containment.

The command and observation traces are in
`logs/harness-robocasa-panda-verified-RGPV/` under the main checkout. Reuse the
pinned RoboCasa365 environment from preflight with the archived
`PickPlaceCounterToSink-development-0` directory as input.

```bash
python scripts/eval_robocasa_panda_sink.py "$archived_fixture" "$new_output_directory"
```

This qualifies native contact execution in one development fixture from an
assisted reset. The public-tool test requires a PandaOmron action bridge,
fresh RGB-D and synchronized robot state before a held-out rollout.

The PandaOmron bridge replayed the reference trajectory through the shared
robot client. Two replays passed the upstream predicate and full horizontal
basin containment. On the repeat, the bridge accepted 272 distinct arm and
gripper target packets, the tongs rose 19.9 cm, and the scorer recorded zero
unintended robot contact frames. The independent pose and contact trace is
in `logs/harness-pandaomron-bridge-SqW6/contact-repeat/`.

The production server smoke verified synchronized front RGB-D, wrist RGB,
camera calibration and joint feedback. It moved the measured arm and torso,
closed and opened the gripper, and rejected malformed and out-of-range
targets. The public image tool captured a fresh view before any actuation.
The packets, images and receipts are in
`logs/harness-pandaomron-bridge-SqW6/harness-smoke/`.

### Production CHAT on the PandaOmron sink fixture

The shared agent now skips navigation and gaze startup commands when a
bridge declares those capabilities unavailable. PandaOmron publishes the
native right-side scene camera for this rollout because the centered view
looks through the arm at the pickup reset. RoboSuite's visual-mesh settings
remove collision overlays from the images. RGB-D, wrist RGB and joint feedback
come from the same stamped snapshot. The client preserves its own receive time
so the observed evidence gate can reject stale frames.

With scripted tool selection, production CHAT captured an image and called
`scene_tasks` and `plan_pick_place`. The model request carried the native
camera image. The two task tools returned `observed_scene_unavailable`. The
scorer measured zero actuation and no lift. This run failed the benchmark
transfer predicate. Its transcript, requests, sensor packet and independent
physics score are in
`logs/harness-pandaomron-agent-e0bv/scripted/`.

A sensor-only geometry diagnostic used manual pixel polygons on that capture.
It measured approximately 22 cm of visible tongs surface and a 23 cm difference
in height between sampled counter and basin-bottom patches. The right-side
view therefore supplies useful surface samples for both roles. The autonomous
case still needs tongs tracking, separate support admission and a native
motion refiner. The observed planning contract now accepts different pickup
and destination supports and keeps those bounds in its revalidation checks.

The repeated scripted run is in
`logs/harness-pandaomron-agent-e0bv/verified-scripted/`. The local real-model
runs recorded the vision request. Qwen2-VL-2B exceeded its CPU time budget.
Qwen3.5-0.8B hit a CPU tensor error in a fused kernel, and the PyTorch fallback
exceeded the 120-second run budget. Use the scripted run to reproduce the
public tool and sensor contract result.

### Native transfer ports

PandaOmron supplied synchronized front and wrist RGB-D and joint state to
production CHAT for the development tongs and held-out shaker cases. The
native finger-target smoke measured both fingers within 1 mm of their
targets. The scoring process repeated the simulator-state-assisted tongs
reference successfully with a 19.9 cm lift and a settled, fully contained
placement. That control took 18.6 s wall time for 23.55 s of simulation.

The latest scripted CHAT runs took 17.4 s for the tongs and 12.3 s for the shaker.
Both received a native camera image, reported `observed_scene_unavailable`
on discovery and planning, and issued zero actuation. Manual masks and height
screening on the wrist depth isolated finite counter and basin plane patches
in both fixtures. The development basin patch measured 15 by 13 cm. Its
diagonal is shorter than the 28 cm tongs, so this patch alone cannot certify
the full placement footprint. The measurements and pixel selections are in
`panda-support-diagnostic.json`. The Stretch tabletop
sensing preflight measured its destination from 2,098 depth points and
rejected a stale frame. Two recorded views allowed a numerical fit to an
independently specified cylinder, with approximately 0.2 mm depth residual
and unresolved axial orientation. The live test still lacks qualified
payload tracking and full enclosure. Both ports need live scene and robot
material evidence before an observed transfer can be attempted.

Both cached native MolmoSpaces/rby1 robot profiles compiled. The joint-control
profile exposes 26 actuators, 29 joints and six cameras. The current
Galaxea-based harness spec matched none of the native joint or actuator names.
The upstream finger actuators are motors on coupled joints. The next port
requires a native mapping and measured finger control before running the bowl
case through CHAT. In a gravity-disabled transmission smoke, an arm joint
settled at its 0.08 rad target, and the upstream open motor command produced
about 1.2 mm motion per finger after two seconds. The results are in
`molmospaces-rby1-inventory.json` and `molmospaces-rby1-transmission.json`.

Caliban refused the configured model endpoint connections. The bridge smoke,
independent scores and exported diagnostics are under
`logs/observed-transfer-ports-7xjm/`. These runs used a working checkout and
are diagnostic evidence. The
[porting recipe](../testing/agent_harness_assessment.md#port-a-benchmark-arm)
defines the staged qualification for additional benchmark embodiments.

### PandaOmron multi-view and shaker follow-up

Six scripted CHAT runs covered the native left, center, and right scene
cameras on the two frozen resets. Every run received synchronized scene and
wrist imagery and refused planning before motion because the live evidence
provider is unavailable. In the manually annotated depth diagnostic, all
tested shaker basin patches could fit its isolated-asset hull. None
of the tongs patches could fit its hull. The closest tongs patch was short by
3.7 cm on one axis even after yaw search. The reachable wrist patch was smaller.

The shaker's first 36 simulator-supervised straight approaches all hit the
same cabinet handle. RRT detours cleared the handle. Height filtering within
the manually selected shaker region corrected a pickup pixel that landed on
the counter. A 10 cm lift, native torso planning, and a release height derived
from the measured support plane enabled a quiet placement. Extending the
retreat by 3.5 cm was required to pass RoboCasa's 25 cm hand-to-object check.
The passing run lifted 9.3 cm, retained both pads through carry, settled fully
inside the basin, and finished with 27.3 cm gripper clearance and zero
unintended contacts. It used the native controller and simulator-state
collision supervision. The shaker reset has now been used for tuning.
Combining consecutive straight joint segments preserved the passing result
and reduced simulated duration from 153.7 s to 41.0 s and measured wall time
from 51.8 s to 41.8 s. The compact run used 810 controls and retained two-pad
contact through release.

The observed verifier uses the pickup surface for lift separation and the
destination for settling. A calibrated approach can specify different lift
and retreat distances. The live PandaOmron port still needs qualified
payload tracking and scene clearance before CHAT can execute this route.
The camera selections, robot-only IK probes, and physical rollout reports
are under `logs/pandaomron-followup-FayW/`.

### Live RGB-D placement and guarded probe development

Production CHAT discovered the shaker using synchronized scene and wrist
RGB-D, diagnostic pixel grounding, measured counter and basin patches, and
an axial enclosure from the isolated shaker asset. Scene initialization
uses the robot-only model and measured joints. Starting tracking before the
initial voxel carve sustained six captures in the next run. Planning then
stopped before dispatch when new occupied depth returns disagreed with the
initial free-space evidence. A repeat identified two conflicting voxels,
both attributed to background. The camera-only transfer remains open.

In the next pass, the full depth carve for a recorded scene/wrist pair
fell from 0.54 s to 0.08 s with identical classification of every voxel.
The pickup check used measured counter returns around all four corners
of the shaker footprint to test its resting height. The contact and
swept-clearance checks still use the original depth witnesses.

The pixel audit identified isolated wrist returns in three conflicting
cells. One return moved 9.2 cm toward the wrist between captures while
the camera translated 4 micrometers. Two lossless-depth runs also saw
isolated returns, 24.7 cm and 8.4 cm closer than those in the reference
frames. The planner retained those cells as occupied and refused the
current operation. An explicit follow-up CHAT turn requested a new plan
when the result called for replanning. Trials and CPU measurements are
under `logs/pandaomron-next-TwJk/`.

The subsequent certification audit used a development reset with the torso
2 mm above its lower stop. Measured finger feedback crossed the soft joint
limit with the full-open motor preset. Moving the preset 1 mm inward let
certification pass the joint-limit check. The Panda outer-link bounds now
include both the shared planning margin and their calibrated displacement
error. The next rejection was `occupied_environment` at the stationary
`mobilebase0_g0_col` envelope during the initial gripper check. The final
audit recorded zero actuation. Camera-only execution and the frozen reset
remain qualification gates.

The recorded chassis surface audit counted 291 scene-camera and 828 wrist
returns within 12 mm of the nominal chassis cylinder surface. None passed
the current ownership requirement of 1 mm material erosion plus 3 mm
depth uncertainty. Median signed distances were 1.93 mm and 2.32 mm outside
the nominal surface. These are nearby returns, whose robot identity still
needs verification against calibrated camera rays. Endpoint containment
alone cannot qualify these visible surfaces. The measurements are in
`logs/pandaomron-next-TwJk/chassis-surface-audit.json`. The next ownership
implementation must account for surface and depth uncertainty while
preserving nearer obstacle returns and mixed occupied cells.

The contact fallback uses the virtual wrist wrench and an independent
bridge-side stop with a limited descent and retention watchdog. A
controlled upward 2 N load at the gripper yielded about a -2.1 N signed
axial residual. An empty-wrist probe stopped on unexpected force after
about 0.02 mm. That verifies the stop path. The probe still needs a
loaded-wrist calibration that distinguishes normal descent transients
from support contact and a successful protected placement.

The input profiles, live CHAT reports, and wrench controls are under
`logs/pandaomron-live-placement-TlXE/`.

### Motion-scoped collision and colliding-start development

The native arm planner now accepts an explicit departure policy for a
colliding start. A forward sampling tree searches for a separating prefix,
then connects to the task goal. Contact allowances remain relative to the
original baseline. Tests cover a blocked straight exit, tangential motion,
limited temporary overlap, rejection of new contacts, and departure from
an observed local surface through the real collision checker.

Fixed links retain their starting environmental contact during a joint
command. Moving links still check robot geometry and environmental sweeps.
An optional sparse 5 mm depth map refines manipulation clearance separately
from navigation voxels. Robot surface association uses the visible robot
model at measured joints. Panda's renderer hides collision meshes and
renders visual meshes, which must also be used for depth association.

A simulator-only diagnostic override bypassed environment and
payload-to-robot collision checks. The original robot self-collision check
remained active. The CHAT plan then reached execution and exposed a missing
blocking gripper-open client method. With that method implemented, the
scorer measured 5.1 cm of end-effector displacement and zero unintended
contact frames. A later reference-view tracking control reached 12.6 cm
and 78 unintended contact frames. Both stopped before pickup on visual
association failure. These controls demonstrate native motion through CHAT
and identify gaps in collision and perception integration. They do not
qualify a complete transfer or real-world contact recovery.

Measured trajectories are plotted in
`logs/pandaomron-measured-motion.png`. Trial directories are recorded in
`/tmp/emet-move-measured`, `/tmp/emet-reference-track`, and the subsequent
`/tmp/emet-local-*` pointers. Recovery without a calibrated force guard is
currently an explicit simulation capability. Five fresh evaluation episodes
remain gated on a successful development transfer.

### Bounded visual recovery and collision-bypass comparison (2026-11-22)

CHAT may continue after an explicitly unstarted planning failure for at most two
recovery rounds per user turn. It can inspect current imagery or query the VLM,
revise the view or approach, and request a fresh plan within the original task.
Missing execution state, partial progress, failed execution, and unknown command
outcomes do not qualify. A VLM proposal remains a hypothesis, not a measured
clearance certificate. Existing simulator-only overrides remain visible in tool
receipts and cannot be enabled on a physical session.

The collision-checked development run
`logs/pandaomron-adaptive-recovery-FiAK/report.json` stopped before motion:
pregrasp search exhausted its budget, the alternate IK posture collided, and
tracking exceeded its update budget. The tracking failure now retains its typed
code instead of becoming `operation_failed`.

The explicit simulator collision-bypass comparison
`logs/pandaomron-adaptive-bypass-Fmcu/report.json` admitted a plan and dispatched 22 native commands.
Measured end-effector displacement was 0.1317 m. It stopped in grasp on
`moving_geometry_ambiguous`, with no measured two-pad lift and 169 unintended
contact frames. Neither experiment completed the transfer. The scripted tool
selector did not evaluate actual LLM/VLM recovery quality.

Remaining execution work: obtain collision-clear pregrasp alternatives, maintain
object tracking through hand occlusion, and verify attachment before carry. For
uncertain geometry, request a new view or retain a bounded pose hypothesis with
its uncertainty propagated into collision geometry. Do not relabel unobservable
poses as precise measurements. Planned contact needs explicit contact scope and
monitored stop/retreat behavior. Physical contact recovery still needs calibrated
force/effort feedback. Map cell size does not establish reconstruction accuracy.

A second bypass comparison using both camera views (`logs/pandaomron-adaptive-multiview-2rtE/report.json`)
admitted a plan and dispatched 21 commands, reaching 0.1282 m displacement.
It stopped on `object_pose_ambiguous` during grasp, with no measured two-pad
lift and 108 unintended contact frames. Adding the second view alone did not
complete pickup; the pose-admission step also needs explicit uncertainty-aware
handling through occlusion.

### Cabinet-route diagnosis and fixed-camera tracking (2026-11-22)

The prior bypass runs did not reach the object: their final EE position remained
near the initial high posture. Their contacts were between the right cabinet
handle and link 7 / the hand. The high-level `grasp` operation includes travel
to pregrasp; its failure label alone did not locate the physical failure.

The shared arm planner now accepts bounded posture-route proposals. Every leg
uses the normal sampled joint-limit, self and environment checks. A rejected
proposal falls back to ordinary search within the original shared time budget.
Pregrasp can propose the embodiment's home posture; no proposal is a collision
waiver. An explicitly supplied joint-only demonstration can also propose a
route and appears as `configured_joint_route_proposal` in the Panda receipt.
These development demonstrations come from the previously simulator-supervised
reference, so their use is assisted and does not qualify held-out generalization.

`tracking_views` separates the semantic initialization camera from subsequent
tracking cameras. The fixed scene camera can track a wrist-initialized object,
using new measured depth in every update. There is no stale-pose substitution.
The last candidate fit is retained for diagnosing rank, alternative centers and
search-boundary failures. Planning failure logs identify the rejected link,
map-admission reason and any self contacts.

The fixed-camera, single-heading development control
`logs/pandaomron-fixed-view-client35-YB8f/report.json` rejected the pregrasp
endpoints, issued zero commands, and recorded zero unintended contacts. Its
final candidate fit retained observable translation, zero center spread and
0.46 mm RMS residual; that fit residual is not an accuracy calibration.
The multi-heading and demonstration-route comparisons remain development tests.

To reproduce the archived environment, keep the supported shared client runtime
separate from the RoboCasa server's pinned import overlays: RoboCasa
7bfc60c9d26c02506d4265cbe0e86b35e6e25d42, MuJoCo 3.3.1 and NumPy 2.2.5.
The shared client requires newer MuJoCo bindings. Restored asset directory
symlinks are resolved before mesh identity comparison; the fixture's model and
mesh checks remain active. Missing overlays and initial scene construction
errors are setup failures, not task scores.

Unstarted tracking failures now invalidate the failed source and all cached plan
handles before returning `recovery=replan` with `perception_reset=true`. The next
planning request must ground a new scene from new measurements. Execution errors
do not take this path. A regression covers failure, reset, and fresh successful
planning without commands during the failed request. Route endpoints can seed
pregrasp IK so the planner need not choose a different elbow branch from the
proposed detour. `joint_planning_timeout_s` exposes a bounded diagnostic planning
budget; its default remains 10 seconds for local-map paths.

Follow-up controls reached offline grasp certification. One candidate had 12.0 mm
estimated separation and a 12.7 mm padded requirement; the simulation diagnostic
can now subtract at most 5 mm of planning reserve, preserving model-displacement
and depth-error floors and keeping all occupied geometry. The tested proposal
uses 2 mm, recorded in receipts as `simulation_clearance_slack` and
`clearance_slack_m`. This is not physical qualification.

An opt-in `transient_fit_rejections` policy can reject at most two consecutive
numerical fit failures while retaining only the preceding actual measurement
with its original timestamp. It publishes no frame for a rejected capture,
restores that accepted seed for the next real fit, and retains hard stops for
identity failure, persistent failure or expired evidence. Default behavior stays
unchanged. A world-aligned calibrated grasp can propose a top-down tool frame
without changing the fitted object geometry; placement preserves the resulting
object-to-tool transform. Repeated landing variants no longer retry an identical
pickup after it already failed in the same certification pass.

The first demonstration-route controls mistakenly replaced the recorded torso
trajectory with its initial height. Those proposals are not valid replays of the
successful reference and do not establish that its original route was rejected.
The corrected proposal retains all eight arm/torso joints, including the 15.6 cm
torso lift, and uses the corresponding top-down grasp heading. Validate terminal
FK against the expected waypoint when importing any embodiment's motion prior.


Follow-up collision and recovery audit (2026-11-22)
------------------------------------------------

The corrected eight-joint native demonstration is not a collision-safe motion
prior: its reference scorer omitted robot self contacts, and interpolation
between its stored postures penetrates link 5 against the hand by up to 25.5 mm.
Guide legs now use the configured planner to repair self collisions rather than
assuming that clear endpoints imply a clear segment. The agent evaluator now
scores robot self contacts deeper than 1 mm separately from external contacts;
both must be absent for a passing transfer. Older scores lack that self-contact
qualification. Environment-bypass receipts explicitly say `robot_self_only`.

Measured-joint tracking now distinguishes angular and linear joints. Panda uses
0.01 rad for hinges and 0.001 m for the torso; the old shared 0.03 threshold could
accept a 3 cm torso lag. Runtime validation checks the exact commanded edge,
without substituting an offline guide route. An already-open gripper within
1 mm of its measured preset does not need a redundant opening sweep, while arm
motion still needs its own clearance check. Optional simulator contact-terminated
closing proposes a finite motor corridor and requires measured settling plus
visual lift/retention evidence; stopping short does not prove a grasp. Local
payload-support contact remains bounded by the selected measured support mask,
a 5 mm penetration cap and checks against other obstacles.

A client cProfile run spent 70.7 s in 361,187 WorldSolid initializations, versus
0.135 s in 202 native local-cell distance calls. Per-configuration geometry reuse,
immutable offset changes and per-edge measurement reuse address that Python
construction overhead. Caches do not persist across live evidence updates.
This is profiling evidence for these changes, not an end-to-end speedup claim.

Checked recovery remains blocked by depth/self association. Raw-point inspection
showed several rejected voxels contained robot surface returns missed by the
first-hit ray test, including wrist-camera rays. Collision-proxy association now
uses nominal interiors only, without an expanded outside band. A separate test
matches missed returns against actual visual triangles within at most 3 mm;
nearby obstacle returns 5 mm away remain obstacles in regression tests. No
simulator scene labels are used for that association. The 3 mm band is a supplied
development depth-error assumption, not a real-camera calibration result.

Simple vertical prismatic escape probes now precede guide-route samples so a
long guide cannot consume the whole recovery budget first. An explicit simulator
profile can configure bounded baseline slack (at most 5 mm); generic departure
defaults stay unchanged. Receipts record configured departure bounds. Trials
using 3 mm slack, 1 mm new-contact tolerance and an 8 cm surface-travel envelope
remain diagnostic and do not qualify physical contact recovery.

The checked 1 mm/3 mm visual-association and vertical-first development trials
still stopped before commands at `escape_search_exhausted`; zero scored contacts
in these stationary runs are not task success. The shared regression selection
passed 338 tests after the mesh association and fresh-map changes. The full observed shaker
transfer remains open, along with live qualification of departure revalidation, physical uncertainty
calibration, contact monitoring and independent held-out evaluation.

Recovery revalidation now queries the latest local map instead of its captured
planning-map closure. Environment contact keys use stable integer voxel
coordinates, so fusion array reordering does not relabel a contact. A regression
checks map replacement, inserted/reordered voxels, newly observed contact
rejection and changed-profile rejection. This corrects a runtime validation gap;
it does not establish successful monitored recovery execution.

All six development controls in `logs/pandaomron-self-mask-comparison-20261122.json`
stopped without arm commands. Increasing actual-mesh candidate matching from
16 to 64 triangles and reducing local voxels to 1 mm admitted more search nodes
but did not solve departure. The 1 mm run retained the same 3 mm depth-error
assumption. Further blind changes to tolerances are not a demonstrated solution.


Measured surface-patch departure (2026-11-22)
------------------------------------------

Recovery no longer treats occupied voxel volumes as physical contacts or uses
voxel identity to distinguish old from new contact. Local maps preserve measured
surface samples separately from occupancy centres. A dedicated surface query
measures nominal robot-to-sample distances with micrometre probes, without voxel
volume or depth padding. Ordinary free-space broad-phase checks still use the
conservative occupancy map.

At the initial measured pose, each link forms fixed local contact patches.
Planar evidence permits a 3 cm tangent footprint and a normal association band
of twice the supplied depth error (bounded at 1 cm). Sparse/nonplanar evidence
gets only 2 mm point association. Patches never grow during search or execution.
Within an original patch, the closest measured sample may change as the link
slides. Nominal penetration is compared against the original patch minimum,
using 5 mm default patch slack and the independent 5 mm hard penetration cap.
Changing planning padding does not change nominal penetration. New surfaces
outside the original patches retain the explicit new-contact tolerance; unknown
sweeps, joint limits, robot self contacts and target checks remain active.

The local query uses each newly supplied map during execution; inserted,
reordered or resampled measurements do not create new contact merely because
voxel indices changed. Regressions exercise a real local-depth plane, sliding
across two 5 mm cells, a changed camera sampling alignment, and rejection of a
new surface at the opposite robot face. They also cover patch footprint limits,
normal separation, sparse/collinear evidence, bounded penetration and preserved
raw measured coordinates during map updates.

Escape guide proposals no longer repeat sixteen fractions of every waypoint;
the steering step already bounds joint displacement. Tail connection attempts
now share the escape deadline and have a 2 s per-attempt cap. Diagnostics
separate accepted recovery samples, ordinarily clear configurations and the last
failed connection to the goal. This prevents one unsuccessful connection from
silently overrunning the recovery budget.

The first patch controls stopped before commands. The failure moved from
per-voxel identities to bounded patch penetration, unrelated surface contact and
robot self contact. These are development trials, with scripted tool selection,
assisted grounding/guide proposals and simulator unknown-space optimism; they
are not physical qualification or held-out success. The regression selection
passed 344 tests including the final deadline-diagnostic change. Live Panda transfer
success is still required separately from these implementation regressions.

Final development result: all five patch trials in
`logs/pandaomron-contact-patches-20261122.json` completed with zero actuation
commands and no scored external/self contact frames. The final 5 mm-slack run
accepted 27 recovery nodes but found zero configurations satisfying ordinary
collision checking, so it never attempted a goal connection. Remaining candidate
rejections were hard penetration and contact outside the original patches. This
isolates the outstanding handoff issue: admissible patch recovery samples are
not yet reaching an ordinarily clear configuration. The full transfer is still
incomplete; stationary contact scores do not establish motion safety.


Recovery-to-motion handoff implementation (2026-11-22)
----------------------------------------------------

Original measured near-surface patches now carry a scoped clearance context into
both the goal connection and its runtime suffix. At those original patches only,
planning reserve can be consumed once the nominal surface bound is clear within
the configured new-contact tolerance. Other occupied cells retain ordinary
checks, and free-space checks retain model-displacement and sensor errors while
dropping only the consumed travel buffer. The context is restored after the
path; it is not a global collision bypass. Target and robot self checks remain
active. Contact patches cover the originally queried local measured surfaces,
including near surfaces inside planning reserve, rather than only points within
5 mm nominal distance.

A regression plans a path whose final state remains invalid under ordinary
padding, then verifies it with the retained patch context and verifies the
ordinary rejection returns after scope exit. Runtime tests execute both the
recovery prefix and normal suffix, checking context restoration and actual
commanded joint feedback. Separate tests retain rejection of newly observed
surfaces and insufficient model/sensor uncertainty clearance.

Nominal point clearance uses vectorized conservative solid-distance bounds.
Samples proven clear at an original patch are removed before repeating expensive
occupied-cell distance probes. This changes repeated work, not the physical
penetration limit. Connection attempts share the remaining recovery deadline;
they no longer restart the multi-leg guide after four seconds.

Execution samples permit at most 2 mm surface displacement and 0.005 joint-unit
steps, while collision edges are still checked at 1 mm surface displacement.
Departure feedback tolerance is at most 0.001 joint units (with separate linear
scaling), and its wall deadline scales with waypoint count, capped at 30 seconds.
The former sub-mm motor commands hit a fixed five-second timeout after 3.19 mm
of actual torso motion. A subsequent normal Panda execution reached 7.07 mm of
an 8 mm torso request with no scored unintended external or self contacts; an
already-clear run does not itself qualify the recovery transition.

An explicit simulator-only visual-mesh self-association setting can exceed the
default 3 mm band, up to 15 mm, and is recorded in receipts. The 8 mm control
removed an observed 4.4 mm-offset return that blocked a torso-cover sweep.
Association alone cannot distinguish every close obstacle from self; this
setting remains an assisted simulator diagnostic with independent contact
scoring. Default association remains unchanged. One full-transfer run with that
setting certified pregrasp in 27.3 s and grasp in 2.92 s, then rejected lift.
Other startup captures still failed pregrasp planning; repeatability is open.

Correction to earlier support-contact notes: the supplied support-contact
object's default cap is 1 mm, not 5 mm. An explicit simulator-only
`simulation_support_contact_tolerance_m` can now request 5 mm while retaining
selected-support masking and other-obstacle checks. The generic support-contact
default stays at 1 mm. Tests cover a 2 mm supported penetration rejected by the
default but admitted by the explicit 5 mm mode, with 20 mm penetration and a
separate obstacle rejected in both modes.

`scripts/eval_pandaomron_contact_recovery.py` isolates a short torso motion using
the production observed source, planner and runtime executor. It requires an
actual departure path and nonempty clearance suffix, measured progress, accepted
commands and zero independently scored external/self-contact frames. A declared
additional planning buffer can force a physically clear start to be invalid
under planning reserve, so an already-clear start cannot falsely pass this
transition regression. It neither injects scene geometry nor changes physical
contact limits. This native-executor probe is separate from CHAT tool selection
and full sink-transfer success. The integrated regression selection passes
349 tests; the full observed sink transfer remains incomplete.


Verified recovery handoff execution:
`logs/pandaomron-handoff-forced-buffer-Lx5r/report.json` passed the native Panda
probe. An additional 25 mm planning reserve deliberately made the measured
physically clear start invalid under ordinary checking. The returned path had
both a recovery prefix and nonempty retained-clearance suffix. Five accepted
joint commands achieved 7.039 mm of an 8 mm torso request (7.847 mm maximum
measured simulator EE displacement), with zero scored unintended external and
self-contact frames. The run used the declared 8 mm simulator visual-association
setting and unknown-space optimism; it did not use the global environment bypass.
This verifies that the transition can execute, not the full sink transfer or a
physical-robot uncertainty calibration.

A fresh repeat (`logs/pandaomron-handoff-repeat-3tpu/report.json`) also passed:
five commands, 7.035 mm torso progress, both recovery and clearance suffix used,
and zero scored unintended external/self-contact frames. Both reports are
collected in `logs/pandaomron-handoff-verified-20261122.json`. The same declared
25 mm added planning reserve and 8 mm visual-association simulator setting were
used in both runs. The 349-test integrated suite and lint/diff checks pass.

## 2026-11-22: observed transfer completion work and stack checkpoint

The [stack change map](../testing/observed_manipulation_stack.md) identifies the
shared versus Panda-specific changes and gives the intrinsic-prior/profile
reproduction recipe. The frozen shaker snapshot has been used for tuning and is
explicitly development data. Scripted CHAT routing remains distinct from a
real model-selected trial.

The recovery-to-clearance handoff passed twice as recorded above. Full-transfer
controls exposed additional issues rather than establishing task success:

- Original contact-clearance context was lost between offline phases. It now
  survives clear-start planning, subsequent phases and measured runtime checks;
  hypothetical future poses cannot establish expanding contact exemptions.
- Gripper closure previously predicted the deepest permitted compression.
  Offline contact-terminated closure now predicts first nominal pad contact;
  actual native motor commands remain finite and lift/retention must be verified.
- Support association used one exact depth pixel and counted plane uncertainty
  as nominal penetration. Cross-view association now uses the original measured
  support samples and plane; the explicit simulator cap applies to nominal
  penetration. Raised obstacles and generic 1 mm support limits retain tests.
- The isolated known mesh prior needed the benchmark's intrinsic 1.05 scale.
  Scaling known geometry does not supply live world pose.
- A single scene camera plus wrist camera lost the target behind the approaching
  arm. Exposing both standard scene cameras resolved that loss in a retained
  control. The control then reached a cabinet collision and timed out: disabling
  environment checking did not make the route physically feasible.
- Slow repeated convex subdivision, unbounded aggregate seed search and repeated
  approach searches added avoidable work. Convex/box intersection now has a
  single-thread halfspace solve with conservative fallback. IK proposals share a
  wall budget; native distance kernels warm before live tracking. An already
  open arm at the validated start can explicitly reuse its grasp path while
  every executed segment still checks fresh measured geometry.
- Native arrival feedback is independent of camera frame rate, with atomic joint
  and base feedback and a 250 ms age bound. Geometry still uses the original
  image-aligned encoders. Self-surface qualification is cached per capture only.
- Additional native IK seeds, Cartesian continuation/staging, calibrated grasp
  headings and source-support-derived transport clearance are proposals, never
  collision exemptions. The native reference's destination heading is now tried
  first in the development profile; the observed provider still supplies goals.

Retained controls include `logs/pandaomron-finish-positive-control-jR2d`
(no lift; all-view loss), `logs/pandaomron-finish-best-fit-control-YZeo`
(no lift; all-view loss), `logs/pandaomron-finish-serial-compiler-Rh7p`
(no lift; ambiguous geometry), and the directory referenced by
`/tmp/emet-finish-three-camera` (tracking continued but the unchecked approach
contacted a cabinet and failed motor tracking). None qualifies full transfer.
Checked trials admitted pregrasp, grasp and lift, then exhausted preplace budgets
or lost fresh tracking before actuation. Raw logs preserve those failures.

Native startup aborts occurred inside MuJoCo compilation, before TCP startup or
any commands. The runner now prepares fixture XML before compiling frozen asset
metadata and serializes mesh compilation. Intermittent MuJoCo 3.5 binding crashes
also occurred in narrow synthetic adapter-test selections. A diagnostic allocator
selection passed 92 adapter tests; two standalone 20,000-query collision stress
probes passed. These observations do not establish the underlying native crash
cause. Retain startup/runtime failures rather than counting them as task passes.

The full-task scorer now requires both a successful observed execution receipt
and independently verified physics: a two-pad lift of at least 5 cm, full sink
containment, settling, no scored unintended external/self contacts and permitted
native actuation. Physics-only motion cannot qualify an observed-tool failure.
Full sink-transfer success is still an open gate at this checkpoint.

### Native reset, synchronized auxiliary cameras, and staged arm corridors

The absolute native controller exposed a benchmark reset incompatibility:
RoboCasa's object-settling loop sends zero actions intended as neutral delta
commands. Those commands had driven the arm toward absolute zero. A reset-only
hold now preserves the native starting posture through native control. Measured
joint 4 changed from approximately -0.15 to -2.25 rad. Regression tests check both
successful/failed reset cleanup and unchanged normal absolute command semantics.
Robot-only compilations now request the same serial mesh compiler as scene resets;
the temporary loader binding is restored on failure. Native startup crash causes
remain unresolved; these trials use `PYTHONMALLOC=malloc` diagnostically.

The server-local generic camera list now includes only the primary view. The
Panda bridge explicitly renders the alternate scene and wrist RGB-D from its
snapshot, avoiding a duplicate generic stereo wrist render that could overwrite
the primary acquisition interval. Session metadata reports three cameras. All
auxiliary calibration/depth arrays are saved for offline replay.

The corrected reset required regrounding: shaker and surrounding counter in the
alternate scene camera, destination in the wrist. Offline replay showed that the
old source region cut off the measured plane around the occluded shaker footprint.
Widening that image region passed the existing pickup-support predicate; no
support predicate changed. The checked Cartesian proposal now exits in front of
the initial arm, crosses in front of the target, then approaches it. It translates
before rotating, initializes the same measured clearance scope as joint planning,
and can retain carried orientation until the destination-front waypoint.

`logs/pandaomron-finish-horizontal-corrected-FWVI` (actual directory recorded by
`/tmp/emet-finish-horizontal-corrected`) admitted pregrasp, grasp and lift planning
in 12.63, 1.69 and 1.28 seconds. Transport rejected payload/arm or cabinet
intersections. The subsequent loaded-corridor trial also admitted the first three
phases, then rejected transport. These are planning milestones, not physical lifts.
A wider explicit simulator visual self-association band (up to 30 mm; default
unchanged) did not resolve all transport blockers and is not qualified hardware
self-segmentation. These runs keep environment checks enabled. The separate
`logs/pandaomron-finish-native-reset-control-qboW` run enables environment bypass and must be scored
and reported separately.

The native-reset bypass control (`logs/pandaomron-finish-native-reset-control-qboW`)
planned all six phases and sent 89 allowed native actions. Independent physics
reported 0.458 m maximum end-effector displacement, no unintended external/self
contact frames and **zero two-pad lift**. Execution stopped on
`geometry_tracking_unobservable` before a verified grasp. No task success is
claimed. An alternate calibrated top-down grasp heading is under evaluation to
change wrist visibility without introducing simulator object state.

Validation for the native reset/capture/corridor follow-up: 556 integrated tests
passed with `PYTHONMALLOC=malloc` (29.43 s), plus lint and diff checks. A default
allocator repetition crashed in a native MuJoCo raycast test after earlier
555-test success; retain that failure as an unresolved native runtime issue.

The alternate-heading bypass control
`logs/pandaomron-finish-native-heading-8rf0` sent 127 native actions and moved the
end effector 0.640 m, with zero scored unintended contacts and zero two-pad lift.
It stopped on conflicting camera pose estimates. The opt-in best-view diagnostic
then sent 118 actions in `logs/pandaomron-finish-best-view-control-8ys8`, moved
0.605 m with zero unintended contacts, and stopped on stale object evidence.
It selected no conflicting-view override in that run. These controls remain
incomplete and do not qualify environment planning. A subsequent trial reduces
tracking from three fit starts/120 sampled points to one start/80 points while
retaining the same evidence age checks. The best-view change passed 557 integrated
tests with the diagnostic allocator (29.39 s).

The smaller-budget tracking control
`logs/pandaomron-finish-fast-tracking-control-CZwr` reached pregrasp, grasp, and
native finger closure. It sent 155 native actions with no scored unintended
contacts. There was no positive two-pad lift. Runtime retention stopped before
lift on a 0.1127 rad uncertainty-inclusive angular residual versus 0.1 rad;
translation was 17.1 mm versus 20 mm. The next explicitly assisted trial uses
0.2 rad consistently for runtime retention and visual lift checks, without
relaxing relative translation or lift/placement physics. It also records all
command/measured endpoint events and proposes bounded joint-space shortcuts;
these still require fresh direct-segment collision checks before commands.

`logs/pandaomron-finish-retention-shortcut-gjXI` physically lifted the shaker
**72.4 mm with both finger pads**, with zero unintended external/self contact
frames. It sent 140 allowed actions; the observed executor stopped on
`geometry_tracking_no_match` during lift, so there was no placement success.
The saved wrist image shows the held shaker. The exact assisted profile is now
`configs/eval/pandaomron_sink_assisted_lift_development.json`. The independent
physics milestone must stay separate from an observed verified full task.
Its motion records show 364 pregrasp samples reduced to 101 commands, 49 grasp
samples to 19 commands, and 45 lift samples to 23 commands. A subsequent timing
trial uses two small fitting starts and an explicit 20 mm torso command span;
every resulting chord still receives adaptive collision revalidation. Defaults
and recovery-prefix command resolution remain unchanged.

The larger-slide trial `logs/pandaomron-finish-faster-slide-piIA` reduced pregrasp
from 101 to 49 commands, grasp from 19 to 6, and lift from 23 to 5. It reached
closure, then stopped on `payload_attachment_outside_calibration` before positive
lift. The provider's lever radius omitted the same 20 mm relative-motion allowance
that retention already permits. The corrected fixed calibration includes that
allowance and propagates the larger radius into angular error; retention itself
is not rebased or widened. A source-bound regression admits 15 mm relative drift
and rejects subsequent drift past the original reference. Rejected tracking
packets now retain original raw views, masks, seed poses and intervals for offline
replay; they cannot become accepted evidence.

Validation of the assisted-perception/checked-shortcut/calibration follow-up:
561 integrated tests passed with `PYTHONMALLOC=malloc` (29.65 s), plus lint and
diff checks. Earlier narrow adapter selections still encountered native MuJoCo
segmentation faults; the diagnostic allocator does not establish crash immunity.

The next trial `logs/pandaomron-finish-attachment-bound-UMuA` stopped on a
20.45 mm uncertainty-inclusive retention residual versus 20 mm. Investigation
found a temporal mismatch: runtime retention combined a camera-derived object
pose with newer servo tool feedback. `PayloadGeometryEvidence` now carries the
capture-owned tool transform from the qualified visual frame. Runtime retention
and actual attachment can compare paired observations, while hypothetical
planning retains candidate tool poses. The 20 mm translation threshold remains
unchanged. Source-bound regressions verify asynchronous feedback, true slip,
stale evidence and planning separation. The integrated suite passed 563 tests
with the diagnostic allocator (29.49 s), plus lint and diff checks.

`logs/pandaomron-finish-capture-retention-Lvx2` completed **visual lift
verification** with two distinct fresh observations. Independent physics measured
90.8 mm two-pad lift, zero unintended contacts and 133 mm overall object motion.
It began transport and stopped on uncertainty-inclusive retention estimates
20.53 mm / 0.316 rad versus 20 mm / 0.2 rad. These observations establish a real
and visually verified grasp/lift milestone, not sink placement. The next explicit
simulator diagnostic uses 30 mm / 0.35 rad consistently in runtime retention,
visual lift checking and fixed lever calibration.

`logs/pandaomron-finish-transport-tolerances-QkvD` stopped earlier on a geometry
fit rejection. Offline replay of its retained packet found a wrist candidate
with 95th-percentile surface error 1.01 mm but maximum 16.08 mm: the hard 9 mm
cap rejected an otherwise close fit. Four starts and a 5 mm residual setting did
not admit it; two starts and a 6 mm setting admitted wrist/fused candidates in
38/43 ms, preserving all points. This diagnostic setting is simulator-gated and
separately reported, not a claim of calibrated confidence. The replay also
motivated applying the original learned appearance inside geometry search masks.

A producer bug could refit the same rejected packet on retransmission, then
latch an already-seen capture-ID error. Exact retransmissions of rejected packets
now skip fitting and rejection counting while preserving the last accepted
capture's original age. Regressions verify this, recovery on a future real packet,
and rejection of changed data under the rejected ID.

`logs/pandaomron-finish-residual-retransmission-woCa` again verified lift visually,
then stopped on a 0.484 rad angular retention estimate. The orientation-unconstrained
control `logs/pandaomron-finish-unconstrained-angle-nfBa` stopped on 34.47 mm
uncertainty-inclusive translation during transport. This points to unstable full
pose fitting under gripper occlusion; repeatedly widening retention thresholds
is not a satisfactory tracking solution. The next work is capture-aligned,
vision-checked attachment-motion prediction, with declared conditioning and
independent grip/task scoring. Latest validation: 570 integrated tests passed
with the diagnostic allocator (31.50 s), plus lint and diff checks.

### Carry-aware evidence and physical approach clearance

The carry-aware source conditions orientation on capture-aligned measured tool
motion and the initial attachment. It fits translation to current visible depth,
requires local translation observability, checks all measured residuals and
monitors finger aperture. The original ordinary SE(3) fitter resumes before
release. A conditional fit is explicitly labeled and cannot claim independently
observed orientation. Tests cover contradictory depth, mismatched camera priors,
current tool motion, lost grip aperture and stale pre-closure imagery.

The first carry trial, `logs/pandaomron-finish-carry-prediction-restored-R41b`,
caught a stale aperture reference from before closure. The corrected transition
uses current native finger readings and waits for post-closure captures before
freezing the attachment once and starting lift. The exact benchmark runtime was
restored under `logs/agent-harness-runtime` after temporary dependencies vanished;
versions were verified as RoboCasa commit 7bfc60c, MuJoCo 3.3.1 and NumPy 2.2.5.
`uv --no-config` was required to avoid project-wide version overrides. Existing
asset files were reused; the recorded shaker XML SHA256 remained unchanged.

`logs/pandaomron-finish-carry-transition-IsDe` used 20 mm / 0.2 rad retention,
completed visual lift verification, and carried the shaker across the counter
with 142 carry-predicted sensor updates. Physics measured 110.1 mm two-pad lift
and 817 mm object displacement. It then stopped on controller tracking timeout
near the end of preplace. Independent scoring caught 109 contact frames between
`cab_2_main_group_right_door_g1` and `robot0_link6_collision`; self-contact frames
were zero. This is not placement success. The next route lowers the preplace
height and keeps the grasp heading to avoid the cabinet during final approach.
Integrated validation passed 575 tests with the diagnostic allocator (32.90 s),
plus lint and diff checks. The collision-aware full-transfer gate remains open.

The lower, unchanged-heading run `logs/pandaomron-finish-lower-same-heading-znmN`
stopped near preplace with zero >1 mm contact frames and 10.74 mrad joint error.
A temporary 15 mrad arrival trial (`logs/pandaomron-finish-native-arrival-2KCy`)
also stalled. Offline scorer-only reconstruction identified forearm/faucet
intersection; the angular tolerance change was reverted. No oracle-derived
trajectory was supplied to the agent. The scorer now retains shallow touches in
addition to its unchanged 1 mm penetration metric.

`logs/pandaomron-finish-lower-quarter-turn-B3JA` reported 187 shallow faucet touch
frames and zero >1 mm frames. The narrow-region front-goal run
`logs/pandaomron-finish-front-support-3QJm` likewise reported 144 shallow touches.
Offline RGB-D replay found that the destination image region excluded visible
front floor. Widening it from [575,350,870,660] to [600,350,970,740] expanded the
measured floor patch from Y=[-0.46,-0.33] to [-0.53,-0.33] m, retaining the same
support predicate. Placement proposals now optionally cover both support axes,
with full payload footprint checks.

`logs/pandaomron-finish-wide-front-support-G4Ph` completed preplace and began
lowering. The quarter-turned hand then contacted the front counter rim during
place (17 >1 mm frames, 182 shallow-touch frames); it stopped before release.
The next trial aligns the hand's narrow dimension across the rim and selects an
interior point farther from the faucet. Latest integrated validation: 576 tests
passed with the diagnostic allocator (29.98 s), plus lint and diff checks.

### Full physical transfer and phase-evidence fixes

The aligned-hand/front-floor development run
`logs/pandaomron-finish-aligned-front-witness-oLKY` completed native pick, carry,
placement, release and retreat. Independent physics reported RoboCasa success,
full basin containment, 100.8 mm two-pad lift, 802.7 mm object motion, final speed
4.55 micrometers/s, and zero unintended shallow touches, >1 mm contacts or scored
self contacts. This is a repeatedly tuned development case with environment
collision bypass, supplied object CAD, hand-selected image regions, best-fit
view/hypothesis selection and simulation-only carry conditioning. Public CHAT
tools were invoked through scripted routing; this is not a model-selected or
held-out result.

Carry association now checks actual known-model surfaces rather than treating
every depth return inside the payload enclosure as object material. Support-plane
subtraction is bounded by the full original measured witness footprint: an
unbounded plane erased unrelated held-object points, while using only the
inscribed placement rectangle retained actual support pixels. Initial restored
fixtures settle for 30 native-control steps at the measured absolute posture.

The physical success still returned unknown final visual verification because
the historical lift reference exceeded an arbitrary 30-second age limit. That
limit is removed for immutable historical evidence; original timestamps,
calibration/identity/geometry checks, and fresh ordered post-retreat samples are
preserved. A regression verifies a 126-second-old reference with fresh settling
images, and rejects the same images when current evidence becomes stale. The
full suite passes 580 tests (29.87 s). End-to-end verification is being repeated
under the frozen `pandaomron_sink_assisted_transfer_development.json` recipe.

`logs/pandaomron-finish-fresh-placement-0C58` again passed independent physics:
100.8 mm two-pad lift, 845.7 mm object motion, sink success, zero unintended
touches/contacts/self contacts, final speed 15 micrometers/s. Fresh post-retreat
verification now ran, but rejected settling. The ordinary 10 mm movement bound
includes 5 mm position uncertainty from each capture, leaving zero slack for
any nonzero measured displacement. The assisted transfer profile now explicitly
requests a 20 mm uncertainty-inclusive settling bound (allowed 10–30 mm in the
simulator-only option); ordinary verification remains at 10 mm. Footprint,
support, rotation stability, retreat and freshness checks are unchanged. New
regressions retain both capture errors and reject real motion above the bound.
Placement rejection logs expose geometry and translation/rotation bounds rather
than reporting only a generic placement failure.

The first 20 mm settling trial, `logs/pandaomron-finish-coarse-settling-nAAO`,
lifted and lowered the object into the sink without unintended contact, but did
not release. Pre-release retreat replanning encountered a transient negative
torso reading at its nominal zero stop; the 1 micrometer joint-bound band rejected
the measured start. The shared validator now permits a bounded 0.1 mm slide /
0.1 mrad hinge overshoot, retains the actual state for collision checking, and
keeps nominal IK/sampling ranges. This is numerical stop handling, not permission
to ignore joint limits or a claim of pose-estimation failure.

`logs/pandaomron-finish-bounded-stop-Wn4g` passes both independent physics and
observed execution (`passed=true`, `physics_passed=true`,
`observed_execution_passed=true`). Public tool execution verified grasp/lift and
placement after native release and retreat. Physics measured 100.97 mm two-pad
lift, 845.74 mm object motion, RoboCasa success and full basin containment,
zero unintended shallow touches, >1 mm contacts or scored self contacts, and
2.24 micrometers/s final speed. This closes the assisted development execution
milestone, not the collision-aware/held-out/model-selected/hardware gates.
Validation: 586 integrated tests passed (29.68 s), Ruff and whitespace checks.
The exact profile is tracked as
`configs/eval/pandaomron_sink_assisted_transfer_development.json`; the generated
CAD prior and benchmark assets remain separate as described in the stack recipe.

### One-shot depth pick control (2026-11-23)

At the user's request, isolate a simple tuned pick before adding more tracking.
`scripts/eval_pandaomron_sink_agent.py --open-loop-pick --observed-profile ...`
uses one initial RGB-D estimate, the native robot-only IK model, a staged approach,
finger closure and 10 cm lift. Object CAD, continuous object registration and
environment collision checking are absent. Native measured joint arrival remains
checked. This is a simulation-only diagnostic, not production CHAT execution.

`logs/pandaomron-one-shot-pick-oBx3` reached the grasp without unintended contact,
then stopped on a diagnostic gripper argument-format bug. The corrected
`logs/pandaomron-one-shot-pick-closure-bSUT` completed grasp and lift. Independent
physics measured 92.58 mm two-pad lift, zero unintended shallow touches, scored
external contacts or self contacts. The old runner's full-sink report remains
false because this trial deliberately tested only pick/lift, not transfer. New
runs write a separate `open-loop-score.json` with the task scope recorded.

This successful control used diagnostic image regions, not a runtime VLM call.
An assistant-selected narrower image region then failed before motion in
`logs/pandaomron-one-shot-visual-hint-UyaT`: the reset camera view shifted the target
outside the hint. A broader visual hint is being tested; it must not be described
as autonomous runtime VLM grounding. Preserve this failed attempt in results.

The broader assistant visual hint in
`logs/pandaomron-one-shot-loose-hint-rc0a/open-loop-score.json` passed the separate
pick/lift score: completed closure and lift, 92.57 mm two-pad lift, 92.76 mm object
motion, zero unintended shallow touches, scored external contacts or self contacts.
There were no runtime VLM calls. This is assistant-guided image selection plus
one-shot measured depth, not autonomous grounding. The wrapper returned signal
status 143 after writing the complete score; retain that teardown issue separately
from the independently recorded physical pick result. Eighteen targeted tests and
Ruff passed. These results motivate testing fresh runtime grounding and broader
case coverage before adding mandatory continuous object registration.

### Local runtime VLM + one-shot pick (2026-11-23)

Caliban refused connections. The host RTX 4090 had 23.6 GB free; cached
Qwen/Qwen3-VL-8B-Instruct was loaded in BF16 on a localhost-only service at
127.0.0.1:8765. Warm inference used about 19 GB VRAM, leaving roughly 5 GB.
The diagnostic sends the fresh scene image to the model for object, source,
destination regions and an image-frame closing direction. Depth converts that
direction and region to metric native targets. Requests, responses, decisions and
input images are retained; there is no fallback to saved grounding regions.

The model twice declined the benchmark label "salt shaker". An explicit visual
description, "small brown condiment grinder on the counter", produced regions,
but overly narrow/coarsely placed regions failed depth grounding before motion.
For runtime VLM hints, measured components above the counter now grow from the
hint; a missed hint may snap to a component within 10% of the image diagonal only
when its centroid is clearly preferred over alternatives by 1% of the diagonal.
Distant or ambiguous alternatives remain unavailable. This is image-space
association assistance, not shape completion or a semantic confidence guarantee.

`logs/pandaomron-local-vlm-depth-snap-kP8L/open-loop-score.json` records one actual
runtime VLM grounding call and a completed pick/lift. Independent physics measured
89.06 mm two-pad lift, 89.15 mm object motion, zero >1 mm unintended contacts and
zero scored self contacts. The clean-pick score is **false**: 39 shallow contact
frames involved wrist link6 and cabinets. This demonstrates a physical VLM-guided
pick, not clean execution, full transfer, original-label robustness or held-out
qualification. The fixed staged approach still needs clearance work. A separate
post-pick VLM judgment is saved in `vlm-outcome.json` and does not replace physics.
Nineteen targeted tests and Ruff passed, including narrow/missed hint recovery and
rejection of distant targets. No object CAD or continuous surface tracker was used.

Post-pick Qwen3-VL-8B judgments were false from both the wide scene image and a
second wrist close view (`vlm-outcome.json`, `vlm-wrist-outcome.json`), despite the
independent measured two-pad lift. The wrist answer interpreted the object as
resting below the gripper. Preserve this disagreement: VLM-only still-image
success judgments are not qualified. Future agent verification should combine
fresh views with native finger/hand feedback and depth-backed motion evidence;
the scorer's privileged pad contacts must remain evaluation-only.

### Full no-CAD runtime VLM transfer and matched instructions

The one-shot control now includes carry, lowering, release and retreat, with
separate official physical and clean-contact scores. The initial Qwen3-VL-8B
full attempt (`pandaomron-vlm-full-transfer-zvSl`) lost its grip and stopped during
lowering. A lower surface initially described as a cabinet ledge was incorrectly
identified in that analysis: calibrated RGB-D shows a deeper left sink basin
(about 0.715 m) and a shallower right basin (about 0.773 m). Preserve that correction.
The failure was not evidence that all lower surfaces were outside the sink.

Cached Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, was then
served locally in BF16 with Transformers 5.10.2, using about 19 GB VRAM. The
request includes a coordinate grid, a natural-language visual target description,
and a destination instruction. The model chooses all regions and closing direction.
Measured depth grows/snaps coarse object hints and measures counter support beside
the object. Top-band XY centering and a deeper grasp corrected visible-side bias;
the object remained held through roughly 0.8 m of transport. Initial retries
stopped at shallow forearm/faucet contact near preplace; another stopped on a tiny
out-of-range interpolated command. Commands now stay within nominal joint limits.

Scene-only floor depth missed the front portion of the right basin. Associating
wrist/scene returns with the selected floor height and local footprint supplied
that geometry. Spatial downsampling reduces camera sampling-density bias. The
native control chooses a front interior point from the measured floor; it does
not read a benchmark fixture pose. A preplace waypoint may be accepted within
35 mm / 0.15 rad with an explicit measured-error receipt, allowing downward motion
from shallow contact. Final placement and independent scoring are unchanged.
The first release run then stopped during a new retreat IK solution. Reversing
measured transport configurations solved that final departure.

`logs/pandaomron-vlm35-retrace-transfer-sLBw/open-loop-score.json` records:

- Official RoboCasa success and full basin containment: **true**.
- Native pick, lift, carry, placement, release and retreat: completed.
- Two-pad lift: 93.49 mm; object displacement: 818.78 mm.
- Final object speed: 2.11 micrometers/s.
- Unintended >1 mm contact/self-contact frames: 0/0; shallow touch frames: **355**.
- Physical transfer: **passed**; historical zero-contact score: **failed**. The
  original wrapper used that stricter score for its exit status. Contact policy
  was subsequently updated by the user; see below.

This remains a repeatedly tuned development case, with scripted phase routing,
explicit visual target description and environment collision-check bypass.
It demonstrates a runtime VLM-selected physical transfer without object CAD or
continuous surface registration, not an autonomous-agent or held-out result.
The minimal tracked profile supplies only native gripper targets; its path skips
all of the older CAD/tracking profile initialization.

`logs/vlm-matched-instructions-20261123/report.json` compares two instructions on
identical RGB-D (image hash recorded). Qwen3.5-9B selected the left versus right
basin. Both passed depth grounding, producing floor points approximately
[1.062,-0.452,0.714] and [1.522,-0.494,0.772] m. This is a causal instruction
control on one image: it measures changed grounded goals, not successful physical
execution of both instructions. Reproduce with `scripts/eval_vlm_instruction_pair.py`.
Earlier 8B and narrow-hint failures remain in their separate artifacts.

Before/after VLM verification on the alternate scene view returned unknown after
the successful transfer, correctly avoiding a success claim when the object was
not clear. The second fixed-camera pair returned `right_basin` and `instruction_satisfied=true`,
agreeing with the official physical score (`vlm-front-verification.json`). These semantic
judgments never override physical scoring. Full 20-case qualification, real CHAT
routing, clean-contact repair and the other robot ports remain open. Integrated
validation passed 599 tests (30.33 s), plus targeted geometry/CLI tests and Ruff.

The user subsequently confirmed that contact is acceptable. The primary one-shot
score now follows completed execution and independent task success, with contacts
reported separately. `task-policy-rescore.json` derives `task_passed=true` for the
successful transfer and preserves the original report via its SHA256; the raw
historical report is unchanged. Regression cases ensure contact alone does not
fail a completed task, while unfinished execution, absent lift, failed official
success, wrong containment and continuing object motion still fail. The 355
shallow frames remain visible for interpretation, not a blanket failure gate.

Final task-policy validation: **605 integrated tests passed** (28.00 s), with
Ruff and whitespace checks clean. The alternate fixed-camera verification
(`vlm-front-verification.json`) identified the grinder in the right basin and
reported `instruction_satisfied=true`. The first view's unknown response is
retained; selecting a more informative view resolved this instance without
feeding privileged task state to the model.


### VLM-adaptive expansion screen (2026-11-23)

The harness allows VLM estimates of camera, semantic image regions, grasp
closing direction, body-height fraction and direct versus viewer-side approach.
Regions remain coarse proposals grounded in current calibrated RGBD. Source and
destination can independently choose clearer cameras; no object CAD or exact
asset pose is supplied to the policy. Up to two pre-motion re-estimates receive
grounding failure feedback. These are not post-grasp recovery or production
CHAT routing. Native finger feedback is recorded at closure, lift and release.

`configs/eval/vlm_transfer_expansion_development.json` freezes three archived
fixtures: pear and garlic in layout 1, plus the earlier tongs control in layout
11. `scripts/eval_vlm_transfer_batch.py` verifies fixture hashes, snapshots
policy sources and model metadata, runs one simulator at a time, and reports
first-attempt versus recovered results. These are development screens, not the
20-case qualification pilot. Previously tested fixtures cannot become untouched
held-out cases by renaming their split.

Preserved rounds under root `logs/`:

| Batch | Full transfers | Relevant observation |
| --- | --- | --- |
| `vlm-expansion-screen-20261123-v1` | 0/3 | Pear/garlic approach arrival failures; tongs no physical lift. |
| `vlm-expansion-screen-20261123-v2` | 0/3 | Pear lifted 8.51 cm with two pads but slipped during carry; garlic/tongs no qualifying lift. |
| `vlm-expansion-screen-20261123-v3` | 0/3 | Pear arrival failed with 29 mm tool residual and no contacts; garlic exhausted grounding proposals; tongs reached preplace without a qualifying lift. |

The native Panda server advances physics continuously in `_sim_loop`; wall-time
waits do not pause physics. Synchronized camera capture is distinct from
action-stepped physics. Arrival, retention and surface grounding remain separate
blockers; do not attribute these failures to synchronized stepping. The fourth
round tests independent source/destination camera selection.

Fourth-round result (`logs/vlm-expansion-screen-20261123-v4/report.json`):
**0/3 full transfers**, with no qualifying physical picks. Pear repeated the
29 mm approach residual. Garlic changed source to `scene` and destination to
`wrist` on its second proposal and completed all motion stages, but independent
physics measured no two-pad lift. Tongs reached preplace then timed out, also
without a qualifying lift (82 touch frames). Independent-view grounding thus
unblocked garlic's sequence, but does not establish grasp success. Retain
first-attempt/recovery distinctions and avoid promoting any of these cases.

Validation: **609 tests passed** in 28.88 s; Ruff and whitespace checks passed.
One simulator ran at a time on six assigned CPU cores; the local 9B model used
about 20.3 GB GPU memory with 3.7 GB free at the sampled checkpoint. Next priority
is native approach-error correction and observable grasp/retention recovery,
then a fresh frozen qualification pilot.
