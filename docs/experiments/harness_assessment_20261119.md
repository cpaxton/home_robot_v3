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
