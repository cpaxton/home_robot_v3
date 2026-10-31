# Mars public TAMP controller integration, 2026-10-23

The latest complete simulator teacher run is **16/30**, compared with **4/30**
in the preceding run. All thirty frozen fixtures admitted in both runs. This
does not satisfy the 30/30 public-tool gate. PPO, normal-observation agent
qualification and hardware motion remain gated.

| Snapshot | 20 g | 50 g | 100 g | Total | Source-bound regression tests |
| --- | --- | --- | --- | --- | --- |
| `a17a5734`, public v2 report | 4/10 | 0/10 | 0/10 | 4/30 | 582 |
| `9dee06fc`, public v3 report | 10/10 | 6/10 | 0/10 | 16/30 | 588 |

Both runs use the unchanged
`configs/benchmarks/mars_public_tamp_whole_body_v2.json`: two separate supports,
two layouts, five seeds and three masses. Fixture admission, planning, execution
and end-to-end denominators are separate. The archived v2 raw aggregate omitted
successful executions from its `execution_started` count because successful
receipts lacked that optional field. Its case-level outcomes are unchanged;
the summary includes a separately labeled corrected aggregate. The v3 code
counts successful receipts and names failed independent predicates explicitly.

Complete reports, traces, source/model identities and regression receipts are in
[`public_tamp_v2.json`](../../paper/data/whole_body_control/public_tamp_v2.json) and
[`public_tamp_v3.json`](../../paper/data/whole_body_control/public_tamp_v3.json),
with adjacent compressed full reports and JUnit/receipt files. The earlier 0/30
development sweep and failed frozen-fixture check remain archived too. Nothing
was dropped or converted into success by a planning/admission filter.

The full report archive can be inspected on a clean checkout without rerunning
physics:

```bash
gzip -cd paper/data/whole_body_control/public_tamp_v3.json.gz > /tmp/mars-public-v3-report.json
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m emet.eval.mars_tamp_control plot \
  --report /tmp/mars-public-v3-report.json --case left:50g:0 \
  --output /tmp/mars-public-50g.png
```

## What changed

The planner now considers a short straight base approach plus arm motion before
a larger base transfer. It checks downstream placement/retreat feasibility within
the same bounded IK budget, and screens sampled arm paths for loaded support
margin. Execution recertifies the chosen base goal from measured state.

The controller enters a hold once it reaches the endpoint, then verifies quiet
motion and retained pose accuracy. Previously it kept making small corrections
while waiting to become stationary. An explicit hold disables accumulated base
bias; verified free-space placement motion enables compensation again, and
support contact disables it. Terminal public-tool success now requires a steady
hold, followed by object/support verification. No private stopped-state mutation
or physics reset re-arms a failed stage.

Reference twists and a nonholonomic base tracker follow timed paths; measured
tracking can slow path progress. This is geometric execution with explicit
retiming, not fixed-deadline acceptance. Final wheel slew limits remain opt-in
for the public facade; the legacy simulator control default is preserved.
Stops are bounded by both simulation and wall time, and their measured traces
are retained. Joint-reference previews now use the measured world frame after
sag/base motion rather than the reset frame.

Read-only exports include the nominal hand/base/gripper endpoints for all arm
stages, hand orientation axes, and base transport schedules. The plot overlays
those with the actual commanded/measured hand paths and contact loads:

![Selected successful 50 g transfer](../../paper/data/whole_body_control/public_tamp_v3_50g.png)

Successful full tasks took 65.55–69.00 simulated seconds (median 67.55). They
include closure, lift, transfer, placement, support confirmation, release, retreat
and verification. This remains a slow teacher baseline. One selected 50 g case
had 0.081 rad peak relative object rotation and 3.82 N peak contact load, without
a wheel-lift termination. Tracking diagnostics include all recorded samples and
stop transients; they are not a claim of a 5 mm bound at every instant.

## Remaining failures

- Five 100 g cases lose retention during preplacement; five other 100 g cases
  stop on wheel unloading during base transport.
- One 50 g case loses retention during placement, one stops on base drift during
  arm motion, and two fail bounded planning.
- All ten 20 g cases pass, but ten cases are not evidence of general reliability.

Extra nominal slowing, arm tucking, higher grasp selection and a supported
seating probe did not recover the selected 100 g failure. A lower acceleration
override also worsened a diagnostic; those variants were not promoted. Keep the
geometry, effort limits, frozen goals and 0.1 rad retention gate unchanged while
examining actual contact distribution and grasp stability. A reachable FK target
does not establish a stable grasp, nor does a teacher failure prove intrinsic
trajectory infeasibility.

## Other requirements

Hand-only pose tracking now lets the teacher use a free base when its target is
masked out. Two nearby FK-witness checks passed for Mars and Stretch, with
approximately 0.54 mm/0.0018 rad and 0.94 mm/0.0022 rad endpoint errors respectively.
Both actually moved their bases. These selected free-space checks, now regression
tests, do not qualify manipulation or hardware.

The agent harness uses the real prompt/parser/dispatcher and independent task
scoring, but no LLM trial is qualified: the oracle gate is false and a normal
observed-scene provider is still absent. The existing 298/300 carry and 30/30
single-support result at `b181db61` remains separate, immutable evidence; it does
not qualify the new source tree for training.

The [passive Herman audit](herman_feedback_audit_20261023.md) found source-level
feedback issues and provides an offline-tested upstream patch. No robot motion,
service start, deployment or live actuator reconfiguration occurred.
