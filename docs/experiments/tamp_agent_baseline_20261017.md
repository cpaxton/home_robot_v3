# Native TAMP agent baseline — October 17, 2026

This baseline measures the current PR #213 agent-orchestration branch at
`53e861e811ef4e624df6988fcb2512541c2c9cc8`. It uses the actual `run agent` CLI,
normal native Stretch arm/gripper execution, live ZMQ physics and independent
whole-footprint/support/stillness predicates. Caliban serves
`Qwen/Qwen3-VL-8B-Instruct` at `http://192.168.1.55:8000/v1`.

## Scope and frozen protocol

Six episodes were scheduled before examining outcomes: two process repeats of
three fixed placements. This is six executions of three fixtures, not six unique
problems or a general TAMP benchmark. The left/right placements were originally
held out on October 14 and are now repeated diagnostics, not a new unseen set.
Their complete native reference sequences are not certified; failures must not
be attributed to LLM reasoning alone. The familiar fixture has prior successful
native reference and actual-CLI controls.

Each episode asks for both object footprints to fit world XY goal regions:

- Red cylinder: `[-0.08, -0.86]` to `[0.08, -0.74]`.
- Blue cube: `[-0.08, -0.75]` to `[0.08, -0.58]`.

Inputs are explicitly privileged, stationary simulator geometry. The remote VLM
is used in text/tool mode; no camera image is sent in this baseline. This does not
measure observed grounding, navigation, exploration, native articulation,
clutter relocation or hardware readiness. The earlier observed-camera failures
remain in the [discovery diagnostic report](tamp_observed_discovery_20261014.md).

A first diagnostic user turn requests one deliberately invalid plan execution.
A second explicit user turn authorizes the actual two-transfer task. The same
instruction and 12-tool-round budget apply to every case; the agent must
rediscover after successful execution and stop on partial/unknown outcomes.
A correctly rejected plan or a correct failure stop is not physical task success.

All original goal, collision, retention, tracking and freshness tolerances are
unchanged. Both goals must initially be false, then hold with measured support
and stillness for two seconds of simulated time. The audit must be nonempty and
contain zero forbidden commands. Neither CLI exit code 0 nor assistant prose is
used as evidence of task completion.

VLM weights remain on Caliban. Local processes use CPU execution with
`CUDA_VISIBLE_DEVICES` empty, no cameras, and OSMesa selected defensively.
Two simulations run concurrently, on CPU affinity `0-7,12-31`; individual server
port offsets are 410–460. Parallel execution and remote queueing are part of the
recorded setup. The client leaves sampling parameters unspecified; server
sampling/checkpoint internals were not independently pinned beyond its advertised
model identity. Fresh process repeats are not independent scene samples.

Managed job: `20261017_023941_27d968`.
Artifact root: `/home/cpaxton/runs/emet/tamp-agent-baseline-20261017`.
`manifest.json` freezes case IDs, copied XML hashes, goals, model, endpoint and
source SHA. `caliban-models.json` records the endpoint response. Each episode
retains its CLI protocol, CHAT JSONL, server/agent logs, actuation audit and final
independent predicates. The batch runners and offline aggregation script are
archived there for reproduction. Missing summaries remain infrastructure
failures, visible in the six-attempt denominator.


## Results

| Fixture | Complete tasks | Goals satisfied | Task result | Audited commands |
| --- | ---: | ---: | --- | ---: |
| Familiar | 2/2 | 4/4 | Both sequential transfers verified in both repeats | 830 |
| Frozen left shift | 0/2 | 0/4 | `payload_not_retained` during cylinder lift in both repeats | 352 |
| Frozen right shift | 0/2 | 0/4 | `no_plan_within_budget` before manipulation in both repeats | 14 |
| **All scheduled attempts** | **2/6 (33.3%)** | **4/12 (33.3%)** | **0 infrastructure failures** | **1,196** |

All six summaries were produced. All six episodes began with both goal predicates
false. Both successful episodes also passed the complete acceptance conjunction
(task predicates, nonempty clean audit, observed invalid-plan rejection and
failure-stop checks). All four failed task episodes correctly returned failure
from the evaluator even though the CHAT process itself exited cleanly.

Across all episodes:

- **6/6** deliberately invalid plans were rejected.
- **6/6** episodes had valid structured TAMP results (28 total TAMP calls).
- **6/6** episodes obeyed failure-stop boundaries (10 failure events including the
  six deliberate rejections; no later dispatched tools in a failed user turn).
- Both familiar runs followed `scene_tasks -> plan -> execute -> scene_tasks ->
  plan -> execute`. Successful plans preserved the exact requested object regions.
- Left-shift runs stopped after `scene_tasks -> plan -> execute`; right-shift runs
  stopped after `scene_tasks -> plan`. No held-object recovery or blind replay ran.
- **0 forbidden commands** across **1,196** audited commands. Right-shift runs had
  seven startup commands each and no pick/place execution call.

These checks validate the intended orchestration behavior on this suite. They do
not independently establish free-form semantic goal interpretation or general
multi-step reasoning. The full task success rate remains 2/6; correct refusal or
stopping does not increase that numerator.

## Failure attribution and next gate

Both left-shift failures are partial execution results at `lift`, with
`execution_started: true`, no completed high-level grasp operation, and
`payload_not_retained`. They are physical retention failures. This experiment
does not isolate grasp selection, planned trajectory, contact mechanics and
controller contributions; the frozen traces should be reused by the RL/control
workstream instead of relaxing the retention threshold. The second object was
not attempted after the failed lift.

The right-shift searches used **77 and 76 IK calls**, respectively, against a
**256-call cap**. Rejections were predominantly pregrasp IK (53/54), followed by
preplace IK (5 each), grasp collision (3 each), pregrasp collision (2 each) and
grasp IK (4/3). `no_plan_within_budget` is the public result, but neither run hit
the numerical IK cap. The bounded candidate search found no plan; this is not a
proof of physical infeasibility and not an LLM-selected invalid tool sequence.
A complete physical reference for this shifted task is still needed before a
model-reasoning score or an impossible-task claim is appropriate.

The next physical capability improvements are therefore retention/grasp control
and broader feasible motion search/reference validation. The orchestration
baseline is suitable for paired regressions while those workstreams change.
Observed camera-grounded tasks need their own provider/verification acceptance;
this result does not resolve the documented perception latency/label failures.

## Per-episode timing and provenance

| Episode | Full task | Goal predicates | Wall seconds | Audit commands |
| --- | --- | ---: | ---: | ---: |
| `familiar_r1` | pass | 2/2 | 437.4 | 415 |
| `familiar_r2` | pass | 2/2 | 405.9 | 415 |
| `heldout_left_r1` | fail | 0/2 | 154.1 | 171 |
| `heldout_left_r2` | fail | 0/2 | 170.8 | 181 |
| `heldout_right_r1` | fail | 0/2 | 96.6 | 7 |
| `heldout_right_r2` | fail | 0/2 | 95.8 | 7 |

Wall times include simulator/CLI startup, remote inference/queueing, native
execution, independent settling checks and teardown. The two-worker managed
batch completed in approximately 11.7 minutes. No local GPU inference or rendering
was used. Per-episode request/prompt/tool traces are retained in `logs/chat`.

The before/after Caliban endpoint responses advertise the same model ID. Their
`created` values differ; the metadata does not independently pin a server boot,
checkpoint revision or sampling configuration. This small baseline is not a
checkpoint-controlled comparison against the October 14 experiments.

Machine-readable aggregate: [tamp_agent_baseline_20261017.json](tamp_agent_baseline_20261017.json).
The archived `summarize-tamp-agent-baseline-20261017.py` asserts complete unique
case coverage, checks actual task-turn tool sequences and planned regions, and
recomputes the aggregate from original CLI summaries and CHAT logs. Raw failures
remain intact. No implementation/tolerance tuning occurred during this batch.
