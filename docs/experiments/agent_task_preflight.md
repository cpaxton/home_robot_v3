# Agent task preflight: tools and visual evidence

Roadmap: [implementation plan](../plans/agent_multiroom_environments.md).
Paper entry: `paper/main.tex`. Paid pilot: [planned, disabled](agent_hosted_pilot.md).

## Run the shared task agent

The benchmark now invokes `emet.agent.task.run_agent_task`, also used by
`emet run agent --task-mode`. It shares the interactive agent's model invocation,
Tool registry and dispatcher. Task mode continues after action results instead of
forcing a summary after three rounds. Model conversation history continues across
decisions. It records every prompt, raw response,
command, result, observation and finish/budget outcome.

```bash
uv run emet eval agent-tasks agent --episode cross_room_delivery \
  --model qwen25-3B-Instruct --device cpu --out /tmp/local-agent-delivery
uv run emet run agent --task-suite src/emet/config/benchmarks/agent_tasks.yaml \
  --task-episode cross_room_delivery --task-out /tmp/local-agent-from-app \
  --llm qwen25-3B-Instruct --device cpu --max-tokens 128 --task-max-rounds 24
```

Use the software EGL variables below on this development host. The benchmark
forces offline model loading and accepts only cached local model IDs. CPU Qwen
uses float32 without bitsandbytes; this avoids very slow emulated quantization on
AVX2 hosts. Use `emet jobs` when selecting CUDA on a shared GPU.

This policy preflight has **assisted skills and perception**: room names are
known, and object labels/positions are supplied only for objects with pixels in
the rendered head-camera segmentation. Hidden objects never enter the policy's
inventory. Its small observation cache preserves old evidence until a fresh view
updates it. This cache is not the learned DynaGraph/voxel backend. A source that
moved since it was observed cannot be planned from fresh evaluator coordinates;
the skill requests another observation before planning.

Model completion is a claim: private measured scoring still determines success.
For the moved-object case, first observation and delivery ordering must also pass.
Malformed JSON can have missing closing punctuation repaired, or its fully formed
leading tool calls recovered from a broken batch. Both original and normalized
responses are recorded; missing arguments or values are never invented. Task mode
preserves the shared agent's ordered tool batches; navigation refreshes visible
observations before the next call. Each task has explicit action and decision budgets.
Timeouts/exhaustion are incomplete tasks, never success.

The default skill interface is `atomic`: `pick_place` uses the shared guarded
planning/execution functions in one skill. `--skill-interface plan_execute`
exposes separate planning and one-shot execution handles for interface comparisons.
The policy sees its own pending plan receipts and completed skill receipts, not
evaluator goals. Both interfaces include bounded `find_objects` room search using camera-visible
segmentation; it visits unobserved rooms first and stops after at most one pass
through the room list. Missing-object errors direct the policy to that search
skill without revealing a hidden location. Both interfaces require observed source/destination labels and
reject stale source positions. Navigation immediately returns a fresh observation,
including when a model requests several tools in one turn.

The initial 0.8B/2B and separate-plan 4B diagnostics exposed invalid tool names,
malformed JSON, repeated travel and incorrect plan-handle use. Those traces are
retained as diagnostics, not a controlled ranking of models. Development changed
the prompt, continuation behavior and skill interface during these diagnostics; these runs are not a frozen benchmark comparison.
Restricting execution to one action per reply caused repeated navigation in a
Qwen2.5 diagnostic, so the default retains ordered batches.

## Environment layouts

| Suite | Layout | Intended use |
| --- | --- | --- |
| `agent_tasks.yaml` | Three rooms, unobstructed corridor | Interface debugging |
| `agent_tasks_furnished.yaml` | Three rooms with extra furniture | Geometry/visibility regression |
| `agent_tasks_four_room.yaml` | Four rooms including a study, longer routes | Multi-room task execution |

All suites live under `src/emet/config/benchmarks` and accept `--suite`. All three
placement tasks passed rendered assisted controls in each layout. These are small
procedural MuJoCo fixtures, not native BEHAVIOR/ProcTHOR scenes. The latter remain
a later realism gate.

## Commands

```bash
uv run emet eval agent-tasks list
uv run emet eval agent-tasks preflight
uv run emet eval agent-tasks run --episode cross_room_delivery --out /tmp/agent-control
uv run emet eval agent-tasks run --repeats 3 --out /tmp/agent-certification
uv run emet eval agent-tasks inspect /tmp/agent-control/cross_room_delivery_1
uv run emet eval agent-tasks replay /tmp/agent-control/cross_room_delivery_1
uv run emet eval agent-tasks replay /tmp/agent-control/cross_room_delivery_1 --rerun
uv run emet eval agent-tasks export /tmp/agent-control/cross_room_delivery_1
uv run emet eval agent-tasks compare /tmp/agent-certification/cross_room_delivery_1 /tmp/agent-certification/cross_room_delivery_2
uv run emet eval agent-tasks export /tmp/agent-certification --paper-dir paper
```

Use the existing `emet jobs` lifecycle for GPU rendering. Do not run broad
stale-process cleanup while other worktrees are active. Each native episode is an
owned subprocess with a ten-minute hard limit and process-group cleanup. Runs are
sequential; native failures stop the batch. Existing output directories are never
overwritten. `--no-render` runs geometry/tool checks but cannot pass the visual gate.

For a machine with Mesa EGL but no usable NVIDIA driver, a CPU-only render can use:

```bash
MUJOCO_GL=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
LIBGL_ALWAYS_SOFTWARE=1 \
uv run emet eval agent-tasks run --out /tmp/agent-software-render
```

This override selects software rendering; it does not modify drivers or other jobs.
CPU tests need no renderer:

```bash
uv run python -m pytest src/test/eval/test_agent_tasks.py -q
```

## What is being certified

The bundled `emet/config/benchmarks/agent_tasks.yaml` defines a small, deterministic
three-room fixture. Each case starts a fresh MuJoCo model using the packaged rby1
asset. Navigation follows room-center corridors, sampling every 2 cm against
walls/tables inflated by a 0.40 m footprint radius. This is a conservative geometric
control, not wheel dynamics, carried-object collision checking, or arm IK.

Manipulation invokes the existing CHAT tool registry, opaque one-shot plans and
shared TAMP implementation, with simulator object-pose teleportation. The fixture
adapter performs destination approach before applying a cross-room placement.
Forward kinematics supplies the measured body state and camera renders. The head
camera's 35-degree downward fixture mount is recorded as assistance.

Success means every target's measured center is within its destination's explicit
XYZ region and the target is not held. Missing/nonfinite states fail. These are
**placement-region predicates**, not proof of stable contact or successful grasping.

The moved-object case relocates the cylinder after the first delivery, while the
robot is in another room. Its scripted witness uses private ground truth. Therefore
it verifies event/scoring plumbing and continued solvability, **not learned memory
recovery**. Actual memory fields remain empty until supplied by the shared agent.

`certificate.json` requires three complete successful rendered repeats for every
selected case. It certifies only the explicit assisted fixture profile. The entire
suite requires selection of all three cases; a subset is only a subset certificate.

## Negative controls

```bash
uv run emet eval agent-tasks run --episode two_object_collection --control partial --out /tmp/agent-partial
uv run emet eval agent-tasks run --episode cross_room_delivery --control noop --out /tmp/agent-noop
uv run emet eval agent-tasks run --episode cross_room_delivery --control wrong_destination --out /tmp/agent-wrong
uv run emet eval agent-tasks run --episode cross_room_delivery --control stale_plan --out /tmp/agent-stale
```

These should exit nonzero with `task_failed`, retaining their evidence. `noop`
claims completion without motion; independent scoring must reject that claim.
`partial` completes only the first of two goals. `stale_plan` moves the target
after planning and exercises the real shared plan validator.

## Visual artifacts and provenance

Each episode writes `manifest.json`, hash-linked `events.jsonl`, `metrics.json`,
camera PNGs, and a generated MJCF. `policy` fields and evaluator truth are separate;
these diagnostic files must never be passed wholesale as model context.

Offline exports: self-contained `report.html` with step slider and floor-plan
replay; `episode.rrd`; POV/overhead `episode.mp4`; `overview.png`/`.pdf` and
`storyboard.png`. Missing images are explicit, not silently reused from another
step. `artifacts.json` distinguishes evidence completeness from task outcome.

The HTML/Rerun memory panels display "not recorded" for assisted controls. Do not
substitute evaluator object positions for an agent's graph or invent voxel state.
For paper figures, retain assistance, sample count, source hashes and rendering
commands. Use small checked-in summaries; full runs stay outside the repository.

## Implementation validation (2026-09-13)

All 33 CPU scoring, trace integrity, CLI, publication-integrity and shared-tool
regressions pass. Software EGL renders the fixture without a working NVIDIA
driver. All nine assisted replays (three cases, three resets each) completed with
HTML, Rerun, MP4, overview and storyboard artifacts. All four negative controls
were correctly rejected, retaining complete evidence; partial completion scored
one of two goals. The final repeat battery records implementation and image
hashes; offline exports separately record exporter and finalized artifact hashes.
Its aggregate is exported into the paper data, and the paper builds successfully.

Development-host evidence: `/tmp/emet-agent-preflight-release` (successful controls)
and `/tmp/emet-agent-negative-controls` (expected failures). These large artifacts
are not committed. The small paper summaries retain provenance and figure hashes.

The existing **live ZMQ default-table rby1 control passed** on `e8564ac6` with
software rendering: `plan_pick_place -> execute_pick_place_plan`, 0.1020 m measured
object displacement, and 0.0200 m destination error. Log:
`/tmp/emet-agent-zmq-control-unrestricted.log` on the development host. This is
the existing assisted teleport-placement control, not a new learned benchmark.
The first sandboxed attempt could not bind local port 6201; the bounded retry
outside the sandbox passed. No other process was stopped.

## Remaining acceptance gates

- Multi-room ZMQ/robot integration beyond the existing default-table control.
- Sustained full DynaGraph agent execution beyond the shared task runtime baseline.
- Perception-only object inventories and observed room topology.
- Actual voxel/graph snapshots and learned dynamic-memory recovery (exact model
  inputs and observed-cache revisions are already recorded).
- Frozen local-model baselines across layouts after policy diagnostics pass.

The fixture's successful control runs do not close these gates.

Paper export requires three matched complete witnesses per selected case. It
generates small summaries and a table under `paper/data/agent_task_preflight`,
before/after PDFs under `paper/figs`, and an architecture figure distinguishing
exercised components from pending agent/memory integration. Build with
`./paper/build.sh`; the new appendix is included from `paper/main.tex`.

## Export local policy results

Use `emet eval agent-tasks export-agents RUN... --paper-dir paper` to publish
actual local-model diagnostics into `paper/data/agent_task_policy` and the appendix.
It accepts failed tasks when their evidence is complete, rejects mock/witness runs,
and verifies artifact hashes. Policy tables remain separate from witness certificates.


## Local integration results (2026-09-13)

The shared local policy executed all three tasks across the three layouts. These
are development diagnostics with changing interface code, not matched repeats or
a model ranking. All used cached Qwen2.5-3B-Instruct on CPU; paid cost was $0.

| Task / layout | Measured goals | Decisions | Agent outcome |
| --- | --- | --- | --- |
| Delivery / original three-room | 1/1 | 3 | Completed |
| Collection / furnished three-room | 2/2 | 8 | Repeated actions; decision budget exhausted |
| Moved-object revisit / four-room | 1/2 | 12 | Failed task and response protocol |

The collection run delivered both objects, but did not stop, so overall success
remains false. The revisit run failed to complete the prerequisite green-marker
delivery. The benchmark accepts neither confident narration nor partial placement
as success. Its three-task controlled shared-runtime check passed, including
observation refresh after relocation; this establishes adapter behavior, not local
model competence at recovery.

Reports retained on the development host:

- Delivery: `/tmp/emet-agent-local-baseline/base/cross_room_delivery/report.html`
- Collection: `/tmp/emet-agent-ordered-batch/two_object_collection/report.html`
- Revisit: `/tmp/emet-agent-local-baseline/four_room/moved_object_revisit/report.html`

Each folder also contains camera frames, raw model/tool events, metrics, Rerun,
MP4, overview figures and artifact hashes. The paper summary records each run's
source hashes and assistance. New runs additionally archive the relevant Python
sources. Older development runs retain hashes but did not archive those sources.

Earlier collection variants are retained: the pre-search interface reached 1/2
(`/tmp/emet-agent-local-baseline/furnished/two_object_collection`), and the
single-action variant timed out at 0/2 (`/tmp/emet-agent-final/two_object_collection`).
These failures motivated the bounded search and ordered-batch interface; they are
not omitted successes/failures from a frozen evaluation cohort.

Validation: 103 focused tests passed, covering shared robot-loop compatibility,
task continuation and budgets, parser recovery, observed-state isolation, stale
plans, private temporal scoring and publication integrity. Rendered witness
controls passed all nine task/layout combinations; three additional controlled
shared-runtime cases passed. The full paper builds with the policy table and
figures. Physical manipulation, full learned DynaGraph integration, native
BEHAVIOR scenes and paid providers remain separate gates.
