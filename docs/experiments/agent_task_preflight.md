# Agent task preflight: tools and visual evidence

Roadmap: [implementation plan](../plans/agent_multiroom_environments.md).
Paper entry: `paper/main.tex`. Paid pilot: [planned, disabled](agent_hosted_pilot.md).

## Commands

```bash
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
- Sustained shared-agent execution, including more than three decisions.
- Perception-only object inventories and observed room topology.
- Actual voxel/graph snapshots, model-input references and dynamic-memory recovery.
- Local-model attempts on frozen cases after those contracts pass.

The fixture's successful control runs do not close these gates.

Paper export requires three matched complete witnesses per selected case. It
generates small summaries and a table under `paper/data/agent_task_preflight`,
before/after PDFs under `paper/figs`, and an architecture figure distinguishing
exercised components from pending agent/memory integration. Build with
`./paper/build.sh`; the new appendix is included from `paper/main.tex`.
