# Controlled shared-agent capability study (proposal)

This is a proposed diagnostic suite, not a qualified benchmark or a claim of
novelty. Build on the existing observation case catalogue and native TAMP
fixtures rather than inventing another agent loop. Public EQA/OVMM tasks provide
external relevance; controlled interventions explain which capabilities help.

## Implemented development fixtures (2026-10-28)

`configs/eval/harness_capability_dev.json` now freezes 12 development cases:
seen/unseen recall, far/near approach, unknown-clear/occupied route,
immediate/delayed task intent, unchanged/moved target, and unique/ambiguous box.
The builder creates two ordered MuJoCo snapshots per case, with 20 x 40 x 30 cm
colored target geometry and explicit camera calibration. Color questions do not
name the answer color; recall and delayed-task targets use green/yellow rather
than a single constant blue answer. These are synthetic
ray/contact/evidence controls with a simple planar probe, not full robot models.
The search-then-act prototypes retain a navigation/answer instruction; physical
object transport is deliberately not represented as qualified here.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 12 .venv/bin/python scripts/build_harness_capability_dev.py \
  --output /tmp/capability-dev-geometry
```

Default execution is CPU-only: real MuJoCo rays and contacts, the production
depth-support gate and swept-route validator. All 12 controls pass. Floor cases
distinguish a floor return from an obstacle and validate clear/blocked geometry;
they do NOT qualify a sensor-driven map update or actual wheel execution.

For an agreed idle GPU window, add `--render` under a managed serial GPU job.
This exports 24 RGB frames, metric depth, camera K/poses and scorer-only target
masks. `evidence_manifest.json` references RGB only; `scorer-*.npz` never enters
the evidence loader's model payload. `qualification.json` explicitly reports
`active_agent_qualified=false` and `manipulation_qualified=false`.
Rendering/loader validation is a prerequisite, not model success or an active
benchmark. Real robot bindings, action-triggered interventions and independent
end-to-end scoring remain next-stage work; this status must not be promoted from
a successful geometry test or recorded-QA replay.

`scripts/replay_harness_capability_evidence.py` runs the six color/recall/ambiguity
cases through the production `VLMConversation` and remote client, comparing
full history with current RGB only. It saves exact request packets and raw
responses. Its strict short-answer score is a development diagnostic requiring
manual audit, not an official EQA judge. It cannot measure navigation, floor
recovery, constraint-preserving action sequencing or manipulation; the six
location/floor cases remain geometry controls at this stage. The remote endpoint's
quantization is reported as unknown unless independently established; do not
silently equate it to the prior local int4 pilot.

## First recorded-evidence diagnostic

At `06e8c243`, job `20261028_184222_324368` renders/binds all 12 cases and
24 RGB-D snapshots successfully in about three seconds. The subsequent remote
Qwen job `20261028_184235_7110ff` evaluates the six color/recall/ambiguity cases:
full history 4/6, current RGB only 1/6, zero inference errors. Earlier green/yellow
evidence is recovered with history; both arms guess a color on absent and
ambiguous negatives. All twelve raw final outputs were manually inspected;
the strict parser agrees with the visible color outcomes. This single synthetic
development replay shows an evidence-availability effect, not active-agent,
manipulation, generalization or statistical-superiority evidence.

Artifacts: `~/runs/emet/capability-dev-render-colors-20261028/` and
`~/runs/emet/capability-evidence-replay-20261028/`. Requests retain exact image
history and text; actor packets contain no target masks, references or variants.
Minimal results: `paper/data/shared_harness_pilot_20261028/capability_replay_summary.json`.
The local GPU is reserved only for the brief render; remote replay does not hold
the local GPU lock. Tests: 49 relevant dataset/loader/geometry/dialogue checks pass.

![Audited recorded evidence examples and retained failures](../../paper/figs/harness_capability_dev_overview.png)

Next stage is binding the same interventions to real Stretch/Molmo tasks through
the existing agent, and evaluating active approach/recovery independently. Do
not promote the synthetic planar probe to an implemented robot adapter. Official
EXPRESS/AbstainEQA scoring still needs actual frame packs and evaluator wiring.

## Live agent binding (2026-10-29)

`scripts/run_harness_capability_agent.py` builds full Stretch fixtures for the
12 authored cases and connects a live bridge to the production `run_agent_with_robot`
CHAT loop. All 12 fixture compilation/intervention admission tests pass. The
initial binding uses real RGB-D and measured wheel/head tools, remote Qwen, and
geometry-only memory without learned proposals/segmentation. It therefore cannot
qualify final object localization or manipulation. Scenario turns are recorded:
the live search/act prototype visits a point 0.5 m forward, rather than replaying
the synthetic snapshot camera jump. Do not treat these trajectories as matched.

The evaluator sets only initial fixture/head conditions and the declared
between-turn object relocation. Subsequent base motion uses ordinary wheel
commands. Actual HTTP requests/responses retain image data and dialogue, while
RGB-D/calibration and local/world base/head poses are saved independently.
Physics snapshots and intervention coordinates stay scorer-side. Exit zero
means protocol completion, not a successful task; `success` remains null until
manual evidence/action scoring is qualified.

This binding exposed two real CHAT evidence gaps: `OpenaiClient` silently ignored
the image keyword, and the outer CHAT loop reset history at each user command.
The first was reproduced with a failing test at the actual call boundary; the
repairs now preserve exact image pixels and earlier user/image turns in the
real loop. Existing HTTP-client history bounds remain 50 messages. These repairs
are research-branch changes, absent from the frozen-main benchmark refresh.
The surrounding CHAT/client/dispatch/dialogue tests pass.

The first live attempt correctly refused commands because its physical audit
path was missing. The corrected case uses `actuation.jsonl` before bridge startup;
the refused run remains an infrastructure artifact. First corrected live job:
`20261029_005049_ee39a4` also exposed competing evaluator/agent command owners.
The bridge correctly refused the second owner. The loop now accepts an existing
client and a between-turn input reader; the evaluator and agent share a single
owner without weakening the lease protocol. The perception-disabled mode is
explicit: merely disabling visual servo did not prevent mapper model loads.

At `0a7142bf`, job `20261029_005852_5281e3` completed the first real turn protocol.
The initial RGB was visually checked: green target on a brown box-shaped support.
The final HTTP request retained all three earlier/current images. A 90-degree
turn was rejected for 25 unknown footprint cells; actual floor inspection reduced
that checked pose to zero unknown cells, but measured yaw did not change. The
next instruction said **not to call more tools**, despite remaining rounds, so
Qwen summarized its intent to retry rather than issuing the retry. It answered
"brown" to the later color question. This is not accepted recall success: the
motion was incomplete and the support made the singular box referent ambiguous.
Keep the trace at `~/runs/emet/capability-live-recall-v3-20261029/`.

`458fbaa8` removes the premature info-tool finalizer and permits further authorized
tool calls within the existing budget. Round exhaustion and TAMP failure/unknown
boundaries remain. An actual-loop test exercises sensing then motion; it would
previously end after sensing. `f268d594` changes the support to a cylinder and
declares live fixture version 2 before scoring a new pair. This is a development
fixture repair, not a held-out improvement. New paired job `20261029_010733_da5fb2`
uses `~/runs/emet/capability-live-pair-v4-20261029/`. Qualification still requires
actual request, image, measured motion and final-answer inspection.

The corrected pair is now manually inspected: both protocols complete; the seen
case answers green and the unseen case abstains. **Neither requested turn is
executed**. Both encounter unknown start-footprint cells, and targeted floor
inspection returns `floor_target_unobservable`; measured yaw changes are below
0.0001 degrees. The seen box remains visible at the end, so correct color is
not evidence of recall after turning. These are 0/2 accepted complete tasks,
with the safe refusals and all context preserved. Next isolate broad downward
inspection versus targeted inspection through the same tools, with a scripted
feasibility control, before changing movement rules or crediting recall.
52 CHAT/client/dispatch/dialogue tests and 19 fixture/sweep/export checks pass.

![Actual camera evidence, final answers, and absent physical turn](../../paper/figs/harness_live_evidence_and_motion.png)

Audited inputs and reproduction: `paper/data/harness_live_diagnostic_20261029/`.

## Families and paired interventions

| Family | Matched conditions | What it measures |
|---|---|---|
| Observe, turn, recall | Target in an earlier view; current view has a distractor; repeat with earlier target absent | Evidence identity, useful history, factual final context |
| See, approach, localize | Same object just inside/outside declared sensor grounding range | Whether visible evidence leads to a useful safe approach and metric localization |
| Inspect and replan | Same route with unknown floor, real obstruction, or measured drop | Whether feedback drives sensing/replanning without manufacturing clearance |
| Search, then act | Answer an attribute question or transport the same grounded object after a delay | Shared EQA/OVMM memory, retained task constraints, fresh geometry before action |
| Revisit a changed scene | Move/remove a previously observed object versus unchanged control | Staleness, reacquisition, recovery after unexpected outcomes |
| Evidence insufficient | Present, absent, ambiguous and inaccessible targets with matched language | Grounded answers, appropriate abstention and cost of failed search |

Use readable household objects for basic coverage; declare object dimensions
and projected target pixel area. Tiny objects such as the archived 2.8 cm sugar
cube are an explicit stress stratum, not silently excluded after a failure.
Changes are scripted simulator interventions between actions, never hidden GT
updates to the model's memory. GT masks/poses and feasibility checks are scorer-only.

## Development and assessment splits

First build one positive/control pair per family (12 development episodes).
Validate feasibility with deterministic action sequences or privileged planners
using the SAME collision/contact checks; this is a feasibility control, not a
learned-agent result. Invalid fixtures remain in the audit with a reason.

Before tuning, freeze four layouts per family: two development, two assessment,
with three seeds. Each arm then has 36 assessment episodes (6 x 2 x 3). Use the
same scene/asset/seed population across EQA and OVMM variants where possible;
split by underlying scene, not paraphrase or question ID. Independently review
the tasks and include negative controls. Publish generators, manifests and all
failures. Do not describe historically inspected rooms as unseen holdouts.

## Separate opportunities from outcomes

For every captured frame retain RGB-D, intrinsics/extrinsics, acquisition time,
scope, measured base/head pose, supplied image roles, tool call/results, and
source/config/model hashes. Scorer-side GT masks provide target visibility and
pixel area; visibility alone is not successful localization.

Report end-to-end success AND the diagnostic chain:

1. Target enters a captured view; target evidence actually reaches the model.
2. Model correctly recognizes presence/absence; selected support belongs to target.
3. Admitted geometry meets frozen error thresholds; hypotheses remain distinct.
4. Requested motion is feasible, executes, settles and obtains a useful new view.
5. Task succeeds, or abstains/stops appropriately after a bounded failed search.

Failure codes: missing view, dropped/stale context, semantic miss, invalid support,
localization error, planner rejection, execution failure, reasoning error and
infrastructure failure. Keep denominators for all assigned episodes; report
conditional rates alongside, never instead of, end-to-end outcomes.

Costs: sim seconds, wall seconds, distance/turns, model calls/input images/tokens,
mapping time, planning time, wait time, graph size and retained target recall.
Include failed episodes and timeouts; fewer nodes or early exits alone are not wins.

## Controls and ablations

Three tiers isolate failure boundaries: deterministic tool contracts; scripted
or privileged high-level choices through real physics; learned Qwen through the
same tools. Add a stronger-model subset later as a model x harness comparison;
it is not a replacement for reporting local Qwen results.

Compare pre-stack and merged harnesses as complete systems. Within the merged
harness, change ONE factor at a time: dialogue/context, view selection, evidence
feedback or memory ingestion. Keep VLM, quantization, scene, exploration, budgets,
skill backend and scorer fixed. Existing configurable policies can be ablated;
history-off and feedback-reduction need clean experiment-only switches first.
Do not reintroduce unsafe execution or fabricated motion success as a baseline.

Start with a six-case ablation screen, then freeze one candidate and assess on
the withheld fixtures. Do not run a Cartesian sweep. Discovery tasks assess the
shared harness; RL skill checkpoints are pinned independently and are held
constant within each pair. Robot-specific changes belong only in capability
adapters. Sim oracle/TAMP and learned-perception results remain separate rows.

## External additions

- [EXPRESS-Bench](https://github.com/HCPLab-SYSU/EXPRESS-Bench): prioritize
  exploration/answer grounding. Existing metadata adapter does not provide active
  runtime or official scoring. Qualify evidence replay first, active exploration
  separately; supplied trajectories are not autonomous exploration.
- [AbstainEQA](https://github.com/gibrantaowu/AbstainEQA): matched answerable and
  unanswerable evidence, with answer quality and abstention metrics. Qualify the
  evidence loader and official evaluator before scoring the existing ten-pair
  development selection. This tests evidence use, not manipulation.
- [RoboCasa composite tasks](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html):
  promising external multi-step target, after a visible larger-object fixture
  and observed skill interface qualify. Our robot-merged scenes are adapted
  tasks, not official RoboCasa365 leaderboard results. Pin task horizons and
  simulator/assets; do not change official horizons to repair failures.

Prefer these additions over simultaneously integrating many unrelated simulators.
Our controlled suite supplies causal diagnostics; external tasks test whether
those diagnostics predict broader capability.
