# Controlled shared-agent capability study (proposal)

This is a proposed diagnostic suite, not a qualified benchmark or a claim of
novelty. Build on the existing observation case catalogue and native TAMP
fixtures rather than inventing another agent loop. Public EQA/OVMM tasks provide
external relevance; controlled interventions explain which capabilities help.

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
