# Query-driven memory implementation and pilot

The objective is a shared EQA/OVMM harness with query-driven instance creation,
image-supported annotations, and fresh geometry at the manipulation boundary.
Retrieval candidates are not confirmed objects. No full sweep is authorized.

## Landed prerequisites

- Foundation draft PR #162: ingestion controls, DynaMem EQA fixes, isolated-run tooling.
- TAMP PR #160 merged into main at `4ed75850`; incorporated here. Its documented
  Innate Mars navigation and Sourccey cabinet placement failures remain open.
- Learned Habitat OVMM clients no longer receive semantic labels or placement
  metadata through observations/session data; the evaluator retains ground truth.
- Candidate lifecycle API and additive localization-result fields have focused tests.

## Connected prototype (opt-in; not a validated benchmark row)

The lazy controller now uses the same query-candidate path in agentic EQA and
OVMM find. Voxel retrieval records its source frame ID; repeated queries reuse a
stable handle without adding object nodes. After investigate captures a new voxel
frame, the controller runs detection on demand if streaming masks are absent,
then applies existing depth admission and fusion. Successful promotion exposes
the arrival image as the station evidence; it does not label everything at the
navigation target. The tool result and trace include the grounding outcome.

The search tier admits finite voxel matches at cosine similarity 0.14 or higher,
retaining source frame and score. These are **not** successful localizations;
the OVMM result receives coordinates only after grounded arrival. The original
DynaMem localization threshold remains unchanged for other methods. Since lazy
mapping skips detector initialization, the grounding path loads the shared YoloE
detector only when an arrival actually needs masks.

The graph runtime checkpoint stores candidate references. Reload always revokes
grounding, including when the resumed observation counter matches the old counter.
Old checkpoints without candidate state remain supported.

Enable these parameters before constructing `LazyGraphController` (backend
`lazy_graph`) for either task:

```yaml
query_driven_memory: true
eqa:
  agentic_verify: true
graph_object_fusion:
  enabled: true
  use_instance_nodes: true
```

Keep the other model, mapping, admission, fusion, and budget settings frozen.
The checked-in default remains `query_driven_memory: false`. This policy requires
the agentic loop; the classic arrival-label path is disabled when it is enabled.
OVMM must use agentic find, not the one-shot find adapter.

Single-episode Habitat commands now expose `--query-driven-memory` on
`run-episode --method lazy_graph` and
`run-ovmm-find-episode --backend lazy_graph`. The flag enables the shared agentic
loop without modifying fusion thresholds or budgets. Incompatible backends,
disabled instance fusion, and explicit one-shot OVMM requests are errors. Results
include `query_driven_memory`, so these runs are not mislabeled as ordinary lazy
arrival ingestion. The EQA query row disables simulator semantic observations.

Promotion currently requires one exact normalized detector-label match. A generic
`mug` detection does **not** verify `red mug`, and shared words do not establish
identity. Unsupported attributes remain ungrounded until semantic verification is
connected. Ambiguity is checked before the admission candidate cap, so truncation
cannot turn two possible objects into a unique match. Candidate capacity rejects
new proposals without silently evicting live references.
Fresh failed admission also records negative evidence. That query no longer
retrieves the rejected source frame, and stale cards cannot navigate back to it.
Other query phrases and newly observed source frames remain eligible. Query-mode
traces are enabled by default; a rejected approach is not a verified localization.

## Manipulation handoff

The DynaMem task executor's query policy now requires the existing visual-servo
adapter (`--visual-servo`). It reacquires a candidate immediately before acting
and supplies fresh object points to visual-servo grasp or point-cloud placement.
It rejects unsupported adapters instead of falling through to text-only AnyGrasp,
simulator body lookup, or teleportation. Other policies retain their existing
behavior. Habitat find remains a navigation benchmark, not a manipulation test.

During grasp approach, every new instance mask must match the exact detector class
and the grounded world geometry: at least ten finite points and 80% support within
the observed bounds plus 5 cm. Ambiguous or missing geometry aborts; central-mask,
largest-mask, temporal-mask and open-loop fallbacks cannot select another object.
This is local geometric tracking, not a claim of long-term visual re-identification.

Object identities use stable graph observation IDs, not node indices that graph
maintenance renumbers. After attempted motion, aliases lose grounding and the
moved object's old graph location is retired. A post-action capture is required.
Placement uses observed receptacle points rather than fabricating a surface at
the navigation target. An adapter returning success means its motion routine
completed: `last_query_manipulation.physical_success_verified` remains **false**.
Neither this value nor a successful mock is a benchmark manipulation success.

## Remaining acceptance work

- Connect attribute verification to observation-linked annotations, independently
  of geometric grounding and without weakening target identity checks.
- Run a real learned perception-to-pick/place episode on an appropriate simulator
  or robot with the visual-servo camera/kinematics contract. The GT TAMP executor
  is not a substitute for this gate. Independent physical success verification is
  still required before reporting OVMM manipulation accuracy.
- Controller tests cover on-demand detection, mask-derived geometry, absent and
  ambiguous targets, missing depth, instance opt-out, stale captures, checkpoints,
  and shared EQA/OVMM retrieval. Run live simulator acceptance before claiming
  task success; these tests do not establish answer or manipulation quality.

## Low-load smoke procedure

Use the installed `emet-habitat` wrapper, not its Python directly: the wrapper
sets the Habitat C++ runtime library path. Submit via `emet jobs run --cpu-safe
--need-mib 14000`; it holds the shared GPU lock. Run only one experiment at a time,
with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and
`TOKENIZERS_PARALLELISM=false`. In a worktree without its own environment, supply
the provisioned wrapper's absolute path and point `PYTHONPATH` at that worktree's
`src` and `packages/emet_habitat`; do not reinstall shared editable dependencies.

First run `emet-habitat egl-probe --question-id 1 --json` through the job runner.
Then run the single EQA command with `--query-driven-memory`, explicit
`--no-hm3d-semantics --no-enrich-labels`, and a unique output path/run tag. Smoke
budgets of `EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS=4` and
`EMET_EQA_AGENTIC_MAX_NAV_STEPS=3` are intentionally **not** paper settings.

## September 5 integration evidence

All runs below were sequential, behind the GPU lock, using `--cpu-safe` and
single-thread numerical-library limits. No full sweep or real-robot motions ran.
The final focused regression suite passed **186 tests**, including TAMP bridge
tests. Commit hooks passed. The six worktree-local Habitat CLI tests skip because
the temporary worktree has no provisioned Habitat environment; CLI startup was
instead exercised through the existing environment in the live runs.

| Run | Source | Outcome / limitation |
| --- | --- | --- |
| EGL `20260905_120524_46b7ca` | `1f31fd44` plus integration work | Context and RGB capture passed. |
| EQA q1 `20260905_121523_0f8b5d` | `572f038d` | Correct answer, four decision rounds, zero object nodes; final answer came from MCQ debias, not grounded objects. |
| OVMM lamp/bed `20260905_122109_ab6d21` | `572f038d` | Both find phases failed; exposed strict retrieval gating and missing lazy detector initialization. |
| Same OVMM case `20260905_122936_302915` | `19e4a57b` | Weak candidate led to four approaches; neither target grounded. Exposed missing rejection feedback. |
| Same OVMM case `20260905_124157_215cf1` | `61ac7b2d` | One rejected object approach, then three explores; four receptacle explores. Both find phases still failed, with five total graph nodes. No runtime error. |

The last run's `find_object_agentic_trace.jsonl` records candidate `-3000000`
rejected at round zero as absent or ambiguous. There was no successful promotion
in this live case. Positive promotion and the manipulation handoff currently have
synthetic/controller test coverage, not successful live OVMM acceptance. These
development runs cannot establish accuracy improvements or replace paired pilot
evaluation. The earlier direct-Python launch `20260905_121220_10b487` failed on
the Habitat C++ runtime path before model startup and is not an evaluated episode.

Artifacts are retained under `/tmp/emet-query-eqa-smoke2` and
`/tmp/emet-query-ovmm-smoke{,2,3}` on the development host. Preserve them with the
pilot artifacts before deleting temporary worktrees. The remaining external gate
is selecting a configured visual-servo simulator/robot for learned pick/place;
oracle TAMP attachment must remain a separately labeled benchmark.

## Limited acceptance rotation

- Resume the interrupted EQA random-16, retaining completed artifacts. The old
  no-instance row is diagnostic because its switch bypass was subsequently fixed.
- Compare DynaMem, arrival-only lazy memory, and query-driven memory on those 16
  questions. Freeze source, model, initial states, budgets, and row settings.
- Define six matched OVMM cases before outcomes: visible, occluded/cluttered,
  repeated objects, initially unseen, absent, and moved/revisited target.
- Run TAMP table pick/place, mixed-grasp rejection, rby1 relocation, and kinematic
  agent-tools gates; include at least one learned retrieval-to-manipulation sequence.
- Report per-episode outcomes, errors, grounding success, duplicate candidates,
  stale-target rejection, model calls, runtime, and memory size. Maintain explicit
  known-failure rows rather than silently treating them as passing coverage.

The prototype is incomplete until both EQA and OVMM exercise the shared lifecycle
and a learned target completes pick/place followed by fresh observation.
