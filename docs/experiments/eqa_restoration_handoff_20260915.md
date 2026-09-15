# EQA restoration handoff — September 15, 2026

## Resume here

Current priority from the user: restore the heavily tested EQA path for merge,
then diagnose and fix remaining failures. Keep the single shared EQA/OVMM/TAMP
harness; avoid benchmark-specific rules or requiring object segmentation to
answer a visual question. No full sweep is requested.

**Startup is repaired and tested. Answer-quality diagnosis is in progress; the
new prompt bug described below is NOT patched yet.** No new experiments were
launched during this diagnostic turn. Do not confuse earlier successful test
counts with validation of a future prompt/policy change.

## Working state

- Work in `/tmp/emet-nav-fix`, branch `fix/room-wheel-contact-profile`.
  HEAD before this handoff: `58389f8e`, pushed to the same origin branch.
- PR [#169](https://github.com/cpaxton/home_robot_v3/pull/169) is stacked on
  `fix/paired-regression-audit`, PR
  [#167](https://github.com/cpaxton/home_robot_v3/pull/167). No merge performed.
  Confirm remote status before changing the stack.
- The original `/home/cpaxton/src/home_robot_v3` checkout has unrelated dirty
  work. Preserve it. Do not push to main.
- `/tmp/emet-reach-eval` is the clean frozen evaluation checkout at `fdb441a9`.
  `/tmp/emet-geometry-eval` holds earlier frozen source `0ecc0aa9`.
- Use the original repository's `.venv/bin/python` for unit tests and
  `.venv-habitat/bin/emet-habitat` for Habitat. Habitat needs its environment's
  `lib` on `LD_LIBRARY_PATH`; the CLI wrapper handles this.
- Heavy jobs must run serially through `emet jobs --cpu-safe --gpu-exclusive`.
  CPU cores 8/9 have been unreliable; keep OMP/OpenBLAS/MKL threads at 1.
  No real-robot operations, parallel heavy experiments, or main pushes.

## Completed EQA repair and results

The old six-case run at `~/runs/emet/staged-pregrasp-eqa-20260915` crashed with
exit 134 before answering. Shared A* imports added in `89581c70` transitively
loaded MuJoCo through `base_goal_rank` / `voxel_arm_collision` before Habitat.
Serial isolation reproduced the native OpenGL abort with MuJoCo and the full
runner, but not bare Habitat, OpenCV, Torch, or Open3D individually.

Independently reviewable commits:

- `5d299c9d`: import MuJoCo only when arm FK is actually called; add full-runner
  import-boundary and actual MuJoCo FK regression tests.
- `fdb441a9`: `scripts/check_habitat_runtime.py` preflights full runner imports
  and a real rendered frame before pilot episodes. A bare renderer probe alone
  missed this regression.
- `22627d2c`, `58389f8e`: root-cause and final-result documentation.

Validation: **607 tests passed, 4 skipped** before this diagnostic turn.
No driver/package/model changes were needed. Final findings were posted to
#167 and #169; see [#169 comment](https://github.com/cpaxton/home_robot_v3/pull/169#issuecomment-5687090422).

Frozen retry job `20260915_153507_23eb03`, name `eqa-restored-pilot`, completed
all six processes with exit zero on `fdb441a9`:

| Preset | q15 | q16 | q25 | Accuracy |
| --- | --- | --- | --- | --- |
| hybrid | correct | correct | wrong (B; gold C) | 2/3 |
| qwen_box | correct | correct | wrong (D; gold C) | 2/3 |

Artifacts: `~/runs/emet/eqa-restored-20260915/<preset>_eqa_<qid>/`.
`result.jsonl` links debug bundles, maps, and videos. Exact model inputs and
replies are in the debug bundles; grounding JSON/PNG/NPZ files are in each
case's `evidence/grounding/`.

September 11 comparison (`eed1d868`,
`~/runs/emet/shared-grounding-habitat-retry-20260911`): hybrid T/T/F and
Qwen-box F/T/T. Aggregate accuracy is unchanged, but Qwen-box gains q15 and
loses q25. This is a small, unseeded historical comparison, NOT evidence of
per-question equivalence. Correct q15 negatives also have weak absence evidence.

## New trace diagnosis: do not collapse two distinct failures

Debug root:
`~/.cache/habitat_eqa/episodes/eqa-restored-20260915-<preset>_eqa_25/q0025_lazy_graph/`.
Read `agentic_trace.jsonl`, not just `raw_eqa.txt`: the final answer can come
from the earlier per-view assessor rather than the final multi-image EQA call.

### Qwen-box: premature positive assessment

- Navigates successfully 3.24 m to the voxel proposal for `towels`.
- Candidate and confirmed-view grounding both abstain.
- Per-view `vlm_assess` nevertheless returns `present=true`, `answerable=true`,
  `need_more_views=false`, and “There are some in the bedroom,” claiming a
  white towel on a bed/surface.
- `_maybe_confirm_answerable` accepts `single_view_present`; submission happens
  after **one round / one move / zero exploration**, with budget remaining.
- Final multi-image EQA disagrees, but the confirmed view answer takes precedence.
  This is not a timeout or forced budget-exhaustion answer.
- Manually inspected saved RGB
  `evidence/grounding/grounding-74630766637a46ddb1cba2fe97faa226.png`: it shows a
  crib beside windows with fabric draped over its sides, not a bathroom.
  The fabric's identity is ambiguous; do not claim a definitive towel label.

### Hybrid: coverage/budget failure

- Most assessments explicitly say no towel/bathroom evidence.
- Round 3 suggests an absence answer despite no target, but confirmation
  correctly defers it.
- Finishes **eight rounds / one frontier exploration**, budget exhausted,
  and submits B through the forced-answer/debias path.
- Needs an audit of routing, repeated investigations, frontier choices, and
  available views. Do not “fix” by hardcoding bathroom, gold answer C, or q25.

### Concrete grounding prompt bug (next implementation)

`src/emet/memory/vlm_region_grounding.py:select_vlm_region` currently builds
the box-only prompt as `Locate {description or query!r} in the image`.
EQA supplies the entire MCQ as `description`, including all options.
The saved response explicitly says the quoted question text is not visible,
so it abstains even while mentioning fabric/towels on the crib.

Evidence JSON:
`~/runs/emet/eqa-restored-20260915/qwen_box_eqa_25/evidence/grounding/grounding-74630766637a46ddb1cba2fe97faa226.json`.

Related call paths:

- `src/emet/memory/graph_eqa/agentic/views.py:ground_confirmed_view` passes
  `target_description=executor.question`.
- `src/emet/memory/graph_eqa/agentic/investigate.py` grounds a query candidate
  after arrival; inspect where candidate descriptions are populated too.
- `src/emet/memory/graph_eqa/agentic/assess.py` owns the single-view gate.
- `src/emet/eval/agentic_vlm_assess.py` owns the visual assessment prompt.

Separate the **object query** from **task/question context** in grounding
prompts. Preserve attributes/relations needed by OVMM, but do not tell the
model to locate a full question string or treat hypothetical MCQ options as
observed facts. Apply consistently to box-only and box-plus-point paths.
Add prompt-contract tests in `src/test/memory/test_vlm_region_grounding.py`.

Important: fixing this bug may improve localization but does NOT establish
that q25's answer or coverage will improve. EQA is intentionally allowed to
answer from RGB when geometry abstains; do not replace that with a YOLOE/SAM gate.

## Next bounded work

1. Patch/test the prompt boundary above. Preserve existing geometry and identity
   checks, abstention, whole-object box instructions, and correction behavior.
2. Audit hybrid's action sequence for wasted rounds and candidate text. Its
   trace includes voxel proposals for “going shower now” and “now need grab”
   despite VLM target extraction successfully returning “towels.” Locate the
   retrieval construction before deciding on a minimal shared fix.
3. Assess whether ambiguous positive answers need better visual context or
   corroboration. Do not blindly disable single-view answers: existing tests
   and historical evidence support them for clearly visible objects.
4. Save exact prompts/replies/images; run relevant unit suites first, then a
   frozen serial q15/16/25 comparison at unchanged model/settings/budgets.
   Keep prompt-only versus policy changes separable to attribute results.
5. Update results and TODO honestly. Startup repair can land independently of
   unresolved answer-quality and manipulation work. Never call 2/3 “all passing.”

Existing pilot driver: `scripts/run_shared_grounding_pilot.sh`, `PHASE=eqa`.
It refuses dirty source or reused output directories and exports maps/videos.
Set `HABITAT_BIN` to the original repo's `.venv-habitat/bin/emet-habitat` and
`SAM2_SOURCE` to the original repo's `third_party/segment-anything-2`.
Both presets use Qwen3-VL-8B int4/SDPA, lazy graph/query memory, 20 planning /
10 movement settings, and no HM3D semantic/enriched labels. The agentic trace
shows an internal eight-round/eight-nav budget; do not silently change either.

## Other gates remain open (not this turn's scope)

Room OVMM/manipulation is not accepted and learned TAMP remains gated.
Prior completed pilots: tabletop passes; Molmo staged pregrasp clears the
counter but is too far to reach. The upper-workspace guard in `6a1579e4`
safely rejects the subsequent relocation (`sample_nav_failed`) rather than
overextending. Native and NoSlip RoboCasa can runs hit final-servo extension
timeouts; neither establishes grasp retention. Some older sections of
`shared_grounding_pilot.md` and TODO still say these jobs are pending; reconcile
them against the saved results before claiming current project status.
Hard Habitat OVMM, door/drawer skills, Sourccey/Galaxea and real robots remain
deferred. The user wants one reliable harness, not a succession of detours.
