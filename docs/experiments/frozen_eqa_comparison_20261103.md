# Frozen EQA answer-path comparison

The [GraphEQA connection](grapheqa_decoding_control_20261101.md) is complete.
This is the next fixed-observation comparison from the
[baseline plan](shared_agent_literature_review_20261029.md): compare the model's
answering/context pathways before building a matched active-agent experiment.

## Predeclared development pilot

Four snapshots cover three existing questions: q2/q6/q12 after the fixed initial
scan, and q6 after the reference policy's actual frontier arrival. They are
selected development cases, not four independent questions or a representative
test set. Initial scans provide limited coverage; the arrival snapshot is
conditional on the reference collector's trajectory. No success-frame or
reference-answer selection is used.

The case specification is `configs/evaluation/frozen_eqa_context_pilot.yaml`.
The builder copies CURRENT plus three uniformly spaced earlier PNGs, labels
their sequence/timestamp/estimated pose, checks context/capture alignment and
records source hashes. Older contexts undergo only the already-fixed projection
migration: isolated non-callable frontier nodes are omitted; connected evidence
cannot be dropped. Scorer labels are separate and not read by the builder.

| Arm | Actual implementation / input | Maximum generated-token allowance |
| --- | --- | --- |
| `images_only` | Fork's existing model-only MCQ runner, same four PNGs/question | 2,048 |
| `reference_graph` | Fork's existing GraphEQA planner, same packet plus captured learned graph | 2,048 |
| `emet_answer_packet` | Production `GraphEQAMemory.query_answer(model_final=True)` with an empty graph and the same packet | 64 keyword + 1,984 final = 2,048 |

The emet arm measures the production answer/context path, **not our complete
memory, retrieval, view-selection or active-agent system**. Its genuine keyword
call is counted. An evaluation transport preserves original PNG bytes instead
of the remote client's usual JPEG conversion. The server rejects unlisted,
reordered, cropped or altered images and excess hidden inference. Every arm
makes exactly one image-bearing call. It retains the full request, raw response,
usage and errors for all calls, including keyword extraction.

All arms share one real local Qwen3-VL-8B Instruct int4 model, 512-pixel per-image
preprocessing, and checkpoint decoding defaults. Seeds 0/1/2 reset torch's RNG
at **arm start**, not between the emet helper and final call. These are process
controls, not claimed HTTP seed support or a cross-GPU reproducibility guarantee.
Rotate condition order by case and seed; 4 × 3 × 3 = **36 planned trials**.
Serving uses the normal local HTTP handler and actual model inference; CPU
contract tests use fixtures but the experiment does not use mock model replies.
Manifest records loaded identity, effective generation config and library versions.

Gold is read only after all calls and server shutdown. Scoring maps only an
exact semantic option (case/whitespace normalized) or a sole valid letter; it
does not extract `B Black`, substitute earlier answers or infer labels from free
text. Unknown/abstaining answers are incorrect in the planned denominator.
Truncations, refusals and transport/validation failures remain failed trials.
Report this as `correct_if_stopped_here`, with zero interactions, not episode
success. Differences compare complete input/answering paths, not a causal graph
effect. No tuning decisions or paper-superiority claims follow from four snapshots.

## Tooling verification before inference

The CPU contracts exercise the actual production answer method and assistant
prefill, exact PNG transport, helper/final budget, retained truncation failures,
strict scorer normalization and deferred gold access. A migration test rejects
removal of connected graph evidence. Existing source checks reject tracked or
untracked source edits; both repositories are committed before inference.

Run the fixture builder once, prepare scoring labels in a separate process, then:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frozen_eqa_comparison.py \
  --fork /path/to/graph_eqa --cases /path/to/fixture/cases.json \
  --answers /path/to/scorer-only.json --output /path/to/new-output --seeds 0 1 2
```

Use `emet jobs --gpu-exclusive --cpu-safe`, CPU affinity 12, single BLAS/OpenMP
threads and offline model caches. The experiment checks the shared 20 GiB disk
reserve before every arm. Preserve failures and the declared population. No
parallel simulation/CPU-heavy experiments, robot motion, or full sweep are needed.

## Completed pilot

Job `20261103_182926_ce80b9` ran at emet `e245e011` and GraphEQA `5f3fc51`.
All **36 planned trials** were attempted. The job exits nonzero because one
GraphEQA reply fails its native answer enum (`A. By the dining table` instead
of the exact option value); the other 35 trials finish. This is a retained model
format failure, not an infrastructure crash. Gold was scored after inference
and server shutdown.

| Answer path | Completed / 12 | Correct if stopped here / 12 | Model calls | Mean wall s / trial | Mean prompt tokens / trial | Mean generated tokens / trial |
| --- | --- | --- | --- | --- | --- | --- |
| Images only | 12 | 0 | 12 | 2.44 | 2,087 | 85.5 |
| Reference GraphEQA | 11 | 1 | 12 | 11.97 | 9,577 | 442.5 |
| Emet answer packet, empty graph | 12 | 1 | 24 | 3.61 | 5,625 | 110.2 |

Every trial uses the same 2,048-token maximum allowance, including the emet
keyword call. Actual usage includes every call. Wall time includes API and
answer-path work, including the malformed GraphEQA reply, but excludes capture,
mapping, navigation and model loading (6.47 seconds). Neither these timings nor
the correct-answer counts measure full-agent efficiency or success.

Per snapshot, all three seeds are wrong for the initial q2/q6 packets. The one
correct GraphEQA reply is q6-arrival seed 0; the one correct emet reply is q12-start
seed 2. A correct guess does not establish visual grounding. Incorrect confident
responses number 11 for images-only, 5 for GraphEQA and 7 for the emet packet
path. These are descriptive counts on correlated development trials, not a
calibration study or a superiority result.

## Packet and visual audit

The post-run auditor verifies all planned case/seed/arm tuples, unchanged case
and PNG hashes, identical ordered PNGs across every image-bearing call, and exact
requested caps `[2048]` or `[64, 1984]`. It confirms no hidden inference or
transport errors. A guard added after this pilot also rejects fixture mutations
at arm start; the existing trial packets passed the independent audit, so no
inference rerun is needed for that unchanged-input guard.

Manual inspection of the question/view/path panels finds:

- q2's four views show the living room, without a usable view of the shower rug.
- q6-arrival provides a real change of position, but its packet still shows the
  stove/kitchen area; a ladder cannot be identified in these inspected images.
- q12 includes a narrow bedroom view through the doorway, insufficient to
  confidently count both bedside tables from this packet.

These are human image assessments, not ground-truth visibility labels. They
expose the limited acquisition coverage of this selected fixture. This pilot
qualifies the paired answering machinery and shows its measured cost; it does
not test the full agent's ability to acquire better evidence. Further prompt
tuning on these four cases is not the next evaluation step.

Artifacts:

- Inputs/provenance: `/home/cpaxton/runs/emet/frozen-eqa-context-fixture-20261103/`.
- Complete requests, raw replies, manifests, scoring and `summary.json`:
  `/home/cpaxton/runs/emet/frozen-eqa-answer-paths-20261103/`.
- Four question + ordered-image + estimated-path panels and one cost figure,
  each as PNG/PDF: `figures/` under the result directory. These human panels
  were never model inputs.

Reproduce analysis with `scripts/analyze_frozen_eqa_comparison.py --run ...
--cases ... --runs /home/cpaxton/runs/emet`. Eight new CPU contract tests pass;
combined sensor/diagnostic coverage is 33 tests.

## Next comparison

Collect broader, task-independent trajectories under the same sensor/physical
contract, and retain failed collectors as well as successful ones. Freeze the
eligible histories before comparing current/uniform/selected evidence or
answering paths. Include the question-only control and more representative
questions when evaluating visual value; do not expand this fixture's repeated
snapshots into a paper accuracy claim.

The full active comparison still needs our existing agent bound to the same
estimated-pose basis, body footprint, geometry admission and motion budget as
GraphEQA. Use the existing agent; this study adds no replacement policy. Keep
GraphEQA's connection complete and its mistakes in the denominator. Stronger
models remain optional ablations rather than integration prerequisites.
