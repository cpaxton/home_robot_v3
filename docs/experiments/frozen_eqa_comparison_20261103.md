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
