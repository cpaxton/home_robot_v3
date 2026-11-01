# GraphEQA interaction audit: decoding, history and frontier registry

Follow-up to the [review and planner-contract probes](grapheqa_review_and_contract_probe_20261031.md).
An actual model-selected arrival now reaches a fresh view, but schema-valid
post-motion live acceptance remains open. Passing API tests alone is not that gate.

## Audit finding

The cached `Qwen/Qwen3-VL-8B-Instruct` snapshot
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/generation_config.json` sets
`do_sample: true`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`.
The client passes output budget, beam count and stopping criteria to generation,
but otherwise inherits these decoding settings. The earlier schema-description
and ordering probes did not reset the sampling RNG before paired calls.
Consequently their small differences cannot establish a stable prompt effect.
The API's rejection of unsupported `seed` settings remains correct; no HTTP seed
support or deterministic inference guarantee is being claimed.

## Predeclared diagnostic

Reuse every reference-graph request from the fixed q2/q6/q12 four-image fixture
at `grapheqa-four-images-context-pair-20261030/paired/`. Replay its original
response ordering and exact messages, independently attached images, graph,
labels, model and 2,048-token budget. No gold/scorer input is loaded.

Compare checkpoint-default sampling against an explicit greedy diagnostic for
seeds **0, 1, 2**, resetting torch's RNG before each call. Run all 18 calls
serially; alternate condition order across trials. Save full requests/replies,
effective generation config and seed, including invalid/truncated rows. Greedy
changes only `do_sample` inside a temporary restored config. Repeating it across
seeds checks observed stability; fixed RNG state does not promise bitwise
reproducibility across GPU kernels/platforms.

Measure valid replies, confident stops, uncertain empty plans, and usable
model-selected object/frontier actions. A step paired with confident stopping
is not a usable action. Do not select a winning seed for the live gate or infer
answer-quality gains from action counts. If the greedy condition consistently
produces usable actions without format failures, qualify that explicit profile
on a fresh two-decision q6 interaction. Otherwise retain the model failure and
keep the active gate open; do not add forced navigation or replace its answer.

This is a local inference diagnostic, not a change to the runtime planner or
default model decoding. It uses the real model-server inference/message adapter
in process. Existing HTTP and sensor/executor plumbing have separate checks.

Resources: one GPU lock, CPU 12, one BLAS/OpenMP thread, offline cached models,
20 GiB disk reserve. Package-index/download caches were removed to restore
20.23 GiB before admission; model caches, environments and run evidence remain.

Validation before launch: six diagnostic tests pass, including exact request
preservation and restoration of checkpoint settings after a generation failure.

## Decoding result

Job `20261101_125344_fab70f`, source `e211fe67`, completed all 18 calls with
zero malformed/truncated responses. Output is
`/home/cpaxton/runs/emet/grapheqa-decoding-control-20261101/`.

| Case | Default sampling confident stops / 3 | Greedy confident stops / 3 | Usable actions, sampling / greedy |
| --- | --- | --- | --- |
| q2 | 0 | 0 | 0 / 0 |
| q6 | 2 | 0 | 0 / 0 |
| q12 | 3 | 3 | 0 / 0 |

Each case's three greedy replies were byte-identical across the declared seeds.
Sampling changes q6's confidence, but disabling it does not produce an action.
All answer fields are retained; no gold labels were loaded and no answer-quality
effect is claimed. Total measured request time was 196.47 seconds, excluding
loading/admission. Greedy decoding is **not promoted** to the live profile.
The earlier unseeded schema comparisons remain exploratory, not stable causal
evidence of an interface improvement.

## Independent adapter bug: uncertain empty plans ended episodes

While auditing the legacy loop, `scripts/run_vlm_planner_hmeqa_habitat.py`
continues after an uncertain empty plan within its existing 20-step budget.
The new external `run_active_episode` instead ended immediately with
`no_action_not_confident`. This was a harness decision, not a confident model
stop. It removed the reference controller's opportunity to retry.

The repair keeps the model's confident stop, retries uncertain empty plans only
within the declared decision budget, and requires a fresh capture each time.
An empty plan never invokes the executor. Answer/confidence summaries for these
decisions now remain in history and advance its decision index. Budget exhaustion
still reports `no_action_not_confident`; no movement or answer is invented.
All **44 fork tests** pass, including the empty-plan → chosen action → final answer
sequence and the real planner/provider history/budget integration.

Predeclare a serial fresh q2/q6 diagnostic, four decisions and four independent
uniform sensor images per assessment. q2 exercises the repeatedly uncertain
empty-plan path; q6 controls the model's confident-stop behavior. Keep checkpoint
sampling defaults, original response-field order, safety and scorer separation.
This is an integration check of a reproduced bug repair, not an accuracy or
matched-runtime comparison with the earlier two-decision checks. Require actual
executed motion and a fresh receipt-bearing assessment to close the movement gate.
The active capture loop also checks the disk reserve before every learned sensor
packet, stopping with a retained infrastructure failure if other work consumes it.
Do not discard evidence or silently shorten the run to avoid a failure row.

## First repaired live check: natural action, resource interruption

Job `20261101_130129_138592` used emet `f31ad6a5` and fork `6a18c99`.
q2 now receives all four allowed assessments instead of stopping after the first
empty plan. Every reply is uncertain and empty, and its final A is wrong. It
executes no motion and reports `no_action_not_confident` at budget exhaustion.

q6 naturally selects `frontier_0` while uncertain (confidence 0.5). The executor
begins following that observed-space route. Fifty sensor frames are saved; the
last estimated camera XY is approximately (0.050, 0.350) metres, with tracking
valid and no reported collision. The shared 20 GiB disk guard then stops
recording before route completion. This is a retained infrastructure failure,
not an accepted arrival or a wrong-answer result. No final follow-up assessment
was reached. The original rows/frames/model response remain untouched at
`/home/cpaxton/runs/emet/grapheqa-empty-plan-retry-live-20261101/`.

The interruption also exposed an audit gap: a valid model decision was appended
only after execution returned. An exception could leave physical sensor evidence
but no decision row. The adapter now records that decision in a finally block,
marks execution attempted, and retains the exception without inventing an
unavailable measured receipt. A focused test reproduces the failure.

Storage recovery evicts one old, unshared Torch 2.7.1 **package cache**, retaining
the separate installed root/Habitat environments and every model/run artifact.
Free space rises to 21.52 GiB. Repeat the same q2/q6 order, four decisions and
four images, in a new output directory after freezing the trace repair. Keep
checkpoint sampling; do not choose a winning seed or replay a saved action.
The model must choose again, and the original interrupted run remains a failure.

## Resource-recovered repeat: arrival succeeds, follow-up format fails

Job `20261101_130813_d3dec4`, emet `f6d7b15d`, fork `a2102b6`, repeats the
same two cases. q2 produces two uncertain empty plans, then an invalid answer
value `B Black`; the schema allows only the choice value `Black`. It remains
a retained contract failure, without extracting the letter or repairing text.

q6 naturally selects `frontier_0` while uncertain (0.7). The shared executor
reaches local XY (0.29977, 1.30121) versus goal (0.3, 1.3), position residual
1.23 mm, yaw residual 0.000075 rad, zero replans. These are odometry-relative
residuals, not ground-truth localization accuracy. No collision is reported.
A fresh capture and model call follows: the current-image SHA changes from
`33134b611587fcd5…` to `26051855e69d5d4f…`, and the actual successful motion
receipt reaches that request. The follow-up fails validation (`C In the bathroom`
instead of `In the bathroom`, plus omitted required descriptions). Both planned
episodes therefore remain failures; reaching a view is not answering correctly.

The trace identifies two context problems worth repairing independently of model
quality. History combines a choice ID with its value in an invalid schema-shaped
string. It now records explicit `choice_id` and exact `answer` fields. Also,
39 low-level controller events inflate the follow-up motion receipt to 12,593
characters; the top-level outcome/pose receipt is only 299 characters. Keep all
events in `decisions.jsonl`, while the planner sees the complete high-level
receipt without its internal control polls. No dialogue decision is removed.
Tests verify both the prompt and full artifact so this cannot silently discard
physical evidence. The observed prompt grew from 12,286 to 26,565 tokens; graph
growth also contributes, so no total-token saving is claimed before measurement.

Next repeat the same q2/q6 order and four-decision/four-image protocol after these
context-only fixes. Keep original response ordering and checkpoint decoding,
strict parsing, every failed row, safety and final-model authority. The remaining
gate is a schema-valid post-motion assessment; do not accept a malformed final
answer just because navigation succeeded.

## Context-fixed fresh repeat and frozen follow-up diagnostic

Job `20261101_131529_46f653`, emet `be4aa4a6`, fork `2798dc4`, completes
q2/q6 with five valid model responses. q2 exhausts four uncertain empty plans
and returns B (wrong); q6 stops confidently with C (wrong) before executing its
proposed frontier. No motion is executed in this repeat. Its q6 post-motion path
is therefore unexercised, and it cannot establish that the context change fixes
the earlier malformed follow-up. Retain this zero-correct/two-planned diagnostic;
different stochastic trajectories prevent a causal live comparison.

Rather than repeat navigation until a favorable sample, freeze the two malformed
requests from the preceding run: q2's third assessment and q6's post-arrival
assessment. Replay original vs schema-aligned history with seeds 0/1/2, resetting
the RNG per paired call. Twelve calls run serially. Only the known HISTORY text
changes: choice ID/value fields become explicit JSON, and low-level motion
events leave the prompt while their high-level receipt remains. Current graph,
question, legal response schema, images/order/labels, model and budget remain
exactly unchanged. No response repair or gold answer input is allowed.

Input fixture `/tmp/grapheqa-post-history-fixture-20261101/` copies both failed
traces with source-file hashes and capture context. The experiment uses the
advertised full JSON Schema to measure format adherence; inspect any distinction
from the fork's Pydantic validation when reporting results. This deliberately
selected failure set diagnoses a context mechanism, not representative accuracy
or a full live acceptance pass. Eight diagnostic tests verify the transform and
unchanged surrounding evidence. Keep the motion/receipt execution tests separate.

## Frozen history result

Job `20261101_132158_ff2540`, source `faa82cfa`, completes all 12 calls. Five
original-condition replies fail schema validation, so the job deliberately exits
nonzero; this is a retained negative control, not an infrastructure crash.

| Selected failure request | Original valid / 3 seeds | Corrected history valid / 3 seeds |
| --- | --- | --- |
| q2, third assessment | 1 | 3 |
| q6, post-arrival assessment | 0 | 3 |

Counts agree under the advertised JSON Schema and the actual native Pydantic
response classes in both the root and Habitat environments. Generated native
schemas equal the saved schemas. Reconstructing history with the current fork's
real builder exactly matches the replay transform for both cases. No model reply
was repaired. q6 prompt length drops from 26,565 to 19,249 tokens (7,316 fewer)
with the same current graph, images, labels and generation budget.

This supports the format repair on these selected failures. It does not establish
representative accuracy, reliable action selection, or a fresh end-to-end pass.
Full outputs and both environment validations are at
`/home/cpaxton/runs/emet/grapheqa-frozen-history-contract-20261101/`.

## Another context inconsistency: non-callable frontier nodes

The raw planning projection contains native 3-D frontier nodes even though their
edges are explicitly excluded and the registry is replaced by observed-space
navigation candidates. q2's captured context has 56 frontier nodes but eight
actions; q6's post-arrival context has 67 but eight actions. One failed reply
selects native `frontier_1800`, which is not a callable ID. This is misleading
input, not evidence that unavailable geometry should be admitted or guessed.

Fork `5f3fc51` omits native 3-D frontier nodes from the planning projection and
replaces/clears supplied frontier nodes and their edges on refresh. The native
Hydra reconstruction, objects and room data remain intact. A real wrapper test
covers native vs callable IDs, shrinking and empty registries. All **46 fork
tests** pass. A CPU-only isolated replay of 15 cached learned q6 packets confirms
exact graph/registry equality before and after eight callable frontiers become
available, with identical object IDs to the original contexts. No model,
renderer, simulator pose or annotation assets enter that replay. Output:
`/tmp/grapheqa-frontier-registry-5hf0kp_e/`.

Run one final fresh q2/q6 diagnostic at the corrected projection/history commits,
same case order, four decisions, four uniform images, checkpoint decoding and
strict parsing. Retain all results regardless of which actions sampling chooses.
This checks the corrected input contract; it does not turn prior failures into
successes or establish an accuracy comparison with main.
