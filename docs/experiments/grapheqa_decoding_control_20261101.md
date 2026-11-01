# GraphEQA decoding control before live interaction

Follow-up to the [review and planner-contract probes](grapheqa_review_and_contract_probe_20261031.md).
The active reference adaptation still has no accepted model-selected-motion
episode. Do not substitute passing API tests for that gate.

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
