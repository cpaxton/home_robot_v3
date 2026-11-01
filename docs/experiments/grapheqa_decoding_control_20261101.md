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
