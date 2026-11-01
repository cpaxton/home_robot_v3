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
