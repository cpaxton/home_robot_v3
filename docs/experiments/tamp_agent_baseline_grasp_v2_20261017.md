# Actual run-agent baseline after grasp ordering, October 17

**Full tasks remain 2/6; satisfied goals increase from 4/12 to 5/12.**
The familiar cases remain successful. The left cylinder completes one actual
agent transfer, but neither left two-object episode completes. There is no
recorded payload-retention failure in this batch; two executions stop before
contact, so that is not evidence that retention is solved.

This repeats the [earlier six-episode baseline](tamp_agent_baseline_20261017.md)
after the [grasp-reference change](tamp_grasp_reference_20261017.md).
[Machine-readable results](tamp_agent_baseline_grasp_v2_20261017.json) retain all
six episodes, including failed and partial outcomes.

## Protocol and source

- Actual `run agent` CLI; native physical Stretch execution, stationary base,
  privileged simulator grounding, no cameras, no exploration.
- Same three archived fixture variants, two repeats each. All XML hashes and
  goal regions checked equal to the previous manifest. These are repeated
  diagnostics, not newly held-out tasks or a general benchmark.
- Caliban endpoint `http://192.168.1.55:8000/v1`, advertised model
  `Qwen/Qwen3-VL-8B-Instruct`, text/tool input. No local GPU used. Server-side
  weights and sampling defaults are not independently pinned.
- Same prompts, 12-round bound, independently scored whole-footprint goals,
  measured support/stillness, invalid-plan first turn, and actuation audit.
- Two CPU simulations run concurrently through managed job
  `20261017_090754_403b71`.
- Base source `60c41d4efd9d5a46f619da13928d7ac7d0446439` plus archived
  `implementation.patch`, SHA256
  `511bc8c43550f7647ac12827d825b8cb90c2872b201169772e182ce6c8dc71db`.
  Base commits since the original tested source `53e861e8` changed no production
  code, scripts, or configuration. The only production patch prefers existing
  45-degree grasp candidates before existing 15-degree candidates.
- No model physics, acceptance thresholds, or execution behavior changed during
  the batch. The runner checks the adapter source hash before each episode.

## Results

| Episode | Full task | Goals | Outcome |
| --- | --- | --- | --- |
| familiar r1 | Pass | 2/2 | Both transfers complete |
| familiar r2 | Pass | 2/2 | Both transfers complete |
| left r1 | Fail | 0/2 | Pregrasp revalidation: `invalid_start` |
| left r2 | Fail | 1/2 | Cylinder placed; cube planning fails |
| right r1 | Fail | 0/2 | Cylinder planning fails |
| right r2 | Fail | 0/2 | Pregrasp revalidation: `linear_collision_at_1` |

- Full tasks: **2/6 (33.3%)**, previously 2/6.
- Final satisfied goals: **5/12 (41.7%)**, previously 4/12.
- Infrastructure failures: **0**; all six produce scoring summaries.
- Audit: **1,139 commands, zero forbidden commands**.
- Injected invalid-plan rejection: **6/6**.
- Valid tool envelopes and failure-stop compliance: **6/6** episodes;
  31 TAMP calls, 10 failure boundaries, zero unexpected post-failure dispatches.
- Successful planned regions and task tool ordering checked against raw logs.

The sample is too small to establish a success-rate improvement. Preserved
familiar success does not rule out regressions elsewhere. The right/left cases
still lack certified complete native reference sequences.

## Failure detail and next work

The public envelope reports both pregrasp failures as `operation_failed`; their
more specific reasons come from adapter warnings in `agent.log`. `invalid_start`
does not by itself distinguish a joint-limit violation from collision at the
measured start. `linear_collision_at_1` identifies a collision on the first
revalidated edge sample; these logs do not identify the offending geometry pair.
Neither is a demonstrated grasp-retention failure.

After the left cylinder transfer, cube planning uses **82/256 IK calls** with
rejections: pregrasp IK 51, grasp collision 7, pregrasp collision 4, place
collision 3, place IK 2. Right r1 uses **76/256 calls**, with pregrasp IK 56,
grasp collision 4, lift IK 1, preplace IK 5, pregrasp collision 1. In both cases
the sampled alternatives fail; the numerical call cap is not reached, and this
is not proof of infeasibility.

Next priorities are measured-start/edge collision evidence and robust pregrasp
execution, followed by approach/grasp alternatives for the remaining planning
failures. Do not relax collision checks or label a geometric plan as an RL
success demonstration before live contact replay succeeds. The grasp ordering
is a promising local improvement, not a completed TAMP robustness fix.

## Artifacts

`/home/cpaxton/runs/emet/tamp-agent-baseline-grasp-v2-20261017/` contains frozen
fixtures, manifest, source patch, per-episode server/agent logs, raw CHAT JSONL,
actuation audits, summaries, aggregate metrics, and archived runners under
`validation/`. The final aggregate is `metrics.json`. Original baseline artifacts
remain untouched.
