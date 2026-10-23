# Actual run-agent baseline after ordered pregrasp, October 17

**Full tasks remain 2/6; satisfied goals rise from 5/12 to 6/12. All six
attempted transfers complete.** No pregrasp, retention, or other execution
failure occurs in this batch. Both familiar episodes pass; both left episodes
place the cylinder then reject the cube plan. The right episodes do not execute:
one has a malformed agent argument, and one cannot find a plan.

This repeats the [previous six-episode run](tamp_agent_baseline_grasp_v2_20261017.md)
with the [ordered-pregrasp change](tamp_pregrasp_20261017.md). The
[machine-readable results](tamp_agent_baseline_pregrasp_v3_20261017.json) include
all six episodes without exclusions.

## Comparable protocol

Actual `run agent` CLI, native physical Stretch, privileged stationary simulator
grounding, no cameras or exploration. Caliban serves
`Qwen/Qwen3-VL-8B-Instruct` at `http://192.168.1.55:8000/v1` in text/tool mode;
no local GPU is used. Two CPU simulations run concurrently. Server checkpoint
identity and sampling defaults are not independently pinned.

The three fixed fixture variants, two repeats each, are unchanged: all XML hashes
and goal regions match the prior manifest. These are repeated diagnostics,
not six unique or newly held-out tasks. Prompts, 12-round bounds, invalid-plan
first turn, whole-footprint goal scoring, measured support/stillness, and the
physical actuation audit are unchanged. The runner checks all three changed
production source hashes before every episode and again after the batch.

Source: `60c41d4efd9d5a46f619da13928d7ac7d0446439` plus archived
`implementation.patch`, SHA256
`8e97202e768aaf2b7640bc97c8f5bed996acd7998a7a4bb84ddf7d03ccf27aea`.
The composite patch includes the earlier grasp ordering, ordered candidate and
measured-state pregrasp paths, and richer bounds/collision failure details.
No model physics, collision tolerance, retention limit, or controller gain changes.
Managed job: `20261017_165523_742dad` (done).

## Results

| Episode | Full task | Goals | Outcome |
| --- | --- | --- | --- |
| familiar r1 | Pass | 2/2 | Both transfers complete |
| familiar r2 | Pass | 2/2 | Both transfers complete |
| left r1 | Fail | 1/2 | Cylinder placed; cube planning fails |
| left r2 | Fail | 1/2 | Cylinder placed; cube planning fails |
| right r1 | Fail | 0/2 | Agent sends invalid placement-region shape |
| right r2 | Fail | 0/2 | Cylinder planning fails |

| Metric | Previous grasp-order run | Ordered-pregrasp run |
| --- | --- | --- |
| Full two-object tasks | 2/6 | 2/6 |
| Final satisfied goals | 5/12 | 6/12 |
| Successful / attempted transfer executions | 5/7 | 6/6 |
| Infrastructure failures | 0 | 0 |
| Forbidden actuation commands | 0 | 0 |

The new batch records **1,536 audited commands**, valid result envelopes and
failure-stop compliance in **6/6 episodes**, and **6/6 injected invalid-plan
rejections**. There are 32 TAMP calls, 10 failure boundaries, and zero unexpected
post-failure tool dispatches. Successful plans preserve their requested regions.
The malformed request below is not a valid region and is not hidden by that check.

The observed execution improvement is encouraging, not a statistical robustness
claim. Only six transfers were attempted, and no right-side transfer was exercised.
The source of the earlier left `invalid_start` failure is still not established.

## Remaining failures

Both left episodes finish the cylinder transfer and fail to plan the blue cube.
Each uses **69/256 IK calls**: 51 pregrasp IK rejections, six pregrasp collisions,
and six grasp collisions. The sampled search fails before reaching the numerical
call cap. Rejection counts do not prove the cube task infeasible.

Right r1 is an **agent argument-shape error**, not a motion-planning failure.
The model sends:

```json
{"placement_region": [-0.08, -0.86, 0.08, -0.74]}
```

The required shape, also present in the user instruction, is:

```json
{"placement_region": [[-0.08, -0.86], [0.08, -0.74]]}
```

The tool correctly returns `invalid_placement_region`, performs zero IK calls,
and does not execute. The current continuation policy stops on that failed tool.
This episode stays in the denominator; there is no silent argument reshaping or
retry added during scoring. Future interface work should improve schema/prompt
shape guidance and separately assess any bounded validation-repair policy.

Right r2 uses **74/256 IK calls** with 57 pregrasp IK rejections, two grasp IK
rejections, two grasp collisions, five preplace IK rejections, and one pregrasp
collision. It never starts manipulation.

Next priorities are reference certification and approach/grasp search for the
remaining cube/right planning failures, plus agent argument-shape reliability.
Left/right complete two-object native reference sequences remain uncertified.
No observed grounding, navigation, exploration, hardware sensing, or held-object
recovery capability is inferred from these stationary privileged results.

## Artifacts

`/home/cpaxton/runs/emet/tamp-agent-baseline-pregrasp-v3-20261017/` retains the
manifest, frozen fixtures, source patch, per-episode agent/server logs, raw CHAT
JSONL, actuation audits, independent summaries, `metrics.json`, and a paired
`comparison.json`. Exact runners and Caliban model responses are archived under
`validation/`. Earlier experiment directories are untouched.
