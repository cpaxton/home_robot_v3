# Dialogue repair checkpoint and S0 runtime audit

## Correctness checkpoint

PR #217 head a0a168a9 repairs the public executor facade's private floor-recovery
keyword forwarding. The real-facade regression reproduces the original TypeError
before repair and verifies recovery plus rejection of model-supplied private args
afterward. 79 targeted tests pass; selected lint clean.

Serial rerun `20261018_171125_5f9d90` on that exact head completed S0 distinct
receptacle finding: object and receptacle both grounded, episode subprocess exits
0, validator complete=true/success=true, no agentic_find_error. This is find-only
with perfect sensor depth, physical navigation and local Qwen3-VL-8B int4, not
manipulation acceptance. Source and configuration are frozen in the run manifest.
EQA4 on preceding head was 2/4; Molmo grounds bowl but not microwave. No universal
cross-task or paper-readiness claim.

## Runtime accounting (measured)

Evidence root: `/tmp/emet-dialogue-review-s0-retry-20261018`.

| Component | Seconds | Interpretation |
| --- | ---: | --- |
| Whole episode | 989.13 | 16.5 minutes |
| Agent initialization | 13.58 | Metric-defined init, not whole supervisor startup |
| Initial four-view mapping | 112.35 | Includes scan motion and mapping updates |
| Query phases | 855.71 | Object + receptacle execution |
| Navigation calls within queries | 798.12 | Inclusive planning/execution/map refresh, not pure base motion |
| Floor recovery calls | 8.40 | Two observations, separate from navigation-call totals |
| Logged VLM generation | about 28 | Worker log; compute only, not all HTTP/preprocessing overhead |

Navigation-attempt wall times: object 132.56 + 66.68 s, receptacle 380.51 +
0.12 + 218.25 s. Two floor recoveries cost 4.29 + 4.11 s. Object executor
summary 221.87 s, receptacle 626.98 s. Three investigate decisions and no
frontier exploration: a long chain of model decisions is not the primary cause.

## What the trace establishes

The 380.51 s receptacle call performs four route chunks, refreshing the map and
replanning between them. Planned raw lengths 22/15/14/7 become two executed
waypoints per chunk. The eight-grid-waypoint chunk limit is applied before path
cleaning, so two executed waypoints do not mean only two raw planned samples.
Other calls also include chunking/replans and some unknown-floor rejections.
Success receipts are reported, not timeout aborts in this successful rerun.

The existing timer surrounds the entire navigate_to_target_pose call, including
recursive chunks, endpoint sampling, A*, safety checks, blocking trajectory
execution and intermediate update(). It cannot distinguish those costs. Slow
simulation is possible: the client scales motion waits using sim_to_real_ratio,
but that value is not retained in this run's evidence. Do not infer that the
timeout value is always spent, or that a successful call spent its time in A*.

## Next diagnostic

Profile the same S0 source/settings serially with external timing wrappers around
endpoint sampling, planner.plan/cleaning, map update, execute_trajectory and each
move_base_to/head command. Save command receipts and sim/wall ratio at movement
boundaries. Preserve policies, safety checks, scene, model and budgets; no faster
speed, looser tolerance, larger chunk or history eviction before attribution.
No production runtime change made by this checkpoint.
