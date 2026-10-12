# Shared-agent stack readiness — 2026-10-12

Clear slices independently. Non-draft means evidence is ready for review, not
permission to merge. Keep defaults unchanged; no full sweep or hardware gate.

## Ownership and exact heads

| PR | Ownership | Current gate |
| --- | --- | --- |
| #181 `dc106273` | Opt-in grounding providers, shared assessment dispatch | Exact-head default-path EQA attribution and saved-view provider audit pending. |
| #182 `1319ea6d` | Measured navigation, clearance, wheel contacts | Exact-head route/EQA and short Molmo rby1 route pending. Integrated route result cannot clear this head. |
| #184 `48b56aeb` | Bounded head/floor recovery, measured head-arrival fix, executor delegation | 284 focused ownership-repair tests passed at the acceptance descendant; exact-head simulated recovery and paired EQA pending. |
| #185 | Frozen drivers, evidence isolation, queue ordering, environment discovery | 29 focused driver/metric/discovery tests passed. Live shared-driver pair and remaining failure-path checks pending. |
| #198 `59f70026` | Separate opt-in coverage scan, probe, exact-frame ingestion | 281 extracted coverage/recovery tests; 31 coverage/config-dispatch tests after repair. Corrected live ablation and multi-heading visibility gate pending. |

The measured-head fix moved from integration to #184; evidence isolation and
prerequisite-before-GPU-lock ordering moved to #185. #185 no longer owns the
executor behavior cleanup. #198 is based on #184 and does not block #185.

## Frozen pilot findings

- Baseline `7f65dbef` versus integration `cd92cb1b`, Qwen3-VL-8B-Instruct int4,
  seed 0: EQA q12/q14/q56/q6 scored **4/4 versus 1/4**. Losses on q12/q14/q56
  reproduced. No claim of EQA acceptance or statistical equivalence.
- Both arms failed object and destination find in S0 and RoboCasa. This is not a
  success gate. Molmo returned a wrapper-discovery infrastructure error in both
  arms and is excluded from task accuracy.
- Integration post-wheel route: 50/50 waypoints, maximum measured error
  0.00834215 m / 0.01457790 rad, unchanged 0.02 m / 0.03 rad limits.
- Integration model-free scan: 16 views / 69.19 s. All frames manually inspected;
  views 12/13 clearly show both objects with valid depth. Not a GT-mask benchmark.
- The nominal learned coverage arm recorded `mapping_scan_profile: local`:
  consumers incorrectly read `mapping/scan_profile` after config flattening.
  Fixed only in #198, with a real-loader dispatch regression. Discard that run
  as coverage-effect evidence; preserve it as the bug reproducer.
- Explicit `MOLMOSPACES_PYTHON` discovery resolved a venv symlink to the system
  interpreter. #185 now preserves the venv path; installed venv imports pass.
- Old room driver accepted zero-exit error JSON as job completion. #185 now
  requires a scored episode with boolean outcomes and no infrastructure error.

Artifacts: `/home/cpaxton/runs/emet/acceptance-plan-20261012/` records the initial
protocol. Frozen outputs are `pair-eqa4-{main,candidate}-20261012`,
`pair-room-{main,candidate}-20261012`, `pair-s0-coverage-20261012`,
`postwheel-route-20261012`, and `postwheel-coverage-20261012` under the same run root.
Do not modify their source trees or overwrite results.

## Run and disposition protocol

Use `scripts/run_extracted_acceptance.sh` through CPU-safe, GPU-exclusive jobs.
`SOURCE_DIR` selects the frozen checkout; `SOURCE_SHA` must match; `OUT_DIR` must
be new. `EMET_PY` and `HABITAT_BIN` select installed environments.
`ACCEPTANCE_CONFIG` supplies a frozen matched-provider room config; its default
is the separately advertised opt-in provider configuration, not a matched-main
ablation. `DEV_EQA_IDS`, `DEV_SEED`, and `FIND_EPISODES` select the pilot subset.
Record source and driver hashes. Keep model/seed/budgets identical across pairs.

1. Attribute q12/q14/q56/q6 across main → #181 → #182 → #184. Then complete the
   existing EQA-12 set on passing prefixes. Repeat disagreeing pairs at seeds 1/2;
   a loss on two of three matched seeds blocks the owning slice.
2. For #181, audit all advertised providers on preserved Molmo/RoboCasa RGB-D
   fixtures: accepted masks/geometry, missing/ambiguous targets, invalid depth,
   malformed model responses. Shared dispatch must preserve agent conversation.
3. For #182, run the five-repeat 50-waypoint gate on its head, minimal existing
   TAMP contracts, and a short native Molmo rby1 route. No relaxed tolerances,
   unknown-free assumptions, or navigation teleports.
4. For #184, demonstrate fresh downward evidence enabling same-waypoint recovery.
   Occupied floor remains blocked; impossible gaze and stale observations fail
   explicitly. Require bounded retries and successful intended recovery, not
   merely equally poor aggregate find scores.
5. For #185, validate source/output guards and missing-assets/native-crash/timeout/
   empty/error-metrics handling, then one live EQA pair and one live room pair.
   Correctly reported task failure does not itself block evaluation tooling.
6. For #198, run corrected same-provider local/coverage S0 plus four initial
   headings and an opposite-side fixture with offline GT mask/depth scoring.
   Report image/time cost. Do not condition policy on simulator truth.

For each slice: update its PR with exact parent/head, commands, artifacts and
limitations; mark ready only when its gates pass; request focused review.
After parent merges, retarget and rerun affected gates. No automatic main merge.
