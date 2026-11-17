# Fixed final result

The fixed 131,072-decision continuation completes (212,992 lineage), producing
155,423 physical ticks. No promotion or automatic budget extension.

| Cohort | Source | Curriculum final |
| --- | ---: | ---: |
| Regular | 19/24 | 22/24 |
| Existing coupled | 7/10 | 6/10 |
| Unloaded TAMP | 6/6 | 6/6 |
| Goal changes | 9/16 | 6/16 |

All 56 candidate stops confirm. All ten failed goal cases receive the second
command and satisfy the coarse base region, but miss the 3 mm hand position
bound; four also miss 0.02 rad orientation. All are tracking timeouts. Three
previous successes become 6.27/4.69/7.28 mm misses. Overall worst hand error,
jerk and contact load improve, which does not offset success regressions.

Goal-family training consists of only four completed episodes per training
identity: 31 timeouts, one wheel-lift stop and zero stochastic successes.
Terminal measurements follow stopping and cannot alone establish the active
policy's commanded velocity or explain the equilibrium. Pre-stop traces and
fine-tracking/terminal-credit diagnostics are next.

`results.json` retains every cohort/case outcome and checksums the ignored full
physical report; `results-sha256.json` binds it separately from immutable launch
records. See docs/experiments/goal_change_curriculum_20261117.md for the report.
