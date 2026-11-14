# Recovery budgets and review follow-up

The last [strict coordinator diagnostic](closer_view_feedback_20261113.md)
spent its three navigation calls on two turns and a short drive. The useful
microwave range rejection arrived afterward. That motivates testing a larger
horizon; it does not establish that extra calls will produce a useful approach.

## Budget changes

- `scripts/run_shared_room_pairs.py` exposes `--max-rounds` and
  `--max-nav-steps`, defaulting to **12/8 per find phase**. It writes both limits
  into the frozen manifest and supplies identical limits to both selected arms.
  The previous **6/3** protocol remains explicitly reproducible.
- `configs/emet/shared_find_acceptance.yaml` declares 12 tool rounds and eight
  navigation calls. The acceptance shell now honors the profile instead of
  overriding it with hardcoded 6/3 arguments. General agent defaults are unchanged.
- Relative motion now checks the combined navigation/exploration counter, as
  investigate and exploration already do.
- `budget_status` in final summaries distinguishes navigation exhaustion from
  `budget_hit`, which historically records budget-forced finalization. A model
  may voluntarily submit with exhausted navigation and `budget_hit=false`.

These are still call-count limits. A turn and an entire multi-waypoint
investigation each consume a call; this is not normalized physical work.
Keep measured motion receipts, path traces and runtime alongside those counts.
Simulation-time reporting remains a separate pending measurement task.

## Review disposition

**A — explicit answers.** Fixed the contract rather than silently adjudicating
an explicit commit. In model-final mode, a nonempty `submit_answer(answer=...)`
is authoritative. Omitting the answer, or reaching budget finalization, invokes
the final evidence-reading model. Legacy resolution remains unchanged. Traces
label `submission_mode=explicit` versus `model_query`; explicit commits also
replace stale previous-query exports with a labeled coordinator-submission
record. The original model response remains in the dialogue/requests. Explicit
answer confidence uses the existing verification flag, not a model probability.
Semantic submission creates no object geometry or manipulation permission.

**B — history bounds.** The OpenaiClient HTTP CHAT path already trims before
requests and after ordinary replies, with a default 50-message bound and a
preserved system prefix. Real multi-user CHAT/HTTP tests now exercise trimming,
retained images and a new unrelated user request. They verify context delivery,
not whether a live model answers unrelated questions well. Fixed a real edge
case: requesting a preserved prefix equal to the entire limit caused `[-0:]`
to retain the whole list. At least one slot now remains for the current request.
This is a message-count bound, not token-aware retention or a guarantee for
every local model backend. EQA episode dialogue has its own bounded task horizon.

**C — camera freshness.** All attached camera frames are described as the
latest available frame, potentially cached, with freshness unverified by this
loop. Available frame sequence and acquisition timing/clock metadata are
included. No timestamp is invented and no wall-clock age is computed from an
unrelated sensor/simulation clock.

**D — capability discovery.** The claimed permanent cache of transient failure
is not reproduced. A failed lookup leaves `_native_tools_supported=None`, so
the next property access retries. A regression exercises a 503 followed by an
explicit unsupported declaration and confirms exactly two discovery requests.
Unknown APIs retain the existing compatibility default. No new retry/cache
policy or silent protocol/model switch was introduced.

**E — receipt size.** Audited 11 unique actual receipts from the strict Molmo
run. Maximum rendered payload is 3,036 bytes (floor observation); rejected
rotation is 2,473 bytes and a view receipt 1,828 bytes. Raw feature vectors are
excluded. These are measured sizes for this cohort, not a universal size bound.
No generic truncation was added that might discard geometry or recovery detail.

GraphEQA fork findings on NumPy receipts/history remain resolved at `f41ac89`;
these new changes belong to the home_robot harness.

Validation: **373 focused CPU tests pass**, including answer resolution, the
real CHAT/HTTP boundary, model-final dispatch/stopping, relative motion, profile
resolution and matched command/manifest propagation. Shell syntax and repository
pre-commit checks pass. The JUnit report is
`/tmp/budget-review-final-tests-20261114.xml`; the receipt-size audit is
`/tmp/budget-review-receipt-sizes-20261114.json`. No live larger-budget result is
included in those contract checks.

## Frozen development experiment

Run serially, using the existing managed GPU lease and safe CPU pool, cached
Qwen3-VL-8B int4, physical navigation, four initial scan views, unchanged provider,
2.5 m geometry limit and 1500 s invocation timeout:

| Control | Source | Rounds / navigation calls per phase |
| --- | --- | --- |
| Low-budget candidate | Same new reviewed candidate | 6 / 3 |
| Larger-budget baseline | Frozen main + common clock repair, `35fbabf7` | 12 / 8 |
| Larger-budget candidate | Same new reviewed candidate | 12 / 8 |

Use `molmo_rby1_ithor_s2_idx0`, seed 0. First run the candidate-only low-budget
control, then the larger-budget baseline/candidate pair. This requires a new
low-budget control because explicit-answer handling also changed after the old
diagnostic. Compare that candidate with itself to study budget effects; the
baseline pair compares complete systems. These correlated development runs do
not establish benchmark superiority. Retain failures and inspect actual motion,
range-feedback timing and grounded object IDs, not just the final text.

The older baseline retains its historical relative-motion counter behavior;
check actual combined navigation/exploration use for overshoot before describing
its effective allowance as matched. A separate common counter repair is needed
if such overshoot affects interpretation. Broader acceptance must also compare
against then-current main; this frozen control is for continuity with prior runs.

## Resource status

The larger-budget GPU experiment has **not started**. Available disk fell below
the frozen 20 GiB reserve. Disposable pip cache is only 54 MiB; installed uv
environments and model caches were left alone. Automatic approval review rejected
hardlinking archived simulator models because inode/metadata sharing changes
evidence artifacts outside package-cache cleanup scope.

A read-only, full-SHA-256 plan is available at
`/tmp/budget-archive-dedup-plan-20261114.json`: ten duplicate copies, 5.24 GiB
potential recovery, zero files modified. Applying it requires user approval;
no elapsed wait is approval. The budget driver and review repairs can be reviewed
independently of that resource block. No larger-budget result is claimed.
