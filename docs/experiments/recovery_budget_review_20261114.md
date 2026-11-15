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

## Completed development result

All three episodes completed on November 14; the trace and visual audit was
performed November 15. Candidate source was `defb83b0`, including the explicit
answer and history fixes above. The baseline remained `35fbabf7`. Both manifests
record the driver, sources, weights and declared limits. This is **one stochastic
seed-0 development episode with two sequential find phases**, not three independent
scenes or a manipulation evaluation.

| Arm | Rounds / nav calls per phase | Bowl | Microwave | Query wall s | Episode wall s |
| --- | --- | --- | --- | --- | --- |
| Candidate control | 6 / 3 | Pass | Fail | 170.01 | 257.52 |
| Frozen baseline | 12 / 8 | Pass | Pass | 332.36 | 381.84 |
| Candidate | 12 / 8 | Pass | Pass | 165.33 | 218.42 |

The low-budget task failure remains in the denominator. Its driver returned 1
with `all_complete=true, all_success=false`; it was not an infrastructure crash.
The larger-budget pair returned 0 with both phases passing in both arms. The
cohort supervisor reports completion separately from these task gates.

### What the traces establish

- Candidate executions match the coordinator's selected actions: **10/10** in
  the low control and **13/13** in the larger-budget run, with no action rewrites
  or legacy redirects. Both larger-budget explicit submissions are preserved
  verbatim without a second answer-adjudication call.
- The larger-budget candidate uses four navigation calls for the bowl and two
  for the microwave. Baseline combined navigation/exploration counts are two
  and six, respectively: no observed overshoot of the shared eight-call limit.
  Baseline does use **two legacy `nav_loop_redirect` events** during microwave
  search. This compares complete systems, not identical routing policies.
- Low-budget microwave search gets visual presence but rejects geometry at
  **3.476 m median proposal depth**, with **0/3,192** measured proposal pixels
  inside the unchanged 0.25–2.5 m interval. The rejection reaches the model, but
  all three navigation calls have been spent. Its final semantic location is
  correct; it still has no admitted destination geometry and correctly fails
  the find metric.
- The larger-budget microwave mask covers the actual appliance. Its selected
  proposal has **6,599/6,599** supported depth pixels at **1.646 m median depth**;
  the admitted final mask has 6,360 points. Both predictions have
  `grounded_object` provenance. The microwave touches the upper image edge, so
  this is a usable localization view, not an ideal complete-object framing.

![Low-budget rejected proposal and larger-budget admitted microwave box](figures/recovery_budget_20261114.png)

The figure shows actual captured images and source boxes. Left is a rejected
proposal; right is the accepted detection box. Depth statistics describe proposal
support before final mask admission. Camera trajectories differ between panels.

### Limits and next decision

The candidate's query wall time is about half the baseline's in this pair. That
is a measured outcome, **not an established speedup**. Model sampling, selected
actions and navigation differ; startup/invocation overhead is outside query time.
The candidate keeps moving after grounding the bowl at round 0 and only submits
at round 8. Its first microwave action also references a frontier as an
investigation card, receives `NOT_INVESTIGATE_CARD`, then chooses exploration and
a turn. Errors remain visible; the harness does not substitute a different action.

The bowl detour changes the starting pose for microwave search. The successful
larger-budget microwave phase never encounters the earlier range rejection.
Thus this demonstrates a strict coordinator path to admitted geometry with a
longer horizon; **it does not demonstrate that the model used the range/bearing
receipt to execute deliberate closer-view recovery**. Nor does one contrast
prove a causal budget effect or superiority over main.

Keep the 12/8 opt-in acceptance profile and leave geometry/motion limits unchanged.
Next run matched repeats plus small EQA and another room find check, preserving
all failures and auditing model context. A direct closer-view diagnostic should
start from a common captured out-of-range situation; keep it labeled a diagnostic
rather than replacing the full sequential episode. Broader acceptance must use
then-current main, and pick/place remains unqualified by these find-only results.

### Evidence locations

- Managed job: `20261114_192901_06b2e3`.
- Cohort protocol, copied runner, cleanup audit and test reports:
  `~/runs/emet/recovery-budget-cohort-20261114-r1/`.
- Low control: `~/runs/emet/recovery-budget-low-20261114-r1/`.
- Pair: `~/runs/emet/recovery-budget-high-20261114-r1/`.
- Compact action/budget/grounding audit, including trace SHA-256 hashes:
  `recovery-budget-audit-20261115.json` inside the cohort directory, SHA-256
  `397cfd1ab42b25f3ae2d6e3b5daf265d5ab5416af1002a96f915dad5c0428288`.
  The reproducible read-only `audit_recovery_budget_20261115.py` is beside it.

## Resource status

The user approved the exact ten-file, full-SHA-256 deduplication plan. It was
applied, reclaiming **5.24 GiB**, retaining paths and byte contents via shared
inodes. Archived files are consequently no longer independent writable copies.
The applied audit is copied into the cohort. Automatic approval review rejected
expansion to additional archive paths; those additional files were not changed.
Installed environments and model caches were left alone.

Before launch, the bounded three-episode protocol explicitly revised the
self-imposed disk reserve from 20 to **16 GiB**, based on prior pilots writing
20–66 MiB each. The runner checked this floor every five seconds and serialized
all cases under one managed GPU lease, after the existing Mars jobs finished.
Model, geometry, motion and task limits were unaffected. The original 20 GiB plan
is retained alongside the revised protocol. Cohort free space went from 18.33
to 18.07 GiB; no resource abort occurred.

At the November 15 follow-up only about **13 GiB** remained free, below the declared
floor. No additional simulation was launched. The completed results above are
valid; new live repeats remain pending resource headroom.
