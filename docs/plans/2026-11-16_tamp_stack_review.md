# TAMP merged-stack review follow-up

Branch `fix/tamp-stack-review`, based on main `6386e9d3`. The original stack
[#226](https://github.com/cpaxton/home_robot_v3/pull/226) →
[#231](https://github.com/cpaxton/home_robot_v3/pull/231) →
[#234](https://github.com/cpaxton/home_robot_v3/pull/234) →
[#242](https://github.com/cpaxton/home_robot_v3/pull/242) →
[#243](https://github.com/cpaxton/home_robot_v3/pull/243) →
[#244](https://github.com/cpaxton/home_robot_v3/pull/244) is already merged.
This follow-up addresses the subsequent written reviews together. The five
original inline #226 threads are resolved; the newer reviews are review bodies,
not unresolved inline threads. Do not equate that UI state with fixed code.

## Disposition by review

| PR / finding | Resolution |
| --- | --- |
| #226.1 grasp baseline drops `after` | Optional verifier `observe_after` uses the existing source `collect(..., after=...)`; acquisition must be strictly newer. Cached `snapshot()` protocol stays unchanged. |
| #226.2 inconsistent `cam_intrinsic` | Pinhole metadata uses render FOV/resolution; declared physical sensor sizes retain sensor-unit conversion. Published K/render projection stays unchanged. |
| #226.3 half-pixel convention | Actual-model symmetric GL frustum tests pin zero-based pixel-centre principal point `(W-1)/2,(H-1)/2`. GL window centres occur at `i+0.5`. |
| #226.4 lost payload identity code | Preserve `payload_identity_mismatch` as a safe public code. |
| #226.5 shared VLM race | Per-instance RLock covers generation, history API operations and the entire preserve/restore transaction; concurrency tests use controlled fake requests. Latest-call metrics remain instance-level. |
| #226.6 cached clock-read failure | Cache successful identities only; transient procfs errors can recover. |
| #226.7 hidden historical free cells | Explain original age/2 s expiry and explicit static assumption. This is conditional historical evidence, not robot-material admission. Further ownership qualification remains #245. |
| #231.1 shallow mutable checker clone | Copy robot/exclusion sets and nested support-departure metadata. Success and rejection preserve the original checker. Released checkers remove payload exemptions. |
| #231.2 resting support penetration | Keep the existing strict 1 mm gate. Add 0.9/1.1 mm boundary tests and support-penetration diagnostics. No table exemption or tolerance increase. Solver contact deeper than the bound still fails. |
| #231.3 starved vertical fallback | Reserve 25% of remaining IK calls for the vertical strategy in the opt-in bounded search. The primary limit also covers downstream placement while the generator is suspended. Never renew the global cap; legacy ordering is unchanged. |
| #231.4 internal release failures | Validate attachment identity/parent, rigid free payload, support, controlled ancestor height joint, opening geometry, finite targets and numeric anchors/reserves before search. Return a typed unavailable result. |
| #231.5 missing released checker | Reject missing release state or empty opening geometry before dereferencing or vacuous arrival checks. |
| #231.6 fixed object vocabulary in scorer | Completion helper requires explicit configured object names. The two-object development driver supplies its goal dictionary; arbitrary names work and missing requirements cannot pass. |
| #231.7 surplus last adjustment search | Retain the necessary final fresh certificate, but make it current-pose-only once movement budget is exhausted. Tests require `[True,True,False]` adjustment-search permission across two moves and final verification. |
| #231.8 numeric policy literals | Name descent, axis, target-resolution, settling and polling parameters in `ReleaseLimits`, retaining their original defaults. |
| #231.9 stale raise anchor / targets | Require three finite placement targets. Raise vertically from the measured EE pose/rotation, including measured command/FK shortfall in proposals and lowered probes while preserving the first measured global raise anchor. Check actual solved and sampled FK heights so IK tolerance cannot widen that bound. |
| #234.1 alignment near ties | Document actual three-decimal, half-to-even cosine bins; no epsilon guarantee or ranking/admission change. |
| #234.2 lost closing-axis code | Preserve safe `invalid_gripper_closing_axis`. |
| #234.3 loose endpoint mismatch | Derive the retained-error endpoint from the same full/loose preset as closure. Defaults remain unchanged. |
| #234.4 repeated pinch calibration | Cache immutable-model geometry calibration; returned arrays cannot mutate the cached values. |
| #234.5 inconsistent retention predicate | Use the same exact semantic failure predicate for lift and loaded motion. |
| #234.6 unhelpful placement witness | Add finite public XY/height verification reasons; retain unchanged acceptance bounds. |
| #242.1 slow policy reconstruction | Reuse a matching cached model, construct fresh data/namespace, and retain original snapshot age/signature checks. No timestamp rewriting or longer freshness threshold. |
| #242.2 approach ordering description | Document aligned stationary pre-check, checked base alternatives, then general stationary fallback. No global best-score claim. |
| #242.3 double cancellation / masked reason | Cancel once, retain unconfirmed-stop priority, and record the primary reason in diagnostics. A confirmed cleanup preserves the primary failure. |
| #242.4 quaternion sign | Align free/ball quaternion signs on copies before existing component bounds in initial and scene-change checks. Actual movement still invalidates/rechecks. |
| #242.5 approach/grasp gap | Include arm/coupled stillness during approach settling and require fresh quiet base/arm feedback immediately before grasp. No re-command or relaxed bounds. |
| #242.6 intentional IK-count retention | Explain that task providers charge the consumed delta; restore only temporary limits/deadlines. |
| #242 placement-oracle caveat | Replace algebraic reconstruction with independently transformed off-origin box vertices and nonidentity initial/target rotations; explicitly falsify the historical world-offset formula. Historical datasets remain frozen. |
| #242 stop-witness caveat | Keep reader-confirmed, exact boot/session/sequence evidence. An unseen evicted receipt cannot prove a stop; uncertain cancellation still fails. No invented witness. |
| #243.1 filter precedence / normalization | Parenthesize and normalize robot selectors with strip/casefold. Complete-context object filters remain unsupported in task sessions. |
| #243.2 observed robot identity | Add explicit optional configured robot identity and propagate it to the region provider. Never infer Stretch from an arbitrary profile or simulator object metadata. |
| #243.3 success predicate readability | Parenthesize the typed receipt success test. |
| #243.4 alleged duplicated recovery check | Retain it: the rejection callback appends a result to `checks`. Existing `consumed_handle` regression proves removing the second check incorrectly continues. Add explanation and reject missing/nontrue `passed` values. |
| #244.1 history bound | Document at most two records, each 8192 serialized characters, plus prompt framing. |
| #244.2 mixed-turn policy | Clear after actually dispatched non-TAMP results; ignore dispatch bookkeeping and non-tool prose. Deferred proposals are not dispatched results. |
| #244.3 nullable progress | Preserve explicit null progress fields so absence of a failed step differs from an omitted field. |

## Verification and merge order

1. Complete focused and expanded CPU regressions; retain earlier failed attempts.
2. Run separate native bounded-release control and checked-approach shifted-swap
   controls through the live bridge, unchanged contact/retention/goal thresholds.
   These are privileged simulation development controls, not agent/observed scores.
3. Independent read-only review of final corrections; publish one main-based fix
   PR. Do not reopen or rewrite the merged PRs or their historical result records.
4. Bring this correction into `fix/tamp-observed-admission` and resolve overlaps
   with its existing geometry/receipt repairs. Repeat affected checks there.
   That branch remains unqualified until a full observed transfer passes.

Open #228–#230 are separate perception/infrastructure/planning work, not parents
of this stack. #241 is the ongoing control/RL branch and remains draft under its
own loaded-control and motion-quality gates; this review follow-up does not
promote a policy, modify its branch, or authorize hardware motion.

Raw artifacts: `.tamp-runs/stack-review-20261116/`. Managed expanded jobs:
`20261116_160849_ce61e4` (motion), `20261116_160849_8daeae` (task/CHAT/VLM),
`20261116_160850_f46daa` (serial native controls). Passing software checks are
not a substitute for the recorded native and future observed execution gates.

## Latest results

- Full motion/approach/camera suite: 502 passed, zero failures/errors/skips.
- Focused final review contracts: 163 passed; subsequent explicit-protocol and
  affected release/observed adapter checks: 92 passed, zero failures/errors/skips.
  These suites overlap. The full task/CHAT/VLM run initially had one outdated
  protocol fixture (416/417 passed); the updated full rerun passes all 417 tests,
  zero failures/errors/skips (`20261116_162250_915b21`).
- Corrected live controls: bounded-release two-goal control passes both
  transfers (482 audited commands); bounded-approach shifted swap passes all
  three transfers (787 commands, one completed base approach). Zero forbidden
  commands/malformed audit rows in either. Current/source hashes are recorded
  by the runner; these familiar privileged diagnostics are separate from prior
  cohorts and do not establish a new reliability estimate.
- Preserve both initial live failures. Anchoring raises to measured pose first
  lost search coverage for a measured 18.7 mm command/FK shortfall: offsets up
  to 20 mm could not clear the lowered opening probe. Including that measured
  reserve in candidate construction fixes coverage within the unchanged 40 mm
  global raise bound and unchanged drop/contact/retention/effect gates.
- Independent review agents inspected code without editing or running the live
  jobs. Their final raise-bound, numeric-input and height-chain findings have
  regression tests. Their reviews are not human approvals.
- Actual `run agent` with remote Caliban on clean `5a891c59` passes the contextual
  recovery diagnostic: named one-action progress, second-step preflight refusal
  with no failed-step command, stopped before a separate retry request, then
  fresh discovery/new one-step plan/one remaining transfer. Both original goals
  satisfy independent physical scoring. Eight model calls, 514 audited commands,
  zero forbidden commands or shutdown warnings; clean exit. The refusal is an
  evaluator injection, not actual solver/contact-loss recovery.
- Observed-branch integration of this correction passes 390 affected checks at
  `abf609b0`; full observed transfer remains unqualified.

[Compact evidence record](../experiments/tamp_stack_review_20261116.json).
Native runners began before the runtime commit was made; all 17 frozen source
hashes in each corrected control match `5a891c59`. The actual-agent run used that
clean commit. Source, attempt and cohort distinctions remain explicit.

Disk cleanup is separate from task metrics: prune regenerable package caches,
then remove only the SHA-256-identical duplicate CUDA installer (`.run.1`),
preserving the original installer, installed environments, model weights and
experiment evidence. Observed free space rose from 1.7 GB to 15 GB during cleanup;
concurrent filesystem activity means this is not a precise deletion-byte total.
