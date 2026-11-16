# PR #231 integration decision — November 16

Integrated main `fb9f9bf1` on frozen source `c8a34bfd`, after #226 landed.
GitHub had no submitted review comments on #231 at the time of this audit. This
is author review and fresh validation, not independent approval.

The code conflicts mattered: main added Mars grasp/approach/transport hooks and
model-signature cache checks, while #231 added Stretch grasp fallbacks and policy
identity. The resolution preserves both. Default candidates use each robot's
grasp hook; only the explicit bounded fallback changes the first target. Policy,
robot, model and boot changes invalidate the cached adapter. The Stretch-specific
bounded release profile rejects Mars as `native_policy_unsupported` before any
command; switching back cannot resurrect old handles. Mars keeps legacy native
behavior and its own execution hooks. No controller/contact/retention threshold
is loosened.

The two cache/model tests reuse their already loaded model when rebuilding an
adapter, eliminating unrelated cold-load timestamp expiry. Actual snapshot-age
validation and stale-capture regressions remain intact; no feedback is retimed.

Validation:

- 446 motion tests plus 657 non-overlapping integration tests: **1,103 passed**,
  zero failures/errors/skips. Includes Mars TAMP, configuration, CHAT, sensor,
  navigation, policy identity and public error contracts. Three focused
  compatibility checks also pass; they overlap the larger suite.
- Frozen native smoke on the familiar two-goal control: two transfers complete,
  both independent physical goals hold; 528 audited commands, none forbidden.
- Actual `run agent` with remote Caliban first rejects an invalid plan and stops,
  then handles a separate authorized two-goal request. Both transfers/goals pass,
  successful completion receipt present, clean exit, no dispatch after the failed
  tool in its turn; 509 audited commands, none forbidden.
- Ruff/whitespace and full paper build pass. No local GPU or hardware trial.

These two smokes use independent simulator startups and are not an exact reset
pair, held-out benchmark, or reliability estimate. The historical native 4/8 and
agent 3/4 populations and their failed cases remain unchanged. Source/protocol,
test counts, exact boundary results and nine artifact hashes are in the
[compact record](pr231_main_integration_20261116.json). Raw artifacts remain in
`.tamp-runs/native-stack-20261116/pr231/`, not a public archive.

Merge assessment: land the opt-in policy and its guarded shared interfaces;
keep `legacy` as default and observed/hardware execution unqualified. Follow-up
#234 changes closing/margins and needs its own review; #242's checked base approach
and later agent-recovery capabilities are not claimed by this PR. Preserve
main's separate self-occupancy and tracking-gap limitations from #226.
