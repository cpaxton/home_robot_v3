# Checked native base approach — November 16

The complete original development populations now pass: **8/8 native tasks,
18/18 transfers and 4/4 interface boundaries; 4/4 actual-agent tasks and 10/10
transfers**. Agent completion also requires 8/8 independently scored object
goals. These are privileged simulated Stretch tabletop trials, not held-out
reliability, observed manipulation, household adapters or hardware qualification.

| Frozen cohort | Source | Tasks | Transfers | Other evidence |
| --- | --- | --- | --- | --- |
| Corrected approach pilot | `158fd427` | 3/3 | 7/7 | 2/2 handle checks; 1,858 audited commands |
| Original full native population | `158fd427` | 8/8 | 18/18 | 4/4 boundaries; 4,605 audited commands |
| Actual `run agent` + remote Caliban | `2e4dd8fc` | 4/4 | 10/10 | 8/8 object goals; 2,526 audited commands |
| Focused shutdown-repair replay | `2f993af9` | 1/1 | 3/3 | Shifted swap; 766 audited commands; clean client shutdown |

No cohort above records a forbidden command or infrastructure failure. The
focused shutdown replay is separate, not a fifth case pooled into the agent
baseline. Native, agent and shutdown sources have identical hashes for the
native approach, adapter, placement, policy and release modules. The later
changes repair agent startup and command-client completion bookkeeping.
Historical native **4/8** and agent **3/4** records remain unchanged.

## What changed

- `bounded_approach` checks the current base and bounded wheel-axis alternatives
  using private full-robot geometry. Ordinary controllers execute unloaded base
  motion; measured arrival, quiet pose and confirmed cancellation precede fresh
  grasp planning. No unchecked tuck, loaded base transport or live pose write.
- Loaded placement refreshes measured payload geometry and original-region
  support candidates under the initial primitive's unused planning budget.
  It retains valid original proposals and local reach patches. Full updated
  preplace/place/open-retreat paths must certify every accepted proposal.
- Payload-center offsets are composed through the desired EE orientation.
  This shared coordinate correction also applies to other native profiles;
  checked approach and loaded candidate refresh remain explicitly opt-in.
- The actual chat loop uses shared native-policy validation. An absent audit
  after a CLI startup failure now produces an inspectable failed summary.
- A boot/session/sequence-bound positive stop witness survives receipt eviction
  after hundreds of arm commands. New goals, restarts and uncertain stops cannot
  reuse it. Server cancellation and ownership checks are not weakened.

Collision, retention, release, final-goal checks and robot/contact physics are
unchanged. The default policy selection remains `legacy`. Sweep checks are
sampled static geometry (at most 5 mm / 0.025 rad, with a sampled ±10 mm XY /
±0.05 rad tracking envelope), not continuous or dynamics certificates.

## Actual agent and contract checks

The existing CLI discovers roles, requests `plan_task`, executes the returned
handle, reads the result and stops. The evaluator does not script transfers.
The original four cases all pass with Qwen/Qwen3-VL-8B-Instruct at the remote
endpoint; no local VLM/GPU is used. The cohort makes **16 model calls**, totaling
116.1 seconds of inference wall time. Sampling is server-default, not separately
pinned. Two model responses bundle discovery and planning despite the requested
one-call format; JSON and stop-boundary metrics do not score that format rule.

All four result contracts pass. One deliberately invalid-handle request is
rejected before a separate user turn authorizes the task. No later dispatch
violates that stop boundary. This does not exercise recovery from physical
execution failure. Native invalid/consumed handles, stale-head plans and occupied
placement targets satisfy all four predeclared no-command boundaries.

One successful agent trial emits a shutdown warning after its completed task:
the navigation stop receipt has been evicted by subsequent arm commands, so the
client retries cancellation of an already-stopped goal. A pure command-protocol
reproduction confirms the cause. The follow-up at `2f993af9` passes the complete
shifted swap again and exits without that warning; the earlier outcome is not
rescored. Its three model calls and task are reported separately.

## Verification and retained failures

**407 affected TAMP and command-contract tests pass**, with zero failures,
errors or skips, on `2f993af9` (job `20261116_014531_54b1f9`). Coverage includes
private swept geometry, tracking/stop failures, stale scenes, loaded-candidate
preservation, shared policy validation and receipt eviction/new-goal/restart
ownership guards. Ruff and whitespace checks pass. The updated full paper
builds to 58 pages, with no unresolved references/citations.

The [initial report](tamp_native_base_approach_initial_20261116.md) retains the
first third-transfer placement failure at `d6da3d6d`, the candidate-loss regression
at `2477cd3f`, and its interrupted replay. All four first agent launches at
`158fd427` fail before dispatch because the chat policy allowlist rejects the
new profile. Their job IDs and logs remain recorded; they are not physical
successes or omitted successful agent trials. The complete four-case replay
follows the fix. The corrected-source stationary swap ablation still fails
cube original-reference retention after 1/3 transfers.

![Recorded original-reference retention](media/tamp_native_base_approach_20261116.png)

The plot compares independently initialized same-source development runs, ending
before any loaded binding change. It combines approach/candidate selection and
loaded refinement, not a locomotion-only treatment. Points are sparse simulator
debug samples, not force measurements or a control-rate replay certificate.

[Compact source/job/receipt manifest](tamp_native_base_approach_20261116.json)
and [portable figure data](tamp_native_base_approach_retention_20261116.json).
Reproduce the figure with `scripts/plot_tamp_native_base_approach.py --data
docs/experiments/tamp_native_base_approach_retention_20261116.json --output
paper/figures/tamp_native_base_approach_20261116`. Verbose raw traces stay in
ignored `.tamp-runs/` folders in the two checkouts named by the manifest; this
is not a public raw-data archive.

Next: review/merge the opt-in stack, then freeze new held-out layouts and explicit
execution-failure/recovery cases. Observed self-occupancy/contact admission and
exploration remain separate. [Public policy contract](../api/tamp_task_plans.md#experimental-checked-base-approach)
and [implementation plan](../plans/2026-11-14_native_tamp_base_approach.md).
