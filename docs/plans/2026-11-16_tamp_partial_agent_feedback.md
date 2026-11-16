# Actual-agent partial feedback and authorized recovery

Review: [PR #243](https://github.com/cpaxton/home_robot_v3/pull/243), stacked on #242.

Predeclare two trials in [the JSON protocol](../../configs/eval/tamp_partial_agent_20261116.json)
before live execution. Use the actual `emet.app.run_agent` CLI with a diagnostic
bootstrap that replaces only the native provider constructor with the existing
one-shot evaluator refusal wrapper. Commands, planning, prompts, dispatch and
feedback remain production code. The wrapper refuses admission after one real
measured completed transfer; it does not fake contact, pose or sensor feedback.

The stop-only trial must retain an incomplete task, accurately name the completed
and failed actions, and dispatch nothing after the partial receipt. The separate
recovery trial receives a new stdin user turn only after the expected preflight
refusal, forced final feedback, independently measured supported/settled objects,
and a quiet audit interval. It must rediscover, make a fresh plan and complete
exactly the one remaining transfer. Unexpected outcomes forbid reauthorization.
Closing stdin ends the production input loop naturally; no second request is
silently scheduled before the failure is checked.

Public feedback gains ordered public-role action descriptions and a per-failed-
step dispatch flag, while preserving existing overall-progress fields. Count
only measured-effect-verified actions as completed. Keep failed/unattempted
actions unverified, consumed handles unusable, and full goal verification false
on partial failure. Inspect the model's natural-language report separately from
the machine receipt and final physical goals; a correct stop is not task success.

Commit source/protocol before both live episodes; use remote Caliban with local
GPU hidden, managed CPU-only jobs and at most two simulator bridges. Preserve all
failed attempts and model text. This does not qualify physical grasp-loss,
held-object recovery, observed grounding or hardware. PR242's motion policy and
acceptance thresholds remain unchanged.

The first frozen live pair at `d9fb0273` accurately reports named partial progress
and stops in both trials; stop-only passes. Recovery fails on the subsequent
request: the model proposes an unavailable inspection tool, then the bound
session rejects its own `robot="stretch"` selector. Preserve that failure.
Accept the configured native robot selector, clarify the advertised inspection
tool, and stop mixed TAMP batches at unavailable tools. Repeat both declared
trials on the next clean source; do not replace or rescore the first pair.

The complete replay at `d2b24610` passes both feedback/control tests. Stop-only
correctly leaves the task incomplete. After explicit reauthorization, recovery
completes just the remaining cube action and satisfies both original goals.
All 468 affected tests pass. The model's extra read-only discovery and zero-step
plan after completion are preserved; no efficiency claim is made. See
[the receipts, verbatim feedback and limitations](../experiments/tamp_partial_agent_20261116.md).
