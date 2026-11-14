# Shared native TAMP development replay — November 13

The same eight task candidates now pass **4/8 complete tasks and 11/18 declared
transfers**, with **4/4 boundary checks**, 3,658 audited commands, zero forbidden
commands and zero infrastructure errors. The preceding complete run passed 1/8
and 4/18. The cases informed the fixes, so this is a development replay, not
held-out qualification. Each cell is one live asynchronous MuJoCo trial.

| Initial layout | Complete task | Transfers | Outcome |
| --- | --- | --- | --- |
| Familiar control | pass | 2/2 | complete verified receipt |
| X +15 mm | pass | 2/2 | complete verified receipt |
| X −30 mm | fail | 0/2 | cylinder keeps moving after release |
| X +30 mm | fail | 1/2 | cube lift exceeds 0.1 rad retention bound |
| Y −12 mm, cube yaw +0.12 rad | pass | 2/2 | complete verified receipt |
| Y +12 mm, cube yaw −0.12 rad | fail | 1/2 | cube lift exceeds 0.1 rad retention bound |
| Swap at X −10 mm | pass | 3/3 | complete verified receipt |
| Swap at X +25 mm | fail | 0/3 | joint bounds after release, before retreat |

All four independent boundary checks reject without an additional command:
invalid handle before a separately authorized task, consumed-plan reuse, stale
plan after measured head motion, and a placement target occupied by another
object. These intended errors are not positive task successes.

## What changed

Source `c4fa1668` selects the shared `bounded_release` profile. The ordinary
native policy remains the default. Release waits start from fresh feedback and
permit twelve simulated seconds for measured opening and supported stillness,
matching the existing native final-verification budget. Existing pregrasps
remain first; an 80 mm vertical pregrasp is attempted after they fail under the
same 256-call IK cap. No geometry, controller, friction, collision/retention
threshold, goal region, task instruction or surface sampler changed between
the preceding complete cohort and this replay.

The loaded-payload handoff and bounded release implementation are shared with
production primitive tools and the multi-step provider. The diagnostic adapter
adds tracing rather than replacing execution. Simulator state/contact remains
privileged evidence, not a hardware sensor claim. Native references are not
LLM-agent results; see the separate [actual-agent repeat](tamp_agent_followup_20261113.md).

## Remaining failures and next gate

The X −30 mm cylinder is supported and released but is still moving at the
12-second deadline: 0.1192 m/s and 2.8145 rad/s. It correctly fails. Simply
relaxing the speed gate would accept an unstable placement.

The X +30 mm and Y +12 mm cube grasps retain fingertip contact but rotate by
0.10131 and 0.10042 rad relative to the original payload reference during lift.
They correctly stop at the unchanged 0.1 rad limit. Bilateral contact does not
certify a rigid payload pose or allow renewing a failed bound.

The X +25 mm swap clears release checks, then cannot certify retreat from its
measured arm configuration. The final captured release sample has three arm
segments 7.0–8.7 micrometers above their 0.13 m model limits, beyond the planner's
1 micrometer numerical tolerance. This points to planning/controller margin
at the extension limit, not a need to clear destination clutter. The raw
`joint_limit_diagnostic.json` is a read-only analysis of that captured state;
it does not change the failed task or clamp the live configuration. A follow-up
should reserve achievable joint-limit margins and verify a real escape path,
then replay failures and test new untouched initial layouts. A private saved-state
probe with a 0.5 mm interior margin per arm segment finds a collision-checked
plan in 102 IK calls without changing model limits or sending commands. This
margin is not enabled in production and has no physical success claim.

No full observed task, navigation/exploration, household native integration,
hardware execution or general reliability rate is established.

## Validation and records

The fresh affected suite passes **299 tests**, with zero failures, errors or
skips. The repository-required full configuration suite has **131 passes,
1 failure and 1 skip**. `test_preserved_targets_prevent_torso_collapse_with_upright_base`
fails identically in an isolated `origin/main` checkout at `cd169673`:
`1.4634869868131006 < 0.2` is false. Its test, Rby1 assets, backend and stationary
controller are unchanged by this branch. The skip requires an unavailable
cached merged MJCF. These suite counts overlap and should not be added.

Inputs: [frozen qualification cases](../../configs/eval/tamp_native_qualification_20261113.json)
and [follow-up contract](../../configs/eval/tamp_native_followup_20261113.json).
The [compact manifest](tamp_native_followup_20261113.json) contains source,
job IDs, receipt counts and raw artifact hashes. Full traces remain in ignored
`.tamp-runs/native-followup-20261113/`; the failed preceding cohort and interrupted
first launch remain separate in the [prior report](tamp_native_qualification_20261113.md).

Post-trial validation at `425f180d` passes **302 affected tests**, with zero
failures, errors or skips, including the native verification-code follow-up.
The full paper builds to 58 pages with no unresolved references or citations.
These checks do not change the `c4fa1668` physical/agent trial outcomes.
