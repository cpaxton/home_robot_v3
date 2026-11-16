# Native TAMP qualification — November 13

The first complete prospective population at `cbfebe35` passes **1/8 positive
task candidates and 4/18 declared transfers**. All **4/4 interface-boundary
checks** pass without additional commands. There are **2,372 audited commands,
zero forbidden commands and zero infrastructure failures**. This is native,
privileged, stationary simulated Stretch, with zero model calls. It does not
update the actual-agent 2/4 baseline or establish observed execution.

| Case | Task | Completed transfers | Failure |
| --- | --- | --- | --- |
| Familiar control | fail | 0/2 | release verification |
| Initial X +15 mm | pass | 2/2 | none; consumed-plan rejection also passes |
| X -30 mm | fail | 0/2 | release verification |
| X +30 mm | fail | 0/2 | task search exhausts available motion refinements |
| Y -12 mm, cube yaw +0.12 rad | fail | 0/2 | release verification |
| Y +12 mm, cube yaw -0.12 rad | fail | 0/2 | release verification |
| Swap at X -10 mm | fail | 2/3 | release verification |
| Swap at X +25 mm | fail | 0/3 | task search exhausts available motion refinements |

The invalid-handle check before an explicit renewed task, consumed-handle reuse,
stale plan after ordinary measured head motion, and occupied-target placement
all return their declared error codes with no additional audit commands. Correct
rejection is scored separately from physical task completion. Every candidate
remains in the declared population; none is silently removed as infeasible.
See the [frozen inputs](../../configs/eval/tamp_native_qualification_20261113.json)
and [compact manifest](tamp_native_qualification_20261113.json).

## Shared production implementation and source transition

`agent.native_tamp_policy=bounded_release` now selects the same native adapter
in primitive tools, multi-step task sessions and the controlled reference runner.
The diagnostic wrapper records events; it no longer owns a different release or
handoff implementation for this mode. Default `legacy` behavior is preserved.
The new profile remains explicit, privileged and simulator-only.

The first launch at `4a4a63ad` completed both regression tasks and two boundary
checks. It was not a complete qualification cohort. Local timing found median
7.025 ms model hashing per physics tick versus 2.718 ms physics stepping. Existing
PR #228 was merged into this branch; its cache retains measured state updates and
refreshes on model replacement. A matched private 100-step callback comparison
preserves qpos/qvel/qacc/contacts/fingerprint bit-for-bit while reducing wall time
0.863 s to 0.128 s. This is an offline component measurement, not an end-to-end
speedup or task result. Remaining initial-launch jobs were explicitly interrupted
or cancelled; all artifacts and cancellation records are retained. The complete
unchanged population then ran on `cbfebe35`; its scores above are not mixed with
the initial pair. Faster asynchronous execution exposed the settling bug below.

## Failure diagnosis and development follow-up

A read-only observer during the swapped task finds that the gripper has arrived
at three simulated seconds, but the object is still moving at **0.106 m/s and
2.537 rad/s**. The first arrived-and-quiet sample is at **5.776 simulated seconds**.
The new pre-retreat gate's three-second wait was too short for that supported
release. The existing native final verifier already permits twelve seconds.
The follow-up uses that same bounded settling budget and synchronizes feedback
before anchoring its clock. The 0.01 m/s / 0.1 rad/s limits, no-fingertip-contact,
geometry checks, 0.1 s pre-retreat stillness and independent final verification
remain mandatory. A supported object alone still cannot pass a failed receipt.

The two planning cases are feasible in private replay with a vertical pregrasp
fallback. The shared planner first exhausts every existing candidate, then tries
a pregrasp 80 mm above the grasp using the same path/collision validation and
shared 256-call IK cap. The X +30 mm cylinder plan takes **117 calls**; the
X +25 mm cylinder-to-buffer plan takes **102 calls**. Neither probe commands the
robot or proves physical success. Extra yaw midpoints and several placement
sampling/orientation probes were also retained; unused alternatives are not
introduced into production code. No goal, model, collision or retention threshold
is relaxed by the selected fallback.

The unchanged population is now a **development replay**, since its failures
informed these fixes. Do not relabel it held-out qualification. Preserve this
1/8 result, run fresh native episodes on the follow-up source, and then repeat
the actual production agent using Caliban with explicit failure/recovery turns.
The agent evaluator now additionally requires an authoritative successful
execution receipt; physically satisfied goals after partial execution cannot
become an overall task pass. Earlier agent scores are not rescored.

## Validation and artifacts

The `cbfebe35` affected suite passes **290 tests**, zero failures/errors/skips.
Those checks establish API/control semantics, not task success. Source-bound
JUnit, fixtures, physics/contact traces, endpoint checks, private planning probes,
read-only timing observations and interrupted launch records remain under
`.tamp-runs/native-qualification-clock-20261113/` and
`.tamp-runs/native-qualification-20261113/`. The failed experimental sampler
versions are archived with their probes; only compact manifests enter Git.
