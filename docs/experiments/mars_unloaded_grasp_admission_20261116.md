# Mars unloaded grasp-pose boundary: completed teacher readout

The primitive environment now supports `primitive="grasp_pose"`, through the
actual public-tool approach, pregrasp and grasp pose, **before executor closure**.
The default remains `pregrasp`. On frozen runtime **`9cd439bf`**, the declared
twelve-episode comparison completes with **5/6 successes at each boundary**.
All twelve measured stops confirm. Neither stage admits the full six-case
population; **PPO remains stopped**.

## Same cases and contracts, later boundary

The [predeclared protocol](../plans/mars_unloaded_grasp_admission.md) uses the
original four development and two audit cases, with original reset seeds,
nominal stance, masses, strict stage tolerances, `s_curve` references and
current-COM wheel feedback enabled from construction. No added settling,
task-region enlargement, timeout change, physical-model change or loaded
execution is used. The actor retains history/preview/motion inputs; the
primitive name and case identity remain harness metadata.

| Case | Pregrasp | Grasp pose | Final grasp position / orientation error |
| --- | --- | --- | --- |
| Development left 20 g, seed 34600000 | pass | pass | 1.438 mm / 0.001228 rad |
| Development right 50 g, seed 34600001 | pass | pass | 1.329 mm / 0.000889 rad |
| Development left 50 g, seed 34600002 | lift-path planning rejection before boundary | same rejection | boundary not reached |
| Development right 20 g, seed 34600003 | pass | pass | 0.984 mm / 0.000565 rad |
| Audit left 20 g, seed 34610000 | pass | pass | 1.509 mm / 0.001423 rad |
| Audit right 50 g, seed 34610001 | pass | pass | 0.800 mm / 0.000295 rad |

Each successful new boundary retains measured quiet hold and confirmed stopping,
then independently checks the final pose against **3 mm / 0.02 rad** and checks
open jaws. Individual physical tick rewards remain available, with completion
credit on the final tick. No close, lift or placement commands occur in any of
the twelve attempts. No wheel-lift, invalid-physics, unsafe-contact or retention
flag is recorded. An unloaded approach can contact the object; these results
do not establish a loaded grasp, retention or pick-and-place success.

## Physical parity and what failed

All six default pregrasp runs exactly reproduce the archived corrected-reset
baseline on fourteen recorded physical/outcome fields: success, stop, tick and
decision counts, returns, commands, inference counts, object translation and
rotation, initial-state/model/robot identities, initialization and safety flags.
The known failure's public code is now `planning_failed`, replacing the old
`operation_failed` through the semantic failure repair already in PR #239.
That code change accompanies identical recorded physical results; it is not
an extra successful case.

For each paired reset, the grasp-pose physical trace exactly matches its
pregrasp prefix before the first public grasp command. The five successful
prefixes contain 88, 88, 130, 89 and 90 control ticks; the failed approach's
entire 54-tick trace matches. The comparison excludes measured inference wall
time and retains every other recorded tick field. The command timestamp defines
the handoff: the servo's persistent phase label can still say pregrasp during
this unloaded continuation, so that label alone cannot partition task stages.

The failing case returns semantic `motion_rejection=static_support_margin`,
`failed_stage=lift`, with a confirmed stop. The existing full-task executor
replans pregrasp, grasp and the prospective loaded lift together at measured
state, before executing either selected boundary. Consequently, a later lift
rejection still blocks this unloaded primitive. The new boundary separates
physical execution stages; it does **not** yet separate their planning admission.
The old pregrasp expected-failure exception is deliberately not borrowed by the
new all-case stage readout.

The next planning change should explicitly admit the unloaded reference prefix
under its own task scope, preserving collision/support checks on every requested
motion and leaving full loaded-lift validation authoritative when lift is
requested. Do not fabricate a successful lift witness, lower the support margin,
drop the case, or authorize training from the five surviving cases. Freeze any
such scope change and obtain another stage-specific readout before learning.
The fixed-world-hand/base curriculum remains a separate planned coverage task.

## Evidence and validation

The [compact archive](../../paper/data/whole_body_control/mars_unloaded_grasp_admission_v1/results.json)
preserves all twelve outcomes, final pose residuals, stage readiness flags,
source/row/trace hashes, baseline parity and paired-prefix identities. The
stdlib-only `scripts/summarize_mars_unloaded_primitives.py` reproduces this
archive without stepping physics. Raw traces and the full source manifest stay
ignored under `logs/eval/mars-unloaded-grasp-admission-v1/`.

| Work | Producer | Managed job | Outcome |
| --- | --- | --- | --- |
| Initial focused checks | development tree | `20261116_010149_755d4c` | 107 pass; precedes final endpoint instrumentation |
| Fixed twelve teacher attempts | `9cd439bf` | `20261116_010524_d39c9d` | complete, 105.03 s, both stage admissions false |
| Fresh canonical regressions | `9cd439bf` | `20261116_010523_f3750c` | 1,367/1,367 pass, 760.52 s; receipt independently verified |
| Offline readout checks and paper | diagnostic source | `20261116_011315_aa8bfe` | four tests pass; paper builds at 58 pages with no undefined citations |

Twenty-two pure admission/runner tests pass after final endpoint instrumentation.
The completed canonical gate covers the new native grasp-boundary and reward-timing
test, with zero failures, errors or skips. Its receipt independently verifies
against the pre-consolidation runtime/canonical tests, whose source manifest exactly matches
the teacher freeze. The separate
[validation record](../../paper/data/whole_body_control/mars_unloaded_grasp_admission_v1/validation.json)
binds the XML/receipt and original teacher results; PR #239's older receipt is
not reused. Later changes are offline readout code, separately tested outside the
canonical suite, and documentation/evidence; the frozen runtime is unchanged.
The later #240/#241 consolidation adds the separate directional source/manifest;
its fresh combined validation is recorded in [Active work](../rl/ACTIVE_WORK.md).
It does not turn this historical 5/6 teacher readout into successful admission.

These are six resets observed under two boundaries, rather than twelve independent
trials for a reliability estimate. No new actor, PPO step, policy export, default
promotion, Caliban or hardware run follows from these results.
