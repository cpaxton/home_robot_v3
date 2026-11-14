# Native telescoping-joint planning margin — November 14

This focused change reserves **0.5 mm per telescoping segment (2 mm total)**
inside model limits when the opt-in bounded native policy solves IK targets.
The ordinary legacy policy is unchanged. Physical model limits, measured-start
validation, geometry, collision/retention thresholds, release limits and final
goal verification remain unchanged. A measured state beyond the original limit
still rejects; the planner does not clamp live qpos or silently relabel it valid.
The margin is an empirical reserve, not a certified tracking-error bound.

The original November 13 swap-right25 failure captured arm segments 7–9 µm above
their 0.13 m limits after release, beyond the unchanged 1 µm numerical tolerance.
A private planning probe showed an interior target remained feasible. The new
regressions check real bounded grasp/place paths stay inside this reserve, the
model/state remain unmodified, legacy behavior is preserved, and an 8 µm
out-of-range measured start still rejects without commands.

## Physical results, kept separate by compute condition

The original frozen layouts, instructions, final regions, seeds and thresholds
are retained. Baseline is `935a0c7c`; candidate runtime code is introduced at
`941fa8ff`, with replay-only configuration commits `63fe7a98` and `a4f3e0a1`.
Each condition is one live asynchronous native trial, not an exact initial-state
pair. No LLM, training, hardware, attachments or object-pose teleports are used.

| Compute condition | Policy | Case | Complete task | Transfers | Outcome |
| --- | --- | --- | --- | --- | --- |
| One CPU (20) | old | swap-right25 | fail | 0/3 | planning_timeout; 355.3 s |
| Service CPUs 20–25 | old | swap-right25 | fail | 0/3 | planning_timeout; 201.9 s |
| Service CPUs 20–25 | margin | swap-right25 | fail | 0/3 | planning_timeout; 187.0 s |
| Service CPUs 26–31 | margin | control | pass | 2/2 | complete receipt and independent goals |
| Service CPUs 26–31 | margin | shift15 | pass | 2/2 | complete receipt and independent goals |
| Fast CPUs 0–7 | old | swap-right25 | fail | 1/3 | verification_failed after first placement/retreat |
| Fast CPUs 0–7 | margin | swap-right25 | fail | 1/3 | verification_failed after first placement/retreat |

The 120-second symbolic-planning budget remains unchanged. Expensive feasibility
calls can return after that budget before search rejects: a planning timeout is
still a failed task, even with no task command. The old and new policies both
fail this gate on the slower service CPU set. Do not combine compute conditions
into an aggregate native success rate, erase the timeout failures, or infer
hardware-independent planning performance.

The fast-source baseline completes the first release and retreat and then fails
on the second object. It does **not** reproduce the historical retreat rejection.
The recorded limit-overrun state and unchanged-start rejection regression remain
valid evidence, but this new pair alone cannot prove a causal retreat-success
improvement. At the sampled first release, the baseline's closest segment is 1.05 µm
inside its upper limit; the candidate's is 467.5 µm inside. The achieved
margin differs from the nominal 500 µm target reserve. This is one sampled
state, not a continuous tracking-error or robustness certificate. Both sources
then reject the original-reference cube rotation during the checked loaded
handoff after lift/ordinary hold: 0.10788 and 0.10353 rad respectively, above
the unchanged 0.1 rad bound. Neither task passes, and this development pair
establishes no complete-task success improvement. This attribution does not
retroactively resolve the historical Caliban episode's missing substage.

Measured release positions and complete receipts are retained in
the result manifest rather than substituting controller acknowledgments.

The two regression controls preserve invalid-handle rejection before an explicitly
renewed task and consumed-handle rejection after success, with no extra command
for either error. No automatic physical recovery is inferred from those checks.

## Acceptance and records

152 relevant tests pass. Full affected regression: **304 pass, zero failures, errors or skips**.
The broader parent PR's existing full-config failure/skip are not fixed or
reclassified by this motion-policy change. No blanket all-green claim is made.

Earlier launch scripts created journals in directories required to be new;
those startup errors occurred before simulation and are retained. Single-core
attempts and interrupted runs are also retained. Only the named local replay
jobs were cancelled; controller CPU14/15 workers were not changed. Managed
jobs, source/asset verification and commands are preserved under ignored
`.tamp-runs/joint-margin-20261114/`.

The isolated baseline initially reported hydrated Git LFS assets as modified.
All 417 robot-asset object hashes match the tracked objects, and runtime source
hashes match the baseline commit. The raw dirty flag is preserved with this
independent verification. It is not edited into a clean receipt.

Protocols:
[service replay](../../configs/eval/tamp_native_joint_margin_20261114.json),
[fast CPU pair](../../configs/eval/tamp_native_joint_margin_cpu_replay_20261114.json).
The [compact result manifest](tamp_native_joint_margin_20261114.json) retains
all outcomes, budgets, sources, audit counts, release samples and artifact hashes.

The prior 4/8 native and 3/4 Caliban agent development results remain unchanged.
This does not qualify observed manipulation, whole-body learned control,
household tasks or new sensors. Remaining lift/effect failures need distinct
substage evidence and replay; the shared controller curriculum and trainers
remain separate work.

This PR remains **draft**: the full target-task gate fails. Its planning-margin
change and passing regressions do not authorize merge, default promotion or a
qualified TAMP claim. The next native step is a precise verification-substage
receipt and a physically verified grasp/retention reference for this failure,
coordinated with the controller owners without changing their actor/trainer.
