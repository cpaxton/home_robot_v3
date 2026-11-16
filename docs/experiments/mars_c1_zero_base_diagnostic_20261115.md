# C1 command intervention: base motion hides some learned arm progress

The completed 36-rollout diagnostic reproduces both original C1 actors, then
substitutes the initial actor's **zero base commands** into the final actor at
every live observation. All twelve declared cases are retained in every mode.
This is an in-memory intervention using two policies, not an exported actor or
a replacement whole-body controller. The initial C1 base function is identically
zero; it is not the retained, learned base function of the legacy MLP source.

## Results

| Arm measurement, ten cases | Initial | Shared final | Final with initial zero-base commands |
| --- | ---: | ---: | ---: |
| Full contract | 1/10 | 1/10 | 3/10 |
| Tracking and settling | 1/10 | 1/10 | 4/10 |
| Smoothness | 10/10 | 1/10 | 9/10 |
| Fixed-base contract stops | 0/10 | 5/10 | 0/10 |
| Base-linear jerk failures | 0/10 | 9/10 | 1/10 |
| Terminal-hold failures | 9/10 | 9/10 | 6/10 |

The intervention retains the supported quarter-amplitude success and adds
`roll_left` and quarter-amplitude `pitch_down` full passes. `roll_right` now
tracks and settles, but still fails base-linear jerk. Across all twelve cases,
full passes rise from 2/12 to 4/12. Base hold passes in every mode; base forward
fails in every mode. These are fixed development cases, not new held-out trials.

The effect is partial. Full-amplitude pitch-up/down orientation errors at
termination increase from approximately 0.0190/0.0163 rad to 0.0829/0.0869 rad.
Removing base authority can also remove a direction the learned arm uses to
follow the hand target. Six arm cases still fail tracking/settling. Even with
zero actor base commands, physical arm/base coupling produces one jerk failure.
Freezing policy parameters does not freeze the physical base trajectory.

The result supports investigating base authority, terminal settling and
coordination while preserving learned arm tracking. It does not establish that
permanently zeroing the base solves whole-body control, and it does not identify
which future training intervention will work.

## Replay and provenance

Before the intervention, all **24 unchanged initial/final replays** exactly
match the original reset/reference hashes, result flags, failure reasons and
stop codes. Maximum absolute differences in each recorded base-linear,
base-yaw and joint jerk metric are zero. All twelve intervention identities
also match their original final cases. Original actor/report hashes are checked
before and after execution. There is no PPO update, actor export, threshold
change, hardware motion or promotion.

Managed job `20261115_014324_9d1d19` runs immutable implementation
`afeef3d0b2c24492f492c00efceddde21d420ce5` using its passing 1,249-test
source-bound regression receipt. It uses CPU15, one numerical thread, CUDA
hidden and rendering disabled, with a 300-second limit and 36-rollout budget.
CPU14 remains reserved for the separately declared frozen-base/context trial.

[Compact outcomes and exact archival runner](../../paper/data/whole_body_control/mars_c1_zero_base_diagnostic_v1/results.json)
retain every case, measured-at-stop errors, reference-completion flags, replay
checks and original artifact hashes. The adjacent `plan.json` and `diagnose.py`
are the byte-identical executed protocol and runner, including original local
artifact paths; copy them to scratch storage with those artifacts to replay.
They are an experiment archive, not a portable public evaluation interface.
The adjacent `replication_full_metrics.json` adds a read-only post-run audit:
every continuous metric, not just the three runner-gated jerk metrics, is
exactly equal in all 24 unchanged replays.

The canonical C1 normalization comparison is unchanged. Its own fixed-final
counts remain 0/12 for per-head and 2/12 for shared normalization; the command
intervention must not be substituted into either canonical result.
