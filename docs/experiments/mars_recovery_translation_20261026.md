# Crash recovery and reachable translation curriculum

The reboot removed the temporary worktree, but the four continuation runs had
already completed, followed by their automatic evaluation. All twelve numbered
checkpoint archives verify against their recorded SHA-256 values; the completed
replay archive verifies against its uncompressed checksum. The recovered source
also verifies the original 756-test regression receipt before new changes.
No completed training was repeated. The controller checkout now lives at
`.worktrees/mars-controller-integration` under the persistent primary repository;
the primary workspace remains on its separate review branch. Recovery scripts,
logs and integrity results live in `logs/recovery/20261026/`.

## Recovered continuation results

| Six-second diagnostic | Original | Adaptive, larger restart | Frozen, smaller updates |
| --- | --- | --- | --- |
| Nearby position (mm) | 0.40–0.42 | 1.35–1.39 | 0.29–0.39 |
| Nearby orientation (rad) | 0.023–0.025 | 0.038–0.041 | 0.026–0.028 |
| Wrist orientation (rad) | 0.054–0.059 | 0.095–0.100 | 0.062–0.067 |

Smaller updates with fixed statistics preserve position accuracy better, but
still miss angular and task contracts. All five actors (source plus four
candidates) remain **0/3 programs each**, or 0/15 total program attempts and 0/30
transfers. Ordinary general tracking is also poor: only one candidate succeeds
on one of its two base-only cases; all other candidate/mode pairs score zero.
Retarget/push cases include unsafe-contact stops, retained in the reports.
Each general mode has only two diagnostic cases, not a qualification population.

The joint-reference teacher passes its two base-only cases and fails its other
four general modes. It is not a universal Cartesian teacher: masked hand-only
commands can require base motion, while this baseline tracks an arm joint
witness and masks base feedback. Its failure cannot establish geometric
infeasibility or a learned performance ceiling. Keep those failures visible.

Completed training/source metadata and all replay failures are archived in
`general_continuation_training_v1` and `general_continuation_validation_v1` under
`paper/data/whole_body_control/`. Historical temporary paths in raw reports are
provenance, not current evidence locators.

## Action-audit correction

The original offline audit incorrectly grouped Mars actions as arm-first. The
actual layout is **base [0:2], arm [2:7], grip [7]**. The original per-action
statistics, predictions, model weights and measured pose outcomes are unchanged.
`normalization_action_audit_v2.json` reruns and reproduces the recorded actions,
then uses the corrected grouping, with explicit Mars/Stretch layout tests.
The corrected moving-probe arm-action ratio is approximately **4–9x** for weight
changes versus normalization changes alone, superseding v1's grouped summaries.
This is an offline command counterfactual, not a counterfactual rollout result.

## New general-purpose translation coverage

`translation_curriculum.py` samples joint directions in the angular Jacobian's
nullspace, scaled by native joint speeds. It then checks every finite candidate
by private FK and joint limits. Accepted goals translate the hand **5–20 mm**
while holding its orientation within **0.002 rad**. Failed finite candidates are
rejected, not made reachable by clipping or weakening the goal. The existing
collision/derivative/path admission remains a separate check. Live simulator
positions, velocities and controls are not modified by the sampler; tests cover
both Mars and Stretch coupling. Reset integration starts with Mars.

The fixed-base Mars arm covers a local reachable plane at fixed orientation;
this is not arbitrary three-dimensional translation coverage. Ordinary base,
hand-only, combined, rotation, retarget and disturbance tasks remain necessary.
The sampler adds no observation fields or Jacobian/witness hints to pose policies.
It does not introduce a runtime IK loop, alter geometry, raise velocity limits,
or relax force/retention/wheel/pose requirements.

The 32-target gallery admits all cases geometrically, spanning approximately
5.3–18.6 mm, with orientation changes below 1.1e-7 rad. Upward and downward cases
are both present. The export includes actual task-pose schedules, joint
witnesses, initial state, model/source hashes, rejections and a plot with equal
physical axis scales. It is a geometric diagnostic, not dynamic qualification.

Generate a fresh gallery from the persistent checkout:

```bash
PYTHONPATH="$PWD/src" taskset -c 14 env CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m emet.rl.manipulation.translation_curriculum \
  --samples 32 --output FRESH_DIRECTORY
```

## Physical translation probe and matched run

On eight fixed development seeds, the teacher reaches endpoint hand accuracy in
all eight and settles at least once in all eight, but loses settling in three:
**5/8 final success**, with no force or wheel failures. The original learned
checkpoint has **1/8 final success**, settles at least once in two, and trips
wheel unloading in one. That wheel gap is 0.1007 mm against the unchanged 0.1000
mm threshold; the stop is retained. These cases expose tracking/stability gaps,
not a reason to discard hard cases or count teacher failure as infeasibility.

`mars_pose_translation_curriculum_v1.json` specifies two matched 16,384-step
pilots from the original checkpoint, both with fixed normalization and learning
rate 1e-5 decaying to 2e-6. Only translation coverage differs: control has none;
treatment samples translations in half the nearby-reach resets. The macro mix
remains **70% reach / 30% manipulation**. Expected overall reset shares are 35%
ordinary reach, 14% hold, 10.5% translation, 5.25% local rotation, 5.25% local
combined, and 30% manipulation. Actual subtype episode/step shares are logged to
offline scalar W&B. Probabilities are not guaranteed step shares.

Automatic evaluation compares the source and both candidates on the same eight
translation cases, ten general pose/retarget/push cases, six stationary/nearby/
wrist probes and three public TAMP programs. Keep the original goals and bounds.
No automatic promotion or Herman motion is enabled. Training/evaluation status
is recorded only after launch/completion, not inferred from this specification.

The paired physical probe and complete nominal target gallery are archived in
`pose_translation_curriculum_v1.{json,json.gz}` with matching source manifests
and the exact probe driver. The target plot is
`paper/data/whole_body_control/pose_translation_targets_v1.png`. Portable
checkpoint-evidence wrappers in `paper/data/whole_body_control/recovered_v11/`
replace lost temporary locators while retaining original training sources and
manifest identities. Model weights remain local and are not uploaded.

The final implementation passes **769 fresh source-bound tests**, with
verified XML/receipt `pose-translation-v2-regressions`. The initial full run
exposed a test that expected preview without enabling it; the corrected test
uses the actual direct-velocity pose/preview interface. Its failed run is not
accepted as regression evidence. Runtime package parity, relevant lint and
the updated paper build pass. All five recovered checkpoint-evidence locators
verify against current runtime/model admission while preserving original
training provenance. No qualification or hardware gate is relaxed.

The matched 16,384-step control/treatment runs **started at 2026-10-26T23:14:41.095234+00:00**
from committed source `0c9cd1ba` after verifying all
769 source-bound tests. They run sequentially on CPU 14 with
CUDA disabled and offline scalar W&B. The immutable
`pose_translation_v1_launch.json` records both configurations, source/manifest
hashes, actual environment settings and both persistent drivers. Live status is
`logs/manipulation/mars-pose-translation-v12/status.json`; automatic common
evaluation writes `logs/eval/pose-translation-v12-20261026/`.

Rerunning `logs/recovery/20261026/launch_translation_pair.py` from the same
persistent source/environment preserves completed variants. An unfinished
variant resumes its latest checksummed numbered checkpoint with associated validation files in
a new attempt directory, preserving optimizer/statistics and the remaining
learning-rate schedule. Environment/RNG state resets are recorded, so this is
not claimed bitwise identical continuation. Concurrent launchers are locked
out; partial evaluations are preserved before a fresh replay. Launch is not a
completion or improvement claim. No automatic promotion is enabled.

Both runs subsequently **completed** their 16,384 steps and common evaluation.
All three actors (source, control, treatment) remain 1/8 on translations and 0/3
on the unchanged multi-transfer programs. The control preserves excellent nearby
position accuracy but still times out at public pregrasp; the translation
treatment reaches the path tube bound instead. This is a negative distribution
experiment, not an active training run. Full training/evaluation are archived in
`pose_translation_{training,validation}_v12.*`. The next
[agent acceptance report](mars_agent_controller_20261026.md) diagnoses actual
loaded execution with a separate single-delivery case.
