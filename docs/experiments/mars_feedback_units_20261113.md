# Wheel-feedback interference and clipped jaw observations

Neither diagnostic produces a controller to adopt. The original source remains
retained, and hardware motion is disabled.

## Paired wheel-feedback interventions

The unchanged supported-contact final reproduces all 114 arm/coupled physical
cases exactly, including actions, rewards, traces and outcomes. Every case is
then evaluated under both in-memory interventions, with frozen goals, physics,
teacher budgets and limits.

| Variant | Earlier arm full/tracking | Supported arm full/tracking | Coupled full/tracking |
|---|---:|---:|---:|
| Unchanged final |63/88, 87/88|16/18, 16/18|3/8, 3/8|
| Source wheel readout; final shared context |79/88, 88/88|12/18, 14/18|2/8, 2/8|
| Complete source wheel branch; final arm branch |82/88, 88/88|11/18, 14/18|2/8, 2/8|

Restoring the wheel readout recovers much of the free-space smoothness loss,
but reduces contact and coupled performance. Restoring the complete wheel
branch is a two-actor diagnostic, not an exportable single-policy result.
No force, wheel, invalid-physics or retention flag trips in these diagnostic
tracking cases. No weights or policies are exported from either intervention.

The 24 unchanged-final arm cases failing base jerk move slowly: peak measured
base speed ranges 1.98–3.24 mm/s, with maximum XY motion 0.097–1.829 mm.
Their peak linear jerk is 0.202–0.463 m/s^3. These cases have 0.4905 N reported
peak contact load and still miss the frozen relative smoothness budget. Keep
that failure explicit; these measurements do not qualify a damage limit or
the much harder loaded task.

## Observation-distribution audit

Six supported training parents reproduce their recorded native rollouts
exactly. Compare their unchanged actor inputs with actual failed pregrasp
inputs, retaining samples whose hand error is at least 2 mm. These are
correlated time samples from different trajectories, not independent trials
or a matched causal estimate.

| Measurement | Supported training | Actual public pregrasp |
|---|---:|---:|
| Samples |524|229|
| Jaw position/error clipped at 20 |128/524 each|178/229 each|
| Median first hidden-layer saturation fraction |0.6094|0.7344|
| Median second hidden-layer saturation fraction |0.0391|0.0469|

The frozen jaw statistics came from almost constant apertures. Physical jaw
deflection can exceed their useful range, and clipping then discards differences
between larger deflections. Higher saturation is a conditioning concern;
these data alone do not establish the full cause of failed execution.

A static variance-floor probe leaves the means, weights and all other channels
unchanged, with jaw standard deviation at least one quarter of the nominal
0.3491 rad travel (0.087275 rad). It performs worse without retraining: earlier
arm full63/88→60/88, supported full16/18→12/18, base43/50 and coupled3/8
unchanged. All six paired public-tool episodes fail and stop confirms in all
six; one unchanged episode hits an inference deadline and is inconclusive for
later tracking. The transformed cases time out. This probe is rejected as a
standalone calibration; no runtime transform is adopted.

[Every case, intervention, aggregate input statistic and artifact identity](../../paper/data/whole_body_control/mars_feedback_units_v1/results.json)
is retained. High-volume observations and compressed traces remain ignored.

## Controlled training change

The opt-in `--grip-normalization-floor` changes only three history fields:
measured jaw position, jaw target error and jaw reference. It preserves their
raw measurements and existing means. Other observation units, action semantics,
rewards, robot models, contact limits, masks and executor authority stay fixed.
Default zero preserves existing behavior.

The new trial uses the same source, supported-case mixture, seed90, LR1
optimizer, exploration, reward schedule and fixed262144-transition final as
the completed supported trial. Its only learning intervention is jaw scaling:
keep source units throughout the4096 critic-only steps, then interpolate
standard deviations geometrically to a0.087275-rad minimum over32768 steps.
Update at rollout boundaries and refresh PPO's cached observation from the
same raw measurement, without resetting physics. Log the actual scale fraction.
Checkpoints export ordinary frozen mean/variance statistics; runtime loading
requires no scheduler or action filter. Motion and context must both be trainable.

Before launching: fresh full source-bound regressions; teacher164/164 and
all164 previous teacher cases exact; the same fixed new-seed contact teacher
12/12; exact initialized source and critic-only warmup. Retain the stronger
source unless the fixed final passes nonregression across all original subsets,
four fresh audits, physical safety and actual public-tool execution. Complete
the same nine public-tool comparisons even after a failed tracking gate.
No automatic promotion or hardware test is authorized by this experiment.
