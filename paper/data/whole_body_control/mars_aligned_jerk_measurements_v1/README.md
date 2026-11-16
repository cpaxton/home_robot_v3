# Mars aligned jerk and wheel-feedback diagnosis

`results.json` contains compact measurements and complete outcome/metric rows for
the plant probes. `jerk_channel_columns` declares the three numeric columns used
in each named channel. Raw traces, original full summaries and checkpoints are
excluded from git; their SHA-256 values and producer revisions are retained.
`stationary-feedback.svg` plots the existing teacher's zero-command base hold.
`regressions.json` records the verified 1,284-test source-bound receipt digests.
The full receipt/XML stay in ignored storage; the existing
`emet.rl.regression_gate.verify_regressions` verifies them against local sources.

See the [experiment report](../../../../docs/experiments/mars_aligned_jerk_measurements_20261115.md)
for the fixed protocol, all denominators, invalid first probe and limitations.
No PPO updates, policy promotion or hardware motion occurred.

Ignored artifact root: `logs/eval/mars-aligned-jerk-v1/`.

| Directory | Contents |
| --- | --- |
| `data/` | 24 matched final-actor replays |
| `anchors/` | original zero actor and analytic teacher, stationary case |
| `substeps/` | same two anchors with 2-ms qvel/command/actuator-force samples |
| `feedback/` | invalid first 12-episode probe; intervention restored prematurely |
| `feedback-invalid-readout/` | preserved derived readout of that invalid attempt |
| `feedback-v2/` | corrected four-way teacher probe, 12 episodes |
| `teacher-only-readout/` | preserved intermediate derived readout, before learned comparison |
| `learned/` | cached/current-COM candidate actor, all twelve cases each |
| `full.xml`, `full.receipt.json`, `full.log` | fresh source-bound full regression evidence when complete |
| `*-managed/` | job wrappers and logs |

The checkpoints remain under
`logs/manipulation/mars-reference-base-exploration-v1/{control,candidate}`;
teacher evidence is in its `preflight/` directory. Source archives remain in
`logs/recovery/mars-{jerk-measurements,wheel-feedback}-source-*.tar` with per-file
manifests. The exact producer revisions appear in the report and compact JSON.

From the corresponding immutable producer checkout, with existing artifacts and
environment available, the executable interfaces are:

```bash
python scripts/replay_mars_jerk_measurements.py --output NEW_REPLAY_DIRECTORY
python scripts/replay_mars_jerk_measurements.py --base-hold-anchors --physics-substeps --output NEW_ANCHOR_DIRECTORY
python scripts/probe_mars_wheel_feedback.py --output NEW_TEACHER_PROBE_DIRECTORY
python scripts/probe_mars_wheel_feedback.py --learned-candidate --output NEW_LEARNED_PROBE_DIRECTORY
python scripts/summarize_mars_jerk_measurements.py --root logs/eval/mars-aligned-jerk-v1 --output NEW_READOUT_DIRECTORY
```

Use fresh output directories, one numerical thread and the coordinated CPU lane;
these examples do not override machine resource reservations. Reproduction on a
new source revision is a new diagnostic producer, not reuse of old qualification.
