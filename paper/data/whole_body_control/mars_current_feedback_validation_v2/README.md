# Corrected-reset Mars controller baselines

See the [complete report](../../../../docs/experiments/mars_current_feedback_reset_validation_20261115.md)
and [fixed protocol](../../../../docs/plans/mars_current_feedback_validation.md).
These are development comparisons, not hardware qualification or independent
random-trial confidence estimates. No PPO updates or actor promotion occurred.

- `results.json`: full counts, matched twelve-case comparison, loop stop traces,
  fresh regression receipt digest, actor hashes and compact TAMP outcomes.
- `cases.json`: one row per general case/controller; `columns` declares units
  and semantics. `terminal_velocity_sample_pass` checks one terminal measured
  sample; it is not a stop-confirmation handshake.
- `budget-matrix.json`: all 164 current and original teacher-relative budgets,
  with their real teacher/archive identities preserved.
- `tamp-followup.json`: the separately declared 24-episode frozen-controller and
  pre-planning-settling study. Initial hashes precede optional settling.

Raw data remain under ignored paths:

```text
logs/eval/mars-current-feedback-validation-v1/   # interrupted first producer retained
logs/eval/mars-current-feedback-validation-v2/   # completed reset/baseline/full checks
logs/eval/mars-tamp-tracking-followup-v1/        # additional24 TAMP cases
logs/recovery/mars-current-feedback-source-*.tar
logs/recovery/mars-tamp-tracking-source-*.tar
```

The legacy source bundle is the existing `e7b42569…`; the frozen velocity final
is `38c291b8…` from the prior arm comparison. The restored logical artifact
paths point to existing checksummed files; no new weights were generated.

From a matching checkout and installed RL environment, use fresh output paths
and an available coordinated CPU lane:

```bash
PYTHONPATH=src python scripts/validate_mars_current_feedback.py --phase loop --output NEW_LOOP_DIRECTORY
PYTHONPATH=src python scripts/validate_mars_current_feedback.py --phase baseline --loop-report NEW_LOOP_DIRECTORY/summary.json --output NEW_BASELINE_DIRECTORY
PYTHONPATH=src python scripts/summarize_mars_current_feedback.py --root logs/eval/mars-current-feedback-validation-v2 --output NEW_READOUT_DIRECTORY
PYTHONPATH=src python scripts/probe_mars_tamp_tracking.py --output NEW_TAMP_DIRECTORY
```

The baseline and follow-up require complete matching source evidence. Full
regression receipts are verified using `emet.rl.regression_gate`; old-source
qualification and post-conditioning state identity are not inferred from
self-consistent JSON. Full command arrays are represented by checksums here
and retained in the ignored raw records.
