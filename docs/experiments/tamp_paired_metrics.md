# Paired TAMP metric improvement

Use the existing `scripts/run_tamp_experiments.py` with the same frozen admitted
registry, robot, timeout, case set and repeat protocol for baseline and candidate.
Run each under `emet jobs --cpu-safe --gpu-exclusive` from a clean immutable
checkout. Change one hypothesis at a time; keep a separate held-out scene set.
The runner now records placement defaults/seed and MuJoCo/NumPy/SciPy versions
in `manifest.json.search_protocol`. Registry hashes bind episode seeds/layouts.
Archive simulator assets/configuration and managed-job environment with the run;
the comparator cannot establish identity of external assets or unrecorded overrides.

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_tamp_runs.py \
  /path/to/baseline /path/to/candidate --output-dir /path/to/new-comparison --figure
```

The comparison consumes existing manifests/ledgers; there is no second trial
database. It refuses changed cases, registry, execution mode, arguments, timeout,
search protocol, nonterminal runs, and candidate-specific admission comparisons.
Compare each robot/assistance mode separately. Old manifests missing planner
provenance must be rerun; do not backfill assumed defaults to manufacture a pair.

All scheduled terminal trials remain in the denominator, including crashes,
timeouts and skipped cases. Success requires `completed`, task success and exit
0. Output includes paired gains/losses, first-failure stages, status counts,
source commits and input hashes. Uncertainty resamples scene clusters; fewer
than five scenes produce null rather than a misleading confidence interval.
This interval is descriptive, not an automatic significance or merge criterion.

`--figure` exports PDF and SVG alongside the source `comparison.json`. Figures
show full-task success over scheduled trials and paired gains/losses. Run on
terminal real experiments for paper results; tests of plotting on synthetic
data are not experiment evidence. Retain admission coverage in the separate
`plot_tamp_benchmark.py` figures, and keep GT/observed and assisted/physical
results separate. The comparator does not yet automate latency/safety analysis;
review those artifacts and require zero canary/safety regressions before promotion.

The current three-scene gate is a regression canary, not a generalization study.
Only after it passes should a frozen broader development set drive changes in
approach/orientation sampling or search allocation. Record budget ablations as
success-versus-cost changes. Run untouched holdout at selection milestones and
retain all failed candidates. `promotion: requires_review` is intentional.
