# Shared-harness development pilot

`summary.json` is a minimal paired summary of the final integration artifacts,
not a generalization benchmark or a causal feature ablation. It retains source
pins, exact paired question outcomes, all three S0 seed outcomes/costs, and hashes
of the raw artifacts consumed by the analysis. Main is main-agent plus the
shared task/evaluation patch, not unmodified main. No RGB, reference answers,
model weights or large traces are vendored.

Reproduce on the machine containing the original artifacts (CPU only):

```bash
taskset -c 12 .venv/bin/python scripts/analyze_shared_harness_pilot.py \
  --final-run ~/runs/emet/final-stack-review-20261028 \
  --seed-zero-run ~/runs/emet/s0-camera-feedback-pair-20261028 \
  --output /tmp/shared-harness-pilot-summary.json
diff -u paper/data/shared_harness_pilot_20261028/summary.json /tmp/shared-harness-pilot-summary.json
```

Use an allowed stable CPU and an existing project interpreter on other hosts.
The optional seed-zero population is accepted only if both source revisions
match the final-run manifest exactly. Missing/error/duplicate arms fail analysis.

Results: EQA candidate 8/12 versus control 7/12; S0 both arms succeed on seeds
0 and 2 and miss the receptacle on seed 1. Candidate wall time improves on the
successful pairs and regresses on the failed pair. No consistent speedup is claimed.

`s0_trace_coverage` counts the router's primary picks (not every batched action)
and look-around receipts in final S0 seeds 1 and 2. The candidate uses that tool
only once. These counts describe recorded tool use, not visibility, necessity,
or the causal benefit of any feature. Seed-zero traces are not included in that
coverage field; their outcomes/costs remain in the paired S0 table.

`evidence_loader_audit.json` records pinned official metadata counts/hashes and
the missing official frame qualification. `capability_replay_summary.json`
records the separate six-case synthetic recorded-evidence diagnostic: 4/6 with
history, 1/6 current-only, both failing the two negative cases. Neither result
is an active-agent or official-benchmark score. Full requests and RGB-D/scorer
artifacts remain in the referenced `~/runs/emet/` directories, not in this repo.
