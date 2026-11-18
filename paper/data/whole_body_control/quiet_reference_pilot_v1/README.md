# Stationary-reference exploration pilot — complete

Frozen source `9d0620f9` passes 1,585 source-bound tests. Both 16,384-decision
arms complete; ordinary / quiet scores are 20/24 / 19/24 on new cases,
7/10 / 6/10 existing coupled, 6/6 unloaded TAMP prefixes each. All 120 actor
stops confirm. Quiet adds two contact failures and worsens hand error/linear
jerk, so it is rejected. Its stochastic training holds improve 0/20 → 13/20;
that does not establish better deterministic execution.

`record_results.py` generates compact complete identities/physical readouts in
`results.json`. `status.json` supersedes the historical launch snapshot. The
exact producers and protocol are preserved; raw traces/weights stay ignored
under `logs/manipulation/quiet-reference-pilot-v1/`. The ordinary arm exactly
reproduces the previous independent-critic actor weights. Paper builds to 61
pages. No promotion or hardware use. See the
[completed report](../../../../docs/experiments/quiet_reference_training_20261117.md).
