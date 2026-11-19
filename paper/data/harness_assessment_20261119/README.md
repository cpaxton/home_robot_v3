# First harness audit figure inputs

This snapshot preserves initial contract, recorded and live development
outcomes on the frozen assessment revision. See the
[assessment report](../../../docs/experiments/harness_assessment_20261119.md)
for interpretation and remaining gates.

- [Capability matrix](capability_matrix.md) and [per-case ledger](case_outcomes.json)
  with source and raw-artifact hashes. Null time fields mark unavailable clocks.
- [Outcomes and cost](../../figures/harness_assessment_outcomes_20261119.pdf)
  groups results by evidence tier and retains pending and blocked assignments.
- [Recorded examples](../../figures/harness_assessment_recorded_20261119.pdf)
  show the lowest-ID history success and absent-target failure. The exact
  render and answer references are in [recorded_examples.json](recorded_examples.json).
- [Live Stretch trace](../../figures/harness_assessment_trace_20261119.pdf)
  shows the sampled base poses and turn-boundary RGB. See
  [active_trace_cap-dev-01.json](active_trace_cap-dev-01.json) for frame hashes.
  The displayed frames are the bridge measurements. Model attachments are
  retained separately in the raw HTTP traces.

Build from raw evidence with
[`summarize_harness_assessment.py`](../../../scripts/summarize_harness_assessment.py).
Raw paths resolve on the host under `/tmp/emet-harness-run-20261119`. The
normalized JSONL and generated CSV are in that run directory. This snapshot
is the initial audit and does not pool separate trials or assistance modes.
