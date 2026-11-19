# First harness audit figure inputs

This snapshot preserves initial and recovered cross-simulator outcomes from
the frozen assessment revision. See the
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
The reboot cleared the original temporary artifacts. The ledger retains their
previously recorded hashes and marks their availability. The copied supervisor
log and Habitat bundles are in the persistent
`logs/harness-assessment-resume-20261119/recovered` directory.

- [Recovered Habitat EQA maps](../../figures/harness_assessment_habitat_eqa_20261119.pdf)
  and [map hashes](recovered_eqa_maps.json).
- [Interruption and renderer preflight receipt](reboot_recovery.json).
