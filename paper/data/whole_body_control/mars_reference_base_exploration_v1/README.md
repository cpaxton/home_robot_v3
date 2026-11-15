# Reference-conditioned base exploration evidence

Read the [experiment report](../../../../docs/experiments/mars_reference_base_exploration_20261115.md)
for the conclusions and tables. This directory preserves the machine-readable
evidence behind those tables, including negative outcomes.

| File | Purpose |
| --- | --- |
| `results.json` | All 48 fixed evaluation records, initial-pair checks, training updates/exposure, recipes and original artifact hashes. |
| `protocol.json` | Immutable prelaunch budget, source, resources and supervision identity. |
| `historical-default-replay.json` | Fresh default-control equality audit against the earlier C1 run. |
| `jerk-audit.json` | Reconstruction of the unchanged acceptance jerk from saved base poses. |
| `signed-jerk-audit.json` | Nine analytic controls and the separate signed/unsigned displacement comparison. |
| `review-validation.json` | Separate 50-focused/1,262-full validation for the later review fixes; the experiment's original producer remains unchanged. |
| `*.py.txt` | Exact executed supervision, extraction and read-only audit sources, stored as text rather than portable production tools. |

The JSON is intentionally larger than the human report: retaining every case
and update makes the result auditable without hiding failed cases behind group
averages. There are twelve distinct evaluation cases reused for initial/final
readouts of two trained actors, not 48 independent trials. No reliability
interval is inferred from that repetition. Detailed physics traces, policy
weights and optimizer states remain in ignored artifact storage.

The original training implementation is `3ac722e1`; later documentation/review
changes do not rename that producer. Hashes in the extraction record bind the
original local artifacts. To rerun an extraction, copy the archived text script
to a scratch artifact directory, restore its `.py` name and supply the matching
original inputs/protocol there. Do not run it in this committed archive: the
script writes its output next to itself.

The signed-displacement audit is diagnostic and does not substitute a new gate
or change any original qualification label. Native aligned/substep probes have
their own producer identity and report separately from this training archive.
