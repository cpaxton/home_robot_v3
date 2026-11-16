# Unloaded grasp-pose stage readout

`results.json` is machine-generated compact evidence for the fixed twelve
teacher attempts on runtime `9cd439bf`. Both six-case stages return 5/6, so
stage admission and training remain closed. The
[report](../../../../docs/experiments/mars_unloaded_grasp_admission_20261116.md)
and [declared protocol](../../../../docs/plans/mars_unloaded_grasp_admission.md)
describe the original contracts and retained lift-planning failure.

The JSON binds full source manifests and raw rows/traces by SHA-256. It also
records exact parity on fourteen archived default-pregrasp fields and paired
trajectory prefixes. Raw observations/physics stay ignored; they are required
to rerun the readout, and this directory is not an independent CI benchmark.

From the producer checkout, using retained raw logs:

```bash
python scripts/summarize_mars_unloaded_primitives.py \
  --run logs/eval/mars-unloaded-grasp-admission-v1 \
  --baseline-summary PATH_TO_ARCHIVED_CURRENT_FEEDBACK_BASELINE/summary.json \
  --output FRESH_OUTPUT_DIRECTORY
```

The readout never advances physics. Its own script hash is in `results.json`;
the runtime/canonical-regression source is independently bound. Later software
validation belongs in a separate `validation.json`, preserving the original
teacher producer identity and results.
