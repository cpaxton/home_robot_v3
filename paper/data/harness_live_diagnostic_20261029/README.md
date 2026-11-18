# Live authored diagnostic, not a qualified benchmark result

Source `f268d59442ae5b4f48b82d9f4215e04da4d24867`, managed job
`20261029_010733_da5fb2`; raw artifacts:
`~/runs/emet/capability-live-pair-v4-20261029/`.
Qwen/Qwen3-VL-8B-Instruct is remote; endpoint precision is not established.

Both three-turn protocols completed. Manual image/response audit confirms a
green-box answer in the seen case and abstention in the absent case. However,
both requested turns failed before execution: the initial footprint was unknown
and targeted floor inspection returned `floor_target_unobservable`. Measured
base yaw stayed within 0.0001 degrees of its initial value. The seen target was
still visible in the final frame. Thus neither observe-turn-recall task passes,
and this is not evidence of successful recall after turning or navigation.
The traces show correct safe rejection, actual image/history delivery, and a
remaining sensing/recovery gap. No answer is replaced by the evaluator.

The earlier brown-box-shaped support trace is retained separately and excluded
from scoring because the singular box referent was ambiguous. Version 2 uses a
cylindrical support. These are inspected development cases, not held-out data.

Reproduce the documentary panel (CPU, no inference):

```bash
PYTHONPATH=src .venv/bin/python scripts/render_live_capability_panel.py \
  --run ~/runs/emet/capability-live-pair-v4-20261029 \
  --output /tmp/emet-live-capability-panel
```

`panel_inputs.json` contains measured poses, final raw-message references and
input hashes. `audit.json` records the manual outcome/qualification distinction.
The figure is `paper/figs/harness_live_evidence_and_motion.png`; it is a
diagnostic asset, not a manuscript headline result. Full learned localization,
physical manipulation and the other ten live cases remain unqualified.
