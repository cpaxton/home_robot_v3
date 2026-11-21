# Mars near-limit arm staging

This simulator curriculum starts one arm joint 0.018–0.035 rad inside a
limit, moves it farther inward, and tracks a short wrist gesture with
base and gripper actions suppressed. The start and the full reference
use the intersection of the simulator limits and Herman's published
Innate OS 0.8.0 arm limits when the capture audit is supplied.

The teacher admitted 22 of 30 training episodes and 14 of 20 audit
episodes. The current long-arm specialist passed 7 of the 14
teacher-admitted audit cases with matching references. The bank contains
2,886 moving ticks across eight joint-and-side bins. The upper sides
of joints 2 and 3 had zero admitted training cases. Wheel lift was the
most common teacher failure across training and audit.

Sampling weights divide probability evenly among the occupied
joint-and-side bins. A 40-update smoke run loaded the bank through the
existing distillation trainer. Policy gains from retraining remain to
be measured.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python scripts/collect_mars_near_limits.py --output "$BANK" \
  --limits-audit "$HERMAN_AUDIT" --actor "$LONG_ARM_EXPERT"
python scripts/train_mars_distilled.py --work "$DISTILL_INPUT" \
  --output "$DISTILL_OUTPUT" --workspace-bank "$BANK/report.json"
```

A fresh passive hardware capture paired the version-specific depth
encoding with arm feedback. The recorded rest pose crossed the
published joint 3 limit by about one encoder count. The chassis proxy
reported overlap while the CAD hull pair had roughly 1.1 mm clearance.
Hardware staging remains unverified. [Results and checksums](receipt.json)
