# Hand-pose priority — complete single-factor comparison

Source `9d0620f9`, 1,585 verified regressions. Only hand_pose_weight changes
1 → 4; the completed same-source ordinary control is explicitly reused.
Initial policy/first-observation/first-action pairing passes before updates.
Both final checkpoints have 16,384 decisions. Ordinary / priority: 20/24 /
19/24 new cases, 7/10 / 7/10 existing coupled, 6/6 unloaded TAMP each. Same
three contact-failure identities; no evaluation wheel lift/invalid physics.
Worse worst hand error/contact despite lower yaw jerk; no promotion.

`record_results.py` produces compact complete case identities/physical metrics;
`plot_comparison.py` produces paired-outcomes.svg with explicit active-hand
selection for endpoint error and all tracking cases for jerk. Raw checkpoints
and traces remain ignored. Restored source/archive metadata is preserved. See
[the completed report](../../../../docs/experiments/hand_priority_training_20261117.md).
