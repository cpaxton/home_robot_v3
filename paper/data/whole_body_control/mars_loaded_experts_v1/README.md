# Mars loaded expert curriculum

The simulator carry expert is initialized from a fitted policy for nominal 20–50 g
translation and reorientation. Carry cases begin from a physically established
grasp. The audit checks motion completion, object retention, a confirmed stop,
and post-stop settling.

| Stage | Supervised updates | Held-out ordinary carry | Other audit |
| --- | ---: | ---: | --- |
| Initializer | — | 16/24 | — |
| Diverse nominal | 3,000 | 24/24 | 17/36 on the broader suite |
| Heavier payloads and cylinders | 2,000 | 24/24 | 17/36 broader, 6/6 wider |
| Wider boxes and cylinders | 1,500 | 24/24 | 6/6 wider |

The nominal fit uses 10,000 samples from 95 demonstrations. The second fit
samples stress data with probability 0.20 alongside nominal replay. The final
fit samples nominal, stress, and width data with probabilities 0.75, 0.15,
and 0.10. Stress cases include loads drawn up to 180 g. The width audit covers
22, 28, and 34 mm boxes and cylinders. Failed demonstrations contribute zero
training actions. Training and audit seeds are disjoint.

Both policies in the broader audit matched the teacher's 17 completions. The
shared fixture rejected 17 cases before any policy command, including 12
centered-pinch clearance failures on the native table assets. Two heavy
reorientation cases lost retention. The separate full-task preplace fit
completed 0/4 against 4/4 for the teacher. Its executor retained control of
the base, grip, and phases outside preplace.

[Results](results.json) records individual failures, policy and dataset
digests, and experiment source manifests. Raw simulator traces, datasets and
model weights remain in the local artifact directory
`logs/recovery/expert-progression-v1`. Results were produced with a frozen
simulator runtime and are scoped to the tested simulator conditions.
