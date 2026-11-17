# Longer lower-rate checkpoint fine-tuning

65,536 additional decisions from the completed ordinary checkpoint; actor,
critics and Adam restored, explicit fresh physics/RNG reset. Source 9d0620f9,
1,585-test receipt reverified, same reward/exploration/goals/guards. One native
CPU14 worker, no hardware or promotion. Status is an early snapshot, not a final
performance claim. Raw checkpoints/traces stay ignored. See the
[protocol](../../../../docs/experiments/long_finetune_20261117.md).
