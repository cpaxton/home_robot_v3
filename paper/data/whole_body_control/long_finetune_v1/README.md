# Longer fine-tuning — complete

65,536 additional decisions, same objective/guards, lower LR, actual saved actor/
critic/Adam continuation checked. Reused development trades off accuracy/forces;
fresh confirmation improves 12/24 → 16/24 and safety flags 4 → 1, but peak jerk
increases. No promotion. Results preserve all physical case identities and
explicitly reused baseline rows; raw traces/weights remain ignored.
See [the report](../../../../docs/experiments/long_finetune_20261117.md).
