# Mars shared expert

This simulator experiment trains an actor across base tracking, arm tracking,
coupled tracking, loaded carry, unloaded TAMP approach, lift, and preplace.
The starting policy is the long-arm expert. The data include successful
demonstrations and recoverable learner-state corrections. Tracking, grip, and
motion limits remain the responsibility of the task executor.

Each optimizer update draws four families without replacement and samples
32 actions per family. The loss uses the action channels the executor delegates
for that family. After bootstrapping, each family with corrections uses those
samples with probability 0.35 and otherwise uses the original offline pool.
The current protocol keeps the gripper head frozen.

## Local or GPU training

Use a work directory with `initializer/{manifest.json,policy.zip}` and
`data/{base,arm,coupled,carry,tamp,lift,preplace}.npz`. Each NPZ must contain
finite `history`, `preview`, `motion`, and `action` arrays in the Mars
actor observation and action layout. Successful learner-state corrections
can arrive as atomic `data/online-<family>.npz` files after bootstrapping.
Simulator audit data stay outside these files.

```sh
PYTHONPATH=src python scripts/train_mars_distilled.py \
  --work "$DISTILL_INPUT" --output "$DISTILL_OUTPUT" --device auto \
  --bootstrap 6000 --updates 54000 --hours 4
```

Use `--device cpu` or `--device cuda` to choose a device explicitly. The output
contains an atomically written progress receipt, optimizer recovery state,
and actor-only checkpoints. Resume from the same output directory with matching
inputs and an identical update schedule.

## Evaluation

Score closed-loop simulator outcomes separately for each family. For the
initial screen, the [case manifest](cases.json) declares two audit cases per
family. Reserve these cases from training. Split them across workers with
`--families` if needed. The summary requires complete audit shards bound to
the same actor weights. It checks case uniqueness and verifies stop evidence
on reported passes.

```sh
PYTHONPATH=src python scripts/eval_mars_distilled.py \
  --actor "$ACTOR" --cases paper/data/whole_body_control/mars_shared_expert_v1/cases.json \
  --output "$AUDIT_SHARD"

PYTHONPATH=src python scripts/summarize_mars_family_audits.py \
  --output "$AUDIT_OUTPUT" "$AUDIT_SHARD"
```

The screen is developmental. Broader motion, recovery, and hardware qualification
use independent evidence.

## Initial results

Caliban trained for 54,000 updates. The workstation CPU trained for 6,000 and
its RTX 4090 trained for 12,000. All three received demonstrations and
learner-state corrections for seven families. The run budgets vary.
The [training receipt](training-receipt.json) records the input and checkpoint
digests. See the [Caliban screen](caliban-audit.json),
[CPU screen](local-audit.json), and [RTX 4090 screen](workstation-audit.json)
for per-case outcomes. The RTX run uses the evaluator's stationary-base
lift route and completed 12 of 14 cases.

Caliban completed 12 of 14 cases in the original screen. It missed one
coupled hand path and violated the fixed-base lift envelope once. Applying
the lift executor's arm-only action contract produced two successful lift
replays with the same weights. The evaluator now zeroes learned base
commands for supported lift. The base servo retains its own feedback.

On the [expanded cases](extended-cases.json), the Caliban actor passed 4/4 arm
and 4/4 preplace cases. It passed 3/4 base and 3/8 coupled cases. Base wheel lift
and coupled contact still prevent promotion of those routes. Keep the existing
base specialist and treat the shared actor as a simulator candidate for the
families with passing evidence. [Expanded results](caliban-expanded-audit.json)

The RTX 4090 actor reached 5/8 on the same extended coupled suite. It timed out
on one case, exceeded the path tolerance on one, and hit the recurring unsafe
contact on the final case. [RTX 4090 expanded results](workstation-expanded-audit.json)
