# Moving base goals and native feedback diagnosis

See the [experiment report](../../../../docs/experiments/base_goal_replan_20261116.md)
for interpretation and the unchanged acceptance contracts.

- `results.json` retains complete case outcomes for the final 109-case global
  gain-zero readout, earlier gain-0.5 moving/viewer cases, paired failure
  replays, scalar-gain probes, the scoped 15-case diagnostic, the separate
  physical-stop diagnostic and fresh cold-viewer verification. Producer/source
  identities remain separate; deterministic repeats are not independent trials.
- `protocol.json` fixes the moving-goal audit, including requested updates.
- `moving-goals.svg` shows the rejected gain-zero candidate's reference/tracking
  and the original matched reverse failure with and without retargeting.
- `*.py.txt` preserves exact runners and the aggregation/logging scripts. These
  are evidence producers, not supported controller entrypoints.

Final runtime source is `13c53634901cb44b403b1d134d51b5095ecfaad1`.
The verified final source receipt contains **1,520 passing tests**, no
failures/errors/skips. Archive and receipt hashes are in the JSON. Raw trace
locators and checksums point to ignored local logs; weights, archives and
compressed traces are not committed. Offline W&B run: `5ew43iu9`.

Global gain zero is rejected because coupled/TAMP accuracy regresses. The
scoped per-axis/mask override passes 15/15 diagnostic cases, but has no supported
production contract and is not selectable or promoted. No training steps,
loaded-task qualification or hardware motion occur in this readout. Legacy
stop booleans remain untouched; the separately timed stop diagnostic confirms
both faulted reverse replays quiet after 0.15 s while retaining fault latches.
