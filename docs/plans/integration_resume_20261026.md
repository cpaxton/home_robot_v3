# Integration recovery after host reboot

The host reboot cleared `/tmp` before job `20261026_111539_0dc672` started its
repair/test phase. That job acquired the GPU lock but waited on memory from
the cancelled RoboCasa run; it did not apply the repair or produce a Molmo result.
All commits and experiment artifacts under `~/runs/emet` survived.

Persistent checkouts now live at `.worktrees/harness-integration` and
`.worktrees/harness-baseline` under `/home/cpaxton/src/home_robot_v3`.
The user's dirty detached root checkout remains untouched. The detached repair
commit `c5b71091` is pinned by `recovery/molmo-floor-capability` and its exact
code is restored to the integration branch in `31ca2c88`. The baseline remains
`61da55cb`; no main or owning-PR merge is part of this recovery.

Preserve these results and limits:

- Original combined job `20261026_090114_e985ce`: EQA 9/12 candidate vs 7/12 main;
  272 contracts pass without skips; S0 and floor-on S0 pass. Full gate remains red.
- Setup retry `20261026_104107_cb9eb0`: both corrected live public-tool native
  TAMP trials pass with zero forbidden commands. Its RoboCasa portion was
  cancelled after discovering an invalid task definition, not scored as a miss.
- The RoboCasa registry asks for `jar`, while read-only metadata from the actual
  `obj_main` gives `sugar_cube`. It also asks for `cab`. Read-only scoring checks
  selected `cab_1` for `cab`, but `cab_main` for `cabinet`; simply rewriting the
  word would change the target. Repair and version the task definition explicitly,
  preserving the invalid original and avoiding a post-hoc scoring substitution.
- Molmo's seven unknown footprint cells remain the outstanding real navigation
  failure. The capability-based repair is unit-tested but still needs a fresh
  live test, followed by full exact-source integration requalification.

Next: run focused Molmo on the recovered branch. No automatic full-suite rerun
or blanket merge while the RoboCasa task-definition mismatch remains unresolved.
The integration driver includes every changed test file and a per-PR coverage
matrix; this is scoped integration coverage, not every-world/hardware proof.
