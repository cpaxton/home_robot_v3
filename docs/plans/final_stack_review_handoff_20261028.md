# Final shared-harness stack review handoff

## Objective and canonical review target

Canonical final review: [PR #225](https://github.com/cpaxton/home_robot_v3/pull/225).
This consolidates the five component PRs and their integration-only repairs.

Demonstrate useful capability/contract improvements over main without a material
regression on shared tasks. General reliability, hardware qualification and
paper-scale superiority are not this merge gate. Do not override model answers,
relax geometry/clearance checks, or silently remove a candidate-only failure.

Canonical combined runtime revision:
`7391b5220a7bb2897c19e81bea6f521f82ed542a` on
`test/shared-harness-integration-20261026`.
Main: `61da55cb1c237b456b141d9572aec76cd44a6805` (verified on GitHub 2026-10-28).
Compare with `git diff 61da55cb..7391b522`: 100 files, 4850 additions, 169 deletions.
The later handoff commit changes documentation only. Review the final PR head
as well as this pinned runtime; do not assume the component heads include all repairs.

Frozen candidate checkout:
`/home/cpaxton/src/home_robot_v3/.worktrees/harness-final-review-7391b522`.
Editable integration checkout:
`/home/cpaxton/src/home_robot_v3/.worktrees/harness-integration`.
Do not edit the frozen checkout or the dirty root checkout.

## Component stack (GitHub head snapshot)

| PR | Scope | Base | Pinned head | Draft |
|---|---|---|---|---|
| [#216](https://github.com/cpaxton/home_robot_v3/pull/216) | Final model answer authority | main | `0e97312b90915e9cd359cb94e351bb02f0f93277` | no |
| [#217](https://github.com/cpaxton/home_robot_v3/pull/217) | Episode dialogue, argument validation, request audit | #216 | `4628a7a11a442f30556a75da8b702bec38a26bd1` | no |
| [#220](https://github.com/cpaxton/home_robot_v3/pull/220) | Measured motion/gaze, floor recovery, visual/retrieval separation | #217 | `511141c8cee51f083661fc86c20856bdf21c9d0a` | yes |
| [#221](https://github.com/cpaxton/home_robot_v3/pull/221) | EXPRESS/AbstainEQA metadata adapters only | #220 | `c98c6b04758bf523f7320dda07fdc7c4a3a1b0bc` | yes |
| [#223](https://github.com/cpaxton/home_robot_v3/pull/223) | Floor hypotheses and terrain hazard preservation | #220 | `ca926c244687a37b6f29ab73cf290187e95ecc7f` | yes |

Topology: `main -> 216 -> 217 -> 220 -> {221,223}`. #214 and #224 are a
separate RL/Mars line and are excluded. #199/#202 and the lower navigation/TAMP
slices have already landed in main. Do not reopen or re-merge those.

GitHub reports zero unresolved inline review threads on these five PRs.
Review summaries contain non-inline findings; zero threads is not proof of
review approval. Prior fixes include role/history validation, audit collisions,
motion-envelope normalization, no-op budgets, rotation bounds, terrain scatter
bounds, immutable floor layers, and metadata ID/evidence handling. Verify those
against the pinned heads rather than approving solely from this summary.

## Integration-only repairs requiring final review

These are not all present in the five component heads:

- `31ca2c88`, `85291989`, `1854ccbc`: advertised head limits, remaining-cell
  floor closure, restoration of a measured fresh task-facing view.
- `15477c95`, `a0361d29`, `cdd4591c`, `13852b19`: versioned scorer-only
  RoboCasa protocol and retained invalid/stress cases; sugar cube excluded at
  explicit user request, not relabeled as a pass. RoboCasa replacement remains TODO.
- `a09d04a1`: unverified proposal depth-support diagnostics, not admission.
- `3916b708`, `7391b522`: truthful historical frontier/scan context and
  measured camera-motion feedback. No forced recovery or semantic override.
- `35bdab14`, `3dda1fec`, `835028dc`, `bf7faccd`: frozen integration tooling,
  native task grammar/assets, corrected obsolete 360-degree test expectation,
  explicit Molmo interpreter preflight. Other suffix commits are reports/docs.

Prefer a consolidated final PR of the tested combined tree, with links to the
five component PRs. This avoids merging older component heads while omitting
the live-tested repairs. Keep originals open until final checks/review pass;
then merge the consolidated tree and close originals as superseded with links.
If landing separately instead, port repairs to their owners, refresh descendants,
and verify runtime-tree equivalence before relying on these results.

## Evidence already available

- Contract checkpoint `835028dc`: 976 passed, zero skips/errors/failures.
  Camera-feedback targeted suite at `7391b522`: 70 passed; hooks pass.
- Full prior paired EQA-12: candidate 9/12, main control 8/12. Small pilot,
  not statistically established superiority.
- Scripted and live public-tool TAMP controls pass both arms. Privileged,
  stationary, no learned-perception manipulation claim. No new TAMP runtime
  code since that pilot; final source review must check this boundary.
- Latest S0 pair `s0-camera-feedback-pair-20261028`: both targets grounded in
  both arms. Candidate 479.1 s vs main 979.3 s; receptacle 1 vs 3 rounds.
  Candidate did not call look-around: this does NOT establish that the new
  camera feedback caused recovery. Previous valid S0 candidate failures remain.
- Paired Molmo: candidate bowl localized, microwave missed; main misses both.
  Microwave support around 3.36 m is outside the unchanged 2.5 m grounding
  limit. Do not turn this known limitation into an unconditional merge blocker.

Control source is `a2be9d40a53a9081e204083fe1f437cb0bf42a7f`, main agent plus
the shared task/evaluation-protocol patch. It is not unmodified main. Both arms
use local Qwen3-VL 8B int4 and matched task budgets; native answer policies
differ (legacy vs model_final), so these compare complete systems, not one factor.

## Final checks now queued/running

Job `20261028_103738_4eb9be`, name `final-stack-review-20261028`.
Driver: `.worktrees/run-final-stack-review-20261028.py` (copied/hashed in output).
Output: `~/runs/emet/final-stack-review-20261028/`.
Log: `~/runs/emet/jobs_runs/final-stack-review-20261028/job.log`.

Serial exclusive-GPU phases: fresh combined contracts, paired EQA-12 seed 0,
paired S0 seeds 1 and 2. Seed-0 S0 evidence above is retained separately.
Read `manifest.json`, `progress.json`, `contracts.xml`, per-phase logs,
`s0-seed*/results.jsonl`, and final `report.json`. Driver completion alone is
not acceptance; the report deliberately performs no approval or merge.

Review criteria: valid complete artifacts; no contract failures/skips; assess
paired EQA delta and each S0 arm's grounded outcomes and runtime, retaining
candidate-only losses. If mixed, inspect actual packets/receipts before
claiming noise or a fix. Absolute perfection is not required, but a new material
regression must be understood rather than hidden by aggregate phase scores.

## Finish and approve

Final results and review (2026-10-28): 990/990 contracts pass with zero skips
or errors. Paired EQA is candidate 8/12 versus control 7/12 (candidate-only wins
q65/q68, loss q25). On S0 seed 1 both ground the cylinder and miss the cube:
main 287.2 s, candidate 798.6 s. Seed 2 succeeds in both arms: main 790.4 s,
candidate 478.0 s. Both arms produce complete valid artifacts. The separate
seed-0 pair succeeds in both arms (main 979.3 s, candidate 479.1 s).
There is no consistent per-run runtime gain; prolonged unsuccessful search is
a recorded candidate cost, not an excluded result. These small pilots support
tooling/contract improvements with comparable task outcomes, not statistical
superiority or general reliability.

The consolidated review at #225 reports no blocking code findings and recommends
landing with these results and the seed-1 runtime regression documented. The
component PR comments redirect closeout here without additional blockers. No
eligible GitHub approval is recorded: the reviewing account is the PR author.
The following closeout checklist remains conditional on the final GitHub state.

1. Read final results and visually audit any candidate-only failure and a win.
2. Review integration-only repairs, current review summaries, safety defaults,
   truthful evidence roles and final-model authority.
3. Pin approval/evidence to the final consolidated PR SHA. If GitHub refuses
   self-approval, record that fact; never claim an approval that was not posted.
4. If checks and review support the comparative objective, mark ready, approve
   through an eligible reviewer and merge to main through GitHub. Never push main.
5. Close superseded component PRs with the consolidation link and update TODO/docs.
   New runtime edits require new affected checks; documentation-only changes do not.

Do not run simultaneous heavy experiments, stop unrelated processes, expose GT
annotations to the model, or describe metadata adapters as evaluated benchmarks.
No real-robot test or hardware safety acceptance is included here.
