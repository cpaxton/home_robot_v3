# Non-RL harness integration pilot

This branch combines #216, #217, #220, #221 and #223 with their reviewed fixes.
It excludes the separate #214/#224 RL/controller integration line. Main already
contains #213/#222 and #199/#202; those landed changes are shared by both arms.

Run `scripts/run_shared_harness_integration.py` through `emet jobs run
--cpu-safe --gpu-exclusive --need-mib 22000`. It requires clean frozen main and
candidate worktrees, existing ignored `sam2.1_hiera_small.pt` links, the Habitat
executable, a fresh output directory and a reachable Qwen OpenAI endpoint for
the existing privileged stationary TAMP control. All phases run serially.

Fixed population and gates:

1. Candidate contract tests: dialogue/audit, answer authority, current/pre-turn
   metadata, measured movement/gaze, visual consumers, floor hazards, metadata
   adapters, and public native TAMP API. Missing or skipped checks fail the gate.
2. Paired EQA: q2/6/12/14/15/16/25/28/31/56/65/68, seed 0, local Qwen3-VL-8B
   int4, same exploration budgets. Main uses its native legacy answer policy;
   candidate uses model_final. This is a complete-harness comparison, not a
   single-factor ablation. All 12 records per arm must be valid, with candidate
   accuracy at least main accuracy. Small development results are not paper
   estimates of generalization.
3. Paired S0, RoboCasa single-room and Molmo rby1-proxy FIND checks, same local
   model and budgets, native answer policy per arm. Both arms must complete;
   every candidate task must succeed. A shared baseline failure is reported,
   not grounds to waive a candidate failure or count infrastructure as a miss.
4. Candidate S0 FIND with opt-in floor evidence, explicit simulated Z=0 floor
   calibration. Require task success. Default floor configuration stays unchanged.
5. Paired scripted physical two-object controls and actual live `run agent`
   multi-goal TAMP using public `scene_tasks`, `plan_task`, `execute_task_plan`,
   invalid-plan rejection, independent region/support/settling scoring, and
   forbidden-command auditing. Same Caliban Qwen model endpoint on both arms.
   Require physical task success and compliant stop behavior, not a teleport
   or kinematic substitute. The fixture changes the red object's initial XML
   position to the existing depth-separated two-goal control before startup;
   no runtime teleport or new task tolerance is introduced.

The TAMP control is privileged and stationary, with no camera/perception claim.
It checks production CLI/tool/physics integration, not observed long-horizon
household competence. Floor contracts include synthetic/CPU ray controls, not
hardware cliff qualification. The rby1 simulator adapter remains a proxy.
EXPRESS/AbstainEQA remain metadata-only; their tests do not claim new benchmark
runtime support. These limits remain even if the integration gate is green.

Each phase retains its command, log and source SHA. `manifest.json`,
`progress.json`, contract JUnit, per-episode evidence and `gate.json` distinguish
missing/error results from task failure. A red gate blocks blanket merging:
inspect the first failing boundary, repair its owning slice, then freeze a fresh
revision. Do not edit a running source, retune thresholds to this population,
replace model answers, discard failed cases, or infer green from exit-zero alone.

After a green gate and review closeout, land in dependency order #216, #217,
#220, then #221/#223. Squash/history refreshes must preserve the tested tree;
any runtime change needs appropriate revalidation. Retain all failed artifacts.
