# Planned embodied-QA benchmark extensions

Metadata review closeout (2026-10-25): EXPRESS accepts nonnegative integer IDs
or decimal strings and preserves string spelling, including leading zeroes.
Duplicate normalized string IDs are rejected. AbstainEQA categories belong to
each variant and may coincide or differ; equality is not a malformed evidence
pair. Pilot stratification uses the unanswerable variant's category, while
question/reference/category fields remain isolated from model input.

Status: metadata ingestion adapters implemented in
`emet.eval.embodied_qa_datasets`; **no qualified active runner or scored result**.
Finish the [observation/movement cases](../testing/agent_observation_cases.md)
first. Reuse the same agent/provider/history/artifact contracts; adapters translate
episodes and scoring, not policies.

## Pinned metadata audit — 2026-10-23

- EXPRESS upstream `HCPLab-SYSU/EXPRESS-Bench` commit
  `e8789dadfbd51af6850dc4d2b0526ca0ee78386a`: all 2,044 records parsed;
  deterministic internal-development selection is 14 examples (two/category).
  `data/express-bench.json` SHA256:
  `977ced67fe7b72462745e7de5c52aeff0e6128dd1f8f85d14596953c95d3c59e`.
- AbstainEQA upstream `gibrantaowu/AbstainEQA` commit
  `b970eb97aa6afce7020a0d57baaef79c53e05303`: all 1,636 answerable and
  1,636 unanswerable EM-EQA records parsed with matching pair IDs/evidence.
  `data/QA/emeqa_answerable.json` SHA256:
  `82ba3310a55d03350b58d41abafb0e9e69010c4115de7ef0027487b6c007fb81`;
  `emeqa_unanswerable.json` SHA256:
  `4018cc954a0a72a6ea2870e9f9231c74a568f012a6e9cf1720f5c75270f8ca2b`.
- One official unanswerable category is a non-string NaN. Preserve that record
  as `unlabeled`; exclude it only from category-balanced development selection,
  never silently from full evaluation. The selected pilot has ten pairs.
- `model_input()` exposes only question text, never answerability variant,
  category, reference answer, oracle actions or goal pose. AbstainEQA frame
  indices are retained for its evidence loader. EXPRESS evidence frames are
  not yet parsed; its frame_indices remains empty. Active start-pose binding
  remains separate.
- Eleven fixture tests cover reference isolation, identity, duplicate IDs,
  mismatched evidence, empty input and the upstream unlabeled-category case.
  Run `PYTHONPATH=src .venv/bin/python -m pytest -q src/test/eval/test_embodied_qa_datasets.py`.
- No official data/code is vendored. AbstainEQA's repository declares MIT;
  EXPRESS redistribution terms still require verification. Large frame packs,
  scene downloads, cloud judges and full sweeps have not been run.

AbstainEQA question/category equality is deliberately not required across a
pair: the original question has an OpenEQA task category, while the altered
question has an abstention category. Pilot stratification uses the latter and
keeps both variants. Pair identity and evidence must still match. Its
`episode_history` is retained as an opaque `evidence_id`; normalized `scene`
is None until an explicit scene resolver is qualified. Do not use evidence IDs
as scene-disjoint split keys. EXPRESS retains its explicit scene_id as scene.
Mixed-benchmark pilot input is rejected rather than guessing variant policy.

Metadata audit is not benchmark acceptance: still required are evidence/scene
resolution, shared-model execution, official judge/scorer wiring, failure
handling and manually audited prediction pilots. Supplied trajectories must
not be reported as autonomous exploration.

## EXPRESS-Bench

[Official project](https://hcplab-sysu.github.io/EXPRESS-Bench/) describes
777 trajectories, 2,044 question–trajectory pairs and Exploration–Answer
Consistency (EAC), useful for evaluating whether answers have observed support.

Acceptance sequence:

1. Inspect official code/data/license, scene dependencies, splits and evaluator;
   pin versions/hashes. Supplied-trajectory replay is not active exploration.
2. Add episode/input and official scoring adapters; retain question IDs/raw
   answers. Evaluation annotations must never enter model context.
3. Run a tiny fixed category-balanced development pilot and manually inspect
   images/context/scoring. Report answer quality and EAC separately.
4. Qualify active runtime bindings separately, then use bounded serial pilots.

## AbstainEQA

[Official project](https://abstaineqa.github.io/) describes 1,636 abstention
cases paired with 1,636 original OpenEQA instances, including underspecified
questions and unavailable information.

Acceptance sequence:

1. Inspect official dataset/evaluator release and license; preserve paired IDs,
   splits, evidence and ambiguity categories.
2. Use the existing final-model answer path without semantic answer overrides.
   Retain abstention and clarification outputs, not only correctness.
3. Run paired answerable/abstention examples with official metrics. Report false
   confident answers and unnecessary abstentions separately. Always abstaining
   is not a useful general agent baseline.
4. Qualify episodic-memory input first; active environment bindings separately.

Both adapters need tests for missing assets, empty output, evaluator failure and
ID alignment before a scored pilot. Neither replaces OVMM or TAMP acceptance.
