# Planned embodied-QA benchmark extensions

Status: requested next integrations, **not yet runnable adapters or results**.
Finish the [observation/movement cases](../testing/agent_observation_cases.md)
first. Reuse the same agent/provider/history/artifact contracts; adapters translate
episodes and scoring, not policies.

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
