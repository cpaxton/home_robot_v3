# Final answer authority audit — 2026-10-15

## Finding

The final model is not currently the semantic answer authority. There are three
replacement layers: graph-query postprocessing, agent submission/fallback, and
Habitat scoring. This makes scored accuracy an unreliable measure of final
context quality. A correct final answer can be discarded; a wrong/abstaining
model can also be credited with a correct heuristic answer.

Fixed-response replay of existing artifacts (no new inference):

| Frozen run | Arm | Scored correct | Raw final model correct |
| --- | --- | --- | --- |
| EQA-12 seed 0, repaired evidence profile | main | 9/12 | 5/12 |
| Same paired run | evidence profile | 8/12 | 8/12 |
| Targeted q14/q15/q65, seeds 1/2 | main | 5/6 | 5/6 |
| Same targeted repeats | evidence profile | 1/6 | 5/6 |

All responses in those rows parsed. Raw-model scoring preserves abstentions;
it does not select only confident outputs. The targeted set was selected from
observed disagreements: do not treat it as a representative benchmark score.
This counterfactual holds already-produced final responses fixed. It does not
estimate live changes to trajectories, stopping, prompt history or runtime.
Some correct raw answers still reason from missing evidence (e.g. q15); correct
choice alone is not proof of correct grounding.

Reproduce with `PYTHONPATH=src python scripts/audit_final_answer_authority.py`
and each run's `results.jsonl`:

- `/tmp/emet-reviewed-evidence-profile-eqa12-s0-20261015/`
- `/tmp/emet-reviewed-evidence-disagreements-s12-20261015/`

## Replacement boundaries

1. `graph_answer.query_answer`: salvage inference, location/memory/equipment
   answer substitutions, graph-label confidence gates, human-facing geometry
   rendering. Raw model response is captured before these transformations.
2. `answer._resolve_submit_answer_text`: confirmed assessment beats explicit
   final response, then router preference can beat final response.
3. `answer._forced_answer_fallback`: MCQ rotation voting, earlier/deferred
   assessment letters, deterministic uniform guesses; can replace an existing
   semantic model answer.
4. `runner.run_episode`: equipment substitution and MCQ voting can run again
   unless the summary's final decision is treated as authoritative.
5. Unknown-answer salvage counterfactuals invoke more inference, even though
   they do not change the scored answer. These must not be hidden final calls.

## Isolated repair

The `model_final` policy uses the existing decision-policy selector, not another
agent or reasoning stage. `configs/emet/model_final.yaml` differs from the
evidence-context control only in answer authority. The final query returns its
parsed answer directly, preserving semantic option text and abstentions.
Assessment still controls evidence acquisition; physical safety, freshness and
object-geometry admission remain independent and unchanged. No gold answer is
available to runtime decision-making. Legacy behavior remains reproducible for
the paired control; this is not a default flip or an accuracy acceptance claim.

Tests cover conflicting assessment/tool answers, budget exhaustion, Unknown,
empty outputs, literal count option `None`, model confidence, one inference call
and retained raw traces. The Habitat scorer treats this final decision as
authoritative rather than running another answer selector.

## Separate context defects

- Unverified submission picks a best assessed view for graph IDs but looks up
  retained image arrays using the confirmed ID (often None), losing its packet.
- Stored per-image labels are generic role labels, while assessment used the
  full capture-sequence and camera-pose labels. Final submission drops that
  provenance. Preserve the same packet/labels rather than asserting that its
  earlier answer must be correct.

Repair these independently after freezing the answer-authority comparison.
For the live pilot, compare legacy evidence profile and model-owned final answer
with identical Qwen weights, seed, navigation and image-selection settings.
Report raw/scored disagreement, abstentions, visual grounding and runtime as
well as accuracy. Do not tune answer rules to q14.
