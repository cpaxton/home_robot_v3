# Integration task/view repairs

## Molmo

At `85291989` the bounded floor loop reduced unknown footprint cells to zero,
then completed a measured rotation and exploration. The task still failed.
Manual inspection of router images showed that the head remained aimed at the
floor, so the next task assessment received floor RGB despite restored mobility.

The agent-facing `inspect_floor` now preserves the prior measured gaze, performs
the bounded floor inspection, restores that gaze, verifies measured arrival and
a fresh frame, and maps a new task view. The underlying `observe_floor` remains
a downward-capture primitive so intermediate recovery projections stay valid.
CHAT and EQA use the same inspection wrapper. Failures to restore/capture are
reported explicitly; base drift rejects the operation. Floor/map evidence stays
in the result. No new navigation permission or semantic answer override is added.

## RoboCasa protocol v2

Keep the original `robocasa_pp_s1` and its failures unchanged. Its literal `jar`
does not match `obj_main` (`sugar_cube` in the tested seeded scene), and `cab`
is internal fixture shorthand. Simply renaming it to `cabinet` would silently
switch the legacy selected body from `cab_1` to `cab_main`.

The new `robocasa_pp_s1_v2` is an explicitly different, corrected protocol:

- The task-object noun comes from its declared class annotation using the
  existing `object: obj` resolution, yielding readable language such as
  `sugar cube`. This is task/question generation, not oracle perception.
- The public receptacle query is `cabinet`.
- All six actual cabinet fixtures are declared as acceptable targets before
  prediction. Their IDs and geometry stay scorer-side; the agent receives only
  task language. Missing target annotations invalidate the task rather than
  silently dropping a target or choosing an easier fallback.
- Scoring uses the same planar geometry distance and 0.50 m tolerance. Legacy
  single-ranked-target scoring remains unchanged for old episodes.

Both arms must use the identical versioned registry and evaluator semantics.
The frozen main-agent control receives only the shared evaluation-module and
registry patch; its inference, history, navigation and perception code remain
at `61da55cb`. Record its resulting SHA as main plus evaluation-protocol repair,
not unmodified main. The driver records and rechecks the shared registry hash.
Do not compare corrected scores to old invalid-case scores as an accuracy gain.

434 targeted floor, task-language/scoring, agentic query, and orchestration tests
pass before final checkpoint. New live paired checks are required. The full
integration gate is still red until those and the broader frozen battery pass.
