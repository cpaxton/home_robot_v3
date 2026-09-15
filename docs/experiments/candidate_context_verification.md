# Fixed-candidate context and target-blind verification

Follow-up to the [box ablations](grounding_verification_ablation.md). Test only
final selection: reuse the exact saved measured candidates from both Qwen-box
RGB-D proposals and YOLOE proposals. No new detector, localization, GT, navigation,
or robot calls enter inference. Frames without candidates remain abstentions.

Source `4e1fdc7f`, serial job `20260910_080522_7e8d24`. Local
Qwen3-VL-8B-Instruct int4, 512 new-token budget per call. All calls reset model
conversation context; the blind identity call cannot inherit the requested target.

- **Isolated control:** saved original selection prompt and masked panels,
  rerun with the same system prompt and token budget as the new variants.
- **Context:** original image, plus a context crop and an isolated support panel
  per candidate. Context uses one candidate extent of padding (minimum 32 px),
  keeps RGB intact except a thin yellow support boundary, and explicitly forbids
  treating nearby objects as evidence that the support belongs to the target.
- **Blind context:** first identify every candidate without the query, detector
  scores, labels, or earlier model reasoning. A separate text-only call matches
  those identities against the request. Ambiguous/mixed identities are excluded
  in code even if the matching call selects them. This tests a two-stage decision
  policy, not a pure prompt-only ablation of the context variant.

The recognition call does not receive the target, but the upstream proposals may
be query-conditioned. "Target-blind" does not mean the entire pipeline is blind.
Same-model judgments remain correlated, not independent statistical verification.

Outputs:

- `/home/cpaxton/runs/emet/candidate-context-detector-20260910`
- `/home/cpaxton/runs/emet/candidate-context-vlm-20260910`

Each has `isolated`, `context`, and `blind_context` directories with exact prompts,
responses, per-call time, rendered panels, selected masks, and results. Source RGB
paths and original serialized masks are retained. Score with
`scripts/score_grounding_dataset.py` and the previous cache's `truth.json`.
Latency here excludes original proposal creation/localization.

Supplementary near/high-angle controls are defined in
`configs/ovmm/grounding_clear_controls.yaml` and captured serially after inference,
job `20260910_080618_022b49`. They reuse saved scene XMLs and object placements;
they are oracle-aimed perception fixtures, not navigation evidence or a new
held-out split. Visibility still must be measured rather than assumed from the
filename. Original difficult views remain in the paired evaluation.

Production behavior/defaults remain unchanged. Do not promote a candidate based
on rejection rate alone: report retained correct surfaces, contamination, wrong
identity, missed proposals and runtime separately.

## Completed results

All jobs completed. Each source replays 40 rows through three variants (240 row/
variant evaluations total), but only 13 detector rows and 10 Qwen-box rows have
candidates and therefore invoke the verifier. This is not 240 independent scenes.
The seven visible held-out targets have no cached proposals, so these replays
cannot improve their outcome; all remain rejected. No new holdout success is claimed.

| Fixed proposal source | Verifier | >=95%-pure selections | Accepted below 95% purity |
| --- | --- | --- | --- |
| YOLOE | Matched isolated control | 5 | 8 |
| YOLOE | Context | 5 | 6 |
| YOLOE | Blind context | 5 | 5 |
| Qwen box + RGB-D components | Matched isolated control | 5 | 5 |
| Qwen box + RGB-D components | Context | 5 | 5 |
| Qwen box + RGB-D components | Blind context | 0 | 10 |

The matched isolated control differs slightly from the preceding experiment:
this run uses the common inspection system prompt and 512-token budget. Compare
the new variants against this control, not the previous 192-token results.
The below-purity column includes both background contamination and wrong identity.

**Known mug failure:** detector row 14 is accepted by isolated control, rejected
by context and blind context. Context notes that leaf occlusion makes the identity
ambiguous. Blind recognition calls the candidate a "soap dispenser", then rejects
it as inconsistent with paper towel. The latter is safe rejection through an
incorrect identity, not evidence it correctly recognizes the mug. Both new
variants retain the five pure-surface successes on detector proposals.

**Blind regression:** on Qwen-box RGB-D candidates, blind recognition frequently
labels neighboring background support as the object visible in its context crop.
For example row 12 assigns "paper towel roll" to candidate 0, a wall/counter patch
with zero target overlap, while also assigning that identity to actual target
patches. The text-only matcher selects candidate 0. Saved panels and serialized IDs
are consistent with each other; model descriptions do not reliably bind identity
to support. See `blind_context/12-panel-0.png`, `12-panel-1.png`, and the raw JSON.
This rules out promoting this batch blind-identification policy as-is.

Context-only is the more promising of these two changes: it rejects the known
wrong-object detector case without losing pure successes, and does not regress
the Qwen-box cases in this pilot. It still accepts contaminated surfaces, so this
is not an OVMM final-acceptance solution. Blind identification needs a different
candidate-to-description binding strategy before further use; simply inserting
its descriptions into a text matcher is unsafe in this setting.

The supplementary cache is at
`/home/cpaxton/runs/emet/grounding-clear-controls-20260910`: 20 views, ten with
visible target pixels, ten still blocked. Bottle, bowl, sponge, paper towel,
sugar cube, broccoli, turmeric and pickle slice have at least one visible view;
cabinet bell pepper and lime remain fully hidden. These controls were captured
and inspected, not yet run through model inference. Preserve those failures.

Focused tests: 34 passed. Production defaults and action tolerances are unchanged.
Next: evaluate context-only on the new acquisition controls, then address remaining
support contamination and require fresh multi-view evidence before a bounded
integrated find/OVMM pilot. No new OVMM/task success is claimed here.
