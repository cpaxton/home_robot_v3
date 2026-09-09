# PR #161 reconciliation into the shared-agent stack

Compared #161 at `47a30ed9` with #162 at `44c7710e` and the integrated
navigation stack at `75c3ced1`. No missing production change was identified;
do not cherry-pick the old commits over their replacements.

| #161 contribution | Destination and evidence |
| --- | --- |
| `592eb8a4`: fusion gates, labels/synonyms, keep/blending, growth, SigLIP appearance | Already an ancestor of #162. Fusion configuration, attach, matching, docs and appendix retained. |
| `e986e4b6`: cap the VLM-label ingestion path | #162 `dynamem_graph_hooks.py` retains the cap and adds explicit view/arrival evidence modes. |
| `fb2e58a6`: cap instance-memory fallback | Same hook retains the guard and additionally honors the instance master switch. |
| `4970c495`: disabling instance nodes must disable eval fusion | #162 `apply_eval_graph_fusion_parameters` retains this behavior; backend opt-out enforcement is stronger. |
| `47a30ed9`: commit lazy graph evidence on HM-EQA navigation arrival | #162 `f276e52f` retains the controller hook and implementation; #163 subsequently refines evidence semantics. |
| A/B configs, sweep helpers, fusion docs and paper appendix | Retained in #162; additional paper matrix and voxel baseline tooling are deliberate extensions. |

The foundation also enforces the cap inside direct `apply_detection`, not only
at stream entry. It permits merging existing objects at capacity and records
creation/merge/rejection counters. `test_graph_object_fusion_policy.py` adds
fallback opt-out coverage and strengthens the cap test from checking a setting
to checking the resulting object count.

The old controller-hook test was removed in #162; its assertion that every
arrival produces an object node is not a valid acceptance criterion for the
later view-evidence design. Keep testing arrival evidence without restoring
the assumption that a navigation target localizes every visible label.

Review ownership: fusion/ingestion/configuration in #162; query grounding and
evidence lifecycle in #163; transport/navigation in #164–166. Paper appendix
content is retained for review, not endorsed as a newly validated empirical
claim. Bounded graph growth alone does not establish retained recall or answer
quality. #161 can be closed as superseded, not independently merged.
