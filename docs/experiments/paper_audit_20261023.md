# Paper and agent appendix audit, October 23

Entry: `paper/main.tex`, on `feat/tamp-observed-integration` after source
`fb0fb6cc`. Review covers the main narrative, related work, method, experiment
scope/results and agent/native/platform/model appendices. Runtime code is
unchanged by this paper follow-up. Historical result datasets remain intact;
new repeat outcomes are a separately identified dataset.

## Implemented corrections

| Area | Correction |
| --- | --- |
| Agent architecture | Add Appendix M (`13_combined_agent.tex`) and wire it into main. Distinguish scored EQA_EPISODE, router-off deterministic selection, router-on proposal, and production CHAT. Explain skill/tool groups, provider admission, one-shot handles, measured effects, protected goals and stop boundaries. |
| Budgets and deployment | Distinguish the three-round CHAT default, 1–32 CLI range, 12-round native controls, possible summary call, bounded search/IK and the unwired shared whole-agent ledger. Both remote clients keep telemetry live. Normal CHAT defaults to Dynagraph; camera-free TAMP explicitly uses DynaMem. |
| Current TAMP | Replace the conclusion's old distance-policy Molmo prototype framing with guarded multi-step native tools. Mark contracts/runtime as implemented and suite qualification/observed execution as incomplete. Preserve assisted/native/observed/hardware distinctions. |
| Freshness and absence | Clarify graph age versus sensor-based voxel clearing. Aging a hypothesis does not prove that an occluded object disappeared or certify free space. Graph observation deletion is distinct from retention of the sensor frame log. |
| Historical/current evidence | Remove the obsolete claim that the August 113-question run is still being collected. Preserve its historical source/selection limitations. Mark newer EQA state/progress mechanisms as implementation descriptions, not automatically enabled in old scored rows. |
| Model arithmetic | 2/6 → 5/6 is +50 percentage points, not +50% relative. Keep the underlying table unchanged. |
| Precision attribution | Local int4 versus remote Orin fp16 changes hardware/serving/transport as well as precision. Replace precision-only causal claims. Explain the archived int4 count/clock 6/15 and later 4/15 are distinct runs; only 4/15 + 5/15 yields the recorded 9/30. |
| Cross-policy evidence | Preserve balanced-32 as a cross-policy composite and its 11–16/32 variation; remove wording implying all revisions reliably attain 16/32. No historical score or p-value is recomputed. |
| Embodiments | Remove the incorrect “Stretch-in-Habitat” attribution, distinguish shared protocol from controller/sensor equivalence, add Sourccey profile with explicit lack of native TAMP qualification. |
| Related-work focus | Add prior task-planning work (ReAct, Inner Monologue, SayCan, PDDLStream), and distinguish those methods from our implementations. Add source-backed citations for GraphPad, DREAM, PanoNav, M3Fusion and delta-mem. Shorten the disproportionate LingBot-Map treatment and remove its contradictory fixed-size-state implication. |
| Results consistency | Correct an unrelated quantization-appendix cross-reference in the clutter caption and “all clutter” wording beside 22/24. Preserve the original assisted-control counts and pending cells. |

## Citation corrections checked against primary sources

- [DynaMem](https://arxiv.org/abs/2411.04999) and
  [OK-Robot](https://arxiv.org/abs/2401.12202): lead author is Peiqi Liu, not Song Liu.
- [Explore until Confident](https://arxiv.org/abs/2403.15941): correct author
  Allen Z. Ren and arXiv ID; the previous 2403.11819 is an unrelated Stokes-flow paper.
- [DELTAMEM](https://arxiv.org/abs/2606.03083): correct title/authors and ID;
  the previous 2606.00576 belongs to [DREAM](https://arxiv.org/abs/2606.00576).
- [MolmoSpaces](https://arxiv.org/abs/2602.11337): correct 2026 title, lead author
  and identifier, replacing an incomplete 2025 placeholder.
- [RoboVista](https://arxiv.org/abs/2607.04610): correct 2026 title/authors/ID;
  the previous 2505.14677 belongs to Visionary-R1.
- [SQA3D](https://arxiv.org/abs/2210.07474): Xiaojian Ma and collaborators;
  [HomeRobot OVMM](https://arxiv.org/abs/2306.11565): Sriram Yenamandra and collaborators.
- [LingBot-Map](https://arxiv.org/abs/2604.14141): restore title prefix and the
  current author list's Shangzhan Zhang.
- Added [SayCan](https://arxiv.org/abs/2204.01691),
  [PDDLStream](https://ojs.aaai.org/index.php/ICAPS/article/view/6739),
  [ReAct](https://arxiv.org/abs/2210.03629),
  [Inner Monologue](https://proceedings.mlr.press/v205/huang23c.html),
  [GraphPad](https://arxiv.org/abs/2506.01174),
  [delta-mem](https://arxiv.org/abs/2605.12357),
  [PanoNav](https://arxiv.org/abs/2511.06840), and
  [RoboCasa365](https://arxiv.org/abs/2603.04356).
- M3Fusion metadata is supported by the publisher's English abstract/author block,
  DOI `10.1360/SSI-2025-0293`; TiPToP remains pinned to its v2 citation.

## New evidence, kept separate

The preplanned frozen repeat queue finishes with 3/3 two-goal episodes and 2/3
buffered swaps. The failed swap emits malformed model JSON after planning and
never dispatches execution; it is counted as a failed task. All six outcomes,
3,094 audited commands, zero forbidden commands and provenance hashes are in
`paper/data/native_tamp/agent_control_repeats_20261023.json`. This is not a
rescore of the October 17 2/6 study or observed-input acceptance.

The selected integrated regression set passes 796 checks, and all ten repeats of
the 94-check boundary set pass. These overlapping software checks are not robot
success metrics. The offline Caliban RGBD region diagnostic remains unqualified:
connected depth support does not establish semantic correctness or full pose.

## Still open before publication

1. **Focus and length:** the compiled draft is about 50 pages, with roughly 21
   pages before Appendix A. Select the paper's primary claim and condense to the
   target venue's format; the current document is not submission-ready.
2. **Missing experiments:** OVMM, SQA3D, dynamic exploration and large-scene tables
   still contain explicit pending cells/placeholder figures. Populate from frozen
   scored runs or remove them from the submitted empirical claim.
3. **Primary method validation:** new source-pinned paired EQA runs and isolated
   merge/staleness ablations are still needed; do not transfer historical scores
   to current shared-agent revisions. Keep released GraphEQA selection/parity and
   semantics-sensor provenance audits open (existing P1/P2 ledger).
4. **Manipulation:** observed tracking/geometry, navigation-to-manipulation,
   recovery, native articulation, additional embodiments and household-suite
   certification remain unestablished. Do not replace those gates with unit-test
   counts or successful privileged controls.
5. **Agent reliability:** malformed structured model output caused one scheduled
   task failure despite a valid plan. Any protocol change must be separately
   evaluated; do not repair the archived outcome or silently reshape its JSON.
6. **Model comparisons:** development-selected small slices, omitted/OOM episodes
   and deployment differences prevent general model-ranking or precision-only
   causal claims. Re-audit pinned prompts/versions and full per-run denominators
   before making stronger statements.
7. **Presentation:** existing overfull tables/text, font fallbacks and float
   warnings remain. New appendix pages have been visually checked; compilation
   success alone is not final layout acceptance.

The prior P1–P7 ledger in `shared_agent_paper_update.md` remains open unless its
specific empirical/provenance gate has passed. These writing fixes do not close
unrun experiments or certify hardware readiness.

## Validation

- `./paper/build.sh` succeeds and writes `paper/main.pdf` (50 pages).
- Citation/include checks pass: no missing citation keys, duplicate BibTeX keys,
  unresolved citation/reference warnings, or multiply-defined labels.
- Checked all six new per-episode records against their aggregate: three attempts
  per configuration, 3/3 and 2/3 tasks, malformed-response failure with zero
  transfers, 3,094 audited commands and zero forbidden actions.
- Rendered and visually inspected the new agent appendix pages (42–43 in this
  build); the table and protocol paragraphs are readable. Existing draft-wide
  layout/font warnings remain as noted above.
- `git diff --check` passes. No runtime code changed in this paper-only update;
  the frozen 796-check and ten-repeat validation records retain their own scope.
