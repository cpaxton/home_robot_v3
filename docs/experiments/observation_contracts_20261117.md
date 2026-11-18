# Observation contracts, movement evidence and bounded qualification

The [tool catalogue](../agent_tool_contracts.md) lists all 44 registry names,
their primary arguments, side effects, evidence boundaries and failure contracts.
It distinguishes CHAT and EQA bindings, provider-dependent TAMP tools, historical
views, object geometry, motion receipts and actual task completion. A registry
coverage test keeps the inventory complete. This work adds no tool names.

Three implementation commits were qualified at frozen revision
`2834262efdd8588f6ca3511569766e6fcd511630`:

- `81a12c95`: optional head angles depend on declared adapter capabilities;
  Habitat confirms a synchronous post-arrival capture using acquisition timing,
  without requiring a ZMQ reception sequence.
- `5c38b9b6`: retain bounded RGB/pose/acquisition history independently of voxel
  ingestion, then feed the existing four-image selector. Preserve active query
  images and router request dumps through episode finalization. Clarify partial
  versus total counts and the fallibility of auxiliary hints in the assessment
  prompt; no answer replacement or compulsory extra-view rule.
- `2834262e`: document tools and contracts, with inventory coverage checking.

The focused CPU pack passed **789 tests**. That validates contracts, not task
accuracy. Streaming hardware paths have unit coverage; no real robot ran here.

## Native Habitat and frozen-context checks

Managed job `20261117_171536_3a192a` completed all three stages without an
infrastructure error. Root artifacts:
`/home/cpaxton/runs/emet/observation-contracts-20261117-r1/`.
The driver required clean committed source and a managed exclusive GPU lease;
stages ran serially on CPU cores 12, 14, 20 and 21 with single-thread math.
Configured model: Caliban `Qwen/Qwen3-VL-8B-Instruct` at
`http://192.168.1.55:8000/v1`; actual precision/server revision is unverified.

| Check | Result | Boundary |
| --- | --- | --- |
| Native Habitat gaze, without `_seq_id` | Pass: measured `(0,0)`, advancing native acquisition timestamp, changed pixels, unchanged base | Virtual sensor contract, not mechanical hardware qualification |
| Out-of-range tilt `9` | Rejected; head pose unchanged | No silent clipping or false successful capture |
| Frozen arrival + current, old prompt | Abstains, requests more views | One call per condition; not an accuracy estimate |
| Frozen overview + current, old prompt | Incorrectly answers One | Manually supplied overview; not automatic selector acceptance |
| Frozen overview + current, coverage clarification | Incorrectly answers One | Prompt clarification does not repair this miss |
| Frozen arrival + current, coverage clarification | Abstains, requests more views | Same two-image budget |

The frozen diagnostic uses archived Q12 frames `rgb_0078.png` (current),
`rgb_0075.png` (arrival), and `rgb_0060.png` (overview) from session
`dev.eqa-evidence.CiLw0ao8CN_s0_q12`. The overview visibly contains two bedside
tables. All exact transmitted text, image PNGs and raw responses are retained in
`frozen-packets/<condition>/`; use `request.json` for the actual prompt. The
assessment object's `prompt` field records the production template, before the
diagnostic wrapper removes the new clarification for old-prompt conditions.
Scorer labels were read only after all four inference calls. This manually
selected image test has no reconstructed poses, no claimed online selection
benefit and no statistical superiority claim.

## Live Q12 diagnostic

The candidate-only replay produced complete valid metrics, but **failed the
task**: model-final answer One, gold Two, two decision rounds, no budget hit.
The coordinator explicitly submitted One; the harness preserved that submission.

Movement retention worked: six RGB views retained, thirteen same-viewpoint
frames skipped, and the existing pose selector supplied four images: current,
pre-turn, and two historical movement views. Selection took approximately
8.9 ms. The selector trace records scope, acquisition age, roles, exclusions
and sampling settings. This trajectory explored a frontier rather than
reproducing the earlier investigate route, so it is not a matched trajectory
comparison.

The saved current image itself shows both white bedside tables, one between the
beds and one on the wall side. The assessment nevertheless says one table and
claims sufficient coverage. This run therefore includes a visible model
counting error; it cannot be explained solely by missing movement history.

- [Live metrics](/home/cpaxton/runs/emet/observation-contracts-20261117-r1/live-q12/eqa/q12.jsonl)
- [Current assessment image](/home/cpaxton/.cache/habitat_eqa/episodes/dev.eqa-evidence.IYey3u747M_s0_q12/q0012_lazy_graph/images/vlm_assess_r00_obs1099511627790.png)
- [Assessment/selection trace](/home/cpaxton/.cache/habitat_eqa/episodes/dev.eqa-evidence.IYey3u747M_s0_q12/q0012_lazy_graph/agentic_trace.jsonl)

This replay saved the current assessment PNG and selection metadata, but not
every historical assessment PNG. The follow-up audit repair writes each retained
assessment packet image in order and traces `vlm_packet_pngs`, without overwriting
the canonical current-view image. It respects the existing dump opt-out. The
repair has **45 passing focused tests**, including the real assessment-to-disk
path and debug finalization; it has not been live-rerun in this report.

## Earlier paired micro-pilot

Job `20261117_095311_248038` completed four EQA episodes and two S0 episodes:
`/home/cpaxton/runs/emet/task-gaze-micropilot-20261117-r1/`.
Sources were pre-task-gaze `86180bc7` and candidate `9e8e07ce`, seed zero,
model-final policy, same configured Caliban endpoint and budgets. These
revisions precede movement retention and native Habitat capture repairs above.

| Case | Pre-task-gaze | Candidate |
| --- | --- | --- |
| EQA Q2 | Wrong: White | Unknown, eight-round budget exhausted |
| EQA Q12 | Wrong: One | Wrong: One |
| S0 object + destination localization | 1/2 phases | 2/2 phases |
| S0 episode wall time | 656.7 s | 212.2 s |

The managed pair reports failure because the baseline S0 task gate fails,
not because an episode failed to produce metrics. The room driver calls its
baseline arm `main`, but that arm is explicitly **86180bc7, not current main**.
S0 measures find/localization phases, not completed pick-and-place. Its metrics
omit the runtime endpoint despite the generated config/manifest recording it;
do not claim deployment precision verification. One seed and two EQA questions
do not establish general improvement or non-regression.

## Remaining acceptance

Keep model-final decisions intact. The contracts and retention path are
qualified at their stated levels; **EQA accuracy improvement is unproven**.
Next useful tests are a small matched-source EQA/simple-OVMM pair and inspection
of the complete saved assessment packets. Do not launch a broad sweep or add a
question-specific counting/retry rule to repair Q12. Wider multi-task,
manipulation and hardware acceptance remain separate gates.
