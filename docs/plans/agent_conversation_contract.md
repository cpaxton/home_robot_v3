# Episode conversation and tool-argument repair

User priority: fix history preservation and silently ignored tool arguments
before modifying routing instructions. No routing-rule/system-prompt rewrite,
semantic answer substitution, safety relaxation or default-model change here.

## Contract

- An agent owns one multimodal conversation per explicit episode/session and map
  epoch. OVMM FindObj and FindRec share it; another episode or reset world does
  not. Router requests, assistant replies and every dispatched tool result,
  including graph reads and argument errors, remain available to subsequent
  routing and final-answer calls. Camera snapshots are owned copies.
- Physical model state is isolated per inference while the complete dialogue is
  supplied explicitly. Perception can reset its own prompt without deleting the
  agent dialogue. Failed inference does not commit a half-turn. No history is
  silently evicted; existing episode/action budgets still bound execution.
- Current per-call system instructions remain unchanged and role-specific
  (routing versus final answering); user/assistant/tool-result turns persist
  across that transition. Tool results use explicit user-channel TOOL_RESULT
  records because this harness uses JSON/manual tool dispatch, not native
  assistant tool_call IDs.
- Native Qwen and remote OpenAI-compatible transport preserve prior user images
  and assistant turns. The local VL server no longer flattens all user turns or
  discards assistant messages. Deploy both updated client and server when using
  an older emet remote endpoint. Other VLM providers must implement the native
  conversation-content encoder; unsupported conversion fails explicitly.
- Undeclared argument names are rejected before execution, with supported names
  and a recorded failure returned to the dialogue. For example inspect_graph has
  no query argument: query=bowl is an error, not an applied query. This is not
  a claim of complete JSON-schema type validation. Canonical properties remain
  unchanged; EQA bindings advertise additionalProperties=false.
- Automatic floor-recovery budgets remain internal and cannot be widened by
  model-supplied undeclared arguments. Geometry/freshness/navigation gates stay.
- Router and final-answer debug exports include prior messages and ordered image
  snapshots/hashes, rather than calling a current-state-only export the entire
  conversation. Capture boundary is before model resizing/tokenization.

## Verification and rollout

Commits: 06cf1eba (transport and isolated dialogue), 4e4dd204 (executor integration,
tool boundary and trace export). 470 focused client/server/history/EQA/OVMM/recovery
tests pass. Another 47 room/navigation tests pass with one baseline failure
explicitly excluded: test_voxel_sim_upgrades_full_frame_absent_to_present also
fails on unchanged parent 8216252f. It is not repaired or counted as passing.
Ruff passes. Socket-based tests run with local test-network permission.

Regressions cover perception between turns, episode/world isolation, shared OVMM
phases, native Qwen image encoding, remote client/server assistant history,
final-answer dialogue, failed-call rollback, unsupported arguments with no tool
execution, and transcript/image export. Existing routing-rule files
agentic/tools.py and agent/skills/specs.py are unchanged.

Next: one frozen serial live contract check, EQA q2/6/14/56 and one Molmo rby1
find episode with model_final. Inspect transcript continuity, explicit argument
errors/recovery, final model authority and memory/latency. This is not a full
sweep, matched accuracy result or complete manipulation acceptance. Increasing
history can increase context/GPU cost; measure it before larger pilots.

## Review extraction and completed evidence (2026-10-18)

This slice is extracted onto `fix/eqa-final-answer-contract` (#216), not the
experimental acceptance branch. It contains transport, executor/argument contracts
and request tracing, but no replay drivers, routing rewrites, provenance-warning
experiment or standoff default change. Fresh extracted-source suite: 469 focused
tests pass; selected transport/executor files pass ruff. Live checks below were
on the integrated experimental source, not yet the extracted source.

- Integrated candidate seed-0 EQA12: 8/12 versus previous candidate 6/12 and native
  main 9/12. Main's correct q12 used an answer override despite its final model
  being wrong; candidate's q65 success reasons from missing visual evidence. No
  claim of superiority, grounded state recognition or accepted non-regression.
- Fixed q12/q14 packet replay: full and tool-results-only history both 0/3 for
  each; no-history q14 2/3, q12 0/3. Provenance labels do not recover either;
  q6/q56 controls remain 3/3. Keep the contract fix, not ineffective warnings.
- Wider 0.85 m approach probe: same 2/4 as 0.35 m, extra rounds in two cases.
  Not promoted. q12 still lacks room-wide count coverage.
- q14 handoff audit: later `look_around` and `explore_frontier` really change the
  camera pose. Final slots 1/2 show the current hallway/pre-turn arrival, while
  slots 3/4 retain historical fan views (voxel frames 22/13). Final labels mark
  these as historical; the preceding assessment explicitly cites Image 3. The
  fan evidence was not lost, and a new semantic pin/override is not justified.
- Prior Molmo smoke: bowl grounded, microwave recognized but not localized;
  not a full find pass and not manipulation acceptance. Cross-phase dialogue
  continuity was verified in request exports.

Evidence roots: `/tmp/emet-history-paired12-20261017`,
`/tmp/emet-captured-history-ablation-20261017`,
`/tmp/emet-assessment-provenance-20261017`,
`/tmp/emet-context-standoff-20261017`. q14 handoff bundle:
`/home/cpaxton/.cache/habitat_eqa/episodes/dev.eqa-evidence.cT4LBfnpOK_s0_q14/q0014_lazy_graph`.
The acceptance branch's `docs/plans/context_payload_audit_20261017.md` retains
exact experiment jobs/revisions and limitations. Raw artifacts remain local.

Remaining gate: rerun EQA4 and simple/Molmo OVMM on this extracted source before
making it non-draft. No main merge authorized. Full untruncated history has real
cost (observed peak 21,081 input tokens in EQA12); the TODO history ablation is
required before larger long-horizon rollout. Other VLM providers currently fail
explicitly if they lack the native conversation encoder; local/remote Qwen is
the validated path, not a claim of universal provider support.

Exact-source job `20261018_081808_fbd9dd` completed: EQA4 2/4 with no runtime
errors; Molmo bowl grounded / microwave unlocalized. Simple S0 is invalid as a
task comparison: internal floor recovery raised `handle_tool() got an unexpected
keyword argument 'validate_arguments'`. The public executor facade had not been
updated with the internal dispatcher's new keyword. Mocked recovery tests missed
the interface mismatch. A new test through the real facade reproduces that exact
TypeError before the fix, verifies recovery/replanning afterward and ensures
model JSON cannot request the private bypass/budget. Facade now forwards the
keyword explicitly; 79 targeted recovery/dialogue/answer/OVMM tests pass and lint
is clean. S0 live rerun remains required; no blanket bug-free or paper-readiness
claim. Peak S0 runtime was also substantial and remains a separate performance
concern, not resolved by this signature repair.
