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
