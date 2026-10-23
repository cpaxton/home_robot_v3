# Shared relative movement implementation status

2026-10-23; repair branch `fix/shared-relative-motion-results`.

The CHAT and EQA/OVMM adapters now share `move_forward(meters)` and
`rotate_base(degrees)` schemas. EQA registration requires physical map mode
and the ordinary validated navigation methods; unsupported adapters retain
their existing tool packs. Forward requests are bounded to 0–1.5 m and turns
to ±180 degrees. Invalid/nonfinite inputs cannot command motion.

The shared physical implementation transforms body-relative requests into
world goals, resets the planner, validates the swept footprint without an
exploration-goal exemption, then uses the existing validated waypoint executor.
It reports versioned semantic receipts compatible with the TAMP envelope,
measured body-frame displacement, and independent arrival verification.
EQA counts attempts against its existing navigation budget and captures and
assesses the actual post-motion view, including after partial/failed execution.
Capture or assessment failures do not relabel completed motion as failed.

The existing outer-agent legacy movement path now checks server return values
and measured progress rather than reporting commanded motion as completion.
It remains a compatibility path, not authorization to expose it in autonomous
EQA on adapters without the validated-navigation capability.

The preliminary expanded suite passed 150 tests across movement, shared skill
packs, prompts/results, swept footprint, physical contracts and conversation
preservation. These are unit/contract tests, not live navigation acceptance.
No claim yet that the microwave recovery succeeds or that new tool availability
preserves EQA accuracy. Run serial managed simulation pilots before merging.

PR #219 has merged into #213 (`092d34a2`), not main at last fetch. Do not
implicitly import that pending stack. Its planning/execution tools remain
provider-qualified: this change does not expose privileged native planning
inside an ordinary EQA episode or create a second task-provider registry.

Outstanding: live Molmo/S0 and EQA gates; separate visual presence, retrieval
and geometry statuses so weak retrieval does not summarize a visible target as
ABSENT; qualify other robot adapters through the same navigation capability.
