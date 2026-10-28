# Molmo range feedback follow-up

The frozen `cdd4591c` focused Molmo pilot completed with object FIND passing
and receptacle FIND failing. Floor inspection reduced unknown footprint cells
to zero, restored the measured task gaze, and captured a fresh task image.
Measured rotation and forward movement then executed. This repairs mobility and
camera context; it does not yet qualify the room FIND acceptance gate.

Manual inspection of the retained initial grounding image shows the microwave
above the stove. An approximate interior pixel region has depth 3.2–3.5 m,
beyond this grounding configuration's 2.5 m admission range. The only offered
supported candidate in that attempt is a different appliance. Qwen correctly
rejects that candidate. The agent later turns toward a doorway and exhausts
the six-round budget without admitted microwave geometry.

Evidence directory:
`~/runs/emet/integration-task-view-molmo-20261026/`

The next repair reports per-proposal measured pixel counts, range rejection,
median depth, and image bounds through the existing geometry failure response.
These are explicitly unverified proposals, not object localization or free-space
evidence. Feedback is attached only to the assessed view that produced it.
It neither changes depth limits nor overrides a semantic rejection. Proposal
details are bounded to eight entries; the total proposal count remains visible.

Unit tests check out-of-range abstention, invalid depth exclusion, preserved
semantic rejection, and current-view-only feedback. A new live pilot is required
to determine whether the feedback improves approach behavior. The full stack
remains red; this is not an accuracy or safety qualification claim.
