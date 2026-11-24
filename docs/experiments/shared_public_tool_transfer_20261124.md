# Shared public-tool native transfer controls, 2026-11-24

These controls call canonical `pick_place` through the production CHAT registry,
shared observed adapter and physical executor. Outer tool selection is scripted;
Qwen3.5-9B supplies fresh visual grounding. Object CAD, actor state and saved
semantic regions are absent from the actor. Benchmark metadata initializes and
independently scores the fixture. The development simulation assists are
reported in each receipt; these runs do not qualify hardware collision safety.

Artifacts live at repository-root
`logs/shared-harness-public-transfer-20261124/`. Native drivers and profiles are
retained there. Every retry has its own output directory and free port offset.
`integration-control-summary.json` retains outcomes, including orchestration
failures. There is no model-only or frozen success-rate claim.

| Trial | Commands | Result |
| --- | ---: | --- |
| pear-v1 | 0 | `not_dispatched` |
| pear-v2 | 0 | `support_patch_unavailable` |
| pear-v3 | 0 | `support_patch_unavailable` |
| pear-v4 | 0 | `support_plane_unqualified` |
| pear-v5 | 0 | `geometry_tracking_no_match` |
| pear-v6 | 0 | `task_object_not_settled_on_support` |
| pear-v7 | 0 | `geometry_tracking_no_match` |
| pear-v8 | 0 | `scene_changed_replan` |
| pear-v9 | 0 | `no_observed_transfer_candidate` |
| pear-v10 | 27 | `geometry_tracking_no_match` |
| pear-v11 | 27 | `geometry_tracking_no_match` |
| pear-v12 | 0 | `visual_evidence_incomplete` |
| pear-v13 | 28 | `geometry_tracking_no_match` |
| pear-v14 | 0 | `collision` |
| pear-v15 | 27 | `geometry_tracking_no_match` |

Changes driven by these controls:

- Controller-owned phase/payload state gates model queries. Empty state comes
  from reset or verified placement, never model assertion. Dispatch failures
  preserve unknown load state.
- The scripted runner now actually dispatches a single canonical call, and its
  scorer recognizes the canonical execution receipt. Physical criteria remain.
- Support expansion follows connected measured plane returns, stays local for
  the source and excludes object pixels. Seed bounds retain grown source and
  destination masks instead of recropping them to semantic hints.
- Connected object search tolerates coarse boxes while imposing a 15 cm metric
  bound. Synchronized local, appearance-consistent above-plane surfaces from
  other cameras can augment the CAD-free registration reference.
- Approximate pickup hypotheses tolerate 20 mm footprint changes from partial
  views. Approximate preplanning orientation changes are bounded by 10 mm of
  geometry displacement and 0.2 rad. Contact/collision patches and final-place
  verification are unchanged; exact geometry retains its original check.
- Tracking records per-camera residual witnesses and excludes capture-aligned
  measured robot returns using the existing mapping association.

Trials v10 and v11 reached 27 actor commands and approximately 0.70 m measured
end-effector displacement, with zero unintended contacts recorded by the scorer.
They stopped during pregrasp; no pickup or completed transfer was verified.
The reference-camera occlusion/handoff remains the next native validation gate.

The first full-frame self-association trial (v12) reduced pose capture rate to
roughly one frame per 1.4–1.6 seconds and timed out gathering evidence. The
Panda binding now offers the same ray/triangle association for candidate pixels
only. Full-frame mapping qualification remains separate and lazy. Trial v13
restored tracking throughput and reached 28 commands, then rejected a partial
surface match just outside the 6 mm diagnostic fit limit. Approximate boxes
now use a 10 mm residual floor; exact models retain their existing limits.
The actual tracking limit is recorded in source diagnostics, and approximate
pose scope is explicitly labeled rather than described as unique rigid pose.

Trial v14 refused gripper-close certification for self collision before motion.
The next control removes the driver's zero-aperture override and uses the robot
profile's calibrated closure. No collision checker was disabled for that refusal.

Validation: the final related regression suite passed **725 tests** with two
existing dependency warnings. A focused test process suffered the known native
import segmentation fault; its log and the successful full-suite rerun with
`PYTHONMALLOC=malloc` are retained beside the trial artifacts. Ruff and
whitespace checks pass. No completed shared transfer is claimed.

Trial v15 used calibrated gripper closure, certified all six route phases and
reached 27 commands / 0.701 m end-effector displacement. It still stopped in
pregrasp on `geometry_tracking_no_match`; no pickup or transfer was verified.
Its remaining camera had median reference distance 1.65 mm but 95th-percentile
distance 13.91 mm. The wider residual policy did not resolve this view handoff.
The evidence therefore supports a structural next step: separate bounded
approximate task geometry from mandatory continuous rigid registration, with
observed grasp/lift/place verification at action boundaries. Do not keep
increasing residual limits or score this as a model reasoning failure.
