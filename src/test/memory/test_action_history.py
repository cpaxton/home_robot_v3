# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from emet.memory.graph_eqa.action_history import (
    ActionHistoryEntry,
    ActionSignature,
    ActionTarget,
    ProgressToken,
    decide_candidate,
    render_history_entry,
    stable_digest,
)


def _signature(
    *,
    family: str = "inspect_place",
    approach: int = 0,
    adapter: int = 15,
    view_id: str = "view_8",
) -> ActionSignature:
    return ActionSignature.build(
        tool_name="verify_siglip" if family == "verify_view" else "investigate",
        family=family,
        intent="Find the silver trash can",
        target=ActionTarget(
            kind="place",
            stable_id="place_kitchen_island",
            labels=("kitchen island", "counter"),
            room="kitchen",
            adapter_id=adapter,
            view_id=view_id,
            revision=8,
        ),
        variant={
            "approach_index": approach,
            "view_id": view_id if family == "verify_view" else None,
        },
    )


def _entry(
    signature: ActionSignature,
    before: ProgressToken,
    *,
    after: ProgressToken | None = None,
    outcome_class: str = "no_progress",
    round_index: int = 1,
) -> ActionHistoryEntry:
    return ActionHistoryEntry(
        schema_version=1,
        round_index=round_index,
        selected_by="router",
        signature=signature,
        progress_before=before,
        progress_after=after or before,
        outcome_class=outcome_class,  # type: ignore[arg-type]
        status="test",
        ok=False,
    )


def test_stable_place_identity_survives_obs_adapter_renumbering():
    first = _signature(adapter=15)
    renumbered = _signature(adapter=91)

    assert first.work_key == renumbered.work_key
    assert first.equivalence_key == renumbered.equivalence_key


def test_stable_digest_preserves_coordinate_order_but_not_set_order():
    assert stable_digest("cell", (1, 2)) != stable_digest("cell", (2, 1))
    assert stable_digest("labels", {"sink", "counter"}) == stable_digest(
        "labels",
        {"counter", "sink"},
    )


def test_new_approach_is_alternate_work_variant():
    first = _signature(approach=0)
    alternate = _signature(approach=1)
    token = ProgressToken.build({"view_id": "view_8", "robot_pose_cell": (1, 2)})

    decision = decide_candidate([_entry(first, token)], alternate, token)

    assert decision.allowed is True
    assert decision.disposition == "allowed_alternate"


def test_same_variant_and_progress_token_is_suppressed():
    signature = _signature()
    token = ProgressToken.build({"view_id": "view_8", "robot_pose_cell": (1, 2)})

    decision = decide_candidate([_entry(signature, token)], signature, token)

    assert decision.allowed is False
    assert decision.disposition == "would_suppress_duplicate"
    assert "no new material state" in decision.reason


def test_partial_progress_allows_continuation_from_new_token():
    signature = _signature()
    start = ProgressToken.build({"robot_pose_cell": (1, 2), "view_id": "view_8"})
    progressed = ProgressToken.build({"robot_pose_cell": (2, 2), "view_id": "view_8"})

    decision = decide_candidate(
        [_entry(signature, start, after=progressed, outcome_class="progress")],
        signature,
        progressed,
    )

    assert decision.allowed is True
    assert decision.disposition == "allowed_progress"


def test_progress_outcome_never_blacklists_the_variant():
    signature = _signature()
    token = ProgressToken.build({"robot_pose_cell": (1, 2), "view_id": "view_8"})

    decision = decide_candidate(
        [_entry(signature, token, outcome_class="progress")],
        signature,
        token,
    )

    assert decision.allowed is True
    assert decision.disposition == "allowed_progress"


def test_no_progress_result_suppresses_from_its_post_action_state():
    signature = _signature()
    start = ProgressToken.build({"robot_pose_cell": (1, 2), "view_id": "view_8"})
    stopped = ProgressToken.build({"robot_pose_cell": (3, 2), "view_id": "view_8"})

    decision = decide_candidate(
        [_entry(signature, start, after=stopped, outcome_class="no_progress")],
        signature,
        stopped,
    )

    assert decision.allowed is False
    assert decision.disposition == "would_suppress_duplicate"


def test_transient_failure_gets_one_same_token_retry():
    signature = _signature()
    token = ProgressToken.build({"robot_pose_cell": (1, 2)})
    first = _entry(signature, token, outcome_class="transient", round_index=1)

    retry = decide_candidate([first], signature, token)
    exhausted = decide_candidate(
        [
            first,
            _entry(signature, token, outcome_class="transient", round_index=2),
        ],
        signature,
        token,
    )

    assert retry.disposition == "allowed_transient_retry"
    assert retry.allowed is True
    assert exhausted.allowed is False


def test_completed_verify_is_terminal_for_same_immutable_view():
    signature = _signature(family="verify_view")
    first = ProgressToken.build({"view_id": "view_8", "relevant_evidence": "none"})
    unrelated_change = ProgressToken.build({"view_id": "view_8", "relevant_evidence": "unrelated-room-event"})

    decision = decide_candidate(
        [_entry(signature, first, outcome_class="negative_evidence")],
        signature,
        unrelated_change,
    )

    assert decision.allowed is False
    assert decision.disposition == "suppressed_terminal_view"


def test_semantic_render_leads_with_intent_and_target_not_adapter():
    signature = _signature()
    token = ProgressToken.build({"view_id": "view_8"})
    entry = ActionHistoryEntry(
        schema_version=1,
        round_index=1,
        selected_by="router",
        signature=signature,
        progress_before=token,
        progress_after=token,
        outcome_class="negative_evidence",
        status="ABSENT",
        ok=True,
        verify_status="ABSENT",
        closest_m=0.4,
    )

    text = render_history_entry(entry)

    assert text.startswith('round=1 action=investigate intent="find the silver trash can"')
    assert 'target="kitchen island/counter"' in text
    assert "room=kitchen" in text
    assert "adapter=15" in text
