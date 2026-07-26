# Copyright (c) Chris Paxton 2026

import pytest

from emet.memory.graph_eqa.agentic_policy import (
    AgenticState,
    EvidencePolicy,
    EvidenceRecord,
)


def test_explicit_state_path_and_fused_evidence():
    policy = EvidencePolicy()
    policy.register_hypothesis("graph:7", "woven basket", prior_probability=0.5)
    policy.choose("graph:7")
    assert policy.state == AgenticState.APPROACH
    policy.approached(11)
    assert policy.state == AgenticState.VERIFY
    policy.add_evidence(
        EvidenceRecord(
            hypothesis_id="graph:7",
            obs_id=11,
            phrase="woven basket",
            full_frame_sim=0.13,
            detector_score=0.6,
            detector_backend="owlv2",
        )
    )
    assessment = policy.assess(relation_sufficient=True)
    assert assessment.verified is True
    # Cheap channels propose presence only — never open ANSWER.
    assert assessment.answerable is False
    assert policy.state == AgenticState.REPLAN
    vlm = policy.apply_vlm_assessment(present=True, answerable=True)
    assert vlm.answerable is True
    assert policy.state == AgenticState.ANSWER


def test_siglip_proposal_alone_does_not_verify():
    policy = EvidencePolicy()
    policy.register_hypothesis("graph:8", "towel", prior_probability=0.5)
    policy.choose("graph:8")
    policy.approached(12)
    policy.add_evidence(
        EvidenceRecord(
            hypothesis_id="graph:8",
            obs_id=12,
            phrase="towel",
            full_frame_sim=0.14,
        )
    )
    assessment = policy.assess(relation_sufficient=True)
    assert assessment.verified is False
    assert assessment.answerable is False
    assert policy.state == AgenticState.REPLAN


def test_owl_alone_cannot_open_answer_gate():
    """Holdout q105 regression: OWL detector score must not unlock submit."""
    policy = EvidencePolicy()
    policy.register_hypothesis("view:3", "statue", prior_probability=0.3)
    policy.choose("view:3")
    policy.approached(3)
    policy.add_evidence(
        EvidenceRecord(
            hypothesis_id="view:3",
            obs_id=3,
            phrase="statue",
            detector_score=0.13,
            detector_backend="owlv2",
            full_frame_sim=0.11,
        )
    )
    cheap = policy.assess(relation_sufficient=True)
    assert cheap.answerable is False
    assert policy.state == AgenticState.REPLAN
    deny = policy.apply_vlm_assessment(
        present=True, answerable=False, need_more_views=True
    )
    assert deny.answerable is False
    assert policy.state == AgenticState.REPLAN
    assert deny.verified is True  # present from VLM



def test_stale_view_is_impossible_by_construction():
    policy = EvidencePolicy()
    policy.register_hypothesis("graph:7", "basket")
    policy.choose("graph:7")
    policy.approached(11)
    policy.add_evidence(
        EvidenceRecord(hypothesis_id="graph:7", obs_id=11, phrase="basket", voxel_sim=0.3)
    )
    policy.assess()
    policy.replan()
    policy.choose("graph:7")
    with pytest.raises(ValueError, match="already scored"):
        policy.approached(11)


def test_evidence_must_match_fresh_observation():
    policy = EvidencePolicy()
    policy.register_hypothesis("frontier:1", "clock")
    policy.choose("frontier:1")
    policy.approached(4)
    with pytest.raises(ValueError, match="fresh observation"):
        policy.add_evidence(
            EvidenceRecord(
                hypothesis_id="frontier:1",
                obs_id=3,
                phrase="clock",
                graph_label_match=True,
            )
        )
