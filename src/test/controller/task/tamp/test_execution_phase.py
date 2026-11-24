import pytest

from emet.controller.task.tamp.execution_phase import ManipulationExecutionPhase


def test_unknown_startup_does_not_authorize_model_queries():
    assert not ManipulationExecutionPhase().can_query()
    assert not ManipulationExecutionPhase('held').can_query()
    assert ManipulationExecutionPhase('empty').can_query()


def test_query_permission_tracks_actual_execution_and_verified_placement():
    phase = ManipulationExecutionPhase('empty')
    phase.begin_execution()
    assert not phase.can_query()
    phase.command_dispatched()
    phase.grasp_verified()
    assert phase.snapshot()['payload_state'] == 'held'
    assert not phase.can_query()
    phase.finish_execution(placement_verified=True)
    assert phase.can_query()


def test_ambiguous_command_failure_does_not_claim_empty_gripper():
    phase = ManipulationExecutionPhase('empty')
    phase.begin_execution()
    phase.command_dispatched()
    phase.finish_execution()
    assert phase.snapshot()['payload_state'] == 'unknown'
    assert not phase.can_query()


def test_preflight_refusal_preserves_known_empty_state():
    phase = ManipulationExecutionPhase('empty')
    phase.begin_execution()
    phase.finish_execution()
    assert phase.can_query()


def test_reentrant_execution_does_not_reset_load_state():
    phase = ManipulationExecutionPhase('held')
    phase.begin_execution()
    with pytest.raises(RuntimeError, match='busy'):
        phase.begin_execution()
    assert not phase.can_query()


def test_common_factory_binds_grounder_to_controller_phase(monkeypatch):
    from types import SimpleNamespace

    import emet.controller.task.tamp.observed_arm as module

    installed = []
    source = SimpleNamespace(set_grounder=installed.append)
    backend = SimpleNamespace(get_spec=lambda: SimpleNamespace(name='robot'),
        bind_observed_arm=lambda robot, model: object())
    monkeypatch.setattr('emet.robots.get_robot_backend', lambda name: backend)
    monkeypatch.setattr(module, 'build_observed_arm_adapter',
        lambda *args, **kwargs: SimpleNamespace(execution_phase=kwargs['execution_phase']))
    factory = module.observed_arm_factory(robot_name='robot', load_model=object,
        build_source=lambda robot, arm: source, tasks=(), candidates=None,
        initial_payload_state='empty', grounder_factory=lambda allowed: SimpleNamespace(can_query=allowed))
    adapter = factory(SimpleNamespace(), None)
    assert installed[0].can_query()
    adapter.execution_phase.begin_execution()
    assert not installed[0].can_query()
    adapter.execution_phase.command_dispatched()
    adapter.execution_phase.finish_execution()
    assert not installed[0].can_query()
