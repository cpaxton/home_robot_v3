# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Invalidate task evidence around externally dispatched navigation/view motion."""
from contextlib import contextmanager

WORLD_CHANGING_TOOLS = frozenset({
    'explore', 'find_objects', 'go_home', 'scan_environment', 'rotate_base',
    'face_toward', 'move_forward', 'observe_floor', 'aim_arm_at', 'hand_over',
})


@contextmanager
def task_world_change(name, context):
    """Discard old and in-flight evidence even when motion raises or aborts.

    This is CHAT dispatch admission, not a lock against unrelated controllers.
    TAMP execution owns its own measured per-step state updates.
    """
    if name not in WORLD_CHANGING_TOOLS or context is None:
        yield
        return
    participants = [provider for key in ('observed_tamp_provider', 'tamp_task_session')
                    if (provider := context.get(key)) is not None and callable(getattr(provider, 'invalidate', None))]

    def invalidate():
        for provider in participants:
            provider.invalidate()
        plans = context.get('_tamp_plans')
        if isinstance(plans, dict):
            plans.clear()
        adapter = context.get('_native_tamp_adapter')
        if adapter is not None:
            with adapter.lock:
                adapter.plans.clear()

    invalidate()
    try:
        yield
    finally:
        invalidate()
