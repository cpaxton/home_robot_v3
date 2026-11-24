# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Controller-owned payload/phase state; never inferred from model assertions."""
from threading import RLock


class ManipulationExecutionPhase:
    def __init__(self, initial_payload_state='unknown'):
        if initial_payload_state not in {'unknown', 'empty', 'held'}:
            raise ValueError('invalid_initial_payload_state')
        self._lock = RLock()
        self._payload = initial_payload_state
        self._executing = False
        self._dispatched = False

    def can_query(self):
        with self._lock:
            return not self._executing and self._payload == 'empty'

    def snapshot(self):
        with self._lock:
            return {'execution_phase': 'executing' if self._executing else 'idle',
                    'payload_state': self._payload}

    def begin_execution(self):
        with self._lock:
            if self._executing:
                raise RuntimeError('manipulation_execution_busy')
            self._executing = True
            self._dispatched = False

    def command_dispatched(self):
        with self._lock:
            if not self._executing:
                raise RuntimeError('manipulation_execution_not_started')
            self._dispatched = True
            self._payload = 'unknown'

    def grasp_verified(self):
        with self._lock:
            if not self._executing:
                raise RuntimeError('manipulation_execution_not_started')
            self._payload = 'held'

    def finish_execution(self, *, placement_verified=False):
        with self._lock:
            if not self._executing:
                raise RuntimeError('manipulation_execution_not_started')
            if placement_verified:
                self._payload = 'empty'
            elif self._dispatched:
                self._payload = 'unknown'
            self._executing = False
            self._dispatched = False
