# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Model-independent semantic request boundary for observed manipulation.

Hints express planning preferences, never observation evidence or authorization.
Adapters must explicitly support supplied options before issuing motion.
"""
import json
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class ManipulationRequest:
    object_name: str
    receptacle_name: str
    hints: dict
    hint_mode: str
    constraints: dict

    @classmethod
    def parse(cls, object_name, receptacle_name, *, hints=None, hint_mode='prefer', constraints=None):
        if not all(isinstance(v, str) and v.strip() for v in (object_name, receptacle_name)):
            raise ValueError('invalid_manipulation_target')
        if hint_mode not in {'prefer', 'strict'}:
            raise ValueError('invalid_hint_mode')
        hints = {} if hints is None else deepcopy(hints)
        constraints = {} if constraints is None else deepcopy(constraints)
        if not isinstance(hints, dict) or set(hints)-{'object_region', 'grasp', 'approach', 'placement'}:
            raise ValueError('invalid_manipulation_hints')
        if any(not isinstance(value, dict) for value in hints.values()):
            raise ValueError('invalid_manipulation_hints')
        if not isinstance(constraints, dict) or set(constraints)-{'protected_objects', 'keep_base_stationary'}:
            raise ValueError('invalid_manipulation_constraints')
        try:
            json.dumps({'hints': hints, 'constraints': constraints}, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError('invalid_manipulation_options') from error
        protected = constraints.get('protected_objects', [])
        if not isinstance(protected, list) or any(not isinstance(v, str) or not v.strip() for v in protected):
            raise ValueError('invalid_protected_objects')
        if 'keep_base_stationary' in constraints and type(constraints['keep_base_stationary']) is not bool:
            raise ValueError('invalid_base_constraint')
        return cls(object_name.strip(), receptacle_name.strip(), hints, hint_mode, constraints)
