# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Strict JSON registry validation; design documents cannot launch episodes."""
from __future__ import annotations

import json
from pathlib import Path

from emet.controller.task.tamp.task_contracts import EpisodeSpec, fingerprint


def load_registry(path: str | Path) -> tuple[EpisodeSpec, ...]:
    raw = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('unsupported_registry_schema')
    if raw.get('executable_registry') is not True:
        raise ValueError('registry_is_design_only')
    if set(raw) != {'schema_version', 'executable_registry', 'episodes', 'provenance'}:
        raise ValueError('invalid_registry_fields')
    provenance = raw['provenance']
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(key), str) and provenance[key].strip()
        for key in ('implementation_sha', 'upstream_revision', 'reset_manifest_sha256')
    ):
        raise ValueError('missing_registry_provenance')
    digest = provenance['reset_manifest_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid_reset_manifest_digest')
    if not isinstance(raw['episodes'], list) or not raw['episodes']:
        raise ValueError('empty_episode_registry')
    episodes = tuple(EpisodeSpec.from_dict(row) for row in raw['episodes'])
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError('duplicate_episode_id')
    return episodes


def _reject_constant(value):
    raise ValueError(f'nonfinite_json:{value}')


def describe_registry(path: str | Path) -> dict:
    episodes = load_registry(path)
    return {'schema_version': 1, 'registry_sha256': fingerprint(json.loads(Path(path).read_text())),
            'episodes': len(episodes), 'episode_identities': [e.identity for e in episodes],
            'conditions': sorted({(e.suite, e.robot_track, e.input_mode, e.execution_mode, e.strategy)
                                  for e in episodes}),
            'qualified': False}
