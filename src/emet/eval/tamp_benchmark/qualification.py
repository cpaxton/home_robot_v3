# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Fixed-population accounting; unsupported and infrastructure cases stay visible."""
from collections import Counter

from emet.controller.task.tamp.task_contracts import EpisodeResult, EpisodeSpec

CATEGORIES = ('completed', 'unsupported', 'construction_failure', 'reference_failure',
              'agent_failure', 'infrastructure_failure')


def summarize_population(episodes: tuple[EpisodeSpec, ...], results: tuple[EpisodeResult, ...]) -> dict:
    identities = {episode.identity for episode in episodes}
    if len(identities) != len(episodes):
        raise ValueError('duplicate_episode_condition')
    by_identity = {}
    for result in results:
        if result.episode_identity not in identities:
            raise ValueError('result_outside_frozen_population')
        if result.episode_identity in by_identity:
            raise ValueError('duplicate_episode_result')
        if result.category not in CATEGORIES:
            raise ValueError('invalid_result_category')
        by_identity[result.episode_identity] = result
    counts = Counter(result.category for result in results)
    # Never turn missing/unknown evidence into successful physical acceptance.
    strict = sum(result.category == 'completed' and result.upstream_success is True
                 and result.final_goal_success is True and result.physical_integrity is True
                 for result in results)
    return {'schema_version': 1, 'population': len(episodes), 'reported': len(results),
            'missing': len(identities - by_identity.keys()),
            'categories': {name: counts[name] for name in CATEGORIES},
            'upstream_successes': sum(result.upstream_success is True for result in results),
            'strict_successes': strict,
            'strict_success_fraction': strict / len(episodes) if episodes else None,
            'complete': len(by_identity) == len(episodes)}
