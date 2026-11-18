# Corrected finite-contract critic pilot

Frozen runtime `2acc3b52`; 1,574 source-bound passing tests. Original reward,
16,384 decisions per arm, fixed final checkpoints, no teacher loss or hardware.
`results.json` preserves all case identities, safety outcomes, active-hand errors
and matched-state fingerprints. Full native traces/weights remain under ignored
`logs/manipulation/finite-critic-pilot-v2/`; its complete physical report checksum
is recorded. The producer scripts retain exact ignored output paths.

Retained / joint / independent: 20/24, 19/24, 20/24 new cases; 6/10, 6/10,
7/10 existing coupled; 6/6 unloaded TAMP prefixes each. Same three contact-failure
identities per actor; all 120 actor stops confirm. No candidate is promoted:
independent fitting resolves one old failure but worsens yaw jerk. This does not
qualify loaded carry, full pick/place, agent deployment or hardware.
