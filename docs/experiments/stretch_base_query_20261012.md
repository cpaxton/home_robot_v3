# Stretch simulator base endpoint queries

The full public TAMP flow previously stopped with
`base_pose_validation_unavailable` even though the Stretch oracle relocation
backend was usable by the older benchmark runner. Stretch now implements the
same read-only `check_base_poses` contract used by robosuite.

Queries reconstruct a private scene from the physics process's `scene_qpos`
snapshot and check robot contacts at each proposed world XYT. The capability is
advertised only with a finite correctly-sized measurement and a compatible
free-base model. Missing measurements never fall back to XML defaults. Collision
probing runs outside the publication lock after copying the snapshot.

This is privileged simulator geometry, not a robot sensor or swept-path check.
It does not enable kinematic attachment or native-contact manipulation on Stretch.
Use explicitly labeled teleport controls until the native adapters are ready.

Thirty targeted tests pass, covering moved obstacles, preserved measured state,
invalid queries, missing measurements, capability advertisement, articulation,
and Stretch telemetry. Live endpoint/public-tool validation remains pending.

## Live transport follow-up

The first public-tool run (`20261012_185308_3ed2cc`, source `c15cadab`)
failed with `base_pose_validation_unavailable`: the server endpoint existed but
`StretchZmqClient` did not implement the client method. No timeout occurred.
Both clients now call a shared correlated-receipt query implementation. The
command-runtime regression is parameterized over the actual Generic and Stretch
client methods, verifying successful result transport and unsupported-server
rejection without dispatching actuation. The endpoint and approach suites pass
16 checks. A new live replay is required; the first run remains a failure.
