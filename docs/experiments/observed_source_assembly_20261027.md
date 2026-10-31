# Observed source assembly — October 27

Work on the live source now includes higher-fidelity views, quantization-tolerant
surface proposals, static scan fusion, conservative endpoint ownership and local
simulator acquisition-clock mapping. **Full observed execution remains disabled.**
These are sensor/geometry development experiments, not grasp or agent successes.

## View and geometry findings

The opt-in view uses actual 848×480 head RGB/depth renders and keeps robot
self-occlusion (`EMET_STRETCH_HEAD_CAMERA_SCALE=2`,
`EMET_SIM_HEAD_RENDER_ROBOT=1`). Existing defaults remain unchanged. Rendering
collision proxy groups remains disabled.

The first run (`20261027_192940_da1f5f`) stopped after one Caliban call with
`candidate_overflow`. It remains in the record. Recomputing the same saved depth
in float32 did not resolve the overflow. Replacing adjacent-pixel normal
estimation with a 5×5 measured-neighbourhood tangent fit produced seven proposals
on the same capture, within the existing eight-proposal cap. The fit excludes
distant/incompatible samples and changes neither depth values nor missing pixels.
Colour, normal-band and tracking thresholds were not relaxed.

The repeat (`20261027_193446_5399b1`) passed 5/5 live provider/tool refreshes with
two Caliban initialization calls. The selected surface grew from the earlier
217 pixels to 800 pixels (357 interior pixels), producing five landmarks. That
still fails the unchanged 12-landmark pose admission rule. More pixels improved
admission evidence but did not make this smooth surface pose-observable.

`ObservedWorkspace.fuse_static` combines aligned views with occupied evidence
winning over free evidence. It does not certify a static world or renew evidence
age. `partition(...).excluding(...)` keeps shared obstacle cells and makes
exclusive removed endpoints unknown, not free. Offline reanalysis of five saved
live captures retained ten removed target cells as unknown each time, added zero
free cells, and retained 28 other occupied cells. Offline replay is not a new
freshness qualification or new agent episode.

## Live timing and endpoint-provenance check

Run `20261027_194209_a0cd30` passed 5/5 refreshes with two initialization model
calls and no later inference. The live target/background endpoint partition also
removed ten cells to unknown and added zero free cells. Robot solid-volume and
phase-specific contact handling are still missing.

Each camera publishes an acquisition interval captured under the shared physics
lock, retaining its original time through IPC/publication. RGB and depth intervals
remain separate. A Linux boot/time-namespace identity permits explicit local
simulator monotonic-clock mapping; missing/other-host/device evidence is rejected.
The observed RGB+depth union spanned **203.5 ms**, so timestamp equality or
instantaneous stereo alignment must not be assumed. This is not encoder or
moving-camera alignment qualification; those bounds remain unavailable.

The real public planning/execution tools still refused the unqualified source,
and their admission checks added no commands. No pick/place was attempted.

## Validation and remaining work

The managed CPU regression (`20261027_194238_786ec5`) covers motion, TAMP,
observed CHAT binding, provenance, camera acquisition/registration, self-body
masking, surface proposals and reacquisition. Exact counts and source hashes are
in [the evidence record](observed_source_assembly_20261027.json). All selected
checks passed with no failures/errors/skips; Ruff and `git diff --check` passed.
Software rendering and Caliban left the local GPU unused.

Next requires usable pose hypotheses for the observed object, complete geometry,
robot occupancy/contact policy, observed support, encoder timing/motion bounds,
and the concrete execution-source/refiner assembly. A geometric/symmetry-aware
pose path or an informative measured view is needed; fabricating yaw or lowering
the feature threshold would not fix the current rejection. Keep model-prior debug
controls separate from the observed-voxel tier. Single transfers, swaps and the
navigation/exploration chain have not been qualified by these experiments.
