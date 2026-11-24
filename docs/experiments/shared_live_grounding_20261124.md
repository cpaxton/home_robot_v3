# Shared live visual grounding smoke (2026-11-24)

The shared observed transfer scene now accepts an injected
`VisualTransferGrounder`. Model calls use the common perception implementation;
legacy evaluation imports are compatibility shims. The component proposes image
regions and re-acquires RGB-D before motion planning. It does not issue robot
commands or read benchmark object metadata.

## Behavior exercised

- Qwen predicts object and destination regions; nearby support comes from depth.
  The grounding stage no longer requests unused grasp-angle/height estimates.
- Model inference can exceed the original capture's motion-evidence freshness
  window. A new capture must be independently fresh and newer than the reference.
  Replayed frames, changed views and changed object geometry are rejected.
- An explicit phase callback must permit visual queries. A phase change during
  inference prevents subsequent queries and geometry admission.
- Source/destination failures retain their reason codes and identify which
  region needs revision in retry feedback. Object ambiguity has its own role.
- Generic destination support can be below, level with, or above source support.
  The old sink-only restriction is an explicit legacy-profile constraint.
- Runtime-grounded live scene construction rejects object CAD/enclosure priors.
  Robot geometry, calibration and configured native controllers remain required.

## Native camera smoke

The existing pinned Panda simulator served the frozen pear fixture. The actor
used the existing robot client, robot-only kinematic model and native sample
reader. The shared grounder called local BF16 Qwen3.5-9B and reacquired actual
native sensor captures after inference. No image timestamps were renewed or
fabricated. The phase callback allowed queries because this driver was read-only.

Repository-root artifacts:

- `logs/shared-harness-native-grounding-20261124/pear-read-only/report.json`
- `logs/shared-harness-native-grounding-20261124/pear-read-only-generic-destination/report.json`

Both admitted grounding. The final generic-destination run took approximately
3.7 seconds, with different reference and geometry capture IDs. Requests,
responses, rejected proposals if any, binding metadata and server logs are retained.
The actor invoked no command methods. Simulator reset/stance setup belongs to
fixture initialization and is not counted as agent motion.

This is a live grounding smoke, not a physical pickup, complete tool session,
semantic-identity certification, cross-robot result or success-rate estimate.
The common application still needs to bind this component to authoritative
execution-phase state. Loaded control, grasp candidate generation, observed
verification, drawer skills and full benchmark qualification remain separate
work. No alternative robot execution loop was added for this component.


Validation: 704 related regression tests passed (two existing dependency
warnings), plus Ruff and whitespace checks. Coverage includes live-source
capture rebinding, phase changes, replay rejection, no-CAD enforcement,
region-specific feedback, and lower/same-level/higher destination support.
The native smoke drivers are retained next to their reports; repeated runs need
new output directories and free port offsets.
