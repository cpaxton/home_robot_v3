# Caliban validates targets; frozen public-tool fixtures still block training

The isolated ARM compute snapshot at `01c1d2be` verifies all 3,232 source/asset
files and passes teacher **122/122**. Its full regression attempt passes
**1,077/1,080**. Two public-tool tests stop at frozen-fixture binding; the third
fails because a source archive lacks Git metadata. Initializing an empty Git
repository inside this RAM snapshot fixes the focused artifact-ignore test.
This is a setup repair, not a full-gate pass; source provenance remains the
checksummed snapshot manifest. No qualified PPO training starts on Caliban.

Native ARM recompilation previously differed in compiled floats near 1e-15,
settled wheel phase by 0.00469 rad and derived destination by roughly 40 microns.
Keep the canonical fixture, signatures and tolerances unchanged. A bounded
single-case transport probe instead loads the exact x86 MuJoCo binary model and
integration state on ARM. It reproduces the complete model signature exactly;
qpos, qvel, controls and support positions have zero measured difference, and
hand FK differs by 1.11e-16. Constructing the control object **before** restoring
state matters: the first probe constructed it afterward and changed controls
by 0.00646. Both attempts are preserved.

This probe covers only its listed numeric fields, not the entire admission
protocol. It executes no policy or integration step and implements no production
portable loader. Next: a strict external model/state artifact protocol bound to
the canonical catalog, complete fixture admission, dynamic parity, then the full
ARM regression gate before parallel policy training. Binary models/state remain
in ignored artifact storage; only the [compact evidence](../../paper/data/whole_body_control/mars_caliban_portability_v1/results.json)
is committed. No hardware is used or qualified.
