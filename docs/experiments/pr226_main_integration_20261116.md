# PR #226: current-main integration and foundation review

Merge source `daa26194` integrates main `e065325a` into the prior #226 head
`bac27abb`. The one textual conflict was `paper/references.bib`; both reference
blocks and the automatically merged shared entries are preserved. Main's Mars
controller failure-code mapping and payload-support departure collision checks
remain intact alongside the observed payload snapshot hooks. Conflict resolution
adds no threshold or sensor capability changes beyond main, and does not qualify
observed execution.

All five inline review threads have replies and are resolved. The self-occupancy
thread is resolved as a foundation-scope/documentation disposition, **not as a
working observed planner**. Its actual execution gate stays open in
[issue #245](https://github.com/cpaxton/home_robot_v3/issues/245), with acceptance
criteria linked from the public API, plan and TODO. Raw self-occluding depth still
rejects `invalid_start`; default observed discovery remains read-only.

## Regression results

| Check | Result | Scope |
| --- | --- | --- |
| Expanded motion/TAMP/agent/sensor/VLM/navigation suite | 1,022 passed; zero failures/errors/skips | Includes current main's Mars native TAMP and shared collision tests; 435 s |
| Five review findings plus payload-support integration | 74 passed; zero failures/errors/skips | Actual model camera/constructor checks, possible-dispatch timeout semantics, shared uncertainty margins and self-occluding start rejection |
| Multi-view tracker gap tests | 21 passed; zero failures/errors/skips | Includes two new cases covering motion-envelope expansion and latched loss; overlaps the broader suite |
| Lint and whitespace | Clean incremental code/diff | Main's existing generated artifacts are outside this incremental whitespace check |
| Paper | Builds, 63 pages | No unresolved citations/references; preserves TAMP and controller prior work |

The first expanded run is retained: 12 mesh-load failures from missing LFS
materialization, one model-mismatch assertion obscured by snapshot expiry during
cold model loading, and 42 skips from absent simulator package paths in the
isolated checkout. Assets were restored from hash-verified LFS objects and the
existing simulator packages made available. The test-only model-binding fix
reuses the loaded fixture model when rebuilding that adapter; it does not renew
old captures or loosen the production two-second freshness guard. The repeat is
clean. An earlier focused run had two package-path skips; its repeat has none.
Unit runs began on the identical precommit merged runtime tree; the live
diagnostic below used the committed `daa26194` source.

The shared `get_2d_map()` AST matches current main exactly, including negative
obstacle handling. Explicit remote VLM continuation and default perception resets
remain intentional, as documented in the October 31 review. The previously
renamed task-key namespace, API spacing cleanup, shared margin helper and
camera-intrinsic derivation remain present after integration. Entry-time transport
marking remains conservative because a timeout can follow a successful send.

## Fresh head/wrist diagnostic: retained timing-sensitive failure

The CPU OSMesa repeat completes the privileged native contact transfer but
retains a track for **1/81 captures**, then latches `geometry_tracking_no_match` /
`moving_tracking_lost_reinitialize`. It uses no local GPU or VLM. This failed
tracking repeat is not promoted to an observed task success; the native control
and sensor tracker are scored independently.

The first two head/wrist RGB, depth, intrinsics and optical-pose arrays are
identical to the corresponding archived October 30 captures. The first measured
capture gap increased from **0.884117 s to 1.341875 s**. Under the unchanged
0.25 m/s declared motion bound and 5 mm margin, the association displacement
expands from **0.226029 m to 0.340469 m**. This admits two additional same-colour
pixels: the whole measured union grows from 262 to 264 points, extending to
z=0.792 m rather than the object's measured patch ending at z=0.561 m. Fitting the
complete union rejects rather than hiding those inconsistent points.

Offline replay with each recording's original source clock reproduces **81/81**
for the historical run and **1/81** for the new run. Four tracking/geometry modules
are byte-identical to the pre-merge #226 source. This isolates an inherited
timing sensitivity in the declared association profile, not changed perception
code from main. It does not establish why startup/rendering took longer or a
reliable live capture cadence. No timestamp, speed bound, fitting tolerance or
identity reset is changed to recover the old number. The two added synthetic
gap regressions preserve rejection of newly reachable foreign pixels and latched
loss, even when the images themselves are unchanged.

The historical 81/81 result remains a single development recording, not evidence
of robust live association. Timing/appearance robustness and self-occupancy are
separate prerequisites for the first actual observed transfer. The foundation
can be reviewed and landed with those gates explicit; no full observed executor
or hardware performance is claimed.

## Provenance

[Compact record](pr226_main_integration_20261116.json) includes source identities,
test counts, replay metrics and 17 artifact hashes. Raw logs, original captures,
replay code, control receipts and video remain under
`.tamp-runs/pr226-main-integration-20261116/`; historical captures stay in the
earlier checkout archive. These raw artifacts are not publicly archived. Counts
overlap and the 81 captures are correlated, not independent robot trials.
