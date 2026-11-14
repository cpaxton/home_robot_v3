# Native TAMP qualification and production agent integration

November 6 establishes two successful native development tasks, not broad
robustness or an updated agent baseline. This follow-up moves the exact policy
into the shared native adapter and tests it prospectively. Main was fetched on
November 13 and remains `cd169673`; no merge was needed. Existing evidence and
all failed attempts remain unchanged.

## Shared implementation

`agent.native_tamp_policy=legacy` preserves default behavior. Explicit
`bounded_release` uses the previously tested fine grasp yaw candidates, stable
loaded handoff, and measured release search. The handoff has one production
owner in `native_policy.py`; `TracedNativeAdapter` only wraps events when using
the shared policy. Both primitive tools and the native multi-step provider use
this implementation. Discovery publishes policy identity, caches invalidate on
policy change, and multi-step bindings include the selected identity.

The native path remains privileged, stationary, simulated Stretch. No hardware
trial, camera or local GPU inference is requested. Unsupported observed sources
must not use this profile as a privileged fallback. Final goals, collision and
retention thresholds remain unchanged, as do all previous scored populations.

## Frozen native population

`configs/eval/tamp_native_qualification_20261113.json` declares ten episodes:

- Two regression layouts: familiar and +15 mm initial X. These additionally test
  invalid-handle rejection before an explicitly renewed task, and consumed-handle
  rejection after success.
- Six prospective candidates: X shifts -30/+30 mm; Y shifts -12/+12 mm with cube
  yaw +0.12/-0.12 rad; buffered swaps at X -10/+25 mm. These are unseen by release
  tuning but are not independently reference-certified or a household benchmark.
- Two rejection episodes: stale task plan after ordinary measured head motion,
  and collision-blocked placement into the other object's occupied footprint.

There are eight positive task candidates (18 declared transfers) and four
interface-boundary checks. Rejection episodes have no positive-task success
claim. Keep every candidate in its declared population; report construction,
planning, execution and infrastructure failures separately. Do not exclude a
failed case or replace it with an easier position. An intended rejection passes
its boundary check only with its declared public code, error status and no
additional audit commands. Success after rejection is a separate explicit
scripted request, not automatic continuation after a failed tool batch.

Before live launch, all ten fixtures must compile, retain the canonical model
signature, validate their task JSON, and archive resolved positions/orientations.
This preparation is not a feasibility or execution certificate. Run at most two
CPU simulators concurrently using `emet jobs`; queued pairs wait for both prior
supervisors. Freeze source and inputs before launch, retain all raw traces under
`.tamp-runs/native-qualification-20261113/`, and commit only compact results.

## Agent gate and merge

After inspecting the native population and resolving concrete failures, repeat
actual `run agent` two-goal and buffered-swap tasks using Caliban's remote VLM.
Include the existing `--reject-first` separate-user-turn recovery scenario.
Read final measured goals independently; valid JSON, a planned skeleton or a
correct stop is not task success. Keep native reference scores separate from
agent task success, failure-boundary correctness and inference cost.

The November 13 endpoint check found `Qwen/Qwen3-VL-8B-Instruct` at Caliban's
existing service. Recheck before launching; record model, sampling, source,
inputs, tool calls, audits and every failure. The prior agent 2/4 remains intact.

Review the shared policy/config integration as one cohesive change on the
existing branch. Broader qualification, observed full-arm/contact sources,
navigation/exploration and household native bindings remain distinct acceptance
gates; do not make claims for them from stationary privileged success.

## Common simulator clock repair

At source `4a4a63ad`, both regression tasks and their two boundary checks pass.
An offline local diagnostic measures median model-fingerprint hashing at 7.025 ms
versus 2.718 ms for a physics step under the current load. Existing PR #228 caches
the compiled-model fingerprint and refreshes it on model replacement, retaining
per-tick qpos/qvel/timestamps. It is merged into this feature branch before the
complete qualification run. The completed pair is preserved separately; the
remaining eight old-launch jobs are explicitly cancelled (some began startup).
They are interruptions, not task successes or expected rejections.

Rerun the entire unchanged ten-case population on one new frozen source after
repair validation. Do not combine its task scores with the initial pair.
Preserve cancellation/transition records and the paired offline callback check
under `.tamp-runs/native-qualification-20261113/`; the complete replacement run
uses `.tamp-runs/native-qualification-clock-20261113/`. This repair changes
throughput, not physics parameters or task acceptance. The RGBD GPU throughput
probe from #228 is not rerun here; native CPU qualification remains camera-free.
