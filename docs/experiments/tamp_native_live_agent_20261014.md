# Live native-contact agent trials and IK binding investigation

This follows the paused-physics regional agent tests in
[the regional experiment](tamp_native_regions_20261013.md). The new evaluator uses
the CHAT prompt, parser, tool schemas, and dispatcher over a live Stretch ZMQ
server. Physics continues while the model deliberates. It does not launch the
interactive `run agent` CLI and does not establish exploration or observed-input
performance.

## Implementation and fixed protocol

Branch `fix/native-tamp-live-validation`, based on merged main `b2b71216`:

- `c296f632`: live ZMQ facade and audited regional agent evaluator.
- `14a7dcb8`: explicit simulation-only, camera-free client startup.
- `69392852`: flat-array pose IK and inference timing/protocol provenance.

The frozen Stretch fixture starts with both goals unmet. The model must move
the red cylinder into XY region `[[-.08, -.86], [.08, -.74]]` and the blue cube
into `[[-.08, -.75], [.08, -.58]]`. Initial transforms are set in the scene before
startup, not through runtime object writes. Inputs and independent scoring are
explicitly simulator ground truth. Acceptance requires both entire footprints
inside their regions (5 mm tolerance), released table support, and stillness
through two seconds of measured simulation time. Every inference request must
advance simulation time; the server audit must be nonempty and contain zero
forbidden actions. Grasp retention, collision checks, and IK tolerances were not
relaxed.

Inference uses Caliban's existing OpenAI-compatible server at
`http://192.168.1.55:8000/v1`, model `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, float16, greedy decoding
(`do_sample=False`). Server source is `d07af78ac75d844ebcede886eada7e3b3b231828`;
server-file SHA256 is
`a3a20dc0d560176d574284e7a3cce94df018a373164b0abfa32ce4e3966c3ad1`.
The local seed does not control a remote server's sampling settings.

The final trials run concurrently on separate ports (offsets 320 and 330), with
camera rendering disabled and CPU affinity `0-7,12-31`. They require no local
model weights or GPU reservation. Cameras are deliberately absent: this is a
text-tool integration test, not a visual perception demonstration. Artifacts
contain the scene/config copy, source SHA, prompt, raw responses, tool outcomes,
simulation timestamps, execution events, audit, and server log.

## Results

Artifact directories below are under `/home/cpaxton/runs/emet/`.

| Run | Source | Outcome | Managed job | Artifact directory |
| --- | --- | --- | --- | --- |
| Live pilot | `14a7dcb8` | Both goals passed; 410 audited commands, zero forbidden | `20261014_015456_6bd598` | `native-live-agent-state-only-20261014` |
| Final repeat 1 | `69392852` | Both goals passed; 436 audited commands, zero forbidden | `20261014_020034_25c6bb` | `native-live-agent-final-1-20261014` |
| Final repeat 2 | `69392852` | Both goals passed; 410 audited commands, zero forbidden | `20261014_020035_19edb0` | `native-live-agent-final-2-20261014` |

Final-source result: **2/2 two-goal episodes, 4/4 transfers, 846 audited commands,
zero forbidden actions**. Both goals were initially unmet, and every inference
call advanced physics. Each episode took six model turns, including one malformed
JSON response rejected before tool dispatch; the model recovered on its next turn.
Both final objects independently passed region, released support, and stillness
checks. No IK/native crash occurred in these episodes.

Do not pool the pilot with the final source, earlier paused-physics trials, or
scripted single-transfer controls. Repeats use the same fixture and goals; this
is integration evidence, not held-out task generalization.

Earlier infrastructure attempts are retained, not silently discarded:

- `20261014_014626_576f17`: GPU pilot cancelled while queued, before execution.
- `20261014_014809_a5b403`: software-camera diagnostic cancelled after partial
  execution because rendering made physics impractically slow. No completed
  episode result; artifacts in `native-live-agent-software-20261014`.
- `20261014_015156_faead2`: camera-free startup failed before LLM execution.
  The client waited for image streams intentionally disabled by `no_cameras`.
  `state_only=True` now accepts state-only startup only when simulation metadata
  explicitly advertises integer `num_cameras=0`. Hardware and default clients
  retain their stream requirements. Artifacts in
  `native-live-agent-no-cameras-20261014`.

## Native crash: evidence and remaining uncertainty

MuJoCo 3.5.0 / CPython 3.10 crashed at the same instruction in `_structs` as the
earlier pose-IK incident. Managed probe `20261014_014556_772f85`, pinned to CPU 8,
reproduced SIGSEGV while repeatedly reading named joint views. It performed no
IK, physics stepping, rendering, or robot actuation. GDB evidence is in
`ik-named-core8-20261014/probe.log` (ELF instruction offset `0x1375c9`).
GDB's successful exit is not a crash-free probe result.

The same probe passed on CPU 0 (85,500 lookups) and the safe CPU mask (90,000).
However, subsequent pure-MuJoCo and optional-import isolation probes also passed
on CPU 8. The crash is intermittent; these results do **not** establish a
particular import as its cause or prove a hardware defect. No host settings or
installed package binaries were changed. A bounded diagnostic entry point is
now `scripts/diagnose_mujoco_named_access.py`; the original failing runner and
probe must be retained alongside it because the minimized probe has not reliably
reproduced the failure.

Pose IK now resolves names with `mj_name2id` and uses flat arrays, avoiding
`model.joint()`, `model.body()`, and `data.body()` entirely. A regression test
forbids those methods while solving a nontrivial pose. This removes the observed
binding path from pose IK; it does **not** claim to repair every possible native
runtime fault elsewhere in the process. The CPU affinity mitigation remains
part of the recorded trial protocol.

Controls are archived in `ik-named-diagnosis-20261014`,
`ik-named-core0-20261014`, `ik-pure-probe-20261014`,
`ik-pure-ordered-20261014`, `ik-import-bisect-20261014`, and
`ik-pure-global-20261014`.

## Regression checks and next gate

At final runtime source `69392852`, the combined motion/contact/payload,
native adapter, TAMP API, agent dispatch/prompt/tools, regional scoring,
live-facade, client-thread, and core ZMQ checks passed: **245 passed, no skips**
(`native-live-agent-final-1-20261014/validation/native-live-final-tests.xml`).
The same validation directory archives diagnostic runners and their checksums.
A separate observation-frame batch had
21 passed and 7 pre-existing skips for unavailable Robocasa. Ruff passed on the
changed Python files. These are targeted regression suites, not all-repository
acceptance.

The next gate is held-out initial placements and explicit failure/recovery
episodes through the same live interface, followed by the actual `run agent`
session with observed grounding, navigation, and exploration. Do not infer
those capabilities—or hardware sensor availability—from privileged stationary
success. Native-runtime root-cause investigation remains separate from task
success metrics.
