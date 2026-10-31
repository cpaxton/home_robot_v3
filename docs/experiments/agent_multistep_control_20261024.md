# Agent-selected multi-step controller cases, 2026-10-24

Single-pose tracking is insufficient to assess TAMP execution. These development
pilots use a real model selecting public tools in existing physical simulator
scenes, retain state between transfers, and score object outcomes independently.
They exercise release, regrasp and controller/planner handoffs. No learned actor
or hardware was evaluated.

| Scene | Selector | Complete episodes | Verified transfers | Result |
| --- | --- | ---: | ---: | --- |
| Stretch, two object-specific table regions | Real-model CHAT tools | 1/1 | 2/2 | Both final goals independently verified |
| Stretch, same regions and object order | Scripted public tools | 1/1 | 2/2 | Complete reference also passes |
| Mars, delivery and return between supports | Scripted public tools | 0/1 | 1/2 | Return planning exhausts budget |
| Mars, same delivery/return | Real-model CHAT tools | 0/1 | 1/2 | Same return-planning failure |

These are selected development fixtures, not a held-out success rate. Do not pool
the robot, selector or transfer denominators. The Mars case lacks a successful
complete reference and is not admitted as an agent acceptance benchmark.

## Agent and provenance

The existing Caliban endpoint `http://192.168.1.55:8000/v1` advertised
`Qwen/Qwen3-VL-8B-Instruct`. These runs use the normal CHAT prompt builder,
parser, tool schemas and dispatcher, with an existing remote inference service.
They do not launch the interactive agent CLI, use camera observations, start a
model server or establish observed grounding. Exact server weight revision was
not reverified; raw responses are archived. Local seeds do not control the
remote sampler, which used server defaults.

The Stretch goals come from the existing frozen two-object regional fixture.
The model selects object order and plans, rather than authors these regions.
Both goals initially failed. It completed the cylinder and then the box in five
model turns. Each transfer uses physical grasp/place tools, and the final scorer
checks both whole footprints, released support and stillness over two seconds.
The in-process run has no server-side actuation audit; zero audit entries must
not be described as a validated zero-forbidden-command audit.

For Mars, the model authored an instruction and declarative support sequence
within a constrained delivery/return template. The program was saved before
either execution. This is not broad household task generation. The parser
accepts only known objects/supports and two to four alternating transfers; no
generated code, object relocation, attachment or reset instruction is executed.
The pilot is frozen fixture `left:20g:0`, with the unchanged 20 g payload and
original placement/retention/force/wheel limits.

Each Mars stage requires fresh discovery, a matching support/task handle, the
exact frozen region and a newly returned plan handle. Successful tool receipts
are followed by independent placement/support/grip/robot-stop checks and two
seconds of stability. Failures remain failures; an unverified tool success is
reported as partial and stops execution. No fault is cleared between stages.
Physics advances during motion and verification, and is paused during model
inference. Therefore this is not live asynchronous or disturbance-reactivity
qualification.

## The useful Mars regression

The scripted reference and agent start from exactly matching qpos, qvel, actuator
controls and simulation time. Both physically complete the delivery. The first
placement remains supported and stable. The subsequent return fails to produce
a plan within the existing **256 IK-call budget**: 39 pregrasp IK rejections,
one lift IK rejection and three surface-candidate rejections are retained.
Budget exhaustion is not proof of physical infeasibility.

No wheel-unloading or unsafe-force event occurred in these Mars pilots. Peak
recorded contact force was **3.32 N**. The agent reaches the same planning
failure, then exhausts its 20-turn budget; do not attribute that to learned
tracking accuracy. In particular, a successfully tracked isolated carry does
not establish the ability to regrasp from the released placement/posture.

The next controller/planner case is this exact post-placement return: investigate
grasp frames, clearances, base approach and reachable certificates without
changing the task or safety thresholds. Export accepted pose schedules and
command/measured traces for tracking comparisons, but never label the rejected
return as intrinsically infeasible or use its unknown feasibility as a negative
value-function target.

![Commanded and measured first-transfer paths and loads](../../paper/data/whole_body_control/agent_mars_shuttle_prefix.png)

The plot shows the agent-selected episode using the reference controller. Its
first transfer completes; there is no motion trace for the rejected return.
`FAIL` refers to the full two-stage task. The 5 mm line is a tracking diagnostic,
not a claim of a per-instant bound throughout the executed path.

## Learned-controller handoff

The existing learned Mars bundle was checked against the two-support fixture
before actor execution. It fails with `manipulation model mismatch`: bundle
compatibility currently binds the whole training scene, so adding the external
support changes its fingerprint. This check was not bypassed. The pilot facade
also uses the reference controller; it has no learned-actor execution switch.
The successful Stretch/Mars transfers above are therefore not learned-policy
results and cannot be compared with the earlier 0/5 policy validation.

The integration work is to separate verified robot/actuator/dynamics compatibility
from per-case world geometry, preserve independent scene collision/contact
validation, and wire the learned actor to the same measured pose/gripper command
stream. An unexplained model mismatch must remain a rejection. Keep this task
distinct from fixing the failed return planner and do not silently relax bundle
checks to obtain an apparent policy result.

## Reproduction and next gate

The [development inventory](../../configs/benchmarks/agent_multistep_development_v1.json)
records these families. It was assembled after the pilots and is not a pre-run
held-out catalog. Raw goal programs, model outputs, commands, stage predicates,
pose-reference schedules, source identities and measured traces are archived in
[`agent_multistep_v1.json`](../../paper/data/whole_body_control/agent_multistep_v1.json)
and its adjacent compressed full report, with an exact Python source snapshot of the
pilot implementation. Later regression-binding/telemetry additions must not
silently relabel those pilot source identities.

The new simulator-only runner is:

```bash
taskset -c 14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m emet.eval.mars_tamp_sequences \
  --output logs/eval/mars-new-multistep-run --case left:20g:0 \
  --base-url http://192.168.1.55:8000/v1 \
  --remote-model Qwen/Qwen3-VL-8B-Instruct --max-turns 20
```

The canonical production agent readiness gate remains unchanged. This runner
is an explicitly separate development path for generating diagnostics while
that gate is false; its outputs do not authorize promotion or hardware motion.
Do not replace the unchanged canonical suites with easier generated tasks.

After a complete Mars reference is available, freeze additional nearby layouts,
payloads, multi-object tasks and interruptions before evaluation. Compare the
teacher and learned actor on exactly the same goal programs and initial states,
reporting planning admission, controller execution, retention, smoothness,
contact loads, wheel support and complete episode success separately. Full
RoboCasa/MolmoSpaces scene bindings for Mars remain unimplemented; configuration
files marked design-only are not executable benchmarks.

The final runner/regression-binding source passed **606 tests**, with no failures,
errors or skips. The archived source-bound
[receipt](../../paper/data/whole_body_control/mars-sequence-regressions.receipt.json)
and adjacent XML include the new program-validation, multi-transfer dispatcher
and frozen-region/no-motion tests. The pilot Python source snapshot predates
the final telemetry/receipt-binding additions and is archived separately so the
experimental source identities remain exact. Runtime package parity and relevant
lint pass.
