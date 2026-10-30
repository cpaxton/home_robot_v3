# GraphEQA model API connection and paired context smoke

The external baseline now has a model-provider fork:
[cpaxton/graph_eqa, PR #1](https://github.com/cpaxton/graph_eqa/pull/1), branch
`feat/model-api`. It starts at upstream `d3fbbf14`. Its
[operator guide](https://github.com/cpaxton/graph_eqa/blob/feat/model-api/doc/model_api.md)
lists the adaptations, model profiles, input schema and commands.

## What is connected

Planner and room inference use a configurable OpenAI-compatible Chat Completions
endpoint. The provider/model is selected explicitly rather than inferred from
the model-name string. Profiles support provider JSON Schema, JSON-object mode,
or text-mode JSON with local validation; there is no silent mode/model fallback.
Room inference uses the profile's model unless a separate room model is declared.

The controller no longer receives a gold answer. Its target enums encode
instance IDs; repeated object labels no longer alias the second object to the
first. The frozen context adapter preserves the reference action registry and
uses the selected observed node's geometry. The controller's answer is retained
without an earlier-assessment override.

`scripts/run_paired_context_eval.py` in the fork runs models serially in two
conditions: image-only and reference graph. Both receive exactly the same
captured PNG and question/choices, with the same per-profile token limit. Gold
labels are accepted only as a separate scorer file read after all model calls.
Every request, image data URL, raw response, validation result and reported model
usage is saved. Failed rows stay in the planned denominator.

This runner can compare supplied model endpoints with repeated `--model-config
LABEL=PATH` arguments. Separate profile files declare each model/endpoint. The
local emet server advertises its canonical model ID through `/v1/models`.

## emet API repairs

The live connection exposed existing server contracts that were unsuitable for
model comparisons. Source commits `d4422b0a` and `cede090f` repair them:

- Reject a requested model that is not actually loaded. Report the canonical
  serving ID plus actual HF checkpoint and quantization, rather than echoing any
  requested model name as if it had run.
- Reject unsupported generation settings and structured-output claims before
  inference. This prototype server supports text mode; do not silently accept
  `temperature`, `seed` or provider-enforced JSON modes it does not implement.
- Preserve all leading schema/policy/question system messages and prior
  assistant turns when adapting to the HF vision dialogue contract.
- Advertise the supported tool protocol. CHAT reads the declared capability
  before selecting native API schemas; text-JSON providers retain the existing
  prompt-based tools. Explicit unsupported tool arguments are rejected rather
  than silently discarded, including before mutating client history.
- Mark a timed-out model handle unavailable before another dispatch, because a
  backend worker may still own CUDA resources.
- Forward measured Qwen token counts and EOS/token-limit receipts. Other
  backends without counters report unknown usage, not zero-cost inference.

The check driver `scripts/check_grapheqa_model_api.py` starts a local Qwen server
under the managed GPU lock, checks its actual loaded checkpoint, runs the fork's
pair, and terminates only the process group it created. No real robot is used.

## Live result

Managed job `20261030_130750_9a67a2` passed at emet source `cede090f` and fork
source `21b6876`. The server reported `Qwen/Qwen3-VL-8B-Instruct`, int4, CUDA.
Both requests used a 2,048 generated-token cap and decoded to EOS.

| Frozen condition | Prompt tokens | Generated tokens | Client wall time | Chosen MCQ letter | Confident? |
| --- | ---: | ---: | ---: | --- | --- |
| Current-image only | 445 | 99 | 2.75 s | B | No (0.0) |
| Reference graph + current image | 8,792 | 378 | 9.31 s | A | No (0.7) |

Artifacts:
`/home/cpaxton/runs/emet/grapheqa-model-api-pair-20261030-r2/`.
Its `interop.json` pins both sources and the loaded model; `paired/` contains the
manifest, predictions, scorer output, full model exchanges and copied image.
The owned server exited by SIGTERM after both calls completed; no GPU job remains.

The case is the first existing development question, q2, using the earlier ten
newly rendered RGB-D views, online YOLOE labels and local command odometry.
The frozen image is frame 9; the graph retains the earlier views. Thus this is a
**current-image control**, not a matched-history causal graph ablation. Neither
condition executes a target. This is connection validation, not active EQA or
navigation acceptance, and not evidence of superior answers.

The scorer-only label is B: the image-only guess happens to be correct, while the
graph guess is wrong. Both are uncertain. The current image visibly contains
window blinds, a plant and wooden flooring, with no visible rug. The learned
graph includes an unverified `shower cabin` label; its estimated center projects
near the upper window/curtain region in the current view (approximately pixel
473, 60). The graph response treats that label as bathroom evidence and guesses a rug
color from common knowledge. This is a manually observed perception/reasoning
problem to study in the active loop, not a basis for an answer override.

The first API attempt is preserved separately at
`/home/cpaxton/runs/emet/grapheqa-model-api-pair-20261030/`. Its image-only call
completed, but the graph call emitted a nested answer object instead of the
root planner object, repeated text and ended with incomplete JSON. It was
recorded as failed. The matched retry kept the case, model and token cap;
the generic text formatter now presents the root schema before nested
definitions, and the server reports genuine decode receipts. No answer
extraction, budget increase or model substitution was used to turn it into a pass.

Validation: 16 CPU tests in the fork cover provider modes, duplicate target
labels, identical evidence across different supplied model endpoints, scorer
isolation and retained errors. The emet server/Qwen/history regression set has
40 passing server/client/Qwen tests, plus the CHAT call-boundary regressions.
The native graph/context export also runs without credentials
or model calls in the isolated Spark DSG runtime.

## Remaining active-evaluation gates

1. Resolve the pose-input policy: matched benchmark RGB-D/pose on both sides,
   excluding semantic/object/map ground truth, or estimated odometry without
   simulator pose. The latter requires separate localization qualification.
   No choice is inferred from elapsed time; active motion has not been launched.
2. Connect observed-space frontier generation and common safe execution to the
   reference controller. Its current frozen context has no frontier actions;
   both live calls proposed no motion. API acceptance does not qualify target
   arrival, floor coverage, collision handling or closed-loop evidence collection.
3. Freeze a small paired active population and budgets across emet and the
   learned-perception GraphEQA adaptation. Preserve the model's decision and
   actual context; do not add answer-locking or task-specific recovery rules.
4. Separately evaluate current/uniform/retrieved/selected history for the stronger
   image-only controls. Record actual input images, model calls/tokens, physical
   receipts, sim time and wall time before considering a wider sweep.

See the [sensor-only input audit](grapheqa_sensor_only_qualification_20261030.md)
for the upstream semantic-sensor, enrichment, dataset-filter and navmesh accesses
that must remain outside this active comparison.
