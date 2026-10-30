# Shared agent literature review and baseline design

Review date: 2026-10-29. The target is one agent for EQA, object search, OVMM
and multi-step manipulation. The immediate evaluation question is whether our
memory and tools improve on the same model receiving images alone.

The most consequential comparisons are **images-only inference, simple image
retrieval, a minimal active agent, 3D-Mem, GraphEQA, ToolEQA and NavMCP**.
Static graph versus lazy graph compares two internal system configurations; it
does not isolate memory effects or establish the value of the harness over a
model-only baseline. The priorities
and proposed experiments below are our assessment, not reported paper results.

## Closest work

| Work and primary source | Relevant mechanism | Implication for our experiments |
|---|---|---|
| [OpenEQA, CVPR 2024](https://open-eqa.github.io/) | Separates answering from recorded experience (EM-EQA) and gathering evidence through action (A-EQA). | Evaluate the answer/context layer separately from exploration. Include the question-only control and direct multi-image VLM. |
| [GraphEQA, updated 2025 paper](https://arxiv.org/html/2412.14480v2) | Hierarchical 3D scene graph, current RGB plus two task-relevant images, navigation to object/frontier nodes, and confidence-based stopping. | Our `static_graph` is an internal adaptation. A restricted action schema alone does not reproduce the reference. See the protocol below. |
| [3D-Mem, CVPR 2025](https://arxiv.org/abs/2411.17735) | Memory snapshots group co-visible objects; frontier snapshots expose unexplored regions. Incremental construction and retrieval support repeated tasks. | A high-priority comparator for our image selection and scene coverage. It uses RGB-D/pose and object information; it is not raw RGB-only inference. |
| [ToolEQA, 2025 preprint](https://arxiv.org/html/2510.20310v2) | Explicit planning and tool execution, including navigation, object localization, crops and visual QA. Reports zero-shot and EQA-RT-trained variants. | Closest direct tool-use baseline. Match the frozen backbone for a zero-shot adaptation; report trained variants separately. |
| [NavMCP, August 2026 preprint](https://arxiv.org/html/2608.30396v1) | Couples a reasoning VLM and navigation foundation model through intent, trajectory evidence and persistent memory. | Closest context/interface comparator. Test terminal-frame-only versus intermediate keyframes with the same navigator. Its main and controlled runs use much larger reasoning backbones than our local 8B model. |
| [FAST-EQA, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Zhang_FAST-EQA_Efficient_Embodied_Question_Answering_with_Global_and_Local_Region_WACV_2026_paper.html) | Question-conditioned region/target hypotheses, bounded memory and efficient exploration. | Important for the runtime/coverage claim: bounded memory and faster EQA already have direct prior art. Compare quality at matched acquisition and compute budgets. |
| [MemoryEQA, 2025](https://arxiv.org/abs/2505.13948) | Global scene-map memory and local observation/state memory feed exploration, stopping and answering. Introduces multi-target, multi-region MT-HM3D. | A stronger memory test than only single-object questions. Multi-region comparison can expose failures hidden by a good final close-up. |
| [DIVRR / DynHiL-EQA, March 2026 preprint](https://arxiv.org/abs/2603.09541) | View refinement and selective memory admission address ambiguous, occluded and changing evidence. | Directly relevant to delayed graph admission and looking again before committing geometry. Avoid claiming these ideas are unique to our system. |

### Other memory comparisons worth retaining

| Work and primary source | Relevant mechanism | Role |
|---|---|---|
| [ReMEmbR, 2024](https://arxiv.org/abs/2409.13682) | Retrieval over descriptions with spatial, temporal and image information for long-horizon robot queries; introduces NaVQA. | Practical lightweight retrieval comparator and recorded-robot-data evaluation. It is not a complete OVMM stack. |
| [LVMem, October 2026 preprint](https://arxiv.org/abs/2610.27104v1) | Adds original-image retrieval to ReMEmbR-style memory, with optical-flow frame selection and richer object descriptions. | Especially close to the graph-free alternative. Tests whether retrieving the right image suffices. New preprint; code/runtime reproduction still needs qualification. |
| [Embodied VideoAgent, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.pdf) | Persistent object memory combines egocentric video with depth/pose, updates after actions, and supports navigation/pick/place primitives. | Strong overlap with a shared memory/action story. Distinguish its memory and task demonstrations from our measured physical execution gates. |
| [HIMM, 2026](https://arxiv.org/abs/2602.15513v2) | Separates episodic retrieval with visual verification from reusable semantic memory. | Useful for long-history and repeated-task comparisons. Keep cross-episode learned/extracted rules separate from within-episode history. |
| [R4, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Sohn_R4_Retrieval-Augmented_Reasoning_for_Vision-Language_Models_in_4D_Spatio-Temporal_Space_CVPR_2026_paper.pdf) | Object observations are indexed in space and time for structured retrieval and reasoning. | Comparator for moved objects and temporal queries; persistent object/time indexing is established prior art. |
| [Embodied-RAG, updated 2025](https://arxiv.org/abs/2409.18313v5) | A semantic forest supports retrieval at different spatial/semantic resolutions for navigation and language output. | Relevant to scaling memory across environments and query granularity, rather than a drop-in manipulation policy. |

### Navigation and manipulation lineage

[Explore until Confident / Explore-EQA](https://arxiv.org/abs/2403.15941)
combines semantic-guided exploration with conformal stopping. It supplies a
relevant active baseline and a distinct stopping-policy comparison; calibration
needs its own calibration split rather than our development answers.

[DynaMem](https://arxiv.org/abs/2411.04999) already provides online dynamic
spatio-semantic memory for object localization and mobile manipulation.
[OK-Robot](https://arxiv.org/abs/2401.12202) establishes the importance of
integrating pretrained perception, navigation and grasping modules. Their
reported task success is not directly comparable to our EQA percentages or
privileged stationary simulator control.

[ConceptGraphs](https://arxiv.org/abs/2309.16650) provides an object-centric
open-vocabulary scene representation; [SG-Nav](https://arxiv.org/abs/2410.08189)
uses online scene graphs and re-perception for object navigation. They are
representation/search references, not complete matched EQA-to-manipulation
baselines. [SayPlan](https://arxiv.org/abs/2307.06135) combines hierarchical
scene-graph search, classical planning and feedback-driven replanning;
[Inner Monologue](https://arxiv.org/abs/2207.05608) studies closed-loop language
feedback. Tool orchestration and replanning should therefore not be presented
as new on their own.

[VoxPoser](https://arxiv.org/abs/2307.05973) grounds language in composable 3D
value maps for closed-loop manipulation. [TiPToP, version 2](https://arxiv.org/abs/2603.09971v2)
combines foundation-model perception with GPU-accelerated task-and-motion
planning and evaluates real/simulated manipulation. Use these to position the
manipulation layer; our recent ground-truth stationary Stretch pass does not
establish comparable learned perception or broad manipulation performance.

## Baseline families

### Fixed observations first

Freeze recorded trajectories before comparing context strategies. Each arm
receives the same question and eligible RGB history; sampling must not use
ground-truth targets, reference answers, success frames or future observations.
This isolates evidence presentation and answering from navigation.

| Arm | What the model receives | What it tests |
|---|---|---|
| Question only | Question/options, no images or memory | Language priors and answer leakage |
| Current image | Question and the current captured RGB | Whether history is needed |
| Context images | Current image plus uniformly spaced historical images; no tools, graph, voxel queries or captions | Primary model-only baseline requested by the user |
| Retrieved context | Current image plus simple text-image top-K retrieval from the same history | Whether learned retrieval suffices without a graph or interactive tools |
| Selected context | Our view-selection policy, otherwise identical to context images | Incremental value of coverage/proximity selection |
| Full memory context | Same image budget plus our retrieved semantic/graph evidence | Incremental value of structured memory beyond RGB |

Evaluate frame budgets such as K=1, 4, 8 and 16 where the backbone permits, with
the current frame counted in K and pinned consistently. Compare the main rows
at one predeclared budget, then inspect a small budget curve. Uniform selection
must be chronological, deterministic and free of answer annotations. Include a
recent-window check if uniform coverage misses transient evidence.

Give every image an unambiguous sequence index and provenance in the baseline
with image metadata; run RGB-only versus RGB plus timestamps/poses as a separate
factor. A baseline fed our question-conditioned selector is already benefiting
from part of the harness, so retain uniform context as an independent row.
Candidate paths/IDs are attribution metadata, not semantic answer hints.

Replay each arm on trajectories from multiple collectors, including a fixed
task-independent exploration policy. Using only our successful trajectories
would privilege our acquisition policy and hide failed search. Counterfactual
replay measures answer quality conditional on a captured trajectory; it cannot
establish that an alternative agent would acquire those views. Static replay
also cannot measure the physical consequences of earlier stopping.

Report accuracy, answerability/abstention, relevant-evidence retention, image
count, visual/text tokens and inference latency. Inspect paired answer changes
and exact request packets. The model supplies the answer; evaluation code must
not replace it with an earlier assessment or add an answer-selection heuristic.

### Active agents next

A zero-tool agent cannot physically explore or manipulate. For active tests,
the simplest useful comparator is the same VLM with a small action interface
and RGB history: movement, gaze/observe, answer, and the shared manipulation
primitive when the task requires it. Keep action execution, robot limits,
collision checks, sensor provenance and outcome reporting identical.

| Arm | Agent-facing additions to the minimal interface | Controlled comparison |
|---|---|---|
| Minimal active VLM | None: RGB history and measured action results | Value of semantic memory and higher-level tools |
| Minimal + image retrieval | Retrieve previous RGB evidence | Value of retrieval without explicit object/graph memory |
| Voxel memory | Semantic voxel retrieval; no object graph | Value of spatial retrieval; internal DynaMem row |
| Restricted graph controller | Graph context plus object/frontier navigation and answer/stop | GraphEQA-style adaptation with matched perception |
| Shared lazy agent | Full current memory/tools | Integrated-system outcome |

Removing semantic voxel retrieval is different from removing the geometric
map used by a collision checker. Preserve the common low-level navigation map
when ablating agent-facing memory, and name that row accordingly. A literal
RGB-only navigator changes the control stack and belongs in a separate complete
system comparison. Likewise, disabling optional recovery means disabling its
automatic policy, not fabricating success or removing measured error feedback.

Screen targeted differences rather than a Cartesian product: semantic voxel
retrieval off, graph off, optional high-level tools off, and intermediate
trajectory evidence versus terminal evidence. Keep the model, prompts unrelated
to the removed interface, history policy, perception and motion budgets fixed.
Grounding and geometry required to execute a grasp remain the same low-level
capability in both manipulation arms. Count internal retries and hidden VLM
perception calls in cost; one high-level call can contain substantial work.

## A faithful GraphEQA comparison

The reference paper specifies a layered Hydra graph, room/frontier enrichment,
structured history and current state, object/frontier navigation, and a stopping
policy. Its visual packet uses two SigLIP-relevant historical images plus the
current view. Simulation uses HM3D-Semantics masks; real robot experiments use
Detic. These are material method/input differences. [Paper methods and Appendix A](https://arxiv.org/html/2412.14480v2).

Keep three names separate:

1. **Internal static graph:** the existing implementation, useful for our memory
   regression tests and system comparisons.
2. **GraphEQA controller adaptation:** implement the reference planning/context
   policy with our frozen backbone, common sensor-perception provider, navigation
   and eligible observation stream. Restrict actions to graph object/frontier
   navigation and answer/stop. Declare deviations from Hydra, enrichment,
   segmentation, prompts or stopping. This isolates policy more cleanly but is
   still an adaptation.
3. **Upstream GraphEQA reproduction:** use the upstream implementation/config,
   released episode selection and initial states, graph construction, camera
   settings, visual selection, perception inputs, prompts and stopping behavior.
   Freeze a source revision, then qualify a small pilot. A different VLM is a
   named model-swapped reproduction rather than the original published setting.

For the adaptation, sharing the detector/mask provider is a legitimate control;
there is no need to force our production harness to depend on Detic. For the
reproduction, changing perception or substituting our object graph must be
reported as a deviation. Keep privileged-semantics and sensor-only rows separate.
The graph-only and vision-only ablations in GraphEQA itself are useful starting
points for the comparison design.

Use the released question IDs and poses, not an equally sized prefix. In
particular, the reference paper's 113-question HM3D A-EQA subset is not our
historical first-113 HM-EQA slice. [Dataset definitions](https://arxiv.org/html/2412.14480v2).

## Task coverage and paper claims

[EXPRESS-Bench / Fine-EQA](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Beyond_the_Destination_A_Novel_Benchmark_for_Exploration-Aware_Embodied_Question_ICCV_2025_paper.html)
emphasizes evidence acquisition and exploration-answer consistency.
[AbstainEQA](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_When_Robots_Should_Say_I_Dont_Know_Benchmarking_Abstention_in_CVPR_2026_paper.pdf)
pairs answerable and abstention-requiring questions. Both remain useful for our
existing adapter plan: metadata ingestion alone does not implement their active
protocols or official scorers. Preserve the official passive-vision setting for
abstention; adding actions that reveal missing evidence defines a different task.

MT-HM3D is a particularly relevant additional candidate for multi-region memory;
NaVQA offers long-horizon recorded robot histories. Qualify one small frozen
population before adding dataset loaders or infrastructure. Support the shared
agent claim with an EQA -> find -> move -> revisit/answer sequence in the same
scene and persistent memory, including stale observations and ambiguous targets.
Use learned object geometry for the end-to-end row; privileged TAMP remains a
separate physical feasibility/control condition.

Proposed order:

1. Implement and qualify fixed-trajectory current/uniform/retrieved/selected RGB
   inference. Reuse already captured histories for a cheap paired screen.
2. Add the minimal active VLM and a clearly labeled GraphEQA controller
   adaptation on the existing EQA/small-OVMM pilot. Qualify the original GraphEQA
   reproduction separately, without blocking the simple controls.
3. Prioritize 3D-Mem and zero-shot ToolEQA for executable external comparisons;
   audit NavMCP's release/dependencies and its terminal-only ablation next.
   A paper/project link is not a qualified runnable baseline.
4. Evaluate graph/voxel/tool removals only where the integrated pilot actually
   exercises them. Retain failures and infrastructure denominators.
5. Repeat across scenes/seeds and one additional fixed backbone after the
   development screen. Freeze the assessment set before tuning.

Measure task quality, evidence coverage, localization error, physical completion,
stale-target handling, memory growth, model calls/tokens, path length, simulator
time, wall time and failure stages. Report mapping, perception, planning and
execution costs separately. Similar accuracy at lower runtime is an efficiency
result, not proof of superior reasoning. Published numbers from different models,
scenes, semantics or scorers remain contextual.

The plausible contribution to investigate is an economical, transferable
evidence-to-geometry-to-action lifecycle across tasks. Tool calling, dynamic
memory, visual retrieval and feedback-driven replanning already exist in the
literature. Graph admission has to earn its complexity against the image and
voxel baselines; a good graph-free result remains a valid system outcome.

## Review scope

Method details above were checked against linked primary papers/project pages.
Recent preprints are labeled as such. This is a targeted review for selecting
baselines, not an exhaustive survey or a reproduction of external leaderboard
scores. Runtime source/dependency audits and local pilots for external systems
are still qualification work. No additional model/simulator sweep was launched
for this review.
