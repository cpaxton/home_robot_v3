# GraphEQA without privileged scene inputs

The comparison must construct perception and memory online. Do not supply
ground-truth semantic masks, object IDs/coordinates, precomputed scene graphs,
maps, room labels, annotated target keywords, or answers to either agent.
Cached model weights and a fixed global vocabulary are allowed. Segmentation
must run on the episode's RGB, with its inference cost included in runtime.

## Audit of the reference runner

Pinned reference: [GraphEQA `d3fbbf14`](https://github.com/SaumyaSaxena/graph_eqa/tree/d3fbbf14bdb05631a1633556d7d3056601374cd3).
Setting `use_semantic_data=False` is insufficient:

- `envs/habitat_interface.py` still declares a semantic sensor and constructs
  a global navigation graph from the simulator navmesh.
- `scripts/run_vlm_planner_hmeqa_habitat.py` uses full-scene bounds and shortest
  paths. Its annotation-enrichment file is also read independently of the flag.
- `utils/data_utils.py` selects annotated scenes when semantics are enabled and
  **nonannotated** scenes otherwise. Toggling the flag changes the population.
- The learned Detic branch needs missing Detectron2/Stretch dependencies and
  has a user-specific absolute import path. The no-semantics Hydra initializer
  refers to `config/label_spaces`, while the current files live in `cfg`.

Our historical Habitat rows with `--no-hm3d-semantics` disable semantic ground
truth, but can still use navmesh-assisted movement. Keep that qualification;
do not relabel those results as a fully sensor-only navigation comparison.

## Minimal mapper qualification

`scripts/probe_grapheqa_sensor_only.py` renders the initial view for explicit
HM-EQA question ID 2, then makes two 10-degree left turns. Question 2 is the
first existing development ID, chosen before seeing probe results. The question
row is used only to select the renderer's scene and prescribed spawn.

The detector uses the reference's **global** HM3D vocabulary and our cached
YOLOE model. It predicts masks afresh from each RGB frame; undetected pixels
remain unknown, including when detector class zero is present. This is a
perception substitution, not an unchanged upstream Detic reproduction.

Only RGB, metric depth, calibration, camera pose from commanded in-place turns,
timestamps and the newly predicted label image enter Hydra. The initial local
base pose is zero; camera height/tilt are mount calibration. No simulator global
pose is sent to the mapper. This stationary probe does **not** qualify odometry
under translation, collision or real-world drift.

The native process has no GPU, network, question files, simulator scene assets,
or annotation directories mounted. It receives only those packets, a static
vocabulary, the two small driver/contract files and its output directory. Its
dependency image and reference label-space revision are pinned. The manifest
records model/source/config hashes, fresh-packet hashes and native receipts.
Input/schema failures, detector errors, rejected frames, empty graphs and
container failures stop the probe rather than substituting semantic ground truth.

Example, from a clean committed checkout with cached local weights:

```bash
PYTHONPATH="$PWD/src" EMET_UV_RUN=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  /home/cpaxton/src/home_robot_v3/.venv/bin/emet jobs run \
  --name grapheqa-sensor-probe --need-mib 4000 --cpu-safe --gpu-exclusive \
  -- taskset -c 12 /home/cpaxton/src/home_robot_v3/.venv-habitat/bin/python \
  scripts/probe_grapheqa_sensor_only.py capture \
  --upstream /home/cpaxton/src/home_robot_v3/.worktrees/grapheqa-upstream-20261030 \
  --output /home/cpaxton/runs/emet/grapheqa-sensor-probe-20261030
```

The cached dependency image is
`sha256:1cab965fa1815e43854248d62bbf910454b12c791584df95111f3b87d0fabde3`.
Hydra source inside it is `af6f2c234f46651734b1adbaed3404543b42592b`.
The native import emits duplicate `asl`/`stdout` registration messages. Actual
frame integration and graph serialization, not import success, are the next gate.

## Gates before an active comparison

1. Complete the mapper probe and inspect rendered RGB, predicted masks and the
   resulting graph. A mapper pass is not an answer or navigation success.
2. Retain the reference's graph/context/action/stopping flow while substituting
   only learned perception, question-derived text and the fixed comparison VLM.
   Keep the gold answer exclusively in the scorer; save actual model requests.
3. Use the same observed-map planner/executor, calibrated sensors and declared
   pose-estimation source for both methods. Derive bounds/frontiers from observed
   space or a fixed local workspace, never whole-scene geometry. A simulator may
   enforce physical collisions internally; its route oracle is not an agent tool.
4. Freeze explicit episode IDs before running, without inspecting semantic asset
   availability. Failures remain in the denominator. Match model, movement,
   image/token and wall/sim-time budgets. Do not tune on the final test set.
5. Run a small paired active pilot before the full sweep. Call this a **learned-
   perception GraphEQA adaptation** and enumerate changes. Published scores with
   different perception, navigation, models or populations are not matched rows.

Translation odometry and the observed-map execution adapter remain qualification
work. No active GraphEQA score is claimed by this probe.
