# Robot capability media

`examples.json` selects real run media for the qualitative appendix included by
`paper/main.tex`. Existing absolute paths identify the source artifact store on
the experiment workstation; update them if that store is mounted elsewhere.
No image generation model is used. The historical controls are not current
physical acceptance evidence.

```bash
MPLCONFIGDIR=/tmp/emet-mpl .venv/bin/python scripts/render_capability_media.py \
  paper/data/capability_media/examples.json \
  --output /tmp/emet-capability-gallery --figure paper/figs/capability_media.png
```

Open the generated `index.html`. The output directory is a portable gallery:
original stills/videos, evidence, and a JSON manifest with source paths and SHA-256
hashes. Upload the whole directory to the usual artifact host when available;
link its index from the PR. Videos stay out of the source tree.

For each new capability PR, include a short real recording, before/after stills,
run ID and implementation commit, measured outcome, assistance labels, and a
representative failure where relevant. Navigation should show camera and map;
manipulation should show arm/payload and its destination; opening should show
closed/open states. Keep failed runs, and never label assisted motion physical.

The live CHAT smoke supports `--record-mp4 --video-out PATH`. It saves per-tool
stills and a sibling JSON outcome manifest. Recording adds one-second dwells
before and after each action for visibility; those runs are not latency
benchmarks. Camera stills are sampled after receipts but do not independently
certify measured success: use the paired JSON tool outcomes. Assisted joint
teleports are shown as discrete changes, without fabricated interpolation.

The original waiting capture `20261009_173350_a866a0` was cancelled and superseded
by managed job `20261009_175205_1cc9bd` at frozen source `3f9a599a`. It records the
open/close cycle, full placement regression, RBY1 and Innate Mars controls under
`/home/cpaxton/runs/emet/placement-refresh-media-20261009/`. It is queued; inspect
`suite.json`, case logs and actual frames before claiming new footage or success.

## Task and skill overlays / robot comparisons

Both smoke runners accept `--video-overlay none|banner|border` and optional
`--video-flags TEXT`. Border mode places task, active skill/tool and flags outside
the camera view; none preserves clean video. The stepwise TAMP runner labels
approach/grasp/place with semantic object and receptacle names. The CHAT runner
labels the active public tool; it does not claim visibility into internal skill
transitions. `GT` means ground-truth geometry; `ASSISTED` flags enumerate simulator
help rather than claiming physical manipulation.

`run_capability_media_suite.py` records independent open/close, full placement,
RBY1 and Innate Mars cases under a managed GPU job. RBY1 currently uses the Galaxea
R1 model. Each case keeps its log and process outcome even when another fails.
A zero process exit is not an independent physical-success certificate.
