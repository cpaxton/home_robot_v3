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

Open/close capture queued: job `20261009_173350_a866a0`, implementation `098f2e45`,
output `/home/cpaxton/runs/emet/capability-media-20261009/`. Do not claim footage
or success until the job completes and its frames/outcomes have been inspected.
