# Habitat EQA data

HM-EQA needs two things on disk:

1. **Question CSVs** — small; fetched from Explore-EQA / GraphEQA sources  
2. **HM3D scene meshes** — large; train split for full HM-EQA (~27GB habitat format)

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HABITAT_EQA_DATA_DIR` | `~/.cache/habitat_eqa/data` | `questions.csv`, `scene_init_poses.csv` |
| `HM3D_DATA_PATH` | `~/.cache/habitat_eqa/hm3d` | Root passed to `habitat_sim.utils.datasets_download --data-path` |
| `HM3D_SCENE_DIR` | `$HM3D_DATA_PATH/scene_datasets/hm3d/train` | Train split scene root (HM-EQA scenes) |

Override only when mirroring data elsewhere. If you previously set `HM3D_SCENE_DIR=~/.cache/habitat_eqa/hm3d/train`, **unset it** — that path is wrong after `datasets_download` (see [On-disk layout](#on-disk-layout)).

## Download helper

All commands run from the repo root (main `.venv` is fine):

```bash
# Print paths + credential checklist
uv run python scripts/download_habitat_eqa_data.py --instructions

# HM-EQA CSVs only
uv run python scripts/download_habitat_eqa_data.py --fetch-csv

# HM3D splits (requires ./scripts/install_habitat.sh first)
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d example   # ~150MB, no auth
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d minival   # ~400MB, API tokens
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train      # ~27GB, API tokens
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d val       # val split, API tokens

# Check a specific question's scene file
uv run python scripts/download_habitat_eqa_data.py --verify-question 0

# HM3D-Semantics (GraphEQA sim — GT instance masks; separate from train meshes)
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d-semantics train
uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics
```

`--fetch-hm3d` uses `.venv-habitat/bin/python -m habitat_sim.utils.datasets_download`.

See [HM3D-Semantics](data.md#hm3d-semantics-ground-truth-perception-in-sim) for why only ~37/113 paper questions have GT labels.

## HM-EQA CSVs

Fetched to `HABITAT_EQA_DATA_DIR`:

| File | Source |
|------|--------|
| `questions.csv` | [Explore-EQA questions.csv](https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/questions.csv) |
| `scene_init_poses.csv` | [Explore-EQA scene_init_poses.csv](https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/scene_init_poses.csv) |

`questions.csv` columns: `scene`, `floor`, `question`, `choices`, `question_formatted`, `answer`, `label`.

Example — **question 0** (first data row):

| scene | floor | question |
|-------|-------|----------|
| `00004-VqCaAuuoeWk` | 1 | Is the lamp next to the bed on? |

That scene is in the **HM3D train** split, not the free `example` pack.

`scene_init_poses.csv` uses GraphEQA layout: `scene_floor`, `init_x`, `init_y`, `init_z`, `init_angle` (scene id and floor joined as `00004-VqCaAuuoeWk_1`).

## Matterport credentials (HM3D train / val / minival)

### Use API tokens — not your website login

`habitat_sim.utils.datasets_download` talks to `api.matterport.com` with HTTP basic auth. Habitat expects a **token ID + token secret** pair from Matterport Developer Tools.

**Do not** use your Matterport account email and password. That fails with `Unauthorized` and leaves a tiny stub file (~56 bytes), not a real tarball:

```json
{"code":"request.unauthorized","message":"Unauthorized"}
```

### Where to get tokens

1. Log in to Matterport.
2. Go to **Profile → Settings → Developer Tools**  
   (direct link: [my.matterport.com/settings/account/devtools](https://my.matterport.com/settings/account/devtools))
3. Scroll to **Habitat dataset** / HM3D and **request access** if you have not already.  
   Approval is required before downloads work — a token alone is not enough.  
   Also see [matterport.com/partners/meta](https://matterport.com/partners/meta) and [aihabitat.org/datasets/hm3d](https://aihabitat.org/datasets/hm3d/).
4. After access is **approved**, create a **new** API token on the same page (regenerate if you made a token before approval).
5. Copy **both** values when shown — the **secret is only displayed once**.

| Matterport UI | Shell variable | Example shape |
|---------------|----------------|---------------|
| Token ID | `MATTERPORT_USERNAME` | short hex string from Developer Tools |
| Token secret | `MATTERPORT_PASSWORD` | longer secret string (not your account password) |

Export in your shell (do not commit these, do not paste into issues):

```bash
export MATTERPORT_USERNAME='<token-id-from-devtools>'
export MATTERPORT_PASSWORD='<token-secret-from-devtools>'
```

Smoke auth with a small split before the full train download:

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d minival
```

HM-EQA needs train:

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train
uv run python scripts/download_habitat_eqa_data.py --verify-question 0
```

If a previous attempt failed with `Unauthorized`, remove the bad stub before retrying:

```bash
rm -f ~/.cache/habitat_eqa/hm3d/hm3d-*-habitat.tar
```

### HM3D splits

| Split | Auth | Size (approx.) | HM-EQA |
|-------|------|----------------|--------|
| `example` | No | ~150MB | No — install smoke only |
| `minival` | API tokens | ~400MB | No — dev / path checks |
| `train` | API tokens | ~27GB | **Yes** — all HM-EQA scenes |
| `val` | API tokens | varies | No — unless you add val questions |

Upstream reference: [Habitat-Sim DATASETS.md — HM3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).

## On-disk layout

After `datasets_download --data-path ~/.cache/habitat_eqa/hm3d`:

```
~/.cache/habitat_eqa/
├── data/
│   ├── questions.csv
│   └── scene_init_poses.csv
└── hm3d/
    ├── scene_datasets/hm3d/          # symlink into versioned_data
    │   ├── train/
    │   │   └── 00004-VqCaAuuoeWk/
    │   │       └── VqCaAuuoeWk.basis.glb    # note short id, not full scene id
    │   ├── val/
    │   └── example/
    └── versioned_data/hm3d-0.2/...
```

### GLB path convention

Habitat names meshes with the **short id** (suffix after the first `-`):

| Scene id | GLB filename |
|----------|--------------|
| `00004-VqCaAuuoeWk` | `VqCaAuuoeWk.basis.glb` |
| `00005-yPKGKBCyYx8` | `yPKGKBCyYx8.basis.glb` |

Resolved path (default train root):

```
$HM3D_SCENE_DIR/<scene_id>/<short_id>.basis.glb
```

Implemented in `src/emet/habitat/config.py` (`hm3d_scene_short_name`, `hm3d_scene_glb_path`).

### Semantic GLB path convention

When HM3D-Semantics is installed, each annotated scene also has:

```
$HM3D_SCENE_DIR/<scene_id>/<short_id>.semantic.glb
```

Example: `.../00004-VqCaAuuoeWk/VqCaAuuoeWk.semantic.glb` next to `VqCaAuuoeWk.basis.glb`.

Habitat-Sim loads the annotated scene-dataset config (`hm3d_annotated_basis.scene_dataset_config.json`) and exposes a **semantic sensor** (per-pixel instance ids). Our harness maps those ids to category names (`chair`, `bed`, …) and builds **GraphEQAMemory** object nodes without running Detic or a captioning VLM on every frame.

## HM3D-Semantics (ground-truth perception in sim)

GraphEQA’s Habitat experiments use **`use_semantic_data: True`**: ground-truth HM3D instance masks feed the scene graph. **Detic is for real-world runs only.** See the parity appendix (`paper/sections/appendix/05_habitat_eqa_parity.tex`).

### Sim vs real world

| Setting | Scene graph labels | Notes |
|---------|-------------------|--------|
| **GraphEQA Habitat (paper)** | GT HM3D-Semantics instance masks | Oracle segmentation; benchmark stresses exploration + VLM QA |
| **GraphEQA real world** | Detic open-vocabulary detections | Imperfect perception |
| **Our harness (semantics on)** | GT masks when `.semantic.glb` exists | Matches paper sim assumption for that episode |
| **Our harness (semantics off / missing file)** | VLM or keyword labels from voxels | Harder; **not** paper GraphEQA sim |

When semantics are enabled, the Habitat runner sets `use_sensor_perception=False` so the EQA VLM is reserved for answering questions, not captioning every navigation frame (`packages/emet_habitat/emet_habitat/runner.py`).

### Why it feels weird (and why the paper still uses it)

HM-EQA is an **embodied** benchmark (navigate, explore, limited views), but GraphEQA **does not** ask the sim agent to solve open-vocabulary detection. GT masks isolate “given perfect object segmentation, how well does exploration + graph memory + VLM reasoning work?” Real deployment swaps in Detic.

### Download (separate from train meshes)

Train **meshes** (`--fetch-hm3d train`, ~27GB) and train **semantics** (`--fetch-hm3d-semantics train`) are separate Matterport packages. Both need API tokens:

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d-semantics train
```

Verify one scene:

```bash
uv run python scripts/download_habitat_eqa_data.py --verify-semantics 00004-VqCaAuuoeWk
```

### Coverage report

```bash
uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics
uv run emet habitat info   # one-line summary when CSVs exist
```

On a machine with full train + semantics downloads, expect roughly:

| Pool | Annotated scenes | Notes |
|------|------------------|--------|
| HM3D train split | **~145 / ~800** (~18%) | HM3DSem v0.2 annotates 145 train scans ([dataset page](https://aihabitat.org/datasets/hm3d-semantics/)) |
| HM-EQA paper (113 Q, 49 unique scenes) | **~14 / 49** scenes, **~37 / 113** questions | Overlap is small because Explore-EQA scenes were not chosen from the annotated subset |

### Slice taxonomy (what to run / cite)

| Slice | How | n | Use |
|-------|-----|---|-----|
| Overnight triad | `run_overnight_habitat_eval.sh` | 8 / 32 / 20 | Fast dynagraph vs graph_eqa iteration |
| Annotated semantics | `IDS="$(uv run python -c 'from emet.habitat.hm3d_semantics import hmeqa_annotated_question_ids as f; print(",".join(map(str,f())))')"` or overnight `annotated37_*` phases | ~37 | Fairer perception vs GraphEQA GT path |
| Paper HM-EQA | `emet-habitat run-batch --paper-subset` | **113** | GraphEQA Table 1 *n* |
| Explore-EQA full | `emet-habitat run-batch --all-questions` | up to ~500 | Beyond GraphEQA paper (stretch) |

See also [experiments/habitat_eqa_results.md](../experiments/habitat_eqa_results.md#hm-eqa-slice-taxonomy). Helpers: [`scripts/run_hmeqa_annotated37_h2h.sh`](../../scripts/run_hmeqa_annotated37_h2h.sh), [`scripts/run_hmeqa_paper113_h2h.sh`](../../scripts/run_hmeqa_paper113_h2h.sh).

### Why are we “missing” semantics?

**Two distinct cases — check the report before assuming a broken download.**

1. **Download gap (fixable)**  
   `train_scenes_with_semantics` is far below ~145 (e.g. 0–50).  
   → Run `--fetch-hm3d-semantics train` with valid Matterport tokens.

2. **Dataset gap (not fixable)**  
   Download shows ~145 train semantic meshes, but an HM-EQA scene still has no `.semantic.glb`.  
   → That scan was **never annotated** in HM3DSem. Matterport labeled 216 HM3D spaces total; HM-EQA uses 49 train scenes and most were not in that set. **No download command can create labels that do not exist.**

**What we can do for unannotated episodes today:**

- **Automatic fallback** — depth voxels + VLM/keyword graph labels (current default when `.semantic.glb` is absent).
- **Paper-parity scoring** — restrict to annotated questions only. The report prints their ids; helper `hmeqa_annotated_question_ids()` in `src/emet/habitat/hm3d_semantics.py` returns the same list for batch scripts.
- **Future** — optional Detic path for unannotated sim scenes (closer to real-world GraphEQA, still not identical to paper sim).

Do **not** compare local HM-EQA numbers to GraphEQA Table 1 (63–67%) without noting mixed perception modes unless you score the annotated subset **and** use a comparable EQA VLM.

### Wrong paths (common mistake)

| Wrong | Right |
|-------|-------|
| `.../hm3d/train/<scene_id>/...` | `.../hm3d/scene_datasets/hm3d/train/<scene_id>/...` |
| `.../<scene_id>.basis.glb` | `.../<short_id>.basis.glb` |

## Verify before running

```bash
uv run emet habitat info
uv run python scripts/download_habitat_eqa_data.py --verify-question 0
uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics
```

Expected for question 0 when train is installed:

```
expected glb: .../scene_datasets/hm3d/train/00004-VqCaAuuoeWk/VqCaAuuoeWk.basis.glb
status: OK
```

## OpenEQA (optional, future)

OpenEQA JSON can live at `HABITAT_EQA_DATA_DIR/open-eqa-v0.json`. Full OpenEQA Habitat eval is not wired in the default CLI yet; see the engineering plan.
