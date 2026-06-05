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
```

`--fetch-hm3d` uses `.venv-habitat/bin/python -m habitat_sim.utils.datasets_download`.

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

`habitat_sim.utils.datasets_download` authenticates to `api.matterport.com` with HTTP basic auth. You must use **API token ID + secret** from the Matterport developer page.

**Do not** use your Matterport account email and password. The API returns `{"code":"request.unauthorized","message":"Unauthorized"}` and leaves a tiny `hm3d-train-habitat.tar` stub (~56 bytes) that is not a real archive.

### Setup steps

1. **Request HM3D access** (if needed): [matterport.com/partners/meta](https://matterport.com/partners/meta)
2. Log in at [matterport.com](https://matterport.com)
3. Open **Settings → Developer Tools → Habitat dataset** (wording may vary)
4. Create or export an API token pair
5. Export in your shell (do not commit these):

```bash
export MATTERPORT_USERNAME='<token-id>'
export MATTERPORT_PASSWORD='<token-secret>'
```

6. Download:

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train
```

If a previous attempt failed with `Unauthorized`, remove the bad stub before retrying:

```bash
rm -f ~/.cache/habitat_eqa/hm3d/hm3d-train-habitat.tar
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

### Wrong paths (common mistake)

| Wrong | Right |
|-------|-------|
| `.../hm3d/train/<scene_id>/...` | `.../hm3d/scene_datasets/hm3d/train/<scene_id>/...` |
| `.../<scene_id>.basis.glb` | `.../<short_id>.basis.glb` |

## Verify before running

```bash
uv run emet habitat info
uv run python scripts/download_habitat_eqa_data.py --verify-question 0
```

Expected for question 0 when train is installed:

```
expected glb: .../scene_datasets/hm3d/train/00004-VqCaAuuoeWk/VqCaAuuoeWk.basis.glb
status: OK
```

## OpenEQA (optional, future)

OpenEQA JSON can live at `HABITAT_EQA_DATA_DIR/open-eqa-v0.json`. Full OpenEQA Habitat eval is not wired in the default CLI yet; see the engineering plan.
