# Habitat troubleshooting

## `Habitat wrapper not found`

```
Habitat wrapper not found. From the project root run:
  ./scripts/install_habitat.sh
```

Run `./scripts/install_habitat.sh` from the repo root. Confirm `.venv-habitat/bin/emet-habitat` exists.

## `habitat_sim not found` during download

```
habitat_sim not found. Run ./scripts/install_habitat.sh first.
```

`--fetch-hm3d` uses the Habitat sub-env, not the main `.venv`. Install Habitat first, then retry the download script.

## `HM3D scene not found` (wrong path or missing train split)

Example error:

```
Error: HM3D scene not found: .../hm3d/train/00004-VqCaAuuoeWk/00004-VqCaAuuoeWk.basis.glb
```

Checklist:

1. **Train split installed?** Question 0 needs `--fetch-hm3d train`, not `example`.
2. **Scene root** — default is `.../scene_datasets/hm3d/train`, not `.../hm3d/train`.
3. **GLB name** — file is `VqCaAuuoeWk.basis.glb`, not `00004-VqCaAuuoeWk.basis.glb`.
4. **Stale `HM3D_SCENE_DIR`** — unset or point at `scene_datasets/hm3d/train`.

```bash
uv run python scripts/download_habitat_eqa_data.py --verify-question 0
uv run emet habitat info
```

Details: [data.md](data.md#on-disk-layout).

## Matterport download: `Unauthorized` / tiny `.tar` file

Symptom: download finishes in seconds; `tarfile.ReadError: not a gzip file`; file size ~56 bytes.

```bash
cat ~/.cache/habitat_eqa/hm3d/hm3d-train-habitat.tar
# {"code":"request.unauthorized","message":"Unauthorized"}
```

**Cause:** Web login email/password were used instead of **API token ID + secret**.

**Fix:**

1. Matterport → Settings → Developer Tools → Habitat dataset → create API token  
2. `export MATTERPORT_USERNAME='<token-id>'` and `MATTERPORT_PASSWORD='<token-secret>'`  
3. `rm -f ~/.cache/habitat_eqa/hm3d/hm3d-train-habitat.tar`  
4. Retry `uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train`

See [data.md — Matterport credentials](data.md#matterport-credentials-hm3d-train--val--minival).

## `HM3D train requires Matterport credentials`

Set both env vars before `--fetch-hm3d train|minival|val`. The `example` split does not need auth.

## `pip install habitat-sim` fails on Linux

PyPI does not ship current Linux wheels. Use `./scripts/install_habitat.sh` (micromamba + `aihabitat-nightly`), not `uv pip install habitat-sim` in the main env.

## `questions.csv exists: False`

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-csv
```

Or set `HABITAT_EQA_DATA_DIR` to a directory that already contains the CSVs.

## Pose / scene init errors

`scene_init_poses.csv` keys scenes as `scene_floor` (e.g. `00004-VqCaAuuoeWk_1`). If you hand-edit CSVs, match that format. Loader: `src/emet/habitat/datasets.py`.

## Still stuck?

1. `uv run emet habitat info`  
2. `uv run python scripts/download_habitat_eqa_data.py --instructions`  
3. `.venv-habitat/bin/python -c "import habitat_sim; print(habitat_sim.__version__)"`  
4. Open an issue with the full error, output of `verify-question`, and whether train or example split was downloaded (do **not** paste API tokens).
