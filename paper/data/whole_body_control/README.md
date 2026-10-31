# Control experiment evidence

Keep compact summaries, benchmark definitions, plots and source manifests in Git.
Generated compressed traces, training streams and model weights belong in ignored
artifact storage. `.gitignore` excludes gzip files even under `paper/data/`;
`scripts/check_generated_artifacts.py` also rejects force-added control archives.

`artifacts.json` records the path, size and SHA-256 of each externalized artifact.
The verified local store is `logs/artifacts/whole_body_control/`. It is not included
in a clone and has not been uploaded to W&B or another external service. The
original branch is backed up separately before history cleanup. Historical
experiment docs that say compressed reports were committed describe the earlier
layout; use this catalog for their current location.

Given a copy of the artifact store:

```bash
python scripts/control_artifacts.py verify --store /path/to/control-artifacts
python scripts/control_artifacts.py restore --store /path/to/control-artifacts
```

Restored archives remain ignored. Current report readers can resolve cataloged
artifacts directly from the default local store and verify their hashes; absent
artifacts produce a retrieval instruction. A missing report is never accepted as
qualification evidence. Do not decompress traces into large tracked JSON files
or use force-add to bypass the ignore policy.
