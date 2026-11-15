# Lossless archival of old debug arrays — November 15, 2026

The approved cleanup archived **108,112 raw NumPy arrays** from **559** completed
memory-debug directories. It reclaimed **23.76 GiB**: 53.69 GiB of loose
array allocations became 29.93 GiB of compressed archives. Plan/index
storage adds a small overhead; concurrent writers can change the observed free
space. The earlier download/lint-cache cleanup reclaimed another 10.40 GiB.
Free space after archival was about **40 GiB**.

Only the explicitly inventoried `.npy` files were removed from their loose
locations, after verification. All JPEG/PNG visuals, metrics, text, checkpoints,
environments and recent logs stayed. Directories named in the inspected
checkouts' docs/TODO were excluded, as were directories with files newer than
60 days. The final metadata fingerprint of **27,351 retained
non-array files** exactly matches the initial fingerprint.

## Verification and recovery

Archive directory:

`/home/cpaxton/src/home_robot_v3/logs/.compressed-debug-20261115/`

Each of **214 batches** passed `zstd --test` and GNU tar's byte-for-byte
`--compare` against all source members. Sources were checked for changes and
active use; archives were flushed before removing the corresponding loose
arrays. `index.json` records each archive's SHA-256 hash and verification
status, `plan.json` retains the complete original file inventory, and
`batch-*.files.txt` maps original relative paths to batches. The cleanup used
one low-priority CPU after the pilot ended, with a 16 GiB free-space abort floor.
Managed job: `20261115_135955_556875`.

A test extraction preserved the array's original nanosecond timestamp and size,
and NumPy loaded its RGB data successfully (`uint8`, shape 424 × 240 × 3).
The record is `restore-check.json` in the archive directory.

For example, restore that single array with:

```bash
tar --zstd -xf /home/cpaxton/src/home_robot_v3/logs/.compressed-debug-20261115/batch-0000.tar.zst \
  -C /home/cpaxton/src/home_robot_v3/logs \
  memory_2026-08-26_10-02-47/debug/rgb1.npy
```

To find another array, search `batch-*.files.txt` for its path relative to
`logs/`, then select the corresponding `.tar.zst`. Omitting the final member
name restores that entire batch. Retain the archives: they now contain the raw
arrays, and deleting them would destroy those archived data.

This is an archival operation with an audit trail, not a new automatic
retention policy. No existing simulator-model archive files were deduplicated
as part of this operation.
