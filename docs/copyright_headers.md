# Copyright headers in Python files

## Policy

| File | Header |
|------|--------|
| **Pre-existing** modules (Hello Robot / upstream heritage) | `docs/license_header.txt` — Copyright (c) Hello Robot, Inc. |
| **New files** you create in this repo | `docs/license_header_chris_paxton.txt` — Copyright (c) Chris Paxton 2026 |

When you **edit** an existing file, keep its original header. Do not replace Hello Robot headers on legacy files.

## Pre-commit

The `insert license header (Chris Paxton)` hook runs [`scripts/insert_license_header.py`](../scripts/insert_license_header.py). It decides from **file content**, not from a path list:

- A file whose first 15 lines contain `Copyright` or `SPDX-License-Identifier` is left untouched — legacy Hello Robot headers, vendored SPDX notices, and upstream attributions all survive unchanged.
- Any other Python file gets the Chris Paxton stub, inserted below a shebang or encoding line.
- Empty files are skipped.

**New files need no configuration.** Just create the file; the hook stamps the right header.

Vendored trees are excluded by path in `.pre-commit-config.yaml` (`third_party/`, `scripts/scannet/`, `src/emet/simulation/molmo_occupancy/`) because their upstream license lives in a `NOTICE` rather than a per-file header.

Behavior is covered by [`src/test/utils/test_insert_license_header.py`](../src/test/utils/test_insert_license_header.py), including a repo-wide guard against a file carrying both a Hello Robot and a Chris Paxton header.

> Earlier versions of this hook used two mirrored path allowlists (one per header). Keeping them in sync failed silently: a new file missing from both lists was stamped **Hello Robot**, which the project copyright rule forbids. Do not reintroduce path-based selection.

## Examples

New module (Chris Paxton):

```python
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
```

Existing module (unchanged when patching):

```python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
# ...
```
