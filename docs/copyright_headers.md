# Copyright headers in Python files

## Policy

| File | Header |
|------|--------|
| **Pre-existing** modules (Hello Robot / upstream heritage) | `docs/license_header.txt` — Copyright (c) Hello Robot, Inc. |
| **New files** you create in this repo | `docs/license_header_chris_paxton.txt` — Copyright (c) Chris Paxton 2026 |

When you **edit** an existing file, keep its original header. Do not replace Hello Robot headers on legacy files.

## Pre-commit

- Default `insert-license` applies the Hello Robot header.
- `insert-license (Chris Paxton)` applies the Chris Paxton header only to paths listed in `.pre-commit-config.yaml` under that hook.

When adding a **new** Python file, add its path to:

1. The **exclude** list on the Hello Robot `insert-license` hook, and
2. The **files** list on `insert-license (Chris Paxton)`.

Or use a shared regex (e.g. `scripts/tier4_.*\.py`) if you add a family of scripts.

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
