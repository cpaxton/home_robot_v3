#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI for HuggingFace push/pull of emet scene map caches.

Examples::

    export EMET_SCENE_MAP_HF_REPO=org/emet-scene-maps
    uv run python scripts/scene_map_cache_hub.py list
    uv run python scripts/scene_map_cache_hub.py pull robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt
    uv run python scripts/scene_map_cache_hub.py push robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List complete local cache keys")
    p_list.add_argument("--root", default=None)

    p_pull = sub.add_parser("pull", help="Download one key from HuggingFace")
    p_pull.add_argument("key")
    p_pull.add_argument("--repo", default=None, help="Override EMET_SCENE_MAP_HF_REPO")

    p_push = sub.add_parser("push", help="Upload one local key to HuggingFace")
    p_push.add_argument("key")
    p_push.add_argument("--repo", default=None, help="Override EMET_SCENE_MAP_HF_REPO")
    p_push.add_argument("--public", action="store_true", help="Create/use a public dataset repo")

    args = ap.parse_args()
    from emet.eval.scene_map_cache_hub import list_local_cache_keys, pull_scene_map, push_scene_map

    if args.cmd == "list":
        keys = list_local_cache_keys(root=args.root)
        for k in keys:
            print(k)
        if not keys:
            print("(no complete caches)", file=sys.stderr)
        return 0

    if args.cmd == "pull":
        path = pull_scene_map(args.key, repo_id=args.repo)
        if path is None:
            print(f"pull failed or incomplete for key={args.key!r}", file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.cmd == "push":
        ok = push_scene_map(args.key, repo_id=args.repo, private=not args.public)
        print(f"pushed {args.key} ok={ok}")
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
