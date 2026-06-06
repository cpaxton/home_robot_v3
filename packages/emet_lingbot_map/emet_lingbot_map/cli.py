# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""CLI for LingBot-Map batch inference (runs in .venv-lingbot-map)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from emet_lingbot_map.episode_loader import load_episode
from emet_lingbot_map.inference import (
    InferenceConfig,
    run_inference_on_episode,
    save_predictions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emet_lingbot_map", description="LingBot-Map episode inference")
    sub = parser.add_subparsers(dest="command", required=True)

    infer_p = sub.add_parser("infer", help="Run streaming inference on a recorded episode")
    infer_p.add_argument("--episode", type=Path, required=True, help="Episode dir with metadata.jsonl")
    infer_p.add_argument("--output", type=Path, required=True, help="Output dir for lingbot depths/poses")
    infer_p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model .pt path (default: LINGBOT_MAP_CHECKPOINT env)",
    )
    infer_p.add_argument("--image-size", type=int, default=518)
    infer_p.add_argument("--keyframe-interval", type=int, default=None)
    infer_p.add_argument("--num-scale-frames", type=int, default=8)
    infer_p.add_argument("--use-sdpa", action="store_true", help="Use PyTorch SDPA instead of FlashInfer")
    infer_p.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    infer_p.add_argument("--window-size", type=int, default=64)
    infer_p.add_argument("--first-k", type=int, default=None, help="Only use first K frames from episode")

    stream_p = sub.add_parser("stream-server", help="JSON-lines streaming server (stdin/stdout)")
    stream_p.add_argument("--checkpoint", type=Path, default=None)
    stream_p.add_argument("--use-sdpa", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "infer":
        ckpt = args.checkpoint or Path(os.environ.get("LINGBOT_MAP_CHECKPOINT", ""))
        if not ckpt.is_file():
            parser.error(f"Checkpoint not found: {ckpt} (set LINGBOT_MAP_CHECKPOINT)")

        episode = load_episode(args.episode)
        if args.first_k is not None and args.first_k > 0:
            episode.frames = episode.frames[: int(args.first_k)]

        cfg = InferenceConfig(
            checkpoint=ckpt,
            image_size=args.image_size,
            keyframe_interval=args.keyframe_interval,
            num_scale_frames=args.num_scale_frames,
            use_sdpa=args.use_sdpa,
            mode=args.mode,
            window_size=args.window_size,
        )
        preds = run_inference_on_episode(episode, cfg)
        out = save_predictions(preds, args.output, episode=episode)
        print(f"Wrote LingBot predictions to {out}")
        return 0

    if args.command == "stream-server":
        from emet_lingbot_map.stream_server import run_stream_server

        ckpt = args.checkpoint or Path(os.environ.get("LINGBOT_MAP_CHECKPOINT", ""))
        if not ckpt.is_file():
            parser.error(f"Checkpoint not found: {ckpt}")
        run_stream_server(ckpt, use_sdpa=args.use_sdpa)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
