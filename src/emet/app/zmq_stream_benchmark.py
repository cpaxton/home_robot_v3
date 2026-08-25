# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Sustained ZMQ full-observation stream benchmark (CPU only, no models/GPU).

Connects a bare SUB socket to the robot/sim observation port (the same wire path the
``GenericZmqClient`` uses) and measures frame rate, wire/payload bytes, jitter, and
(client-side) JPEG/JP2 decode cost — without instantiating any perception or VLM model.

Mirrors the real client socket options (``SNDHWM=1``/``RCVHWM=1``/``CONFLATE=1``) so
results reflect what an actual mapping client sees. ``--no-conflate`` on the CLI drops
conflation to expose server-side drops via ``step`` gaps.

CLI entrypoint: ``scripts/benchmark_zmq_obs_stream.py``.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from statistics import fmean, median, pstdev
from typing import Any

import numpy as np
import zmq

import emet.utils.compression as compression
from emet.core.zmq_obs_codec import ZMQ_IMAGE_ALIASES

# Image-ish payload keys used for the "duplicate JPEG" slim-format check.
IMAGE_KEYS: tuple[str, ...] = (
    "rgb",
    "rgb_right",
    "rgb_tertiary",
    "depth",
    "head_cam_left/image",
    "head_cam_right/image",
    "ee_cam/image",
    "third_person_image",
)

CANONICAL_IMAGE_KEYS: tuple[str, ...] = ("head_cam_left/image", "head_cam_right/image", "ee_cam/image")
LEGACY_IMAGE_KEYS: tuple[str, ...] = ("rgb", "rgb_right", "rgb_tertiary")


def payload_size(val: Any, _seen: set[int] | None = None) -> int:
    """Estimated in-memory payload bytes for a decoded ZMQ value (bytes counted exactly)."""
    if val is None:
        return 0
    if isinstance(val, (bytes, bytearray)):
        return len(val)
    if isinstance(val, str):
        return len(val.encode("utf-8", errors="replace"))
    if isinstance(val, dict):
        total = 0
        for key, value in val.items():
            total += payload_size(key) + payload_size(value)
        return total
    if isinstance(val, (list, tuple, set, frozenset)):
        return sum(payload_size(v) for v in val)
    if isinstance(val, np.ndarray):
        return int(val.nbytes)
    try:
        if isinstance(val, (np.floating, np.integer)):
            return int(val.nbytes)
    except AttributeError:
        pass
    return len(str(val).encode("utf-8", errors="replace"))


def measure_message_sizes(msg: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Return ``(total_payload, per_key_bytes)`` for a full-observation message."""
    per_key: dict[str, int] = {}
    total = 0
    for key, value in msg.items():
        size = payload_size(value)
        per_key[key] = size
        total += size
    return total, per_key


def _decode_image(key: str, value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if key == "depth":
            return compression.from_jp2(value) / 1000
        return compression.from_jpg(value)
    except Exception:
        return None


def decode_message_images(msg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """JPEG/JP2-decode every image key (CPU only) and return ``(total_ms, per_key_ms)``.

    Mirrors the per-frame cost ``GenericZmqClient`` pays before a mapping step.
    """
    per_key: dict[str, float] = {}
    total_ms = 0.0
    for key in IMAGE_KEYS:
        value = msg.get(key)
        if value is None:
            continue
        t0 = time.perf_counter()
        _decode_image(key, value)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_key[key] = elapsed_ms
        total_ms += elapsed_ms
    return total_ms, per_key


def slim_duplicates(msg: dict[str, Any]) -> list[str]:
    """Legacy alias keys that are still duplicated on the wire (non-slim server)."""
    problems: list[str] = []
    for legacy, canonical in ZMQ_IMAGE_ALIASES:
        leg = msg.get(legacy)
        canon = msg.get(canonical)
        if leg is not None and canon is not None:
            same = leg is canon or (isinstance(leg, (bytes, bytearray)) and leg == canon)
            if not same:
                problems.append(f"{legacy} ({payload_size(leg)}B) != {canonical} ({payload_size(canon)}B)")
            else:
                problems.append(f"{legacy} duplicates {canonical}")
    return problems


@dataclass
class FrameSample:
    seq: int
    t_recv: float
    wire_bytes: int
    payload_bytes: int
    per_key: dict[str, int] = field(default_factory=dict)
    pickle_ms: float = 0.0
    decode_ms: float = 0.0
    decode_keys: dict[str, float] = field(default_factory=dict)
    step: int | None = None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def aggregate_benchmark(samples: list[FrameSample], *, wall_seconds: float | None = None) -> dict[str, Any]:
    """Collapse per-frame samples into summary stats (pure; no I/O)."""
    if len(samples) < 2:
        raise ValueError(f"need at least 2 samples, got {len(samples)}")
    duration = wall_seconds if wall_seconds is not None else samples[-1].t_recv - samples[0].t_recv
    duration = max(duration, 1e-6)

    gaps = [samples[i].t_recv - samples[i - 1].t_recv for i in range(1, len(samples))]
    gaps_sorted = sorted(gaps)

    wire_total = sum(s.wire_bytes for s in samples)
    payload_total = sum(s.payload_bytes for s in samples)

    pickle_ms = [s.pickle_ms for s in samples]
    decode_ms = [s.decode_ms for s in samples if s.decode_ms > 0]

    per_key_bytes: dict[str, list[int]] = {}
    for s in samples:
        for key, size in s.per_key.items():
            per_key_bytes.setdefault(key, []).append(size)
    per_key_stats: dict[str, dict[str, float]] = {}
    for key, sizes in per_key_bytes.items():
        avg = fmean(sizes)
        per_key_stats[key] = {
            "avg_bytes": avg,
            "min_bytes": float(min(sizes)),
            "max_bytes": float(max(sizes)),
            "share_pct": 100.0 * sum(sizes) / max(payload_total, 1),
        }

    server_steps = [s.step for s in samples if s.step is not None]
    server_hz: float | None = None
    dropped_steps = 0
    step_gaps = 0
    if server_steps:
        span = server_steps[-1] - server_steps[0]
        if span > 0:
            server_hz = span / duration
        for i in range(1, len(server_steps)):
            diff = server_steps[i] - server_steps[i - 1]
            if diff > 1:
                step_gaps += 1
                dropped_steps += diff - 1

    return {
        "n_frames": len(samples),
        "duration_s": duration,
        "fps": (len(samples) - 1) / duration,
        "server_hz": server_hz,
        "period_ms_avg": fmean(gaps) * 1000.0,
        "period_ms_median": median(gaps) * 1000.0,
        "period_ms_p95": _percentile(gaps_sorted, 0.95) * 1000.0,
        "period_ms_max": gaps_sorted[-1] * 1000.0,
        "period_ms_stdev": pstdev(gaps) * 1000.0,
        "wire_mbps": wire_total * 8.0 / duration / 1e6,
        "wire_bytes_per_frame": wire_total / len(samples),
        "payload_mbps": payload_total * 8.0 / duration / 1e6,
        "payload_bytes_per_frame": payload_total / len(samples),
        "pickle_ms_avg": fmean(pickle_ms),
        "pickle_ms_max": max(pickle_ms),
        "decode_ms_avg": fmean(decode_ms) if decode_ms else None,
        "decode_ms_max": max(decode_ms) if decode_ms else None,
        "decode_pct_of_median_period": (
            (fmean(decode_ms) / (median(gaps) * 1000.0) * 100.0) if decode_ms else None
        ),
        "step_gaps": step_gaps,
        "dropped_steps": dropped_steps,
        "per_key": per_key_stats,
        "image_keys_seen": sorted(
            {key for s in samples for key in s.per_key if key in IMAGE_KEYS and s.per_key.get(key, 0) > 0}
        ),
    }


def format_benchmark_result(stats: dict[str, Any], *, robot_id: str | None = None) -> str:
    """Human-readable summary table for a benchmark result."""
    lines: list[str] = []
    header = "ZMQ obs stream benchmark"
    if robot_id:
        header += f"  robot={robot_id}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"frames       {stats['n_frames']}   ({stats['fps']:.1f} fps over {stats['duration_s']:.1f}s)"
    )
    server_hz = stats.get("server_hz")
    if server_hz is not None:
        lines.append(f"server hz    {server_hz:.1f}  (from step spans)")
    lines.append(
        f"period (ms)  avg {stats['period_ms_avg']:.1f}  median {stats['period_ms_median']:.1f}  "
        f"p95 {stats['period_ms_p95']:.1f}  max {stats['period_ms_max']:.1f}  "
        f"stdev {stats['period_ms_stdev']:.1f}"
    )
    lines.append(
        f"throughput   wire {stats['wire_mbps']:.2f} MB/s ({stats['wire_bytes_per_frame']:.0f} B/frame)  "
        f"payload {stats['payload_mbps']:.2f} MB/s ({stats['payload_bytes_per_frame']:.0f} B/frame)"
    )
    lines.append(f"unpickle     avg {stats['pickle_ms_avg']:.2f} ms  max {stats['pickle_ms_max']:.2f} ms")
    decode_avg = stats.get("decode_ms_avg")
    if decode_avg is not None:
        lines.append(
            f"decode (CPU) avg {decode_avg:.2f} ms  max {stats['decode_ms_max']:.2f} ms  "
            f"({stats['decode_pct_of_median_period']:.1f}% of median period)"
        )
    if stats.get("step_gaps"):
        lines.append(
            f"DROPS        {stats['step_gaps']} gaps / {stats['dropped_steps']} steps skipped "
            "(subscriber fell behind; check --no-conflate mode)"
        )
    lines.append("")
    lines.append(f"{'key':<30} {'avg_bytes':>12} {'share':>8}")
    grand = max(sum(v["avg_bytes"] for v in stats["per_key"].values()), 1)
    for key in sorted(stats["per_key"], key=lambda k: -stats["per_key"][k]["avg_bytes"]):
        v = stats["per_key"][key]
        flag = "  ← image" if key in IMAGE_KEYS else ""
        lines.append(f"{key:<30} {v['avg_bytes']:12.0f} {v['share_pct']:7.1f}%{flag}")
    lines.append(f"{'TOTAL':<30} {grand:12.0f}")
    if stats.get("image_keys_seen"):
        lines.append("")
        lines.append("image keys: " + ", ".join(stats["image_keys_seen"]))
    return "\n".join(lines)


@dataclass
class BenchmarkOptions:
    host: str
    port: int = 4401
    seconds: float = 10.0
    frames: int = 0
    timeout_ms: int = 15000
    conflate: bool = True
    decode: bool = False
    warmup_frames: int = 2
    verbose: bool = False


@dataclass
class BenchmarkResult:
    stats: dict[str, Any]
    robot_id: str | None = None
    capabilities: dict[str, Any] | None = None
    last_message_keys: list[str] | None = None
    slim_problems: list[str] | None = None


def run_benchmark(opts: BenchmarkOptions, *, on_sample: Any = None) -> BenchmarkResult:
    """Stream ``seconds`` (or ``frames``) of full-observation messages and aggregate stats."""
    if opts.seconds <= 0 and opts.frames <= 0:
        raise ValueError("set --seconds or --frames")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    if opts.conflate:
        sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, opts.timeout_ms)
    sock.connect(f"tcp://{opts.host}:{opts.port}")

    robot_id: str | None = None
    capabilities: dict[str, Any] | None = None
    last_keys: list[str] | None = None
    slim_problems: list[str] | None = None
    samples: list[FrameSample] = []
    try:
        warmup = 0
        while warmup < opts.warmup_frames:
            try:
                if sock.recv() is not None:
                    warmup += 1
            except zmq.Again:
                raise TimeoutError(f"no ZMQ obs on tcp://{opts.host}:{opts.port} within {opts.timeout_ms} ms")

        start_wall = time.monotonic()
        seq = 0
        while True:
            if opts.frames > 0 and seq >= opts.frames:
                break
            if opts.seconds > 0 and opts.frames == 0 and time.monotonic() - start_wall >= opts.seconds:
                break
            try:
                t_recv = time.monotonic()
                raw = sock.recv()
            except zmq.Again:
                raise TimeoutError(f"no ZMQ obs on tcp://{opts.host}:{opts.port} within {opts.timeout_ms} ms")

            t_unpickle = time.perf_counter()
            msg = pickle.loads(raw)
            pickle_ms = (time.perf_counter() - t_unpickle) * 1000.0
            if not isinstance(msg, dict):
                continue
            payload_total, per_key = measure_message_sizes(msg)
            decode_ms, decode_keys = decode_message_images(msg) if opts.decode else (0.0, {})
            sample = FrameSample(
                seq=seq,
                t_recv=t_recv,
                wire_bytes=len(raw),
                payload_bytes=payload_total,
                per_key=per_key,
                pickle_ms=pickle_ms,
                decode_ms=decode_ms,
                decode_keys=decode_keys,
                step=msg.get("step"),
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample, msg)
            if opts.verbose:
                print(
                    f"  [{seq}] step={sample.step} wire={len(raw)}B payload={payload_total}B "
                    f"gap~{sample.t_recv - (samples[-2].t_recv if len(samples) >= 2 else sample.t_recv):.1f}ms"
                )
            seq += 1
            robot_id = msg.get("emet_robot_id") or robot_id
            caps = (msg.get("emet_session") or {}).get("capabilities")
            if caps:
                capabilities = caps
            last_keys = list(msg.keys())
            if last_keys is not None:
                slim_problems = slim_duplicates(msg)
    finally:
        sock.close()
        ctx.term()

    stats = aggregate_benchmark(samples, wall_seconds=None)
    return BenchmarkResult(
        stats=stats,
        robot_id=robot_id,
        capabilities=capabilities,
        last_message_keys=last_keys,
        slim_problems=slim_problems,
    )


def summarize(result: BenchmarkResult, *, json_out: str | None = None) -> str:
    """Print summary + optional JSON dump; returns the human-readable text."""
    import json

    text = format_benchmark_result(result.stats, robot_id=result.robot_id)
    if result.capabilities:
        text += f"\ncapabilities: {json.dumps(result.capabilities, sort_keys=True)}"
    if result.slim_problems:
        text += "\nSLIM FORMAT VIOLATION — duplicate JPEG aliases on the wire:\n  " + "\n  ".join(
            result.slim_problems
        )
    if json_out:
        payload = {
            "robot_id": result.robot_id,
            "capabilities": result.capabilities,
            "slim_problems": result.slim_problems,
            "stats": result.stats,
        }
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return text