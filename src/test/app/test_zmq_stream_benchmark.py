# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import time

import numpy as np
import pytest

import emet.utils.compression as compression
from emet.app.zmq_stream_benchmark import (
    BenchmarkOptions,
    FrameSample,
    aggregate_benchmark,
    decode_message_images,
    lidar_wire_notes,
    measure_message_sizes,
    payload_size,
    run_benchmark,
    slim_duplicates,
)


def test_payload_size_counts_bytes_and_recurses():
    assert payload_size(b"abcd") == 4
    assert payload_size(None) == 0
    assert payload_size("hello") == 5
    assert payload_size(np.zeros((4, 6, 3), dtype=np.uint8)) == 4 * 6 * 3
    assert payload_size({"a": b"xx", "b": [1, np.zeros(4, dtype=np.float64)]}) == 37


def test_measure_message_sizes_totals():
    msg = {"rgb": b"x" * 100, "depth": b"y" * 50, "step": 3}
    total, per_key = measure_message_sizes(msg)
    assert per_key["rgb"] == 100
    assert per_key["depth"] == 50
    assert total == 100 + 50 + len("3")


def test_decode_message_images_measures_cpu_cost():
    rgb = np.full((64, 48, 3), 200, dtype=np.uint8)
    depth = np.full((16, 16), 1200, dtype=np.uint16)
    msg = {
        "rgb": compression.to_jpg(rgb),
        "head_cam_right/image": compression.to_jpg(rgb),
        "depth": compression.to_jp2(depth),
    }
    total_ms, per_key = decode_message_images(msg)
    assert set(per_key) == {"rgb", "head_cam_right/image", "depth"}
    assert all(v >= 0.0 for v in per_key.values())
    assert total_ms >= 0.0


def test_slim_duplicates_detects_legacy_dupes():
    jpg = b"same"
    assert slim_duplicates({"rgb": jpg, "head_cam_left/image": jpg}) == ["rgb duplicates head_cam_left/image"]
    assert slim_duplicates({"head_cam_left/image": jpg}) == []


def test_slim_duplicates_flags_distinct_jpegs():
    assert len(slim_duplicates({"rgb": b"a", "head_cam_left/image": b"b"})) == 1


def test_lidar_wire_notes_flags_float64():
    pts = np.zeros((360, 2), dtype=np.float64)
    notes = lidar_wire_notes({"lidar_points": pts})
    assert len(notes) == 1
    assert "float64" in notes[0]
    assert lidar_wire_notes({"lidar_points": pts.astype(np.float32)}) == []
    assert lidar_wire_notes({}) == []


def test_aggregate_benchmark_stats_math():
    t0 = 100.0
    samples = []
    for i in range(11):
        samples.append(
            FrameSample(
                seq=i,
                t_recv=t0 + i * 0.1,
                wire_bytes=1000 + i,
                payload_bytes=800,
                per_key={"rgb": 700, "depth": 100},
                step=i,
            )
        )
    stats = aggregate_benchmark(samples)
    assert stats["n_frames"] == 11
    assert stats["fps"] == pytest.approx(10.0, rel=0.01)
    assert stats["server_hz"] == pytest.approx(10.0, rel=0.01)
    assert stats["wire_bytes_per_frame"] == pytest.approx(1005.0)
    assert stats["payload_bytes_per_frame"] == 800
    assert stats["dropped_steps"] == 0
    assert stats["per_key"]["rgb"]["avg_bytes"] == 700
    assert stats["image_keys_seen"] == ["depth", "rgb"]


def test_aggregate_benchmark_detects_step_drops():
    samples = [
        FrameSample(seq=0, t_recv=0.0, wire_bytes=100, payload_bytes=80, per_key={}, step=1),
        FrameSample(seq=1, t_recv=0.1, wire_bytes=100, payload_bytes=80, per_key={}, step=4),
        FrameSample(seq=2, t_recv=0.2, wire_bytes=100, payload_bytes=80, per_key={}, step=5),
    ]
    stats = aggregate_benchmark(samples)
    assert stats["step_gaps"] == 1
    assert stats["dropped_steps"] == 2


def test_aggregate_benchmark_decode_pct_uses_ms_units():
    t0 = 100.0
    samples = [
        FrameSample(seq=i, t_recv=t0 + i * 0.1, wire_bytes=100, payload_bytes=80, per_key={}, decode_ms=1.0)
        for i in range(11)
    ]
    stats = aggregate_benchmark(samples)
    assert stats["decode_ms_avg"] == pytest.approx(1.0)
    assert stats["decode_pct_of_median_period"] == pytest.approx(1.0)  # 1ms vs 100ms period


def test_aggregate_benchmark_single_frame_is_diagnostic():
    sample = FrameSample(seq=0, t_recv=1.0, wire_bytes=500, payload_bytes=400, per_key={"rgb": 400}, step=3)
    stats = aggregate_benchmark([sample])
    assert stats["n_frames"] == 1
    assert stats["fps"] == 0.0
    assert stats["wire_bytes_per_frame"] == 500
    assert stats["payload_bytes_per_frame"] == 400
    assert stats["period_ms_avg"] == 0.0


def test_aggregate_benchmark_empty_raises():
    with pytest.raises(ValueError, match="no observation frames"):
        aggregate_benchmark([])


def test_run_benchmark_end_to_end(monkeypatch):
    import pickle
    import threading

    import zmq

    msgs = [
        {"emet_robot_id": "innate_mars", "emet_session": {"capabilities": {"zmq_obs_slim": True}}, "step": i}
        for i in range(5)
    ]

    def _publish():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.bind("tcp://127.0.0.1:5511")
        time.sleep(0.3)
        for msg in msgs:
            sock.send_pyobj(msg)
            time.sleep(0.01)
        time.sleep(0.5)
        sock.close()
        ctx.term()

    threading.Thread(target=_publish, daemon=True).start()
    result = run_benchmark(
        BenchmarkOptions(host="127.0.0.1", port=5511, frames=4, timeout_ms=2000, warmup_frames=1)
    )
    assert result.robot_id == "innate_mars"
    assert result.capabilities == {"zmq_obs_slim": True}
    assert result.stats["n_frames"] == 4