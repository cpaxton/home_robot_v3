# Copyright (c) Chris Paxton 2026
"""Tests for emet.utils.cpu_affinity."""

from __future__ import annotations

from pathlib import Path

from emet.utils.cpu_affinity import (
    cpus_at_or_above_mhz,
    format_taskset_list,
    safe_cpu_ids,
)


def _write_cpu(sysfs: Path, cpu_id: int, max_khz: int, *, online: bool = True) -> None:
    cpu = sysfs / f"cpu{cpu_id}"
    (cpu / "cpufreq").mkdir(parents=True)
    (cpu / "cpufreq" / "cpuinfo_max_freq").write_text(f"{max_khz}\n")
    if cpu_id != 0:
        # cpu0 is always online and often has no online file
        (cpu / "online").write_text("1\n" if online else "0\n")


def test_format_taskset_list_ranges() -> None:
    assert format_taskset_list([0, 1, 2, 3, 7, 12, 13, 14]) == "0-3,7,12-14"
    assert format_taskset_list([8]) == "8"
    assert format_taskset_list([]) == ""


def test_safe_cpu_ids_excludes_turbo(tmp_path: Path) -> None:
    # Mimic i9-14900KF: 0-7 @ 5.7 GHz, 8-11 @ 6.0 GHz, 12-15 @ 5.7 GHz
    for i in range(8):
        _write_cpu(tmp_path, i, 5700000)
    for i in range(8, 12):
        _write_cpu(tmp_path, i, 6000000)
    for i in range(12, 16):
        _write_cpu(tmp_path, i, 5700000)

    turbo = cpus_at_or_above_mhz(6000.0, sysfs_root=tmp_path)
    assert turbo == [8, 9, 10, 11]

    kept = safe_cpu_ids(exclude_min_mhz=6000.0, sysfs_root=tmp_path)
    assert kept == [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    assert format_taskset_list(kept) == "0-7,12-15"


def test_safe_cpu_ids_explicit_allow_and_exclude(tmp_path: Path) -> None:
    for i in range(4):
        _write_cpu(tmp_path, i, 5700000)
    kept = safe_cpu_ids(
        exclude_min_mhz=None,
        explicit_allow=[0, 1, 2, 3],
        explicit_exclude=[1],
        sysfs_root=tmp_path,
    )
    assert kept == [0, 2, 3]


def test_safe_cpu_ids_never_empty(tmp_path: Path) -> None:
    for i in range(2):
        _write_cpu(tmp_path, i, 6000000)
    # Excluding everything falls back to online rather than pinning nowhere.
    kept = safe_cpu_ids(exclude_min_mhz=6000.0, sysfs_root=tmp_path)
    assert kept == [0, 1]
