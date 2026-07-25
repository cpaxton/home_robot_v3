# Copyright (c) Chris Paxton 2026
"""CPU affinity helpers for long Habitat / VLM evals on unstable turbo cores.

On this i9-14900KF workstation, logical CPUs whose ``cpuinfo_max_freq`` is
6.0 GHz (two P-cores / four SMT threads) concentrate invalid-opcode and
``libcuda`` segfaults. Prefer excluding those cores for HM-EQA rather than
manually remembering ``taskset -c 0-7,10-31`` (which still leaves the second
6 GHz core online).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _cpu_sysfs_root() -> Path:
    return Path("/sys/devices/system/cpu")


def online_cpu_ids(sysfs_root: Path | None = None) -> list[int]:
    root = sysfs_root or _cpu_sysfs_root()
    ids: list[int] = []
    for path in sorted(root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        online = path / "online"
        if online.exists() and online.read_text().strip() == "0":
            continue
        ids.append(int(path.name[3:]))
    return ids


def cpu_max_mhz(cpu_id: int, sysfs_root: Path | None = None) -> float | None:
    """Return ``cpuinfo_max_freq`` in MHz, or None if unavailable."""
    root = sysfs_root or _cpu_sysfs_root()
    freq_path = root / f"cpu{cpu_id}" / "cpufreq" / "cpuinfo_max_freq"
    if not freq_path.exists():
        return None
    try:
        # sysfs reports kHz
        return int(freq_path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cpus_at_or_above_mhz(
    min_mhz: float,
    *,
    sysfs_root: Path | None = None,
    cpu_ids: list[int] | None = None,
) -> list[int]:
    """Logical CPUs whose advertised max frequency is >= ``min_mhz``."""
    ids = cpu_ids if cpu_ids is not None else online_cpu_ids(sysfs_root)
    out: list[int] = []
    for cpu_id in ids:
        mhz = cpu_max_mhz(cpu_id, sysfs_root)
        if mhz is not None and mhz + 1e-6 >= min_mhz:
            out.append(cpu_id)
    return out


def safe_cpu_ids(
    *,
    exclude_min_mhz: float | None = 6000.0,
    explicit_exclude: list[int] | None = None,
    explicit_allow: list[int] | None = None,
    sysfs_root: Path | None = None,
) -> list[int]:
    """CPUs suitable for long CUDA / Habitat evals.

    Priority:
    1. ``explicit_allow`` if provided (after subtracting ``explicit_exclude``).
    2. Else online CPUs minus turbo cores (``exclude_min_mhz``) and
       ``explicit_exclude``.
    """
    online = online_cpu_ids(sysfs_root)
    exclude = set(explicit_exclude or [])
    if exclude_min_mhz is not None and exclude_min_mhz > 0:
        exclude.update(cpus_at_or_above_mhz(exclude_min_mhz, sysfs_root=sysfs_root, cpu_ids=online))
    if explicit_allow is not None:
        base = [c for c in explicit_allow if c in online]
    else:
        base = list(online)
    kept = [c for c in base if c not in exclude]
    if not kept:
        # Never return empty — fall back to online so callers do not pin nowhere.
        return list(online)
    return kept


def format_taskset_list(cpu_ids: list[int]) -> str:
    """Compact ``taskset -c`` list (e.g. ``0-7,12-31``)."""
    if not cpu_ids:
        return ""
    ids = sorted(set(cpu_ids))
    ranges: list[str] = []
    start = prev = ids[0]
    for cpu_id in ids[1:]:
        if cpu_id == prev + 1:
            prev = cpu_id
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu_id
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def apply_affinity(cpu_ids: list[int], pid: int | None = None) -> None:
    """Pin ``pid`` (default: current process) to ``cpu_ids``."""
    target = os.getpid() if pid is None else int(pid)
    os.sched_setaffinity(target, set(cpu_ids))


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-min-mhz",
        type=float,
        default=float(os.environ.get("EMET_EXCLUDE_CPU_MIN_MHZ", "6000")),
        help="Exclude CPUs with cpuinfo_max_freq >= this (MHz). 0 disables.",
    )
    parser.add_argument(
        "--exclude",
        default=os.environ.get("EMET_CPU_EXCLUDE", ""),
        help="Extra CPUs to exclude (csv / ranges), e.g. 8-11",
    )
    parser.add_argument(
        "--allow",
        default=os.environ.get("EMET_CPU_ALLOW", ""),
        help="If set, only these CPUs (minus --exclude) are kept",
    )
    parser.add_argument("--print-csv", action="store_true", help="Print compact taskset list")
    parser.add_argument("--print-json", action="store_true", help="Print JSON summary")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply affinity to this process (children inherit after exec)",
    )
    parser.add_argument(
        "--apply-pid",
        type=int,
        default=None,
        help="Apply affinity to an existing PID (e.g. bash $$)",
    )
    args = parser.parse_args(argv)

    exclude_mhz = args.exclude_min_mhz if args.exclude_min_mhz > 0 else None
    allow = _parse_int_list(args.allow)
    exclude = _parse_int_list(args.exclude) or []
    turbo = cpus_at_or_above_mhz(exclude_mhz, cpu_ids=online_cpu_ids()) if exclude_mhz else []
    kept = safe_cpu_ids(
        exclude_min_mhz=exclude_mhz,
        explicit_exclude=exclude,
        explicit_allow=allow,
    )
    compact = format_taskset_list(kept)

    if args.print_json:
        import json

        print(
            json.dumps(
                {
                    "exclude_min_mhz": exclude_mhz,
                    "turbo_cpus": turbo,
                    "extra_exclude": exclude,
                    "allow": allow,
                    "kept": kept,
                    "taskset": compact,
                }
            )
        )
    elif args.print_csv or not (args.apply or args.apply_pid is not None):
        print(compact)

    if args.apply or args.apply_pid is not None:
        apply_affinity(kept, pid=args.apply_pid)
        if not args.print_csv and not args.print_json:
            print(f"applied affinity pid={args.apply_pid or os.getpid()} cpus={compact}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
