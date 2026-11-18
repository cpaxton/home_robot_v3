#!/usr/bin/env python3
"""Read a WBC run ledger without importing a simulator or changing a checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def ledger_rows(ledger):
    if not isinstance(ledger, dict) or not ledger:
        raise ValueError("nonempty control-tick ledger required")
    for name, item in ledger.items():
        for key in ("control_ticks", "episodes"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError(f"{name}: nonnegative integer {key} required")
        weight = item.get("requested_fraction")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise ValueError(f"{name}: finite requested fraction required")
    if not math.isclose(sum(x["requested_fraction"] for x in ledger.values()), 1, abs_tol=1e-6):
        raise ValueError("requested ledger fractions must sum to one")
    return ledger


def summarize(record):
    outer = ledger_rows(record["mixture"])
    total = record["control_ticks"]
    if type(total) is not int or total < 0 or sum(x["control_ticks"] for x in outer.values()) != total:
        raise ValueError("family control ticks must sum to recorded total")
    rows = []
    for family, item in outer.items():
        inner = record.get("general_mixture") if family == "general" else None
        if inner:
            inner = ledger_rows(inner)
            if sum(x["control_ticks"] for x in inner.values()) != item["control_ticks"]:
                raise ValueError("general child ticks must sum to family ticks")
        for name, child in (inner or {family: item}).items():
            rows.append(
                {
                    "use_case": f"{family}/{name}" if inner else family,
                    "target_fraction": item["requested_fraction"] * child["requested_fraction"]
                    if inner
                    else item["requested_fraction"],
                    "control_ticks": child["control_ticks"],
                    "actual_fraction": child["control_ticks"] / total if total else None,
                    "episodes_started": child["episodes"],
                }
            )
    return {
        "decisions_in_run": record["decisions"],
        "source_decisions": record.get("source_decisions", 0),
        "source_lineage_decisions": record.get("source_cumulative_decisions", record.get("source_decisions", 0)),
        "control_ticks": total,
        "complete": record.get("complete", False),
        "rows": rows,
        "clock": "physical control ticks; episode counters count starts, not completions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run.json, including a still-running checkpoint")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    try:
        contents = args.run.read_bytes()
        result = summarize(json.loads(contents))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    result["run_sha256"] = hashlib.sha256(contents).hexdigest()
    if args.format == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(f"{result['decisions_in_run']:,} decisions in this run; {result['control_ticks']:,} physical control ticks.")
    print("\n| Use case | Target | Actual | Control ticks | Episodes started |\n| --- | ---: | ---: | ---: | ---: |")
    for row in result["rows"]:
        actual = "—" if row["actual_fraction"] is None else f"{100 * row['actual_fraction']:.2f}%"
        print(
            f"| {row['use_case']} | {100 * row['target_fraction']:.2f}% | {actual} | {row['control_ticks']:,} | {row['episodes_started']:,} |"
        )
    print(
        "\nEpisode starts may include an unfinished final episode. Shares use physical ticks, not episode or PPO-decision counts."
    )


if __name__ == "__main__":
    main()
