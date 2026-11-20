"""Join disjoint simulator audit shards for one Mars actor."""

import argparse
import json
from collections import Counter
from pathlib import Path

from emet.rl.manipulation.arm_spline_benchmark import save

FAMILIES = {"base", "arm", "coupled", "carry", "tamp", "lift", "preplace"}


def summarize(paths, minimum_per_family):
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report.get("complete") is not True or report.get("mode") != "audit" for report in reports):
        raise ValueError("All shards must be complete held-out audit reports")
    weights = {report["weights_sha256"] for report in reports}
    if len(weights) != 1:
        raise ValueError("Audit shards must use identical actor weights")
    manifests = {report.get("cases_sha256") for report in reports}
    if len(manifests) != 1:
        raise ValueError("Audit shards must use the same case manifest")
    rows = [row for report in reports for row in report["rows"]]
    if len({(row["family"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate case in the audit")
    counts = Counter(row["family"] for row in rows)
    if set(counts) != FAMILIES or min(counts.values()) < minimum_per_family:
        raise ValueError("Every family must meet the declared audit sample count")
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("Audit rows require an explicit success result")
        if row["success"]:
            predicates = row.get("final_predicates")
            stop = row.get("stop_confirmed") is True or (
                isinstance(predicates, dict) and predicates.get("robot_still") is True
            )
            flags = row.get("flags", {})
            if not stop or any(flags.values()) or (isinstance(predicates, dict) and not all(predicates.values())):
                raise ValueError(f"Inconsistent success evidence for {row['id']}")
    passed = Counter(row["family"] for row in rows if row["success"])
    return {
        "weights_sha256": weights.pop(),
        "cases_sha256": manifests.pop(),
        "scope": "closed-loop simulator screening by task family",
        "family_results": {family: {"passed": passed[family], "cases": counts[family]} for family in sorted(FAMILIES)},
        "all_cases_passed": passed == counts,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-per-family", type=int, default=2)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.minimum_per_family < 1 or args.output.exists():
        parser.error("Require a positive audit count per family and a fresh output path")
    report = summarize(args.shards, args.minimum_per_family)
    save(args.output, report)
    for family, row in report["family_results"].items():
        print(f"{family} {row['passed']}/{row['cases']}")


if __name__ == "__main__":
    main()
