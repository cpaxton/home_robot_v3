#!/usr/bin/env python3
"""Compare fixed-registry TAMP runs; emit JSON and optionally a provenance-linked figure."""
import argparse
import json
from pathlib import Path

from emet.eval.tamp_comparison import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    result = compare_runs(args.baseline, args.candidate)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (out / "comparison.json").write_text(text)
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = result["scheduled_pairs"]
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = [result["baseline_successes"], result["candidate_successes"]]
        bars = ax.bar(["Baseline", "Candidate"], [v / n for v in counts])
        ax.bar_label(bars, labels=[f"{v}/{n}" for v in counts], padding=3)
        ax.set(ylim=(0, 1.12), ylabel="Full-task success / scheduled trials",
               title=f"Paired evaluation: +{result['gains']} gains, −{result['losses']} losses")
        fig.tight_layout()
        for extension in ("pdf", "svg"):
            fig.savefig(out / f"paired_success.{extension}")
        plt.close(fig)
    print(text, end="")


if __name__ == "__main__":
    main()
