"""Record/render Mars curriculum mosaics, publish media and export 3D coverage."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.visualization.curriculum_cases import FAMILIES, cases_for
from emet.visualization.curriculum_coverage import export
from emet.visualization.curriculum_gallery import publish
from emet.visualization.curriculum_record import record, sha
from emet.visualization.curriculum_video import render


def run(output, families, *, fps=8, resume=False, loop_shorter=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "curriculum-render-job-v1",
        "started": datetime.now(timezone.utc).isoformat(),
        "families": list(families),
        "complete": False,
        "hardware_used": False,
        "producer_sha256": sha(__file__),
    }

    def update(phase, family=None):
        state.update(phase=phase, current_family=family, updated=datetime.now(timezone.utc).isoformat())
        save(output / "progress.json", state)

    publish(output)
    try:
        for family in families:
            update("recording", family)
            report_path = output / family / "report.json"
            if resume and report_path.exists():
                report = json.loads(report_path.read_text())
                from emet.visualization import curriculum_record

                if (
                    report.get("complete") is not True
                    or report["sources"] != source_hashes()
                    or report["producer_sha256"] != sha(curriculum_record.__file__)
                    or report["cases_sha256"] != digest(cases_for(family))
                ):
                    raise ValueError("partial or changed recording; use a fresh output directory")
            else:
                record(output, [family])
            update("rendering", family)
            receipt_path = output / (family + "-video.json")
            if resume and receipt_path.exists():
                from emet.visualization import curriculum_video

                receipt = json.loads(receipt_path.read_text())
                if (
                    receipt["producer_sha256"] != sha(curriculum_video.__file__)
                    or receipt["source_report_sha256"] != sha(report_path)
                    or any(sha(output / name) != expected for name, expected in receipt["assets"].items())
                ):
                    raise ValueError("changed render; use a fresh output directory")
            else:
                render(output, family, fps=fps, loop_shorter=loop_shorter)
            publish(output)
        update("coverage")
        export(output)
        publish(output)
        state["complete"] = True
        update("complete")
    except BaseException as exc:
        state["error"] = str(exc)
        update("failed", state.get("current_family"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loop-shorter", action="store_true")
    args = parser.parse_args()
    families = args.family or (
        "arm_splines_l0",
        "arm_splines_l1",
        "arm_splines_l2",
        "hand",
        "base",
        "goal_changes",
        "tamp",
    )
    run(args.output, families, fps=args.fps, resume=args.resume, loop_shorter=args.loop_shorter)


if __name__ == "__main__":
    main()
