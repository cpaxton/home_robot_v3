# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Serial, bounded replay of cached datasets; no scene generation or robot actions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, help="Run a separate local candidate plus matched verifier controls")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest, truth = [], []
    for folder in args.datasets:
        manifest.extend(yaml.safe_load((folder / "manifest.yaml").read_text()))
        truth.extend(json.loads((folder / "truth.json").read_text()))
    (args.output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (args.output_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    if args.preset:
        from emet.core.parameters import get_parameters

        parameters = get_parameters(str(args.preset))
        spec = parameters.get("eval", {})["grounding"]
        variants = [*spec["controls"], spec["verifier"]]
        (args.output_dir / "resolved_parameters.json").write_text(json.dumps(parameters.data, indent=2))
        (args.output_dir / "preset.yaml").write_text(args.preset.read_text())
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_vlm_regions.py",
                "--config",
                str(args.preset),
                "--manifest",
                str(args.output_dir / "manifest.yaml"),
                "--output-dir",
                str(args.output_dir / "proposals"),
                "--strategy",
                "depth_candidates",
                "--box-ablation",
                spec["box_ablation"],
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_candidate_context.py",
                "--config",
                str(args.preset),
                "--baseline",
                str(args.output_dir / "proposals/results.json"),
                "--output-dir",
                str(args.output_dir / "verification"),
                "--variants",
                *variants,
            ],
            check=True,
        )
        for variant in variants:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/score_grounding_dataset.py",
                    "--truth",
                    str(args.output_dir / "truth.json"),
                    "--results",
                    str(args.output_dir / "verification" / variant / "results.json"),
                ],
                check=True,
            )
        return
    for variant in ("baseline", "expand_25", "whole_object", "verify", "repair"):
        command = [
            sys.executable,
            "scripts/audit_vlm_regions.py",
            "--manifest",
            str(args.output_dir / "manifest.yaml"),
            "--output-dir",
            str(args.output_dir / variant),
            "--strategy",
            "depth_candidates",
            "--remote-image-format",
            "png",
            "--box-ablation",
            variant,
        ]
        if variant in ("expand_25", "verify", "repair"):
            command += ["--replay-boxes", str(args.output_dir / "baseline/results.json")]
        print(f"=== {variant}: {len(manifest)} fixed queries ===", flush=True)
        subprocess.run(command, check=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/score_grounding_dataset.py",
                "--truth",
                str(args.output_dir / "truth.json"),
                "--results",
                str(args.output_dir / variant / "results.json"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
