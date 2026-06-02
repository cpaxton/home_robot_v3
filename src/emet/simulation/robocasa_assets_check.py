# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Robocasa kitchen asset completeness checks (basic fixtures + LightWheel registry)."""

from __future__ import annotations

from pathlib import Path


def robocasa_package_dir(project_root: Path | None = None) -> Path:
    """Path to ``third_party/robocasa/robocasa`` (the importable package)."""
    root = project_root or Path(__file__).resolve().parents[3]
    return root / "third_party" / "robocasa" / "robocasa"


def fixture_registry_layout_ok(robocasa_pkg: Path) -> bool:
    """True when per-type registry YAMLs exist (not wiped by fixtures_lw extract)."""
    from emet.simulation.robocasa_registry_sync import missing_required_registry_stems

    return not missing_required_registry_stems(robocasa_pkg)


def basic_fixtures_present(robocasa_pkg: Path) -> bool:
    """True when the base fixtures zip was extracted (e.g. ``sinks/white_sink/model.xml``)."""
    sentinel = robocasa_pkg / "models" / "assets" / "fixtures" / "sinks" / "white_sink" / "model.xml"
    return sentinel.is_file()


def lightwheel_fixtures_present(robocasa_pkg: Path) -> bool:
    """True when LightWheel fixtures are usable (registry lists ``Sink025``, etc.).

    The ``fixtures_lw`` zip often adds mesh dirs without updating YAML; call
    :func:`emet.simulation.robocasa_registry_sync.ensure_lightwheel_registry` first.
    """
    from emet.simulation.robocasa_registry_sync import ensure_lightwheel_registry

    return ensure_lightwheel_registry(robocasa_pkg)


def objaverse_objects_ready(robocasa_pkg: Path) -> bool:
    """True when objaverse MJCFs have ``reg_bbox`` geoms (required for object sampling)."""
    from emet.simulation.robocasa_objaverse_bbox import objaverse_reg_bbox_present

    return objaverse_reg_bbox_present(robocasa_pkg)


def robocasa_kitchen_assets_complete(robocasa_pkg: Path | None = None) -> bool:
    """Basic meshes + fixture_registry + LightWheel registry + objaverse bbox for serve robocasa."""
    pkg = robocasa_pkg or robocasa_package_dir()
    return (
        basic_fixtures_present(pkg)
        and fixture_registry_layout_ok(pkg)
        and lightwheel_fixtures_present(pkg)
        and objaverse_objects_ready(pkg)
    )


def format_robocasa_assets_incomplete_message(
    *,
    detail: str | None = None,
    missing_lightwheel: bool = False,
    missing_basic: bool = False,
    missing_registry: bool = False,
    missing_objaverse_bbox: bool = False,
) -> str:
    """User-facing instructions when Robocasa scene generation cannot resolve fixture styles."""
    lines = [
        "=" * 60,
        "  Robocasa kitchen assets are missing or incomplete.",
    ]
    if detail:
        lines.append(f"  {detail}")
    if missing_basic:
        lines.append(
            "  Missing: base fixture meshes (textures/fixtures zip). "
            "Run from the project root:"
        )
        lines.append("    uv run python scripts/download_robocasa_assets.py --yes")
    if missing_registry:
        lines.append(
            "  Missing: fixture_registry YAML layout (often deleted when extracting fixtures_lw). "
            "Restore vendored registry files from the robocasa submodule, then sync LightWheel entries:"
        )
        lines.append(
            "    git -C third_party/robocasa checkout -- robocasa/models/assets/fixtures/fixture_registry/"
        )
        lines.append("    uv run python scripts/sync_robocasa_lightwheel_registry.py")
    if missing_objaverse_bbox:
        lines.append(
            "  Missing: objaverse object bounding-box geoms (reg_bbox). "
            "The objaverse zip is raw MJCF; run the one-time post-process:"
        )
        lines.append("    uv run python scripts/process_robocasa_objaverse_reg_bbox.py")
    if missing_lightwheel or (not missing_basic and detail and "Did not find style" in detail):
        lines.append(
            "  Missing: LightWheel fixture pack (``fixtures_lw``) or registry sync. "
            "Kitchen styles reference IDs like Sink025; the lw zip adds mesh folders under "
            "fixtures/sinks/Sink025/ but registry YAML must list them (run sync below)."
        )
        lines.append("  After downloading fixtures_lw, sync registry entries:")
        lines.append("    uv run python -c \"from emet.simulation.robocasa_registry_sync import sync_lightwheel_registry; sync_lightwheel_registry()\"")
        lines.append("  Download everything needed for Robocasa sim (recommended):")
        lines.append("    ./scripts/install_simulation.sh -y")
        lines.append("  Or only fetch missing packs:")
        lines.append("    uv run python scripts/download_robocasa_assets.py --yes")
        lines.append("  (installs base fixtures + fixtures_lw automatically)")
        lines.append("  See docs/simulation.md — section \"Install Robocasa\".")
    lines.append("=" * 60)
    return "\n".join(lines)


def diagnose_robocasa_assets(
    robocasa_pkg: Path | None = None,
) -> tuple[bool, bool, bool, bool, bool]:
    """Return ``(complete, basic_ok, registry_layout_ok, lightwheel_ok, objaverse_ok)``."""
    pkg = robocasa_pkg or robocasa_package_dir()
    basic = basic_fixtures_present(pkg)
    layout = fixture_registry_layout_ok(pkg)
    lw = lightwheel_fixtures_present(pkg)
    obj = objaverse_objects_ready(pkg)
    return (basic and layout and lw and obj, basic, layout, lw, obj)
