# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Shared mapping update loop for ``emet capture`` and ``emet stream``.

Used by :mod:`emet.app.zmq_obs` after :mod:`emet.app.stream_agent_factory` builds the agent.
See ``docs/zmq_obs.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import click

from emet.app.stream_agent_factory import (
    StreamAgentBundle,
    create_stream_agent,
    format_stream_stats,
    stream_agent_update,
    stream_stats,
)
from emet.config.stream_config import (
    StreamZmqObsConfig,
    apply_stream_logging_config,
    resolve_stream_verbose,
    should_log_every_step,
)


@dataclass
class MappingSessionResult:
    bundle: StreamAgentBundle
    steps: int
    final_stats: dict[str, Any]

    @property
    def agent(self) -> Any:
        return self.bundle.agent


def run_mapping_session(
    *,
    backend: str,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool = True,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    hz: float = 1.0,
    max_steps: int = 0,
    no_sensor_perception: bool = False,
    no_instance_graph: bool = False,
    compare_to_gt: bool = False,
    dynav_from_default: bool = True,
    config_overrides: list[str] | None = None,
    rerun_hold_s: float = 0.0,
    stop_agent: bool = True,
    on_ready: Callable[[StreamAgentBundle], None] | None = None,
    on_status: Callable[[int, dict[str, Any]], None] | None = None,
    verbose: bool = False,
    stream_cfg: StreamZmqObsConfig | None = None,
) -> MappingSessionResult:
    """Run ``agent.update()`` until Ctrl+C, robot disconnect, or ``max_steps``."""
    if stream_cfg is None:
        stream_cfg = StreamZmqObsConfig()
    effective_verbose = resolve_stream_verbose(cli_verbose=verbose, yaml_value=stream_cfg.verbose)
    apply_stream_logging_config(stream_cfg, verbose=effective_verbose)
    status_interval_s = float(stream_cfg.status_interval_s)
    log_every_step = should_log_every_step(max_steps=max_steps, cfg=stream_cfg, verbose=effective_verbose)
    if backend == "voxel_only":
        click.echo("Voxel-only: skipping SigLIP/YoloE/VLM/scene-graph models (voxel + depth only).")
    bundle = create_stream_agent(
        backend,  # type: ignore[arg-type]
        robot=robot,
        host=host,
        port_offset=port_offset,
        dynav_config=dynav_config,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow_missing_depth,
        cpu_only=cpu_only,
        use_sensor_perception=not no_sensor_perception,
        use_instance_graph=not no_instance_graph,
        compare_to_gt=compare_to_gt,
        dynav_from_default=dynav_from_default,
        config_overrides=config_overrides,
    )
    agent = bundle.agent
    resolved_backend = bundle.backend
    if resolved_backend == "ground_truth":
        from emet.app.graph_nav_gt import ensure_ground_truth_ready

        ensure_ground_truth_ready(agent, context="stream")

    if on_ready is not None:
        on_ready(bundle)

    period_s = 1.0 / max(0.05, float(hz))
    step = 0
    last_status_t = 0.0
    final_stats: dict[str, Any] = {}
    interrupted = False
    try:
        while agent.robot.is_running():
            t0 = time.monotonic()
            stream_agent_update(agent, resolved_backend)
            step += 1
            now = time.monotonic()
            final_stats = stream_stats(agent, resolved_backend, dynav_resolved=bundle.dynav_resolved)
            if on_status is not None and (log_every_step or now - last_status_t >= status_interval_s):
                on_status(step, final_stats)
                last_status_t = now
            if max_steps > 0 and step >= max_steps:
                break
            elapsed = time.monotonic() - t0
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if stop_agent:
            agent.robot.stop()
        if enable_rerun and rerun_hold_s > 0 and not interrupted:
            url = "http://localhost:9090?url=ws://localhost:9877"
            click.echo(f"Rerun viewer: {url}  (holding {rerun_hold_s:.0f}s)")
            time.sleep(float(rerun_hold_s))

    if max_steps > 0 and step >= max_steps:
        click.echo(f"Finished {step} update(s) (--max-steps {max_steps}); stopping.")

    return MappingSessionResult(bundle=bundle, steps=step, final_stats=final_stats)


def echo_mapping_status(step: int, stats: dict[str, Any]) -> None:
    backend = stats.get("backend", "?")
    click.echo(f"  [{backend}] step {step}: {format_stream_stats(stats)}")
