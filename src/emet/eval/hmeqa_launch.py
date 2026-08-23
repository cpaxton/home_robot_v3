# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build ``env KEY=VAL …`` parts for ``emet hmeqa h2h`` / resume job launches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emet.habitat.config import default_habitat_eqa_data_dir, default_hm3d_scene_dir
from emet.llms.remote_ops import DEFAULT_LLM_PORT, DEFAULT_VL_PORT, openai_base_for_host

HMEQA_RUN_MANIFEST_SCHEMA = "emet.hmeqa.run_manifest"
HMEQA_RUN_MANIFEST_VERSION = 2

DEFAULT_DECISION_POLICY = "legacy"
DEFAULT_GRAPH_EVIDENCE_MODE = "off"
DEFAULT_ROOM_HISTORY_MODE = "off"
DEFAULT_ROOM_POLICY = "canonical"
DEFAULT_ROOM_TARGET_HINTS = True
DEFAULT_INVESTIGATE_STAMP = False
DEFAULT_ATTEMPT_LEDGER_MODE = "off"
DEFAULT_USE_HM3D_SEMANTICS = False
DEFAULT_USE_ENRICH_LABELS = False
DEFAULT_VARIANT_ID = "legacy"

DEFAULT_EQA_HF_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_EQA_VL_FAMILY = "qwen3_vl"
DEFAULT_EQA_VL_QUANTIZATION = "int4"
DEFAULT_EQA_ANSWER_MAX_NEW_TOKENS = 384
DEFAULT_EPISODE_TIMEOUT_SECONDS = 7200
DEFAULT_MAX_PLANNING_STEPS = 20
DEFAULT_MAX_MOVEMENT_STEP = 10
DEFAULT_AGENTIC_MAX_TOOL_ROUNDS = 8
DEFAULT_AGENTIC_MAX_NAV_STEPS = 8

_DECISION_POLICIES = frozenset({"legacy", "grounded_v2"})
_VISIBILITY_MODES = frozenset({"off", "shadow", "agent"})
_ROOM_POLICIES = frozenset({"canonical", "llm"})
_AGENTIC_VERIFIERS = frozenset({"none", "owlv2", "yoloe"})
_ARMS = frozenset({"classic", "agentic"})
_VARIANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_SOURCE_PATHS = {
    "ARMS": ("evaluation.arms",),
    "HOLDOUT_IDS": ("ids.question_ids",),
    "EMET_EQA_AGENTIC_VERIFIER": ("evaluation.agentic_verifier",),
    "EMET_EQA_AGENTIC_REQUIRE_VERIFIED": ("evaluation.require_verified",),
    "EMET_EQA_AGENTIC_ROUTER": ("evaluation.agentic_router",),
    "EMET_EQA_AGENTIC_DECISION_POLICY": ("variant.agentic_decision_policy",),
    "EMET_EQA_GRAPH_EVIDENCE_MODE": ("variant.graph_evidence_mode",),
    "EMET_EQA_ROOM_HISTORY_MODE": ("variant.room_history_mode",),
    "EMET_EQA_ROOM_POLICY": ("variant.room_policy",),
    "EMET_EQA_ROOM_TARGET_HINTS": ("variant.room_target_hints",),
    "EMET_EQA_ROOM_STAMP_INVESTIGATE": ("variant.investigate_stamp",),
    "EMET_EQA_ATTEMPT_LEDGER_MODE": ("variant.attempt_ledger_mode",),
    "EMET_EQA_ATTEMPT_LEDGER": ("variant.attempt_ledger_mode",),
    "EMET_HMEQA_USE_HM3D_SEMANTICS": ("evaluation.use_hm3d_semantics",),
    "EMET_HMEQA_USE_ENRICH_LABELS": ("evaluation.use_enrich_labels",),
    "EMET_HMEQA_VARIANT_ID": ("variant.id",),
    "EQA_HF_MODEL_ID": ("model.hf_model_id", "model.requested_hf_model_id"),
    "EQA_VL_FAMILY": ("model.vl_family",),
    "EQA_VL_QUANTIZATION": ("model.vl_quantization",),
    "EMET_LLM_HOST": ("model.host", "model.vl_endpoint"),
    "EMET_VL_ENDPOINT": ("model.vl_endpoint",),
    "HMEQA_VL_PORT": ("model.vl_port", "model.vl_endpoint"),
    "HMEQA_LLM_PORT": ("model.llm_port",),
    "TIMEOUT": ("budgets.episode_timeout_seconds",),
    "HMEQA_MAX_PLANNING_STEPS": ("budgets.max_planning_steps",),
    "HMEQA_MAX_MOVEMENT_STEP": ("budgets.max_movement_step",),
    "EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS": ("budgets.agentic_max_tool_rounds",),
    "EMET_EQA_AGENTIC_MAX_NAV_STEPS": ("budgets.agentic_max_nav_steps",),
    "EMET_EQA_ANSWER_MAX_NEW_TOKENS": ("budgets.answer_max_new_tokens",),
    "HABITAT_EQA_DATA_DIR": ("inputs.data_dir",),
    "HM3D_SCENE_DIR": ("inputs.hm3d_root",),
}
_HMEQA_RUNTIME_ONLY_ENV = frozenset({"EMET_EQA_TRACE"})
_HMEQA_NONCANONICAL_BEHAVIOR_ENV = frozenset(
    {
        "EMET_ATTEMPT_LEDGER_MAX",
        "EMET_ATTEMPT_LEDGER_PERSIST_ABSENT",
        "EMET_DYNAGRAPH_EXPLORE_UNCOVERED",
        "EMET_DYNAGRAPH_MCQ_DEBIAS",
        "EMET_DYNAGRAPH_MEMORY_SUMMARY",
        "EMET_HABITAT_PAD_OBSTACLES",
        "EMET_VLM_FRONTIER_SCORING",
        "EMET_WORLD_SESSION_ID",
        "HMEQA_SEED",
    }
)


class HmeqaRunManifestError(ValueError):
    """Raised when an HM-EQA run cannot be created or resumed reproducibly."""


def _choice(name: str, value: Any, choices: frozenset[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise HmeqaRunManifestError(f"{name} must be one of {allowed}; got {value!r}")
    return normalized


def _bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HmeqaRunManifestError(f"{name} must be a boolean; got {value!r}")


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise HmeqaRunManifestError(f"{name} must be an integer; got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise HmeqaRunManifestError(f"{name} must be >= {minimum}; got {result}")
    return result


def _resolved_path(value: str | os.PathLike[str] | None, fallback: Path) -> str:
    path = Path(value).expanduser() if value is not None and str(value).strip() else fallback
    return str(path.expanduser().resolve())


def validate_hmeqa_runtime_environment(
    env: Mapping[str, str],
    *,
    config: Mapping[str, Any],
) -> None:
    """Reject ambient HM-EQA policy overrides that are not frozen in the manifest."""
    unsupported = {
        name
        for name, value in env.items()
        if str(value).strip()
        and (
            (name.startswith("EMET_EQA_") and name not in _ENV_SOURCE_PATHS and name not in _HMEQA_RUNTIME_ONLY_ENV)
            or name in _HMEQA_NONCANONICAL_BEHAVIOR_ENV
        )
    }
    normalized = normalize_hmeqa_run_config(config)
    openai_base = str(env.get("EMET_OPENAI_BASE_URL", "")).strip().rstrip("/")
    host = str(normalized["model"].get("host") or "").strip()
    if openai_base:
        expected = openai_base_for_host(host, int(normalized["model"]["llm_port"])).rstrip("/") if host else ""
        if openai_base != expected:
            unsupported.add("EMET_OPENAI_BASE_URL")
    for name in ("EMET_CALIBAN_HOST", "OPENAI_BASE_URL"):
        if str(env.get(name, "")).strip():
            unsupported.add(name)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise HmeqaRunManifestError(
            "unfrozen HM-EQA behavior environment is set: "
            f"{names}. Add the control to the canonical run config or unset it."
        )


def _csv_words(name: str, value: str, choices: frozenset[str]) -> list[str]:
    words = [part.strip().lower() for part in str(value or "").split(",") if part.strip()]
    if not words:
        raise HmeqaRunManifestError(f"{name} must contain at least one value")
    invalid = [word for word in words if word not in choices]
    if invalid:
        raise HmeqaRunManifestError(
            f"{name} contains invalid value(s) {', '.join(invalid)}; expected {', '.join(sorted(choices))}"
        )
    if len(words) != len(set(words)):
        raise HmeqaRunManifestError(f"{name} contains duplicate values: {value!r}")
    return words


def _csv_ids(name: str, value: str) -> list[int]:
    result: list[int] = []
    for raw in str(value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            qid = int(raw)
        except ValueError as exc:
            raise HmeqaRunManifestError(f"{name} contains a non-integer id: {raw!r}") from exc
        if qid < 0:
            raise HmeqaRunManifestError(f"{name} contains a negative id: {qid}")
        result.append(qid)
    if not result:
        raise HmeqaRunManifestError(f"{name} must contain at least one question id")
    if len(result) != len(set(result)):
        raise HmeqaRunManifestError(f"{name} contains duplicate question ids: {value!r}")
    return result


def normalize_hmeqa_vl_endpoint(raw: str) -> str:
    """Return ``openai@http://…/v1`` (or pass through an already-prefixed spec)."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("openai@"):
        return s
    # Bare URL or host:port → openai@base
    if "://" in s:
        base = s.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"openai@{base}"
    # host or host:port
    if ":" in s and not s.startswith("["):
        host, _, port_s = s.partition(":")
        try:
            port = int(port_s)
        except ValueError:
            port = DEFAULT_VL_PORT
        return f"openai@{openai_base_for_host(host, port)}"
    return f"openai@{openai_base_for_host(s, DEFAULT_VL_PORT)}"


def build_hmeqa_run_config(
    *,
    arms: str,
    ids: str,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    use_hm3d_semantics: bool = DEFAULT_USE_HM3D_SEMANTICS,
    use_enrich_labels: bool = DEFAULT_USE_ENRICH_LABELS,
    decision_policy: str = DEFAULT_DECISION_POLICY,
    graph_evidence_mode: str = DEFAULT_GRAPH_EVIDENCE_MODE,
    room_history_mode: str = DEFAULT_ROOM_HISTORY_MODE,
    room_policy: str = DEFAULT_ROOM_POLICY,
    room_target_hints: bool = DEFAULT_ROOM_TARGET_HINTS,
    investigate_stamp: bool = DEFAULT_INVESTIGATE_STAMP,
    attempt_ledger_mode: str = DEFAULT_ATTEMPT_LEDGER_MODE,
    variant_id: str = DEFAULT_VARIANT_ID,
    eqa_hf_model_id: str | None = None,
    eqa_vl_family: str | None = None,
    eqa_vl_quantization: str | None = None,
    eqa_answer_max_new_tokens: int = DEFAULT_EQA_ANSWER_MAX_NEW_TOKENS,
    host: str | None = None,
    vl_endpoint: str | None = None,
    vl_port: int | None = None,
    llm_port: int = DEFAULT_LLM_PORT,
    episode_timeout_seconds: int = DEFAULT_EPISODE_TIMEOUT_SECONDS,
    max_planning_steps: int = DEFAULT_MAX_PLANNING_STEPS,
    max_movement_step: int = DEFAULT_MAX_MOVEMENT_STEP,
    agentic_max_tool_rounds: int = DEFAULT_AGENTIC_MAX_TOOL_ROUNDS,
    agentic_max_nav_steps: int = DEFAULT_AGENTIC_MAX_NAV_STEPS,
    data_dir: str | os.PathLike[str] | None = None,
    hm3d_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the canonical behavior-affecting configuration frozen for one H2H OUT."""
    variant = str(variant_id or "").strip()
    if not _VARIANT_ID_RE.fullmatch(variant):
        raise HmeqaRunManifestError(
            "variant_id must start with an alphanumeric and contain only "
            "alphanumerics, '.', '_' or '-' (maximum 128 characters)"
        )

    host_s = str(host or "").strip()
    endpoint = normalize_hmeqa_vl_endpoint(str(vl_endpoint or "").strip())
    effective_vl_port = int(vl_port) if vl_port is not None else DEFAULT_VL_PORT
    effective_llm_port = _positive_int("llm_port", llm_port)
    if host_s and not endpoint:
        endpoint = f"openai@{openai_base_for_host(host_s, effective_vl_port)}"
    remote_vl = bool(endpoint or host_s)
    requested_hf_model_id = str(eqa_hf_model_id or "").strip() or None
    hf_model_id = None if remote_vl else (requested_hf_model_id or DEFAULT_EQA_HF_MODEL_ID)
    family = str(eqa_vl_family or "").strip() or DEFAULT_EQA_VL_FAMILY
    quantization = str(eqa_vl_quantization or "").strip() or DEFAULT_EQA_VL_QUANTIZATION

    return {
        "variant": {
            "id": variant,
            "agentic_decision_policy": _choice("agentic_decision_policy", decision_policy, _DECISION_POLICIES),
            "graph_evidence_mode": _choice("graph_evidence_mode", graph_evidence_mode, _VISIBILITY_MODES),
            "room_history_mode": _choice("room_history_mode", room_history_mode, _VISIBILITY_MODES),
            "room_policy": _choice("room_policy", room_policy, _ROOM_POLICIES),
            "room_target_hints": _bool(room_target_hints, name="room_target_hints"),
            "investigate_stamp": _bool(investigate_stamp, name="investigate_stamp"),
            "attempt_ledger_mode": _choice("attempt_ledger_mode", attempt_ledger_mode, _VISIBILITY_MODES),
        },
        "evaluation": {
            "arms": _csv_words("arms", arms, _ARMS),
            "agentic_verifier": _choice("agentic_verifier", agentic_verifier, _AGENTIC_VERIFIERS),
            "require_verified": _bool(require_verified, name="require_verified"),
            "agentic_router": _bool(agentic_router, name="agentic_router"),
            "use_hm3d_semantics": _bool(
                use_hm3d_semantics,
                name="use_hm3d_semantics",
            ),
            "use_enrich_labels": _bool(use_enrich_labels, name="use_enrich_labels"),
        },
        "model": {
            "hf_model_id": hf_model_id,
            "requested_hf_model_id": requested_hf_model_id,
            "vl_family": family,
            "vl_quantization": quantization,
            "host": host_s or None,
            "vl_endpoint": endpoint or None,
            "vl_port": effective_vl_port if host_s else None,
            "llm_port": effective_llm_port if host_s else None,
        },
        "budgets": {
            "episode_timeout_seconds": _positive_int("episode_timeout_seconds", episode_timeout_seconds),
            "max_planning_steps": _positive_int("max_planning_steps", max_planning_steps),
            "max_movement_step": _positive_int("max_movement_step", max_movement_step),
            "agentic_max_tool_rounds": _positive_int("agentic_max_tool_rounds", agentic_max_tool_rounds),
            "agentic_max_nav_steps": _positive_int("agentic_max_nav_steps", agentic_max_nav_steps),
            "answer_max_new_tokens": _positive_int(
                "eqa_answer_max_new_tokens",
                eqa_answer_max_new_tokens,
                allow_zero=True,
            ),
        },
        "ids": {
            "question_ids": _csv_ids("ids", ids),
        },
        "inputs": {
            "data_dir": _resolved_path(data_dir, default_habitat_eqa_data_dir()),
            "hm3d_root": _resolved_path(hm3d_root, default_hm3d_scene_dir()),
        },
    }


def normalize_hmeqa_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a serialized H2H run configuration."""
    try:
        variant = config["variant"]
        evaluation = config["evaluation"]
        model = config["model"]
        budgets = config["budgets"]
        ids = config["ids"]
        inputs = config["inputs"]
        if not all(isinstance(section, Mapping) for section in (variant, evaluation, model, budgets, ids, inputs)):
            raise TypeError("manifest config sections must be mappings")
        return build_hmeqa_run_config(
            arms=",".join(str(value) for value in evaluation["arms"]),
            ids=",".join(str(value) for value in ids["question_ids"]),
            agentic_verifier=str(evaluation["agentic_verifier"]),
            require_verified=_bool(evaluation["require_verified"], name="require_verified"),
            agentic_router=_bool(evaluation["agentic_router"], name="agentic_router"),
            use_hm3d_semantics=_bool(
                evaluation.get("use_hm3d_semantics", DEFAULT_USE_HM3D_SEMANTICS),
                name="use_hm3d_semantics",
            ),
            use_enrich_labels=_bool(
                evaluation.get("use_enrich_labels", DEFAULT_USE_ENRICH_LABELS),
                name="use_enrich_labels",
            ),
            decision_policy=str(variant["agentic_decision_policy"]),
            graph_evidence_mode=str(variant["graph_evidence_mode"]),
            room_history_mode=str(variant["room_history_mode"]),
            room_policy=str(variant["room_policy"]),
            room_target_hints=_bool(variant["room_target_hints"], name="room_target_hints"),
            investigate_stamp=_bool(variant["investigate_stamp"], name="investigate_stamp"),
            attempt_ledger_mode=str(variant["attempt_ledger_mode"]),
            variant_id=str(variant["id"]),
            eqa_hf_model_id=model.get("requested_hf_model_id") or model.get("hf_model_id"),
            eqa_vl_family=str(model["vl_family"]),
            eqa_vl_quantization=str(model["vl_quantization"]),
            eqa_answer_max_new_tokens=int(budgets["answer_max_new_tokens"]),
            host=model.get("host"),
            vl_endpoint=model.get("vl_endpoint"),
            vl_port=model.get("vl_port"),
            llm_port=model.get("llm_port") or DEFAULT_LLM_PORT,
            episode_timeout_seconds=int(budgets["episode_timeout_seconds"]),
            max_planning_steps=int(budgets["max_planning_steps"]),
            max_movement_step=int(budgets["max_movement_step"]),
            agentic_max_tool_rounds=int(budgets["agentic_max_tool_rounds"]),
            agentic_max_nav_steps=int(budgets["agentic_max_nav_steps"]),
            data_dir=str(inputs["data_dir"]),
            hm3d_root=str(inputs["hm3d_root"]),
        )
    except (KeyError, TypeError, ValueError, HmeqaRunManifestError) as exc:
        if isinstance(exc, HmeqaRunManifestError):
            raise
        raise HmeqaRunManifestError(f"invalid HM-EQA run config: {exc}") from exc


def hmeqa_run_config_digest(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 over canonical behavior-affecting values."""
    normalized = normalize_hmeqa_run_config(config)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def hmeqa_git_state(project_root: Path) -> dict[str, Any]:
    """Return full commit plus a digest of the current tracked/untracked dirty state."""

    def _git(*args: str, text: bool = True) -> str | bytes:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(project_root),
                check=True,
                capture_output=True,
                text=text,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HmeqaRunManifestError(f"could not inspect git state in {project_root}: {exc}") from exc
        return result.stdout

    commit = str(_git("rev-parse", "HEAD")).strip()
    status = str(_git("status", "--porcelain=v1", "--untracked-files=all")).splitlines()
    dirty = bool(status)
    dirty_digest: str | None = None
    if dirty:
        diff = _git("diff", "--binary", "HEAD", "--", text=False)
        untracked = _git("ls-files", "--others", "--exclude-standard", "-z", text=False)
        assert isinstance(diff, bytes)
        assert isinstance(untracked, bytes)
        digest = hashlib.sha256()
        digest.update("\n".join(status).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0tracked-diff\0")
        digest.update(diff)
        for raw_path in sorted(path for path in untracked.split(b"\0") if path):
            digest.update(b"\0untracked\0")
            digest.update(raw_path)
            path = project_root / os.fsdecode(raw_path)
            try:
                if path.is_symlink():
                    digest.update(os.fsencode(os.readlink(path)))
                else:
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
            except OSError as exc:
                raise HmeqaRunManifestError(f"could not hash untracked file {path}: {exc}") from exc
        dirty_digest = f"sha256:{digest.hexdigest()}"
    return {
        "commit": commit,
        "dirty": dirty,
        "dirty_digest": dirty_digest,
        "status": status,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise HmeqaRunManifestError(f"could not hash HM-EQA input {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def hmeqa_external_input_state(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint the small datasets and freeze the HM3D asset root path."""
    normalized = normalize_hmeqa_run_config(config)
    inputs = normalized["inputs"]
    data_dir = Path(inputs["data_dir"])
    hm3d_root = Path(inputs["hm3d_root"])
    questions = data_dir / "questions.csv"
    init_poses = data_dir / "scene_init_poses.csv"
    for path in (questions, init_poses):
        if not path.is_file():
            raise HmeqaRunManifestError(f"required HM-EQA input is missing: {path}")
    if not hm3d_root.is_dir():
        raise HmeqaRunManifestError(f"HM3D scene root is missing: {hm3d_root}")
    return {
        "data_dir": str(data_dir),
        "questions": {
            "path": str(questions),
            "sha256": _sha256_file(questions),
        },
        "scene_init_poses": {
            "path": str(init_poses),
            "sha256": _sha256_file(init_poses),
        },
        # HM3D meshes are too large to hash at launch. The canonical path is
        # frozen; the manifest does not claim content identity for scene assets.
        "hm3d_root": str(hm3d_root),
    }


def load_hmeqa_run_manifest(out_dir: Path) -> dict[str, Any]:
    """Load and minimally validate ``OUT/run_manifest.json``."""
    path = Path(out_dir) / "run_manifest.json"
    if not path.is_file():
        raise HmeqaRunManifestError(
            f"cannot resume {out_dir}: run_manifest.json is missing; refusing to mix an unfrozen historical run"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HmeqaRunManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise HmeqaRunManifestError(f"{path} must contain a JSON object")
    if manifest.get("schema") != HMEQA_RUN_MANIFEST_SCHEMA:
        raise HmeqaRunManifestError(
            f"{path} has unsupported schema {manifest.get('schema')!r}; expected {HMEQA_RUN_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("schema_version") != HMEQA_RUN_MANIFEST_VERSION:
        raise HmeqaRunManifestError(
            f"{path} has unsupported schema_version {manifest.get('schema_version')!r}; "
            f"expected {HMEQA_RUN_MANIFEST_VERSION}"
        )
    return manifest


def _validate_hmeqa_run_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    git_state: Mapping[str, Any],
    external_inputs: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    try:
        frozen_config = normalize_hmeqa_run_config(manifest["config"])
        frozen_digest = hmeqa_run_config_digest(frozen_config)
    except (KeyError, HmeqaRunManifestError) as exc:
        raise HmeqaRunManifestError(f"invalid frozen run manifest: {exc}") from exc
    declared_digest = str(manifest.get("config_digest") or "")
    if declared_digest != frozen_digest:
        mismatches.append(f"manifest config digest is corrupt ({declared_digest or 'missing'} != {frozen_digest})")
    requested_digest = hmeqa_run_config_digest(config)
    if requested_digest != frozen_digest:
        mismatches.append(f"config digest {requested_digest} != frozen {frozen_digest}")

    frozen_git = manifest.get("git")
    if not isinstance(frozen_git, Mapping):
        mismatches.append("manifest git state is missing")
    else:
        for key in ("commit", "dirty", "dirty_digest"):
            if frozen_git.get(key) != git_state.get(key):
                mismatches.append(f"git {key} {git_state.get(key)!r} != frozen {frozen_git.get(key)!r}")
    if manifest.get("external_inputs") != external_inputs:
        mismatches.append("external input paths or dataset hashes differ from the frozen manifest")
    if mismatches:
        raise HmeqaRunManifestError("refusing HM-EQA resume: " + "; ".join(mismatches))


def prepare_hmeqa_run_manifest(
    out_dir: Path,
    *,
    project_root: Path,
    config: Mapping[str, Any],
    sources: Mapping[str, str] | None = None,
    resume: bool,
    git_state: Mapping[str, Any] | None = None,
    external_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or validate the immutable, versioned manifest for an H2H output."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized = normalize_hmeqa_run_config(config)
    current_git = dict(git_state or hmeqa_git_state(Path(project_root)))
    current_inputs = dict(external_inputs or hmeqa_external_input_state(normalized))
    for key in ("data_dir", "hm3d_root"):
        if current_inputs.get(key) != normalized["inputs"][key]:
            raise HmeqaRunManifestError(
                f"external input state {key} {current_inputs.get(key)!r} "
                f"does not match config {normalized['inputs'][key]!r}"
            )
    path = out / "run_manifest.json"
    if path.exists():
        if not resume:
            raise HmeqaRunManifestError(
                f"run_manifest.json already exists in {out}; use resume or choose a new output directory"
            )
        manifest = load_hmeqa_run_manifest(out)
        _validate_hmeqa_run_manifest(
            manifest,
            config=normalized,
            git_state=current_git,
            external_inputs=current_inputs,
        )
        return manifest
    artifacts = _hmeqa_run_artifacts(out)
    if artifacts and not resume:
        raise HmeqaRunManifestError(
            f"refusing to replace an existing HM-EQA run in {out}; "
            "choose a new output directory or resume it from a frozen manifest"
        )
    if resume:
        # Give the more specific missing-manifest diagnostic.
        return load_hmeqa_run_manifest(out)

    digest = hmeqa_run_config_digest(normalized)
    manifest: dict[str, Any] = {
        "schema": HMEQA_RUN_MANIFEST_SCHEMA,
        "schema_version": HMEQA_RUN_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": current_git,
        "external_inputs": current_inputs,
        "config_digest": digest,
        "config": normalized,
        "sources": dict(sorted((sources or {}).items())),
        # Keep the most frequently audited values directly visible without
        # requiring consumers to know the nested canonical config layout.
        "variant": normalized["variant"],
        "model": normalized["model"],
        "budgets": normalized["budgets"],
        "ids": normalized["ids"],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return manifest


def _env_value(env: Mapping[str, str], key: str, fallback: Any) -> Any:
    value = env.get(key)
    return fallback if value is None or value == "" else value


def hmeqa_run_config_from_env(
    env: Mapping[str, str],
    *,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve direct-script environment values, reusing a frozen config on resume."""
    base = (
        normalize_hmeqa_run_config(base_config)
        if base_config is not None
        else build_hmeqa_run_config(
            arms="classic,agentic",
            ids="15,68,105,17",
            agentic_verifier="none",
            require_verified=False,
            agentic_router=False,
        )
    )
    variant = base["variant"]
    evaluation = base["evaluation"]
    model = base["model"]
    budgets = base["budgets"]
    ids = base["ids"]
    inputs = base["inputs"]

    ledger_mode_default = variant["attempt_ledger_mode"]
    if "EMET_EQA_ATTEMPT_LEDGER_MODE" not in env and "EMET_EQA_ATTEMPT_LEDGER" in env:
        ledger_mode_default = (
            "agent" if _bool(env["EMET_EQA_ATTEMPT_LEDGER"], name="EMET_EQA_ATTEMPT_LEDGER") else "off"
        )

    return build_hmeqa_run_config(
        arms=str(_env_value(env, "ARMS", ",".join(evaluation["arms"]))),
        ids=str(
            _env_value(
                env,
                "HOLDOUT_IDS",
                ",".join(str(value) for value in ids["question_ids"]),
            )
        ),
        agentic_verifier=str(_env_value(env, "EMET_EQA_AGENTIC_VERIFIER", evaluation["agentic_verifier"])),
        require_verified=_bool(
            _env_value(
                env,
                "EMET_EQA_AGENTIC_REQUIRE_VERIFIED",
                evaluation["require_verified"],
            ),
            name="EMET_EQA_AGENTIC_REQUIRE_VERIFIED",
        ),
        agentic_router=_bool(
            _env_value(env, "EMET_EQA_AGENTIC_ROUTER", evaluation["agentic_router"]),
            name="EMET_EQA_AGENTIC_ROUTER",
        ),
        use_hm3d_semantics=_bool(
            _env_value(
                env,
                "EMET_HMEQA_USE_HM3D_SEMANTICS",
                evaluation["use_hm3d_semantics"],
            ),
            name="EMET_HMEQA_USE_HM3D_SEMANTICS",
        ),
        use_enrich_labels=_bool(
            _env_value(
                env,
                "EMET_HMEQA_USE_ENRICH_LABELS",
                evaluation["use_enrich_labels"],
            ),
            name="EMET_HMEQA_USE_ENRICH_LABELS",
        ),
        decision_policy=str(
            _env_value(
                env,
                "EMET_EQA_AGENTIC_DECISION_POLICY",
                variant["agentic_decision_policy"],
            )
        ),
        graph_evidence_mode=str(_env_value(env, "EMET_EQA_GRAPH_EVIDENCE_MODE", variant["graph_evidence_mode"])),
        room_history_mode=str(_env_value(env, "EMET_EQA_ROOM_HISTORY_MODE", variant["room_history_mode"])),
        room_policy=str(_env_value(env, "EMET_EQA_ROOM_POLICY", variant["room_policy"])),
        room_target_hints=_bool(
            _env_value(env, "EMET_EQA_ROOM_TARGET_HINTS", variant["room_target_hints"]),
            name="EMET_EQA_ROOM_TARGET_HINTS",
        ),
        investigate_stamp=_bool(
            _env_value(
                env,
                "EMET_EQA_ROOM_STAMP_INVESTIGATE",
                variant["investigate_stamp"],
            ),
            name="EMET_EQA_ROOM_STAMP_INVESTIGATE",
        ),
        attempt_ledger_mode=str(_env_value(env, "EMET_EQA_ATTEMPT_LEDGER_MODE", ledger_mode_default)),
        variant_id=str(_env_value(env, "EMET_HMEQA_VARIANT_ID", variant["id"])),
        eqa_hf_model_id=_env_value(
            env,
            "EQA_HF_MODEL_ID",
            model.get("requested_hf_model_id") or model.get("hf_model_id"),
        ),
        eqa_vl_family=str(_env_value(env, "EQA_VL_FAMILY", model["vl_family"])),
        eqa_vl_quantization=str(_env_value(env, "EQA_VL_QUANTIZATION", model["vl_quantization"])),
        eqa_answer_max_new_tokens=int(
            _env_value(
                env,
                "EMET_EQA_ANSWER_MAX_NEW_TOKENS",
                budgets["answer_max_new_tokens"],
            )
        ),
        host=_env_value(env, "EMET_LLM_HOST", model.get("host")),
        vl_endpoint=_env_value(env, "EMET_VL_ENDPOINT", model.get("vl_endpoint")),
        vl_port=int(_env_value(env, "HMEQA_VL_PORT", model.get("vl_port") or DEFAULT_VL_PORT)),
        llm_port=int(_env_value(env, "HMEQA_LLM_PORT", model.get("llm_port") or DEFAULT_LLM_PORT)),
        episode_timeout_seconds=int(_env_value(env, "TIMEOUT", budgets["episode_timeout_seconds"])),
        max_planning_steps=int(_env_value(env, "HMEQA_MAX_PLANNING_STEPS", budgets["max_planning_steps"])),
        max_movement_step=int(_env_value(env, "HMEQA_MAX_MOVEMENT_STEP", budgets["max_movement_step"])),
        agentic_max_tool_rounds=int(
            _env_value(
                env,
                "EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS",
                budgets["agentic_max_tool_rounds"],
            )
        ),
        agentic_max_nav_steps=int(
            _env_value(
                env,
                "EMET_EQA_AGENTIC_MAX_NAV_STEPS",
                budgets["agentic_max_nav_steps"],
            )
        ),
        data_dir=_env_value(env, "HABITAT_EQA_DATA_DIR", inputs["data_dir"]),
        hm3d_root=_env_value(env, "HM3D_SCENE_DIR", inputs["hm3d_root"]),
    )


def hmeqa_config_env(config: Mapping[str, Any]) -> dict[str, str]:
    """Translate canonical run config to the environment consumed by H2H/Habitat."""
    normalized = normalize_hmeqa_run_config(config)
    variant = normalized["variant"]
    evaluation = normalized["evaluation"]
    model = normalized["model"]
    budgets = normalized["budgets"]
    ids = normalized["ids"]
    inputs = normalized["inputs"]
    env = {
        "ARMS": ",".join(evaluation["arms"]),
        "HOLDOUT_IDS": ",".join(str(value) for value in ids["question_ids"]),
        "EMET_EQA_AGENTIC_VERIFIER": evaluation["agentic_verifier"],
        "EMET_EQA_AGENTIC_REQUIRE_VERIFIED": str(int(evaluation["require_verified"])),
        "EMET_EQA_AGENTIC_ROUTER": str(int(evaluation["agentic_router"])),
        "EMET_EQA_AGENTIC_DECISION_POLICY": variant["agentic_decision_policy"],
        "EMET_EQA_GRAPH_EVIDENCE_MODE": variant["graph_evidence_mode"],
        "EMET_EQA_ROOM_HISTORY_MODE": variant["room_history_mode"],
        "EMET_EQA_ROOM_POLICY": variant["room_policy"],
        "EMET_EQA_ROOM_TARGET_HINTS": str(int(variant["room_target_hints"])),
        "EMET_EQA_ROOM_STAMP_INVESTIGATE": str(int(variant["investigate_stamp"])),
        "EMET_EQA_ATTEMPT_LEDGER_MODE": variant["attempt_ledger_mode"],
        # Shadow and agent both collect rows; router state renders them only in
        # agent mode so collection can be audited without policy leakage.
        "EMET_EQA_ATTEMPT_LEDGER": str(int(variant["attempt_ledger_mode"] != "off")),
        "EMET_HMEQA_USE_HM3D_SEMANTICS": str(int(evaluation["use_hm3d_semantics"])),
        "EMET_HMEQA_USE_ENRICH_LABELS": str(int(evaluation["use_enrich_labels"])),
        "EMET_HMEQA_VARIANT_ID": variant["id"],
        "EQA_HF_MODEL_ID": str(model.get("hf_model_id") or ""),
        "EQA_VL_FAMILY": model["vl_family"],
        "EQA_VL_QUANTIZATION": model["vl_quantization"],
        "EMET_LLM_HOST": str(model.get("host") or ""),
        "EMET_VL_ENDPOINT": str(model.get("vl_endpoint") or ""),
        "HMEQA_VL_PORT": str(model.get("vl_port") or DEFAULT_VL_PORT),
        "HMEQA_LLM_PORT": str(model.get("llm_port") or DEFAULT_LLM_PORT),
        "TIMEOUT": str(budgets["episode_timeout_seconds"]),
        "HMEQA_MAX_PLANNING_STEPS": str(budgets["max_planning_steps"]),
        "HMEQA_MAX_MOVEMENT_STEP": str(budgets["max_movement_step"]),
        "EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS": str(budgets["agentic_max_tool_rounds"]),
        "EMET_EQA_AGENTIC_MAX_NAV_STEPS": str(budgets["agentic_max_nav_steps"]),
        "EMET_EQA_ANSWER_MAX_NEW_TOKENS": str(budgets["answer_max_new_tokens"]),
        "HABITAT_EQA_DATA_DIR": inputs["data_dir"],
        "HM3D_SCENE_DIR": inputs["hm3d_root"],
        "EMET_HMEQA_CONFIG_DIGEST": hmeqa_run_config_digest(normalized),
    }
    if model.get("host"):
        env["EMET_OPENAI_BASE_URL"] = openai_base_for_host(
            str(model["host"]),
            int(model.get("llm_port") or DEFAULT_LLM_PORT),
        )
    return env


def missing_hmeqa_snapshot_artifacts(
    bundle_dir: Path,
    *,
    arm: str,
    env: Mapping[str, str],
) -> list[str]:
    """Return required evidence artifacts missing from an H2H bundle snapshot."""

    def enabled(name: str, default: bool) -> bool:
        raw = str(env.get(name, "")).strip().lower()
        if not raw:
            return default
        return raw not in {"0", "false", "no", "off"}

    bundle = Path(bundle_dir)
    required_files: list[str] = []
    required_dirs: list[str] = []
    if enabled("EMET_EVAL_EXPORT_COMPACT_MEMORY", False):
        required_files.append("compact_memory/manifest.json")
    if arm == "agentic":
        if str(env.get("EMET_EQA_GRAPH_EVIDENCE_MODE", "off")).strip().lower() != "off":
            required_files.append("world_evidence.json")
            if enabled("EMET_EVAL_EXPORT_WORLD_EVIDENCE_RGB", True):
                required_dirs.append("world_evidence_views")
        if str(env.get("EMET_EQA_ATTEMPT_LEDGER_MODE", "off")).strip().lower() != "off":
            required_files.append("attempt_ledger.json")
        if str(env.get("EMET_EQA_ROOM_HISTORY_MODE", "off")).strip().lower() != "off":
            required_files.append("room_events.json")

    missing = [
        relative
        for relative in required_files
        if not (bundle / relative).is_file() or (bundle / relative).stat().st_size == 0
    ]
    missing.extend(f"{relative}/" for relative in required_dirs if not (bundle / relative).is_dir())
    return missing


def _flatten_config_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return [prefix]
    result: list[str] = []
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.extend(_flatten_config_paths(value[key], path))
    return result


def _hmeqa_run_artifacts(out: Path) -> list[Path]:
    artifact_names = ("classic.jsonl", "agentic.jsonl", "orchestrator.log", "progress.json", "DONE")
    artifacts = [out / name for name in artifact_names if (out / name).exists()]
    artifacts.extend(out.glob("classic_q*.jsonl"))
    artifacts.extend(out.glob("agentic_q*.jsonl"))
    return artifacts


def prepare_hmeqa_run_manifest_from_env(
    out_dir: Path,
    *,
    project_root: Path,
    env: Mapping[str, str] | None = None,
    resume: bool,
) -> dict[str, Any]:
    """Script entry: reuse frozen values, apply explicit env overrides, and validate."""
    environ = dict(os.environ if env is None else env)
    out = Path(out_dir)
    existing: dict[str, Any] | None = None
    effective_resume = resume
    if resume and (out / "run_manifest.json").is_file():
        existing = load_hmeqa_run_manifest(out)
    elif resume:
        # The overnight orchestrator can mark a newly-created next phase RESUME=1
        # after an earlier phase completed. Permit only a truly empty H2H OUT;
        # historical partial/scored directories still fail closed.
        artifacts = _hmeqa_run_artifacts(out)
        if artifacts:
            raise HmeqaRunManifestError(
                f"cannot resume {out}: run_manifest.json is missing but run artifacts exist; "
                "refusing to mix an unfrozen historical run"
            )
        effective_resume = False
    raw_config = environ.get("EMET_HMEQA_RUN_CONFIG_JSON", "").strip()
    if raw_config:
        try:
            parsed = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise HmeqaRunManifestError(f"invalid EMET_HMEQA_RUN_CONFIG_JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise HmeqaRunManifestError("EMET_HMEQA_RUN_CONFIG_JSON must contain a JSON object")
        config = normalize_hmeqa_run_config(parsed)
    else:
        config = hmeqa_run_config_from_env(
            environ,
            base_config=existing.get("config") if existing is not None else None,
        )
    validate_hmeqa_runtime_environment(environ, config=config)

    expected_digest = environ.get("EMET_HMEQA_CONFIG_DIGEST", "").strip()
    actual_digest = hmeqa_run_config_digest(config)
    if expected_digest and expected_digest != actual_digest:
        raise HmeqaRunManifestError(f"launch config digest {expected_digest} does not match effective {actual_digest}")

    raw_sources = environ.get("EMET_HMEQA_RUN_SOURCES_JSON", "").strip()
    if raw_sources:
        try:
            parsed_sources = json.loads(raw_sources)
        except json.JSONDecodeError as exc:
            raise HmeqaRunManifestError(f"invalid EMET_HMEQA_RUN_SOURCES_JSON: {exc}") from exc
        if not isinstance(parsed_sources, Mapping):
            raise HmeqaRunManifestError("EMET_HMEQA_RUN_SOURCES_JSON must contain a JSON object")
        sources = {str(key): str(value) for key, value in parsed_sources.items()}
    else:
        source = "frozen_manifest" if existing is not None else "script_default"
        sources = dict.fromkeys(_flatten_config_paths(config), source)
        for env_name, paths in _ENV_SOURCE_PATHS.items():
            if environ.get(env_name, "").strip():
                for path in paths:
                    sources[path] = f"environment:{env_name}"

    return prepare_hmeqa_run_manifest(
        out,
        project_root=Path(project_root),
        config=config,
        sources=sources,
        resume=effective_resume,
    )


def hmeqa_h2h_env_parts(
    *,
    arms: str,
    ids: str,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    use_hm3d_semantics: bool = DEFAULT_USE_HM3D_SEMANTICS,
    use_enrich_labels: bool = DEFAULT_USE_ENRICH_LABELS,
    decision_policy: str = DEFAULT_DECISION_POLICY,
    graph_evidence_mode: str = DEFAULT_GRAPH_EVIDENCE_MODE,
    room_history_mode: str = DEFAULT_ROOM_HISTORY_MODE,
    room_policy: str = DEFAULT_ROOM_POLICY,
    room_target_hints: bool = DEFAULT_ROOM_TARGET_HINTS,
    investigate_stamp: bool = DEFAULT_INVESTIGATE_STAMP,
    attempt_ledger_mode: str = DEFAULT_ATTEMPT_LEDGER_MODE,
    variant_id: str = DEFAULT_VARIANT_ID,
    resume: bool = False,
    eqa_hf_model_id: str | None = None,
    eqa_vl_family: str | None = None,
    eqa_vl_quantization: str | None = None,
    eqa_answer_max_new_tokens: int | None = DEFAULT_EQA_ANSWER_MAX_NEW_TOKENS,
    host: str | None = None,
    vl_endpoint: str | None = None,
    vl_port: int | None = None,
    llm_port: int = DEFAULT_LLM_PORT,
    episode_timeout_seconds: int = DEFAULT_EPISODE_TIMEOUT_SECONDS,
    max_planning_steps: int = DEFAULT_MAX_PLANNING_STEPS,
    max_movement_step: int = DEFAULT_MAX_MOVEMENT_STEP,
    run_config: Mapping[str, Any] | None = None,
    config_sources: Mapping[str, str] | None = None,
) -> list[str]:
    """Explicit env assignments injected into the jobs-wrapped H2H script.

    Parent-shell exports are **not** inherited by the Habitat child unless listed here.
    """
    config = (
        normalize_hmeqa_run_config(run_config)
        if run_config is not None
        else build_hmeqa_run_config(
            arms=arms,
            ids=ids,
            agentic_verifier=agentic_verifier,
            require_verified=require_verified,
            agentic_router=agentic_router,
            use_hm3d_semantics=use_hm3d_semantics,
            use_enrich_labels=use_enrich_labels,
            decision_policy=decision_policy,
            graph_evidence_mode=graph_evidence_mode,
            room_history_mode=room_history_mode,
            room_policy=room_policy,
            room_target_hints=room_target_hints,
            investigate_stamp=investigate_stamp,
            attempt_ledger_mode=attempt_ledger_mode,
            variant_id=variant_id,
            eqa_hf_model_id=eqa_hf_model_id,
            eqa_vl_family=eqa_vl_family,
            eqa_vl_quantization=eqa_vl_quantization,
            eqa_answer_max_new_tokens=(
                DEFAULT_EQA_ANSWER_MAX_NEW_TOKENS if eqa_answer_max_new_tokens is None else eqa_answer_max_new_tokens
            ),
            host=host,
            vl_endpoint=vl_endpoint,
            vl_port=vl_port,
            llm_port=llm_port,
            episode_timeout_seconds=episode_timeout_seconds,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
        )
    )
    parts = [
        "EMET_ALLOW_SDPA_ATTN=1",
        "EMET_EQA_TRACE=1",
        f"COVERAGE_QIDS={coverage_qids}",
        f"EPISODE_COOLDOWN_SEC={int(cooldown)}",
        f"NATIVE_CRASH_POLICY={crash_policy}",
        f"NATIVE_CRASH_STREAK_ABORT={int(streak_abort)}",
    ]
    for key, value in hmeqa_config_env(config).items():
        # Empty local-model values are intentionally omitted for remote VL.
        if value != "":
            parts.append(f"{key}={shlex.quote(value)}")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    parts.append(f"EMET_HMEQA_RUN_CONFIG_JSON={shlex.quote(config_json)}")
    if config_sources:
        sources_json = json.dumps(dict(config_sources), sort_keys=True, separators=(",", ":"))
        parts.append(f"EMET_HMEQA_RUN_SOURCES_JSON={shlex.quote(sources_json)}")
    if resume:
        parts.append("RESUME=1")
    return parts


def hmeqa_h2h_vl_endpoint_from_env_parts(parts: list[str]) -> str | None:
    """Return the ``EMET_VL_ENDPOINT`` value from env parts, if present."""
    for p in parts:
        if p.startswith("EMET_VL_ENDPOINT="):
            return p.split("=", 1)[1].strip("'\"")
    return None
