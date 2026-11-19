# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Frozen-evidence controls around the production EQA answer path.

This is an answer-packet experiment, not another agent or navigation policy.
Its empty memory deliberately isolates the production answer/context module.
"""

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

from emet.core.parameters import Parameters
from emet.llms.graph_eqa_vlm import GraphEQAVLMClient
from emet.llms.openai_server import LLMServeState
from emet.llms.openai_vllm_client import OpenaiVLLMClient
from emet.llms.prompts.hmeqa_eqa_prompt import HMEQA_EQA_PROMPT
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def png_hashes(images):
    return [hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() for item in images]


def semantic_choice(answer, choices):
    """Scorer-only exact choice/letter normalization; never repair model text."""
    if not isinstance(answer, str):
        return None
    text = " ".join(answer.split())
    letters = [chr(65 + index) for index in range(len(choices))]
    if text in letters:
        return text
    matches = [
        letter
        for letter, choice in zip(letters, choices, strict=True)
        if text.casefold() == " ".join(choice.split()).casefold()
    ]
    return matches[0] if len(matches) == 1 else None


class PinnedPNGClient(OpenaiVLLMClient):
    """Use the original, whitelisted PNGs instead of remote-client JPEG encoding."""

    def __init__(self, images, *, base_url, model, api_key):
        super().__init__(model=model, base_url=base_url, api_key=api_key, image_max_side=0)
        self._openai = OpenAI(base_url=base_url, api_key=api_key, timeout=240, max_retries=0)
        self._pngs = {}
        for item in images:
            path = Path(item["path"])
            arr = np.asarray(Image.open(path).convert("RGB"))
            self._pngs[(arr.shape, arr.tobytes())] = path.read_bytes()

    def _rgb_to_jpeg_data_url(self, rgb):
        arr = np.asarray(rgb)
        data = self._pngs.get((arr.shape, arr.tobytes()))
        if data is None:
            raise ValueError("Production answer path requested an image outside the frozen packet")
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class FrozenServeState(LLMServeState):
    """Real model inference with an auditable per-arm allowance and PNG boundary."""

    def begin_arm(self, output, images, allowance, *, expected_hashes=None):
        self.trace_path = Path(output) / "transport_calls.jsonl"
        self.expected_images = png_hashes(images)
        if expected_hashes is not None and self.expected_images != expected_hashes:
            raise ValueError("Frozen PNG files changed since the comparison manifest was captured")
        self.allowance = allowance
        self.requested_tokens = 0
        self.records = []

    def complete(self, body):
        record = {"request": body}
        started = time.monotonic()
        try:
            cap = body.get("max_tokens", body.get("max_completion_tokens"))
            if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                raise ValueError("Every comparison call needs an explicit positive output cap")
            if self.requested_tokens + cap > self.allowance:
                raise ValueError("Production path exceeded its total per-arm output allowance")
            hashes = []
            for message in body["messages"]:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if not url.startswith("data:image/png;base64,"):
                            raise ValueError("All conditions must send the same independent PNGs")
                        hashes.append(hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest())
            if hashes and hashes != self.expected_images:
                raise ValueError("Image bytes/order differ from the frozen packet")
            self.requested_tokens += cap
            record["image_sha256"] = hashes
            response = super().complete(body)
            record["response"] = response
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["wall_s"] = time.monotonic() - started
            self.records.append(record)
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def validate_arm(self):
        for record in self.records:
            if "error" in record:
                raise ValueError(record["error"])
            choice = record["response"]["choices"][0]
            if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                raise ValueError("Incomplete/refused response remains a failed trial")
        if sum(bool(record.get("image_sha256")) for record in self.records) != 1:
            raise ValueError("Each answer path must make exactly one image-bearing call")


def run_emet_packet(case, *, base_url, model, api_key, allowance=2048):
    """Run actual keyword extraction + model-final query_answer on an empty graph."""
    keyword_cap = 64
    if allowance <= keyword_cap:
        raise ValueError("The arm allowance must include keyword and answer calls")
    remote = PinnedPNGClient(case["images"], base_url=base_url, model=model, api_key=api_key)
    memory = GraphEQAMemory(
        parameters=Parameters(
            eqa={"prompt_variant": "hmeqa", "answer_format": "json"},
            eqa_vl={"eqa_max_images": len(case["images"]), "answer_max_new_tokens": allowance - keyword_cap},
        ),
        image_description_client=GraphEQAVLMClient(remote, system_prompt=None, max_tokens=keyword_cap),
        eqa_client=GraphEQAVLMClient(remote, system_prompt=HMEQA_EQA_PROMPT, max_tokens=allowance - keyword_cap),
        defer_llm_clients=True,
    )
    images = [np.asarray(Image.open(item["path"]).convert("RGB")) for item in case["images"]]
    question = (
        case["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(case["choices"]))
    )
    _, answer, confidence, _, _, attached = memory.query_answer(
        question,
        evidence_images=images,
        evidence_labels=[f"Image {i + 1}: {item['label']}" for i, item in enumerate(case["images"])],
        evidence_roles=["assessed_view"] + ["historical_context"] * (len(images) - 1),
        model_final=True,
    )
    if len(attached) != len(images) or any(not np.array_equal(a, b) for a, b in zip(attached, images, strict=True)):
        raise ValueError("Production answer path changed the frozen image packet")
    if memory.store.nodes or memory.last_eqa_salvage_used:
        raise ValueError("Packet-only comparison must not create graph objects or salvage answers")
    return {"answer": answer, "is_confident": confidence, "raw": memory.last_eqa_model_raw}
