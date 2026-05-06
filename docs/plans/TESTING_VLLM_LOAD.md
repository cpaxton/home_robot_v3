# Heavy VLLM load tests (`vllm_load`)

Tests marked `@pytest.mark.vllm_load` download Hugging Face weights and run a minimal multimodal forward pass. They are **excluded from the default suite** via `[tool.pytest.ini_options] addopts` in `pyproject.toml` (`-m "not vllm_load"`).

## When to run

- **Manual** (GPU recommended): after `uv sync`, from the repo root:

  ```bash
  uv run pytest -m vllm_load -o addopts='-ra -v' src/test/llms/test_vllm_load_smoke.py
  ```

  The `-o addopts=...` clears the default `-m "not vllm_load"` so the marked tests are selected.

- **Nightly / CI**: run the same command on a machine with CUDA, a long timeout, and optional `HF_HOME` cache. Not required for PR or pre-commit runs.

## Requirements

- GPU strongly recommended for Qwen3-VL / Gemma multimodal smoke.
- Sufficient VRAM for the chosen checkpoint (override with `VLLM_LOAD_TEST_MODEL` in the smoke test file if needed).
- Hugging Face access where models are gated (accept license on the model card when prompted).

## Fast tests

Registry and factory unit tests under `src/test/llms/` (e.g. `test_vllm_registry.py`) run with the normal `uv run emet test` and do not load weights. The project pins `transformers>=4.51` for Qwen3-VL (`Qwen3VLForConditionalGeneration`).
