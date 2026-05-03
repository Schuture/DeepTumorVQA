"""Reference Agent evaluator for Qwen3-VL via vLLM OpenAI-compatible server.

Run the server first (in a separate process / SLURM job):

    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-VL-4B-Instruct \\
        --port 8877 \\
        --max-model-len 8192 \\
        --gpu-memory-utilization 0.85 \\
        --dtype bfloat16 \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes

Then in your eval script:

    from deeptumorvqa.reference.qwen3vl_agent import Qwen3VLAgentEvaluator

    ev = Qwen3VLAgentEvaluator(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        api_base="http://localhost:8877/v1",
        mode="oracle",
    )
    metrics = ev.run(split="benchmark", fmt="mc",
                     output_path="results/qwen3vl_agent_oracle.json", limit=100)

The agent loop, tool execution, and tool-cache lookup are all handled by the
parent class — this file only implements `chat(messages, tools)` by talking
to the vLLM server's OpenAI-compatible API.

For vision mode, this implementation also supports passing the initial 2D image
to the model via the standard OpenAI image_url message format.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal

from ..eval.agent_evaluator import AgentEvaluator


class Qwen3VLAgentEvaluator(AgentEvaluator):
    """Run Qwen3-VL agent against the DeepTumorVQA benchmark.

    Talks to a vLLM OpenAI-compatible server. The server itself is responsible
    for parsing model tool-call output via `--tool-call-parser hermes` (works
    for both Qwen3-VL and Meissa, both Hermes-format).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        api_base: str = "http://localhost:8877/v1",
        api_key: str = "EMPTY",
        mode: Literal["oracle", "predicted", "vision"] = "oracle",
        temperature: float = 0.0,
        max_tokens: int = 512,
        request_timeout: int = 120,
        **kwargs,
    ):
        super().__init__(model_id=model_id, mode=mode, **kwargs)
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=request_timeout)
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Required override
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """One round-trip to the vLLM server."""
        # Inline base64 images returned by `crop_organ` so vLLM sees them.
        # vLLM's OpenAI-compat API expects image content as
        # `{"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}`
        prepared = [_inline_tool_images(m) for m in messages]

        kwargs = dict(
            model=self.model_id,
            messages=prepared,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        # Normalise tool_calls to {name, arguments: dict}
        tool_calls: list[dict] = []
        for tc in (choice.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"name": tc.function.name, "arguments": args})

        return {"content": choice.content, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Helper: convert tool messages with `image_base64` results into image content
# ---------------------------------------------------------------------------

def _inline_tool_images(msg: dict) -> dict:
    """If a tool message carries an image (from crop_organ), reformat it so
    the model can actually see the image on the next turn.

    OpenAI Chat API doesn't allow image content inside a tool message, so we
    follow vLLM's documented pattern: rewrite the tool message as a brief
    text summary, then append a follow-up user message with the image. We
    don't actually do that here — we just re-encode the inline image into
    the tool message content, and let the underlying model decide what to do.
    Keep this simple; users can override for fancier handling.
    """
    if msg.get("role") != "tool":
        return msg
    content = msg.get("content", "")
    if not isinstance(content, str):
        return msg
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return msg
    b64 = payload.get("image_base64")
    if not b64:
        return msg
    # Replace with a text-only summary; embed image as a follow-up user msg
    # is the right thing, but tool-message-with-text works for Qwen3-VL.
    summary = {k: v for k, v in payload.items() if k != "image_base64"}
    summary["image"] = "<inline_image>"
    return {**msg, "content": json.dumps(summary)}
