"""Reference VQA evaluator for Qwen3-VL (4B / 8B).

Usage:
    from deeptumorvqa.reference.qwen3vl_vqa import Qwen3VLEvaluator

    ev = Qwen3VLEvaluator(model_id="Qwen/Qwen3-VL-4B-Instruct", backend="vllm")
    metrics = ev.run(split="benchmark", input_mode="2d_image", fmt="mc",
                     output_path="results/qwen3vl_vqa.json", limit=100)

Backends:
    - "vllm" (default, faster): requires `vllm>=0.6.0` and a compatible CUDA build
    - "hf": HuggingFace transformers (slower but no extra deps)

Supports input modes 2d_image and 2d_video. Does NOT support 3d_volume — Qwen3-VL
has no native 3D processor; use a 3D-specific reference impl for that.

This is a TEMPLATE — fork it to plug in your own model. The only required
override is `generate(prompt, images, video_path) -> str`.
"""

from __future__ import annotations

from typing import Literal

from ..eval.vqa_evaluator import HFVLMEvaluator


class Qwen3VLEvaluator(HFVLMEvaluator):
    """Run Qwen3-VL on the DeepTumorVQA benchmark."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        backend: Literal["vllm", "hf"] = "vllm",
        device: str = "cuda",
        max_new_tokens: int = 256,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enable_thinking: bool = False,
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        if backend == "vllm":
            from vllm import LLM, SamplingParams  # type: ignore
            self._llm = LLM(
                model=model_id,
                trust_remote_code=True,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                limit_mm_per_prompt={"image": 4, "video": 1},
                dtype="bfloat16",
            )
            self._sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            self._processor = None
            self._model = None
        elif backend == "hf":
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            self._torch = torch
            self._llm = None
        else:
            raise ValueError(f"backend must be 'vllm' or 'hf', got {backend!r}")

    # ------------------------------------------------------------------
    # Required override
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        images: list | None = None,
        video_path: str | None = None,
    ) -> str:
        if self.backend == "vllm":
            return self._generate_vllm(prompt, images, video_path)
        return self._generate_hf(prompt, images, video_path)

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

    def _generate_vllm(self, prompt, images, video_path) -> str:
        # Build the OpenAI-style multimodal message
        content: list[dict] = []
        if images:
            for im in images:
                content.append({"type": "image", "image": im})
        if video_path:
            content.append({"type": "video", "video": video_path})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        outputs = self._llm.chat(messages, sampling_params=self._sampling)
        text = outputs[0].outputs[0].text
        return _strip_think(text) if not self.enable_thinking else text

    # ------------------------------------------------------------------
    # HF transformers backend
    # ------------------------------------------------------------------

    def _generate_hf(self, prompt, images, video_path) -> str:
        if video_path is not None:
            raise NotImplementedError(
                "HF backend video input requires the qwen-vl-utils pipeline. "
                "Use backend='vllm' for video, or override _generate_hf in a subclass."
            )

        # Build a Qwen3-VL chat-template message
        content: list[dict] = []
        if images:
            for im in images:
                content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        # apply_chat_template returns the formatted text + image inputs
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with self._torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # Strip the prompt tokens
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        return _strip_think(text) if not self.enable_thinking else text


def _strip_think(text: str) -> str:
    """Remove `<think>...</think>` (Qwen3-VL thinking mode)."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    return text
