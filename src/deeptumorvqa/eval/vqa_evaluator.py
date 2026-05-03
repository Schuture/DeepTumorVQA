"""Direct VQA evaluator (no tool use).

Subclass `HFVLMEvaluator`, implement `generate(prompt, images, video_path)`,
and the framework handles dataset iteration, prompt formatting, scoring,
and incremental save/resume.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Literal

from ..data.loader import (
    InputMode,
    Split,
    iter_questions,
)
from ..data.prompt import SYSTEM_PROMPT_VQA, build_prompt
from .metrics import aggregate

SAVE_EVERY = 100  # incremental save cadence


class HFVLMEvaluator(abc.ABC):
    """ABC for direct (non-agent) VQA evaluators.

    Subclasses must implement `generate()`. Optionally override
    `system_prompt` for model-specific system messages.
    """

    system_prompt: str = SYSTEM_PROMPT_VQA

    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # User-supplied
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        images: list | None = None,
        video_path: str | None = None,
    ) -> str:
        """Return the raw model output for one prompt.

        `images` is a list of PIL.Image (one or more frames). `video_path`
        is set when input_mode='2d_video' and the model accepts video files.
        Subclasses may use either or both. CT volumes are passed via
        `video_path` if input_mode='3d_volume' (NIfTI .nii.gz path); a
        subclass that supports 3D should detect and decode it.
        """

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def run(
        self,
        split: Split = "benchmark",
        input_mode: InputMode = "2d_image",
        fmt: Literal["mc", "freeform"] = "mc",
        output_path: str | Path = "results/vqa.json",
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        allow_hf_fallback: bool = False,
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        """Evaluate the model on a benchmark split. Returns metrics."""
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume support
        done_qids: set[str] = set()
        results: list[dict] = []
        if resume and output_path.exists():
            with open(output_path) as f:
                prior = json.load(f)
            results = prior.get("results", [])
            done_qids = {r["qid"] for r in results}
            print(f"[resume] {len(done_qids)} questions already in {output_path}")

        n_processed = 0
        for q, asset in iter_questions(
            split=split,
            mode="vqa",
            input_mode=input_mode,
            data_dir=data_dir,
            cache_dir=cache_dir,
            allow_hf_fallback=allow_hf_fallback,
            limit=limit,
        ):
            qid = q["qid"]
            if qid in done_qids:
                continue

            prompt = build_prompt(q, input_mode, fmt)

            # Resolve image / video kwargs
            images: list | None = None
            video_path: str | None = None
            if input_mode == "2d_image":
                images = [Image.open(asset).convert("RGB")] if asset and asset.exists() else None
            elif input_mode in ("2d_video", "3d_volume"):
                video_path = str(asset) if asset and asset.exists() else None

            try:
                raw = self.generate(prompt, images=images, video_path=video_path)
            except Exception as e:
                raw = f"[ERROR] {type(e).__name__}: {e}"

            results.append({
                "qid": qid,
                "image_id": q["image_id"],
                "question_subtype": q.get("question_subtype"),
                "question_type": q.get("question_type"),
                "raw_output": raw,
                "pred_answer": raw,
                "correct_option": q.get("correct_option"),
                "answer": q.get("answer"),
            })
            n_processed += 1

            if n_processed % SAVE_EVERY == 0:
                self._save(output_path, results, n_processed)

        # Final save
        self._save(output_path, results, n_processed)

        # Score
        bench_index = {r["qid"]: r for r in results
                       if "question_subtype" in r and r.get("question_subtype")}
        # Build full benchmark index for proper aggregation
        from ..data.loader import resolve_root, load_qa
        root = resolve_root(split, "vqa", input_mode, data_dir=data_dir,
                            cache_dir=cache_dir, allow_hf_fallback=allow_hf_fallback)
        qs = load_qa(split, root)
        full_index = {q["qid"]: q for q in qs}
        metrics = aggregate(results, full_index, fmt=fmt)
        self._save_metrics(output_path, metrics, results)
        return metrics

    @staticmethod
    def _save(output_path: Path, results: list[dict], n_processed: int):
        with open(output_path, "w") as f:
            json.dump({"n_processed": n_processed, "results": results}, f, indent=2)

    @staticmethod
    def _save_metrics(output_path: Path, metrics: dict, results: list[dict]):
        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "n_processed": len(results),
                "results": results,
            }, f, indent=2)
