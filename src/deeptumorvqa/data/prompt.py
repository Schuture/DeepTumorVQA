"""Prompt templates for VLM evaluation.

The prompt is built from the QA record + a few formatting rules:
  - "MC mode" appends "Answer with ONLY the single letter (A, B, C, or D)."
  - "Free-form mode" asks for the answer directly with subtype-aware hints
    (numeric subtypes get a "Answer with a number" suffix, etc.).
  - Patient demographics + voxel spacing are included when present in the
    benchmark record (some subtypes need spacing to compute volumes).
"""

from __future__ import annotations

from typing import Literal

from .loader import InputMode  # re-export typing
from ..eval.metrics import SUBTYPE_ANSWER_TYPE


def _modality_blurb(input_mode: InputMode, num_slices: int | None = None) -> str:
    if input_mode == "2d_image":
        return ("The image is a single axial CT slice from the patient's abdomen "
                "(window width 400, level 50).")
    if input_mode == "2d_video":
        return ("The video plays a series of axial CT slices through the patient's "
                "abdomen (window width 400, level 50).")
    if input_mode == "3d_volume":
        return ("You are given the full 3D abdominal CT volume "
                "(NIfTI, native voxel spacing).")
    return ""


def _patient_blurb(q: dict) -> str:
    parts = []
    if q.get("sex"):
        parts.append(f"sex={q['sex']}")
    if q.get("age"):
        parts.append(f"age={q['age']}")
    if q.get("scanner"):
        parts.append(f"scanner={q['scanner']}")
    if q.get("contrast"):
        parts.append(f"contrast={q['contrast']}")
    return f"Patient: {', '.join(parts)}." if parts else ""


def _spacing_blurb(q: dict) -> str:
    """Include spacing/shape only for measurement subtypes that need it."""
    st = q.get("question_subtype", "")
    needs = SUBTYPE_ANSWER_TYPE.get(st) == "numeric"
    if not needs:
        return ""
    spacing = q.get("spacing")
    shape = q.get("shape")
    if not spacing:
        return ""
    return f"Voxel spacing (mm): {spacing}.  Volume shape: {shape}."


def build_prompt(
    q: dict,
    input_mode: InputMode,
    fmt: Literal["mc", "freeform"],
) -> str:
    """Compose the user message text."""
    parts: list[str] = [_modality_blurb(input_mode)]
    pat = _patient_blurb(q)
    if pat:
        parts.append(pat)
    sp = _spacing_blurb(q)
    if sp:
        parts.append(sp)

    if fmt == "mc":
        parts.append(q["mc_question"])
        parts.append("Answer with ONLY the single letter (A, B, C, or D).")
    else:
        parts.append(q["question"])
        atype = SUBTYPE_ANSWER_TYPE.get(q.get("question_subtype", ""), "categorical")
        if atype == "numeric":
            parts.append("Answer with a single number (no units, no extra words).")
        elif atype == "binary":
            parts.append("Answer with 'yes' or 'no'.")
        else:
            parts.append("Answer with the most appropriate short phrase.")

    return "\n".join(p for p in parts if p)


SYSTEM_PROMPT_VQA = (
    "You are a radiology AI assistant. You read abdominal CT scans and answer "
    "clinical questions using the visual evidence and any patient context "
    "provided. Be concise and follow the answer-format instructions exactly."
)

SYSTEM_PROMPT_AGENT = (
    "You are a radiology AI agent analyzing abdominal CT scans. You can call "
    "tools to obtain segmentation results, quantitative measurements, and "
    "clinical reference knowledge. Use tool calls when the question requires "
    "evidence you cannot obtain by visual inspection alone. After gathering "
    "enough evidence, give the final answer in the requested format "
    "(single letter for multiple choice, exact value/word for free-form)."
)
