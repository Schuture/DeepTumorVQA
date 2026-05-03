"""Lazy data loader for DeepTumorVQA_2.0.

Two modes:
  1. HuggingFace mode (default):
     - On first call, fetches only the assets needed for the requested
       (split, input_mode, agent_mode) configuration from
       `tumor-vqa/DeepTumorVQA_2.0` via `huggingface_hub.snapshot_download`.
     - Subsequent calls reuse the local HF cache.

  2. Local mode (`data_dir` given):
     - Reads everything from the given directory (which must mirror the HF
       layout). No network access. Errors clearly if any required artifact
       is missing.

Both modes share the same per-run "allow patterns" rule, so a 2D-image-only
eval never needs to download CT volumes or MP4 videos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

HF_REPO_ID = "tumor-vqa/DeepTumorVQA_2.0"

InputMode = Literal["3d_volume", "2d_image", "2d_video"]
AgentMode = Literal["oracle", "predicted", "vision"]
Split = Literal["benchmark", "train"]
Mode = Literal["vqa", "agent"]


# ---------------------------------------------------------------------------
# Per-run allow_patterns mapping
# ---------------------------------------------------------------------------

def required_patterns(
    split: Split,
    mode: Mode,
    input_mode: InputMode | None = None,
    agent_mode: AgentMode | None = None,
) -> list[str]:
    """Return the minimal HF allow_patterns for this run config."""
    qa_file = "test_qa.json" if split == "benchmark" else "train_qa.csv"
    base = [f"{split}/{qa_file}"]

    if mode == "vqa":
        if input_mode is None:
            raise ValueError("`input_mode` required when mode='vqa'.")
        # Direct VQA always uses whole-volume PNG/MP4 (organ crops are vision-agent-only)
        if input_mode == "2d_image":
            return base + [f"{split}/images_2d/whole/**"]
        if input_mode == "2d_video":
            return base + [f"{split}/videos/whole/**"]
        if input_mode == "3d_volume":
            if split == "train":
                # Train CTs are upstream — direct user to download from AbdomenAtlas
                raise ValueError(
                    "Training-split 3D CTs are not hosted on this dataset. "
                    "Pull them from huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini "
                    "(see train/ct_link.md for the BDMAP IDs we use)."
                )
            return base + [f"{split}/ct/**"]

    if mode == "agent":
        if agent_mode is None:
            raise ValueError("`agent_mode` required when mode='agent'.")
        if agent_mode == "oracle":
            return base + ["tool_cache/benchmark_oracle_tool_cache.json"]
        if agent_mode == "predicted":
            return base + ["tool_cache/benchmark_totalsegmentator_cache.json"]
        if agent_mode == "vision":
            # Vision agent reads pre-extracted organ-focused slices
            return base + [f"{split}/images_2d/organ/**"]

    raise ValueError(f"Unsupported config: split={split} mode={mode}")


# ---------------------------------------------------------------------------
# Resolve dataset root: HF cache OR local --data-dir
# ---------------------------------------------------------------------------

def resolve_root(
    split: Split,
    mode: Mode,
    input_mode: InputMode | None = None,
    agent_mode: AgentMode | None = None,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_hf_fallback: bool = False,
) -> Path:
    """Return the local directory containing the dataset, fetching from HF if needed.

    If `data_dir` is given, validates that the required artifacts exist locally
    and returns it. If `allow_hf_fallback` is also True, missing artifacts are
    fetched into the local data_dir.
    """
    patterns = required_patterns(split, mode, input_mode, agent_mode)

    if data_dir is not None:
        root = Path(data_dir)
        missing = _missing_paths(root, patterns)
        if missing:
            if not allow_hf_fallback:
                raise FileNotFoundError(
                    f"`--data-dir {root}` is missing the following required files for "
                    f"split={split} mode={mode} input={input_mode} agent={agent_mode}:\n  - "
                    + "\n  - ".join(str(p) for p in missing)
                    + "\n\nEither populate the missing files or pass --allow-hf-fallback "
                    "to fetch them from HuggingFace into this directory."
                )
            _hf_download(patterns, local_dir=root, cache_dir=cache_dir)
        return root

    # HF mode
    return Path(_hf_download(patterns, cache_dir=cache_dir))


def _missing_paths(root: Path, patterns: list[str]) -> list[Path]:
    """Return any patterns from `patterns` that don't exist under `root`.

    Patterns ending in `/**` need the directory to exist + be non-empty;
    direct file paths must exist.
    """
    missing: list[Path] = []
    for p in patterns:
        target = root / p.rstrip("/**")
        if p.endswith("/**"):
            if not target.is_dir() or not any(target.iterdir()):
                missing.append(target)
        else:
            if not target.is_file():
                missing.append(target)
    return missing


def _hf_download(
    allow_patterns: list[str],
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Wrap snapshot_download. Returns the local snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(local_dir) if local_dir else None,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


# ---------------------------------------------------------------------------
# QA iterator + image-path resolver
# ---------------------------------------------------------------------------

def load_qa(split: Split, root: Path) -> list[dict]:
    """Load the QA records for a split."""
    if split == "benchmark":
        with open(root / "benchmark" / "test_qa.json") as f:
            data = json.load(f)
        return data["questions"]
    if split == "train":
        import csv
        with open(root / "train" / "train_qa.csv") as f:
            return list(csv.DictReader(f))
    raise ValueError(split)


def asset_path(
    root: Path,
    split: Split,
    image_id: str,
    input_mode: InputMode,
) -> Path:
    """Return the local path to the requested modality artifact (whole-volume)."""
    if input_mode == "2d_image":
        return root / split / "images_2d" / "whole" / f"{image_id}.png"
    if input_mode == "2d_video":
        return root / split / "videos" / "whole" / f"{image_id}.mp4"
    if input_mode == "3d_volume":
        if split == "train":
            raise ValueError("Train CTs are upstream — see train/ct_link.md.")
        return root / split / "ct" / image_id / "ct.nii.gz"
    raise ValueError(input_mode)


def organ_image_dir(root: Path, split: Split = "benchmark") -> Path:
    """Directory containing organ-specific PNGs for vision-agent mode."""
    return root / split / "images_2d" / "organ"


def iter_questions(
    split: Split,
    mode: Mode,
    input_mode: InputMode | None = None,
    agent_mode: AgentMode | None = None,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_hf_fallback: bool = False,
    limit: int | None = None,
) -> Iterator[tuple[dict, Path | None]]:
    """High-level convenience iterator.

    Yields (question_dict, modality_artifact_path). Artifact path is None for
    pure-text agent modes (oracle/predicted) where no image input is shown.
    """
    root = resolve_root(split, mode, input_mode, agent_mode,
                        data_dir=data_dir, cache_dir=cache_dir,
                        allow_hf_fallback=allow_hf_fallback)
    qs = load_qa(split, root)
    if limit:
        qs = qs[:limit]
    for q in qs:
        if mode == "vqa" and input_mode:
            yield q, asset_path(root, split, q["image_id"], input_mode)
        elif mode == "agent" and agent_mode == "vision":
            yield q, asset_path(root, split, q["image_id"], "2d_image")
        else:
            yield q, None


def tool_cache_path(
    agent_mode: AgentMode,
    root: Path,
) -> Path:
    """Path to the tool cache JSON for a given agent mode (oracle/predicted only).

    Vision mode does not use a tool cache — its tools (`crop_organ`,
    `list_available_crops`, `lookup_medical_knowledge`) need only the organ
    image directory and the in-process knowledge base.
    """
    if agent_mode == "oracle":
        return root / "tool_cache" / "benchmark_oracle_tool_cache.json"
    if agent_mode == "predicted":
        return root / "tool_cache" / "benchmark_totalsegmentator_cache.json"
    raise ValueError(f"agent_mode={agent_mode!r} has no tool cache")
