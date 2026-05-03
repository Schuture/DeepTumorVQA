"""Stage all release assets into a single HF-layout directory.

Reads from the research monorepo's scattered locations and produces a clean
mirror of the target HF dataset structure under `--stage-dir`. Everything
is symlinked (not copied) where possible to save disk; only the renamed/
filtered files (test_qa.json, train_qa.csv, reports_and_metadata.csv) are
real copies.

Run this BEFORE `push_to_hub.py`.

Usage:
    python -m deeptumorvqa.scripts.stage_release \\
        --stage-dir /mnt/realccvl15/ychen646/dtv2_stage
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

# Sources in the research monorepo (paths from CLAUDE.md / verified inventory)
DEFAULTS = {
    "benchmark_qa":       "/home/ychen646/TumorVQA/benchmark/benchmark_v4.json",
    "train_qa":           "/home/ychen646/TumorVQA/dataset/Tumor_VQA_dataset_V4_train.csv",
    "val_qa_pool":        "/home/ychen646/TumorVQA/dataset/Tumor_VQA_dataset_V4_test.csv",
    "reports_full":       "/home/ychen646/TumorVQA/dataset/AbdomenAtlas3.0_April_24_2025_reports_and_metadata.csv",
    "subtype_schema":     "/home/ychen646/TumorVQA/data/subtype_program_schema.csv",
    "leaderboard_csv":    "/home/ychen646/TumorVQA/release/metadata/leaderboard.csv",
    "agent_sft_train":    "/home/ychen646/TumorVQA/data/agent_sft_train.jsonl",
    "agent_sft_holdout":  "/home/ychen646/TumorVQA/data/agent_sft_train_holdout.jsonl",
    "tool_cache_oracle":  "/mnt/realccvl15/ychen646/TumorVQA-results/tool_cache_merged.json",
    "tool_cache_pred":    "/mnt/realccvl15/ychen646/TumorVQA-results/tool_cache_predicted_merged.json",
    "tool_cache_train":   "/mnt/realccvl15/ychen646/TumorVQA-results/tool_cache_train_merged.json",
    "bench_whole_png":    "/home/ychen646/TumorVQA/benchmark/slices_whole",
    "bench_organ_png":    "/home/ychen646/TumorVQA/benchmark/slices",
    "bench_whole_mp4":    "/home/ychen646/TumorVQA/benchmark/videos_whole",
    "train_whole_png":    "/mnt/realccvl15/ychen646/TumorVQA_slices_train_whole",
    "train_whole_mp4":    "/mnt/realccvl15/ychen646/TumorVQA_videos_train_whole",
    "ct_root":            "/mnt/data/yixiong/AbdomenAtlas1.1",
}


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def _link_dir_contents(src: Path, dst: Path, glob: str = "*"):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob(glob):
        if f.is_file():
            _link(f, dst / f.name)
            n += 1
    return n


def _copy_renamed(src: Path, dst: Path):
    """Use a real copy (not symlink) when the destination has a different name
    than the source — HF folder uploaders sometimes complain about symlinks."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def stage_benchmark(stage: Path, paths: dict):
    """Stage benchmark/ folder."""
    print("\n[benchmark]")
    bench = stage / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)

    # test_qa.json — copy + rename (drop _v4 in name; content unchanged)
    src = Path(paths["benchmark_qa"])
    dst = bench / "test_qa.json"
    _copy_renamed(src, dst)
    print(f"  test_qa.json: {dst.stat().st_size/1e6:.1f} MB")

    # PNGs (whole + organ)
    n = _link_dir_contents(Path(paths["bench_whole_png"]), bench / "images_2d" / "whole", "*.png")
    print(f"  images_2d/whole/: {n} PNGs")
    n = _link_dir_contents(Path(paths["bench_organ_png"]), bench / "images_2d" / "organ", "*.png")
    print(f"  images_2d/organ/: {n} PNGs")

    # MP4s (whole only)
    n = _link_dir_contents(Path(paths["bench_whole_mp4"]), bench / "videos" / "whole", "*.mp4")
    print(f"  videos/whole/: {n} MP4s")

    # CTs (991 NIfTI + skip segmentations per release decision)
    qa = json.load(open(paths["benchmark_qa"]))
    bench_ids = sorted({q["image_id"] for q in qa["questions"]})
    ct_root = Path(paths["ct_root"])
    n = 0
    for iid in bench_ids:
        src = ct_root / iid / "ct.nii.gz"
        if not src.exists():
            print(f"  [WARN] missing CT: {src}", file=sys.stderr)
            continue
        _link(src, bench / "ct" / iid / "ct.nii.gz")
        n += 1
    print(f"  ct/: {n} CT volumes (out of {len(bench_ids)} expected)")


def stage_train(stage: Path, paths: dict):
    """Stage train/ folder."""
    print("\n[train]")
    tr = stage / "train"
    tr.mkdir(parents=True, exist_ok=True)

    _copy_renamed(Path(paths["train_qa"]), tr / "train_qa.csv")
    print(f"  train_qa.csv: {(tr/'train_qa.csv').stat().st_size/1e6:.1f} MB")
    _copy_renamed(Path(paths["val_qa_pool"]), tr / "val_qa_pool.csv")
    print(f"  val_qa_pool.csv: {(tr/'val_qa_pool.csv').stat().st_size/1e6:.1f} MB")

    # PNG (whole)
    n = _link_dir_contents(Path(paths["train_whole_png"]), tr / "images_2d" / "whole", "*.png")
    print(f"  images_2d/whole/: {n} PNGs")

    # MP4 (whole) — may not be fully extracted yet
    src_dir = Path(paths["train_whole_mp4"])
    if src_dir.exists():
        n = _link_dir_contents(src_dir, tr / "videos" / "whole", "*.mp4")
        print(f"  videos/whole/: {n} MP4s")
    else:
        print(f"  videos/whole/: source dir missing, skipped")

    # ct_link.md — instruct users to fetch CT from AbdomenAtlas
    train_csv = Path(paths["train_qa"])
    train_ids = sorted({row["Image ID"] for row in csv.DictReader(open(train_csv))})
    md = tr / "ct_link.md"
    md.write_text(_build_ct_link_md(train_ids))
    print(f"  ct_link.md: {len(train_ids)} BDMAP IDs listed")


def _build_ct_link_md(train_ids: list[str]) -> str:
    lines = [
        "# Training-split 3D CT volumes",
        "",
        "We do **not** re-host the raw CT NIfTI files for the training split. They",
        "are identical to the cases in",
        "[`AbdomenAtlas/AbdomenAtlas3.0Mini`](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini).",
        "",
        "To download:",
        "",
        "```bash",
        "from huggingface_hub import snapshot_download",
        "snapshot_download(",
        "    repo_id='AbdomenAtlas/AbdomenAtlas3.0Mini',",
        "    repo_type='dataset',",
        "    allow_patterns=[f'image_only/{iid}/ct.nii.gz' for iid in TRAIN_IDS],",
        ")",
        "```",
        "",
        f"## Training BDMAP IDs ({len(train_ids)} total)",
        "",
        "<details><summary>click to expand</summary>",
        "",
        "```",
        *train_ids,
        "```",
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


def stage_agent_sft(stage: Path, paths: dict):
    print("\n[agent_sft]")
    d = stage / "agent_sft"
    d.mkdir(parents=True, exist_ok=True)
    _link(Path(paths["agent_sft_train"]), d / "train.jsonl")
    _link(Path(paths["agent_sft_holdout"]), d / "holdout.jsonl")
    note = d / "coverage_note.md"
    note.write_text(
        "# Agent SFT data — coverage note\n\n"
        "This release ships **20K training trajectories + 20K holdout trajectories** "
        "synthesized from oracle tool traces. Both sets cover the same **283 unique "
        "training images** (a subset of the full 8,334-image train pool).\n\n"
        "Why 283? The oracle tool cache for the training split is currently complete "
        "for only those 283 images (see `tool_cache/training_oracle_tool_cache.json`). "
        "Extending coverage to the full 8,334-image train pool is planned for v2.1.\n\n"
        "The SFT data still covers all 38 task subtypes for which oracle tool traces "
        "are well-defined (excluded: inter-segment comparison, largest lesion slice/"
        "location, adjacent organ — these need spatial information not provided by "
        "the available tools).\n"
    )
    sz_t = (d / "train.jsonl").stat().st_size / 1e6
    sz_h = (d / "holdout.jsonl").stat().st_size / 1e6
    print(f"  train.jsonl: {sz_t:.1f} MB; holdout.jsonl: {sz_h:.1f} MB")


def stage_tool_cache(stage: Path, paths: dict):
    print("\n[tool_cache]")
    d = stage / "tool_cache"
    d.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("tool_cache_oracle", "benchmark_oracle_tool_cache.json"),
        ("tool_cache_pred",   "benchmark_totalsegmentator_cache.json"),
        ("tool_cache_train",  "training_oracle_tool_cache.json"),
    ]
    for key, name in pairs:
        src = Path(paths[key])
        if src.exists():
            _link(src, d / name)
            print(f"  {name}: {src.stat().st_size/1e6:.2f} MB")
        else:
            print(f"  [WARN] missing {key}: {src}", file=sys.stderr)


def stage_metadata(stage: Path, paths: dict):
    print("\n[metadata]")
    d = stage / "metadata"
    d.mkdir(parents=True, exist_ok=True)

    # leaderboard.csv (already in release/)
    _link(Path(paths["leaderboard_csv"]), d / "leaderboard.csv")
    print(f"  leaderboard.csv linked")

    # subtype_schema.csv (link existing)
    src = Path(paths["subtype_schema"])
    if src.exists():
        _link(src, d / "subtype_schema.csv")
        print(f"  subtype_schema.csv linked")
    else:
        print(f"  [WARN] subtype_schema not found at {src}; skipping")

    # reports_and_metadata.csv: filter the full reports CSV to BDMAP IDs we use
    full = Path(paths["reports_full"])
    if not full.exists():
        print(f"  [WARN] reports_full missing: {full}; skipping")
        return
    bench_ids = sorted({q["image_id"] for q in json.load(open(paths["benchmark_qa"]))["questions"]})
    train_ids = sorted({row["Image ID"] for row in csv.DictReader(open(paths["train_qa"]))})
    used = set(bench_ids) | set(train_ids)
    out = d / "reports_and_metadata.csv"
    n_in = n_out = 0
    with open(full) as fin, open(out, "w") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r)
        w.writerow(header)
        # Find BDMAP-id col
        try:
            id_col = header.index("BDMAP_ID")
        except ValueError:
            id_col = 0
        for row in r:
            n_in += 1
            if row[id_col] in used:
                w.writerow(row)
                n_out += 1
    print(f"  reports_and_metadata.csv: {n_out:,} / {n_in:,} rows kept ({len(used):,} BDMAP IDs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-dir", required=True)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["benchmark", "train", "agent_sft", "tool_cache", "metadata"],
                   help="Skip these stage groups (useful for incremental re-runs).")
    args = p.parse_args()

    stage = Path(args.stage_dir).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    print(f"=== Staging into {stage} ===")

    paths = DEFAULTS

    if "benchmark" not in args.skip:
        stage_benchmark(stage, paths)
    if "train" not in args.skip:
        stage_train(stage, paths)
    if "agent_sft" not in args.skip:
        stage_agent_sft(stage, paths)
    if "tool_cache" not in args.skip:
        stage_tool_cache(stage, paths)
    if "metadata" not in args.skip:
        stage_metadata(stage, paths)

    print(f"\n=== Done. Run dry-run summary with: ===")
    print(f"  python -m deeptumorvqa.scripts.push_to_hub --stage-dir {stage} --dry-run")


if __name__ == "__main__":
    main()
