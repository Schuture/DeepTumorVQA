"""Upload the staged DeepTumorVQA_2.0 release directory to HuggingFace.

The script expects a local directory whose layout exactly mirrors what users
will see on HF:

    <stage_dir>/
    ├── README.md                    # dataset card
    ├── benchmark/
    │   ├── test_qa.json
    │   ├── ct/BDMAP_*/ct.nii.gz
    │   ├── images_2d/{whole,organ}/*.png
    │   └── videos/whole/*.mp4
    ├── train/
    │   ├── train_qa.csv
    │   ├── val_qa_pool.csv
    │   ├── images_2d/whole/*.png
    │   ├── videos/whole/*.mp4
    │   └── ct_link.md
    ├── agent_sft/{train,holdout}.jsonl
    ├── tool_cache/{benchmark_oracle,benchmark_totalsegmentator,training_oracle}_tool_cache.json
    └── metadata/{leaderboard,leaderboard_freeform,subtype_schema,reports_and_metadata}.csv

Upload happens folder-by-folder so a network blip only loses one folder.
After everything is uploaded, a SHA256 manifest is generated and pushed.

Usage:
    python -m deeptumorvqa.scripts.push_to_hub \\
        --stage-dir /mnt/realccvl15/ychen646/dtv2_stage \\
        --repo-id  tumor-vqa/DeepTumorVQA_2.0 \\
        --dry-run                       # list everything but don't actually upload

To actually upload, drop --dry-run. You must be `huggingface-cli login`'d as
a user with write access to `tumor-vqa`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Folders to upload, in priority order. Smaller folders first so failures
# during the long video uploads don't lose the small / metadata content.
UPLOAD_ORDER = [
    "metadata",
    "tool_cache",
    "agent_sft",
    "benchmark/test_qa.json",        # single file
    "benchmark/images_2d",
    "benchmark/videos",
    "train/train_qa.csv",            # single file
    "train/val_qa_pool.csv",         # single file
    "train/ct_link.md",              # single file
    "train/images_2d",
    "train/videos",
    "benchmark/ct",                  # 30 GB, last
]


def _walk_files(stage: Path, rel_target: str) -> list[Path]:
    """Return all real files under stage/rel_target (or [path] if it's a file)."""
    p = stage / rel_target
    if p.is_file():
        return [p]
    if p.is_dir():
        return [f for f in p.rglob("*") if f.is_file()]
    return []


def _summarize(stage: Path) -> dict[str, tuple[int, float]]:
    """Return {rel_target: (n_files, size_gb)} for each entry in UPLOAD_ORDER."""
    out: dict[str, tuple[int, float]] = {}
    for rel in UPLOAD_ORDER:
        files = _walk_files(stage, rel)
        size_gb = sum(f.stat().st_size for f in files) / 1e9
        out[rel] = (len(files), size_gb)
    return out


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _build_manifest(stage: Path, output_path: Path):
    """Walk every file in stage/ and write `<sha256>  <relpath>` lines.

    Skips the manifest itself + README.md (so re-running doesn't change them).
    """
    skip = {output_path.name, "README.md"}
    lines: list[str] = []
    for f in sorted(stage.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(stage)
        if rel.name in skip:
            continue
        lines.append(f"{_sha256(f)}  {rel.as_posix()}")
    output_path.write_text("\n".join(lines) + "\n")
    print(f"[manifest] {len(lines)} files hashed -> {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-dir", required=True,
                   help="Local directory mirroring the HF layout.")
    p.add_argument("--repo-id", default="tumor-vqa/DeepTumorVQA_2.0",
                   help="HF dataset repo (must already exist or use --create-repo).")
    p.add_argument("--create-repo", action="store_true",
                   help="Create the repo if it doesn't exist.")
    p.add_argument("--only", nargs="*",
                   help="Only upload specific top-level entries from UPLOAD_ORDER. "
                        "Useful for incremental retries.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be uploaded but don't actually push.")
    p.add_argument("--skip-manifest", action="store_true",
                   help="Don't (re)compute MANIFEST.sha256.")
    p.add_argument("--skip-readme", action="store_true",
                   help="Don't (re)upload README.md.")
    args = p.parse_args()

    stage = Path(args.stage_dir).resolve()
    if not stage.is_dir():
        sys.exit(f"--stage-dir does not exist or is not a directory: {stage}")

    print(f"\n=== DeepTumorVQA_2.0 -> {args.repo_id} ===")
    print(f"Stage dir: {stage}")
    summary = _summarize(stage)
    total_files = sum(n for n, _ in summary.values())
    total_gb = sum(s for _, s in summary.values())
    print(f"\nUpload plan ({total_files:,} files, {total_gb:.1f} GB total):")
    for rel, (n, s) in summary.items():
        marker = "" if args.only is None or rel in args.only else "  [skipped via --only]"
        print(f"  {rel:<35s}  {n:>6,} files  {s:>7.2f} GB{marker}")

    if args.dry_run:
        print("\n[dry-run] no actual uploads will happen.")
        return

    # Heavy imports only when actually uploading
    from huggingface_hub import HfApi, create_repo
    api = HfApi()

    if args.create_repo:
        print(f"\n[setup] Creating repo {args.repo_id} (if not exists) ...")
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    # README first if it exists (so the repo has a card while big uploads happen)
    readme = stage / "README.md"
    if readme.exists() and not args.skip_readme and (args.only is None or "README.md" in args.only):
        print(f"\n[upload] README.md")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Upload dataset card",
        )

    # Each top-level entry as a separate commit
    for rel in UPLOAD_ORDER:
        if args.only is not None and rel not in args.only:
            continue
        files = _walk_files(stage, rel)
        if not files:
            print(f"\n[skip] {rel} — no files in stage")
            continue
        n, gb = summary[rel]
        target = stage / rel
        if target.is_file():
            print(f"\n[upload] {rel}  ({gb:.3f} GB, 1 file)")
            api.upload_file(
                path_or_fileobj=str(target),
                path_in_repo=rel,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload {rel}",
            )
        else:
            print(f"\n[upload] {rel}/  ({gb:.2f} GB, {n:,} files)")
            api.upload_folder(
                folder_path=str(target),
                path_in_repo=rel,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload {rel}/  ({n} files, {gb:.2f} GB)",
            )

    # Manifest at the end
    if not args.skip_manifest:
        manifest_path = stage / "MANIFEST.sha256"
        print(f"\n[manifest] Hashing all files ...")
        _build_manifest(stage, manifest_path)
        print(f"[upload] MANIFEST.sha256")
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo="MANIFEST.sha256",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Upload SHA256 integrity manifest",
        )

    print("\n=== upload complete ===")
    print(f"Browse: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
