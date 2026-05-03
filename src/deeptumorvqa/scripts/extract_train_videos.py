"""Extract whole-volume MP4 videos from train-split CT NIfTI files.

For each BDMAP_XXX in the train split, generate a single whole-volume MP4
(~150 frames, 8 fps) at the configured output dir. Uses the same windowing
(W=400, L=50) and frame sampling logic as the benchmark videos. Output is
re-encoded to H.264 via system ffmpeg (~50% smaller than raw OpenCV mp4v).

Sharded by `--shard-id` / `--num-shards` for SLURM array parallelism. Each
shard handles every Nth image (modular striping). Resume-safe: existing
output files are skipped.

Usage:
    python -m deeptumorvqa.scripts.extract_train_videos \\
        --train-csv /home/ychen646/TumorVQA/dataset/Tumor_VQA_dataset_V4_train.csv \\
        --ct-root /mnt/data/yixiong/AbdomenAtlas3.0Mini/extracted \\
        --output-dir /mnt/realccvl15/ychen646/TumorVQA_videos_train_whole \\
        --shard-id 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Reuse the original extraction utility (must be importable from monorepo root)
EXTRACT_PATH = Path("/home/ychen646/TumorVQA/benchmark")
sys.path.insert(0, str(EXTRACT_PATH))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--ct-root", required=True,
                   help="Dir containing BDMAP_XXX/ct.nii.gz folders.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--codec", choices=["h264", "mp4v"], default="h264",
                   help="h264 (~50%% smaller, requires system ffmpeg) "
                        "or mp4v (raw OpenCV output, no re-encode).")
    args = p.parse_args()

    # Lazy imports (so --help works without nibabel/cv2)
    import nibabel as nib  # type: ignore
    import numpy as np  # type: ignore
    from extract_2d_slices import extract_whole_volume_video, get_axial_axis  # type: ignore

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get unique train image IDs
    image_ids: list[str] = []
    seen = set()
    with open(args.train_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            iid = row.get("Image ID") or row.get("image_id")
            if iid and iid not in seen:
                seen.add(iid)
                image_ids.append(iid)
    image_ids.sort()
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(image_ids)} unique train images")

    # Slice for this shard
    shard_ids = image_ids[args.shard_id::args.num_shards]
    print(f"[shard {args.shard_id}] handling {len(shard_ids)} images "
          f"(first: {shard_ids[0] if shard_ids else 'none'}, last: {shard_ids[-1] if shard_ids else 'none'})")

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for i, iid in enumerate(shard_ids):
        out_path = out_dir / f"{iid}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue

        ct_path = Path(args.ct_root) / iid / "ct.nii.gz"
        if not ct_path.exists():
            n_fail += 1
            print(f"[shard {args.shard_id}] missing CT: {ct_path}", flush=True)
            continue

        try:
            nii = nib.load(str(ct_path))
            ct_data = nii.get_fdata()
            axial_ax = get_axial_axis(nii)

            # Step 1: write raw mp4v via OpenCV to a tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            extract_whole_volume_video(
                ct_data, tmp_path,
                axial_axis=axial_ax,
                fps=args.fps,
                max_frames=args.max_frames,
            )

            # Step 2: re-encode to H.264 (smaller; standard codec for HF/web)
            if args.codec == "h264":
                rc = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-i", tmp_path,
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-pix_fmt", "yuv420p",
                     str(out_path)],
                    capture_output=True, text=True,
                )
                Path(tmp_path).unlink(missing_ok=True)
                if rc.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {rc.stderr[:200]}")
            else:
                shutil.move(tmp_path, str(out_path))

            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"[shard {args.shard_id}] FAIL {iid}: {type(e).__name__}: {e}", flush=True)
            if out_path.exists():
                out_path.unlink()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (n_done + n_skip) / max(elapsed, 1)
            eta_min = (len(shard_ids) - i - 1) / max(rate, 0.01) / 60
            print(f"[shard {args.shard_id}] {i+1}/{len(shard_ids)}  "
                  f"done={n_done} skip={n_skip} fail={n_fail}  "
                  f"{rate:.2f} img/s  ETA {eta_min:.1f} min", flush=True)

    print(f"[shard {args.shard_id}] FINISHED  done={n_done} skip={n_skip} fail={n_fail}  "
          f"total time {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
