#!/bin/bash
#SBATCH --job-name=dtv-smoke-q3vl
#SBATCH --output=/home/ychen646/TumorVQA/release/logs/smoke_q3vl_%j.out
#SBATCH --error=/home/ychen646/TumorVQA/release/logs/smoke_q3vl_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --exclude=ccvl12,ccvl14
#SBATCH --export=NONE

# Reference smoke test: Qwen3-VL-4B on 100 benchmark samples (2D image, MC)
# Expected: ~37.8% (paper Table 1) ±5% on small N

PY=/home/ychen646/.conda/envs/medrax/bin/python
PYTHONPATH=/home/ychen646/TumorVQA/release/src

cd /home/ychen646/TumorVQA/release

PYTHONPATH=$PYTHONPATH $PY -m deeptumorvqa.evaluate \
    --model /mnt/realccvl15/ychen646/llms/Qwen3-VL-4B-Instruct \
    --backend hf \
    --mode vqa --input 2d_image --format mc \
    --data-dir /home/ychen646/dtv_smoke \
    --limit 100 \
    --max-new-tokens 32 \
    --output /home/ychen646/TumorVQA/release/logs/smoke_qwen3vl_100.json \
    --label "Qwen3-VL-4B (smoke)" \
    --no-resume
