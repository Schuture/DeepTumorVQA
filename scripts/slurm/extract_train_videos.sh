#!/bin/bash
#SBATCH --job-name=dtv-train-vid
#SBATCH --output=/home/ychen646/TumorVQA/release/logs/train_vid_%A_%a.out
#SBATCH --error=/home/ychen646/TumorVQA/release/logs/train_vid_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --partition=main
#SBATCH --nodelist=ccvl14
#SBATCH --export=NONE

# Use medrax env explicitly (no bashrc)
PY=/home/ychen646/.conda/envs/medrax/bin/python
PYTHONPATH=/home/ychen646/TumorVQA/release/src

cd /home/ychen646/TumorVQA/release

PYTHONPATH=$PYTHONPATH $PY -m deeptumorvqa.scripts.extract_train_videos \
    --train-csv /home/ychen646/TumorVQA/dataset/Tumor_VQA_dataset_V4_train.csv \
    --ct-root /mnt/data/yixiong/AbdomenAtlas1.1 \
    --output-dir /mnt/realccvl15/ychen646/TumorVQA_videos_train_whole \
    --shard-id ${SLURM_ARRAY_TASK_ID} \
    --num-shards 16 \
    --codec h264
