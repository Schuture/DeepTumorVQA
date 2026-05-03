#!/bin/bash
# Generic SLURM template for running deeptumorvqa-eval on a single node.
#
# Usage:
#   sbatch slurm_template.sh
#
# Adjust the SBATCH lines for your cluster (partition, GPU type, time).
# This example asks for 1 GPU and 8 CPUs, suitable for a 4B-7B VLM.

#SBATCH --job-name=dtv-eval
#SBATCH --output=logs/dtv_%j.out
#SBATCH --error=logs/dtv_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# ---- Inputs ----
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
MODE=${MODE:-vqa}                # vqa | agent
INPUT=${INPUT:-2d_image}         # 3d_volume | 2d_image | 2d_video  (vqa only)
AGENT_MODE=${AGENT_MODE:-oracle} # oracle | predicted | vision      (agent only)
FORMAT=${FORMAT:-mc}             # mc | freeform
BACKEND=${BACKEND:-vllm}         # vllm | hf | openai | custom
LIMIT=${LIMIT:-0}                # 0 = full benchmark
OUTPUT=${OUTPUT:-results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.json}
DATA_DIR=${DATA_DIR:-}           # set to a local mirror to skip HF
LABEL=${LABEL:-}

# ---- Environment ----
# Adjust to your conda env. The reference impl uses HF transformers + vLLM.
# Don't `conda activate` inside SLURM; use the explicit interpreter path
# (avoids common shell-init failures).
PY=/path/to/your/conda/envs/dtv/bin/python

# ---- Build the command ----
ARGS=(
  --model "$MODEL"
  --backend "$BACKEND"
  --mode "$MODE"
  --format "$FORMAT"
  --output "$OUTPUT"
)
if [ "$MODE" = "vqa" ]; then
  ARGS+=(--input "$INPUT")
elif [ "$MODE" = "agent" ]; then
  ARGS+=(--agent-mode "$AGENT_MODE")
fi
[ -n "$DATA_DIR" ] && ARGS+=(--data-dir "$DATA_DIR")
[ -n "$LABEL" ]    && ARGS+=(--label "$LABEL")
[ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")

mkdir -p "$(dirname "$OUTPUT")" logs

echo "Running: $PY -m deeptumorvqa.evaluate ${ARGS[*]}"
$PY -m deeptumorvqa.evaluate "${ARGS[@]}"
