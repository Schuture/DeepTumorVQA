---
language: en
license: cc-by-4.0
task_categories:
- visual-question-answering
- image-classification
tags:
- medical-imaging
- ct-scan
- abdominal-radiology
- vqa
- agent-evaluation
- 3d
pretty_name: DeepTumorVQA v2
size_categories:
- 100K<n<1M
configs:
- config_name: benchmark
  data_files: benchmark/test_qa.json
  default: true
- config_name: train
  data_files: train/train_qa.csv
- config_name: agent_sft
  data_files:
  - split: train
    path: agent_sft/train.jsonl
  - split: holdout
    path: agent_sft/holdout.jsonl
---

# DeepTumorVQA v2

3D abdominal-CT diagnostic Visual Question Answering benchmark with **42
clinical subtypes** and **438K total QA pairs (10K curated benchmark + 428K
training pool)**. Includes pre-extracted 2D and video modalities, 20K agent
training trajectories with tool-use traces, and a paper-locked leaderboard.

## Quick start

```python
from huggingface_hub import snapshot_download
import json

# Just the QA + 2D PNGs (~700 MB)
root = snapshot_download(
    "tumor-vqa/DeepTumorVQA_2.0",
    repo_type="dataset",
    allow_patterns=["benchmark/test_qa.json", "benchmark/images_2d/whole/**"],
)
qa = json.load(open(f"{root}/benchmark/test_qa.json"))
print(qa["num_questions"], "questions across", len(qa["question_types"]), "super-types")
# >> 10000 questions across 4 super-types
```

For a one-line CLI evaluator, install [`deeptumorvqa`](https://github.com/Schuture/DeepTumorVQA):

```bash
pip install deeptumorvqa
deeptumorvqa-eval --model Qwen/Qwen3-VL-4B-Instruct \
                  --mode vqa --input 2d_image --format mc \
                  --output results/qwen3vl.json
```

## Configurations

| Config | Splits | Description |
|---|---|---|
| `benchmark` (default) | test (10,000) | curated 991-image benchmark with all 42 subtypes |
| `train`     | train (428,050) | full training pool, 8,334 unique CTs |
| `agent_sft` | train (20,000), holdout (20,000) | ShareGPT trajectories with tool calls |

Plus loose folders:
- `tool_cache/` — pre-computed tool outputs (oracle, TotalSegmentator, training partial)
- `metadata/`   — paper leaderboard, subtype schema, structured radiology reports

## Folder layout

```
DeepTumorVQA_2.0/
├── benchmark/                                # 10K test QA + all modalities
│   ├── test_qa.json
│   ├── ct/BDMAP_*/ct.nii.gz                  # 991 NIfTI volumes (~30 GB)
│   ├── images_2d/
│   │   ├── whole/*.png                       # 991 whole-volume slices (~672 MB)
│   │   └── organ/*.png                       # 7,160 organ-focused crops (~2.9 GB) for vision agent
│   └── videos/whole/*.mp4                    # 991 MP4s (~3.9 GB)
│
├── train/                                    # 428K training QA
│   ├── train_qa.csv
│   ├── val_qa_pool.csv                       # 47K held-out (the 10K benchmark was sampled from here)
│   ├── images_2d/whole/*.png                 # 8,334 PNGs
│   ├── videos/whole/*.mp4                    # 8,334 MP4s
│   └── ct_link.md                            # CT NIfTI is upstream (AbdomenAtlas3.0Mini)
│
├── agent_sft/                                # 20K + 20K agent trajectories
│   ├── train.jsonl
│   ├── holdout.jsonl
│   └── coverage_note.md
│
├── tool_cache/                               # Pre-computed tool outputs
│   ├── benchmark_oracle_tool_cache.json      # 991 imgs, GT-mask values
│   ├── benchmark_totalsegmentator_cache.json # 991 imgs, auto-seg values
│   └── training_oracle_tool_cache.json       # 283 train imgs (subset, used by agent_sft)
│
├── metadata/
│   ├── leaderboard.csv                       # 30 paper-evaluated models (MC results)
│   ├── leaderboard_freeform.csv              # 24 models, free-form scoring
│   ├── subtype_schema.csv                    # 42 subtypes × clinical citations × required tools
│   └── reports_and_metadata.csv              # AbdomenAtlas reports for the 8334+991 BDMAP IDs we use
│
└── README.md                                 # this file
```

## Per-modality lazy download

The benchmark CT volumes alone are 70 GB. **You don't have to download them**
unless you specifically run `--input 3d_volume`. Each evaluation config only needs:

| Run config | Download size |
|---|---|
| `vqa + 2d_image`    | 0.7 GB |
| `vqa + 2d_video`    | 3.9 GB |
| `vqa + 3d_volume`   | 70 GB |
| `agent + oracle`    | 11 MB (just the cache JSON) |
| `agent + predicted` | 9 MB |
| `agent + vision`    | 2.9 GB (organ PNGs) |

Training assets are similarly modality-split: 6 GB of 2D PNGs vs ~18 GB of
H.264 MP4s. Training CT NIfTI is **not re-hosted** — link to upstream
[AbdomenAtlas3.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)
(see `train/ct_link.md`).

## QA record schema (benchmark/test_qa.json)

```json
{
  "qid": "bench_00017",
  "image_id": "BDMAP_00008446",
  "spacing": "[0.83 0.83 5.0]",
  "shape": "(512, 512, 116)",
  "sex": null,
  "age": null,
  "question": "How would you classify the splenic enlargement based on volume?",
  "answer": "Normal (no splenomegaly)",
  "mc_question": "How would you classify ... A: Normal ... B: Moderate ... C: Severe ... D: Mild ...",
  "correct_option": "A",
  "organ": "spleen",
  "lesion": false,
  "question_type": "medical reasoning",
  "question_subtype": "splenomegaly grading",
  "requires_tools": ["segment_organ", "measure", "lookup_medical_knowledge"]
}
```

## Source data

CT volumes are from **AbdomenAtlas3.0Mini** (linked via the
[upstream HF dataset](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)).
Questions are auto-generated from the AbdomenAtlas structured radiology reports
+ 43-organ-class segmentation masks, with templated programs covering the 42
subtypes. See the paper for details.

**Annotations**: All QA pairs are programmatically generated from validated CT
metadata. A 200-question stratified subset was reviewed by **two practicing
board-certified radiologists** (7 and 13 years experience, independent of the
annotation team) — junior reached 45.0% MC accuracy, senior 54.5% free-form,
demonstrating non-trivial difficulty.

## Considerations for using

- This dataset is for **research and educational use only**. It is not approved
  for clinical decision-making.
- All CT data was collected with appropriate IRB approval and was deidentified
  by the AbdomenAtlas project.
- Models trained on this data may inherit biases reflected in the underlying
  patient demographics; see the paper for a demographic breakdown.

## License

- Derived QA, agent SFT trajectories, tool caches, 2D PNG/MP4: **CC-BY-4.0**
- Upstream CT NIfTI: see
  [AbdomenAtlas3.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)

## Citation

```bibtex
@article{deeptumorvqa2026,
    title   = {{DeepTumorVQA}: A 3D CT Diagnostic VQA Benchmark Across 42
               Clinical Subtypes},
    author  = {…},
    journal = {NeurIPS Datasets and Benchmarks},
    year    = {2026},
}
```
