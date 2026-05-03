# DeepTumorVQA v2 Release Inventory

Final pre-upload checklist. **All code + docs are ready**; the only remaining
data step is the train-set whole-volume MP4 extraction (running in background).
Once that finishes, run `stage_release.py` to assemble the HF stage dir, then
`push_to_hub.py` to upload (requires `huggingface-cli login` first).

## Status overview

| Track | What | Status |
|---|---|---|
| **GitHub repo content** | code + docs + tests + notebooks | ✅ ready (`/home/ychen646/TumorVQA/release/`) |
| **HF dataset content**  | benchmark + train + agent_sft + tool_cache + metadata | ⏳ waiting for train MP4s (~15 min) |
| **Smoke test**          | Qwen3-VL-4B HF backend, 100 samples, MC | ✅ 36.0% (paper 37.8% on 10K — within sampling error) |
| **Round-trip pytest**   | VQA + Agent + CLI custom backend | ✅ 3/3 pass |
| **HF upload**           | `huggingface-cli login` + push | ⏳ pending your authorization |
| **GitHub PR**           | open `v2` branch on `Schuture/DeepTumorVQA` | ⏳ pending your authorization |

---

## A. GitHub repo content (release/)

This whole directory is what goes onto the `v2` branch. Tree:

```
release/
├── README.md                                    # GitHub front page (quickstart + leaderboard)
├── HF_README.md                                 # Will be uploaded to HF as the dataset card
├── LICENSE                                      # Apache-2.0 (code)
├── .gitignore
├── pyproject.toml                               # pip install -e . + entry-point
├── RELEASE_INVENTORY.md                         # this file
│
├── src/deeptumorvqa/
│   ├── __init__.py
│   ├── evaluate.py                              # CLI entrypoint (deeptumorvqa-eval)
│   ├── _data/                                   # bundled package data (CSVs)
│   │   ├── leaderboard.csv
│   │   ├── leaderboard_freeform.csv
│   │   └── subtype_schema.csv
│   ├── data/
│   │   ├── loader.py                            # HF + --data-dir + lazy fetch
│   │   └── prompt.py                            # MC + freeform templates
│   ├── eval/
│   │   ├── metrics.py                           # MC + freeform scoring (MRA / binary / categorical)
│   │   ├── tools.py                             # 5 tools (cache-first; vision uses organ PNGs)
│   │   ├── vqa_evaluator.py                     # HFVLMEvaluator ABC
│   │   ├── agent_evaluator.py                   # AgentEvaluator ABC
│   │   └── leaderboard.py                       # auto-ranking output
│   ├── reference/
│   │   ├── qwen3vl_vqa.py                       # vLLM/HF backend
│   │   └── qwen3vl_agent.py                     # vLLM-server + OpenAI client
│   └── scripts/
│       ├── extract_train_videos.py              # SLURM-shardable MP4 extraction
│       ├── stage_release.py                     # populate HF stage dir
│       └── push_to_hub.py                       # idempotent folder-by-folder upload
│
├── examples/
│   ├── 01_direct_vqa.ipynb                      # Qwen3-VL VQA on 100 samples
│   ├── 02_agent_eval_oracle.ipynb               # Agent oracle + tool trajectory
│   ├── 03_custom_model.ipynb                    # Plug in your own VLM/agent
│   ├── 04_agent_sft_demo.ipynb                  # Load 20K trajectories
│   ├── dummy_evaluator.py                       # 30-line custom-backend example
│   └── cluster/slurm_template.sh                # generic single-node SLURM
│
├── metadata/                                    # canonical CSVs (also bundled in _data/)
│   ├── leaderboard.csv                          # 30 paper-evaluated MC rows
│   ├── leaderboard_freeform.csv                 # 24 paper-evaluated FF rows
│   └── subtype_schema.csv                       # 42 subtypes × 13 attribute cols
│
├── scripts/slurm/                               # SLURM submit scripts (cluster-specific)
│   ├── extract_train_videos.sh
│   └── smoke_test_qwen3vl.sh
│
└── tests/
    └── test_round_trip.py                       # 3 tests, all passing
```

Total source: **2,800 LOC** across 11 modules.

---

## B. HuggingFace dataset content (planned upload)

Layout on `tumor-vqa/DeepTumorVQA_2.0` after upload (~100 GB total, real-measured):

```
benchmark/                                       # self-contained 10K test set (~78 GB)
├── test_qa.json                                 # 10K QA, 42 subtypes (~2.5 MB)
├── ct/BDMAP_*/ct.nii.gz                         # 991 NIfTI volumes (~70 GB) — only fetched for --input 3d_volume
├── images_2d/
│   ├── whole/*.png                              # 991 PNGs (~672 MB) — direct VQA
│   └── organ/*.png                              # 7,160 PNGs (~2.9 GB) — vision agent
└── videos/whole/*.mp4                           # 991 MP4s (~3.9 GB) — direct VQA video

train/                                           # 428K QA training pool (~21 GB)
├── train_qa.csv                                 # 428,050 QA rows (~119 MB)
├── val_qa_pool.csv                              # 47,933 held-out (10K benchmark drawn from here)
├── images_2d/whole/*.png                        # 8,334 PNGs (~6 GB)
├── videos/whole/*.mp4                           # 8,334 H.264 MP4s (~18 GB) — IN PROGRESS, ~80% done
└── ct_link.md                                   # CT NIfTI is upstream (AbdomenAtlas3.0Mini)

agent_sft/                                       # 20K + 20K ShareGPT trajectories (~118 MB)
├── train.jsonl
├── holdout.jsonl
└── coverage_note.md

tool_cache/                                      # pre-computed tool outputs (~25 MB)
├── benchmark_oracle_tool_cache.json             # 991 imgs, GT-mask values
├── benchmark_totalsegmentator_cache.json        # 991 imgs, auto-seg values
└── training_oracle_tool_cache.json              # 283 train imgs (matches agent_sft coverage)

metadata/                                        # ~2 MB
├── leaderboard.csv
├── leaderboard_freeform.csv
├── subtype_schema.csv
└── reports_and_metadata.csv                     # AbdomenAtlas reports for the 9,257 BDMAP IDs we use

README.md                                        # HF dataset card (from HF_README.md)
MANIFEST.sha256                                  # integrity checksums
```

---

## C. Step-by-step upload procedure

### C.1  Authenticate

```bash
huggingface-cli login    # paste a write-access token for tumor-vqa org
```

### C.2  Wait for train-MP4 extraction to complete

Currently running on `ccvl14` as SLURM array `151487_*`:

```bash
squeue -u ychen646 -j 151487
# done when no shards remain
ls /mnt/realccvl15/ychen646/TumorVQA_videos_train_whole/ | wc -l
# expect 8334
```

### C.3  Stage the full HF layout

This creates `/mnt/realccvl15/ychen646/dtv2_stage/` mirroring HF, mostly via
symlinks (no large file copies):

```bash
cd /home/ychen646/TumorVQA/release
PYTHONPATH=src /home/ychen646/.conda/envs/medrax/bin/python \
    -m deeptumorvqa.scripts.stage_release \
    --stage-dir /mnt/realccvl15/ychen646/dtv2_stage
```

Then drop the `HF_README.md` into the stage dir as `README.md`:

```bash
cp HF_README.md /mnt/realccvl15/ychen646/dtv2_stage/README.md
```

### C.4  Dry-run inventory check

```bash
PYTHONPATH=src /home/ychen646/.conda/envs/medrax/bin/python \
    -m deeptumorvqa.scripts.push_to_hub \
    --stage-dir /mnt/realccvl15/ychen646/dtv2_stage \
    --dry-run
```

Should show ~25K files / ~100 GB grouped into 12 upload batches.

### C.5  Real upload (folder-by-folder)

```bash
PYTHONPATH=src /home/ychen646/.conda/envs/medrax/bin/python \
    -m deeptumorvqa.scripts.push_to_hub \
    --stage-dir /mnt/realccvl15/ychen646/dtv2_stage \
    --create-repo
```

Smaller folders (metadata, tool_cache, agent_sft, JSON / CSV singles) push
first; the 30 GB CT folder is last. Each folder is a separate commit so a
mid-upload failure doesn't lose everything. To retry just one folder:

```bash
... push_to_hub.py --stage-dir ... --only benchmark/ct
```

### C.6  GitHub v2 branch

```bash
cd /home/ychen646/TumorVQA/release
git init
git remote add origin https://github.com/Schuture/DeepTumorVQA.git
git checkout -b v2
git add .
git commit -m "DeepTumorVQA v2 release"
git push -u origin v2

# After review:
gh repo edit Schuture/DeepTumorVQA --default-branch v2
gh repo edit Schuture/DeepTumorVQA --description "3D CT diagnostic VQA benchmark — 42 subtypes, agent + VLM eval"
```

---

## D. Known gaps (v2.1 candidates)

1. **Training-set tool cache + agent SFT coverage** — currently 283/8,334
   train images. Extending to all 8,334 needs ~28× compute. Documented in
   `agent_sft/coverage_note.md`.
2. **Train CT NIfTI** — links to AbdomenAtlas3.0Mini upstream; we don't
   re-host. Documented in `train/ct_link.md`.
3. **No organ-specific videos** — only whole-volume MP4s shipped. Vision
   agent uses pre-extracted organ PNGs (sufficient per paper protocol).
4. **`trajectory.py`** — placeholder; full tool-trace metrics
   (Jaccard / param-acc / sequence match) not yet ported from research repo.

---

## E. What's verified

- ✅ `pip install -e .` + entry-point `deeptumorvqa-eval` work
- ✅ All 3 round-trip tests pass (VQA mock, Agent mock, CLI custom backend)
- ✅ Mock VQA evaluator on real benchmark + scoring + leaderboard ranking
- ✅ Mock Agent evaluator on real tool cache + 3-step trajectory
- ✅ Real Qwen3-VL-4B (HF backend) on 100 samples → 36.0% (paper 37.8% on 10K)
- ✅ `--data-dir` offline mode (no HF calls when path provided)
- ✅ Per-modality lazy download (allow_patterns verified for all 6 run configs)
- ✅ Vision-agent organ-PNG resolution (with alias + lesion-fallback)
- ✅ Auto-ranking output puts user model in correct position vs paper
- ✅ Notebook JSON validates (8-10 cells each)
- ✅ Stage script symlinks 9,262/9,257 BDMAP-id rows from full reports CSV
- ✅ Push-to-hub dry-run lists all 12 upload batches correctly
