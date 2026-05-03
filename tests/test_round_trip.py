"""End-to-end round-trip tests using mock evaluators (no GPU required).

Verifies the complete data + scoring + leaderboard pipeline without needing
to download any model or call HF. Run with:

    pytest tests/

Three tests:
  1. VQA + 2D image + MC: a "always answer A" mock evaluator on 20 samples
  2. Agent + oracle + MC: a 3-step deterministic mock agent on 5 samples
  3. Custom-backend CLI: dummy_evaluator.py via the CLI entrypoint

The tests build a tiny --data-dir from the live monorepo paths, which
will only exist on the original development cluster. To run elsewhere,
set $DTV_TEST_DATA_DIR to a directory mirroring the HF layout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# These paths only exist on the development cluster. Skip the suite gracefully
# elsewhere unless DTV_TEST_DATA_DIR is provided.
DEV_BENCH_JSON = Path("/home/ychen646/TumorVQA/benchmark/benchmark_v4.json")
DEV_PNG_DIR = Path("/home/ychen646/TumorVQA/benchmark/slices_whole")
DEV_TOOL_CACHE = Path("/mnt/realccvl15/ychen646/TumorVQA-results/tool_cache_merged.json")


def _pick_dev_data():
    if not (DEV_BENCH_JSON.exists() and DEV_PNG_DIR.exists() and DEV_TOOL_CACHE.exists()):
        pytest.skip("development data paths not available; set DTV_TEST_DATA_DIR")


@pytest.fixture
def vqa_data_dir(tmp_path):
    """Build a 20-question --data-dir mirroring the HF layout."""
    _pick_dev_data()
    root = tmp_path / "dtv"
    (root / "benchmark" / "images_2d" / "whole").mkdir(parents=True)

    bench = json.load(open(DEV_BENCH_JSON))
    small = {k: v for k, v in bench.items() if k != "questions"}
    small["questions"] = bench["questions"][:20]
    small["num_questions"] = 20
    with open(root / "benchmark" / "test_qa.json", "w") as f:
        json.dump(small, f)

    for q in small["questions"]:
        src = DEV_PNG_DIR / f"{q['image_id']}.png"
        if src.exists():
            (root / "benchmark" / "images_2d" / "whole" / src.name).symlink_to(src)
    return root


@pytest.fixture
def agent_data_dir(tmp_path):
    """Build a 5-question --data-dir + symlink to the real tool cache."""
    _pick_dev_data()
    root = tmp_path / "dtv"
    (root / "tool_cache").mkdir(parents=True)
    (root / "benchmark").mkdir(parents=True)

    bench = json.load(open(DEV_BENCH_JSON))
    small = {k: v for k, v in bench.items() if k != "questions"}
    small["questions"] = bench["questions"][:5]
    small["num_questions"] = 5
    with open(root / "benchmark" / "test_qa.json", "w") as f:
        json.dump(small, f)

    (root / "tool_cache" / "benchmark_oracle_tool_cache.json").symlink_to(DEV_TOOL_CACHE)
    return root


def test_vqa_pipeline(vqa_data_dir):
    """HFVLMEvaluator subclass + scoring + leaderboard ranking."""
    from deeptumorvqa.eval.vqa_evaluator import HFVLMEvaluator
    from deeptumorvqa.eval import leaderboard

    class AlwaysA(HFVLMEvaluator):
        def __init__(self):
            super().__init__(model_id="always-A")
        def generate(self, prompt, images=None, video_path=None):
            return "A"

    out = vqa_data_dir / "out.json"
    metrics = AlwaysA().run(
        split="benchmark", input_mode="2d_image", fmt="mc",
        output_path=str(out), data_dir=str(vqa_data_dir), limit=20, resume=False,
    )

    assert metrics["overall"]["n_total"] == 20
    assert 0.0 <= metrics["overall"]["accuracy"] <= 1.0
    assert "by_super_type" in metrics
    # Leaderboard must accept the metrics
    report = leaderboard.report(
        metrics, user_label="AlwaysA", user_mode="direct", top_k=5,
    )
    assert "AlwaysA" in report
    assert "ranks #" in report


def test_agent_pipeline(agent_data_dir):
    """AgentEvaluator subclass + ReAct loop + tool execution."""
    from deeptumorvqa.eval.agent_evaluator import AgentEvaluator

    class MockAgent(AgentEvaluator):
        def __init__(self):
            super().__init__(model_id="mock", mode="oracle", max_steps=3)
            self.step = 0
        def chat(self, messages, tools):
            self.step += 1
            if self.step == 1 and tools:
                return {"content": "Check liver",
                        "tool_calls": [{"name": "segment_organ",
                                        "arguments": {"target": "liver"}}]}
            if self.step == 2 and tools:
                return {"content": "Now measure",
                        "tool_calls": [{"name": "measure",
                                        "arguments": {"target": "liver",
                                                      "measurement_type": "volume_cm3"}}]}
            return {"content": "B", "tool_calls": []}

    out = agent_data_dir / "out.json"
    metrics = MockAgent().run(
        split="benchmark", fmt="mc",
        output_path=str(out), data_dir=str(agent_data_dir), limit=5, resume=False,
    )

    assert metrics["overall"]["n_total"] == 5
    saved = json.load(open(out))
    # First sample's trajectory should have segment + measure + final_answer
    traj = saved["results"][0]["trajectory"]
    assert any(s.get("tool") == "segment_organ" for s in traj)
    assert any(s.get("tool") == "measure" for s in traj)
    assert any("final_answer" in s for s in traj)


def test_cli_custom_backend(vqa_data_dir, tmp_path):
    """The full CLI accepts --backend custom + --custom-module."""
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    out = tmp_path / "cli_out.json"
    env = {**os.environ, "PYTHONPATH": str(examples_dir)}
    rc = subprocess.run(
        [sys.executable, "-m", "deeptumorvqa.evaluate",
         "--backend", "custom", "--custom-module", "dummy_evaluator",
         "--mode", "vqa", "--input", "2d_image", "--format", "mc",
         "--data-dir", str(vqa_data_dir),
         "--limit", "20",
         "--output", str(out),
         "--label", "DummyAlwaysA",
         "--no-resume"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert rc.returncode == 0, f"CLI failed: {rc.stderr[-500:]}"
    assert "Your model ranks #" in rc.stdout
    assert out.exists()
    data = json.load(open(out))
    assert data["metrics"]["overall"]["n_total"] == 20
