"""Minimal example custom evaluator: always answers 'A'.

Use as a template for your own model:

    python -m deeptumorvqa.evaluate \\
        --backend custom --custom-module examples.dummy_evaluator \\
        --mode vqa --input 2d_image --format mc \\
        --data-dir /tmp/dtv_test_data --limit 20 \\
        --output /tmp/dummy.json
"""

from deeptumorvqa.eval.vqa_evaluator import HFVLMEvaluator


class DummyEvaluator(HFVLMEvaluator):
    def __init__(self):
        super().__init__(model_id="dummy-always-A", device="cpu")

    def generate(self, prompt: str, images=None, video_path=None) -> str:
        return "A"


def build_evaluator(args):
    """The CLI calls this to construct the evaluator."""
    return DummyEvaluator()
