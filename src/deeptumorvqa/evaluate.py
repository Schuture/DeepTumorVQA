"""DeepTumorVQA one-click CLI evaluator.

Examples:

  # Direct VQA, 2D image, MC, full benchmark, vLLM backend
  python -m deeptumorvqa.evaluate \\
      --model Qwen/Qwen3-VL-4B-Instruct \\
      --mode vqa --input 2d_image --format mc \\
      --backend vllm --output results/qwen3vl_vqa.json

  # Agent eval in oracle mode, 100-sample smoke test, local data dir
  python -m deeptumorvqa.evaluate \\
      --model Qwen/Qwen3-VL-4B-Instruct \\
      --mode agent --agent-mode oracle --format mc \\
      --backend openai --api-base http://localhost:8877/v1 \\
      --data-dir /mnt/datasets/DeepTumorVQA_2.0 \\
      --limit 100 --output results/qwen3vl_agent.json

  # Custom user model: implement HFVLMEvaluator/AgentEvaluator subclass in
  # `mymodels.eval_runner.build_evaluator()`
  python -m deeptumorvqa.evaluate \\
      --backend custom --custom-module mymodels.eval_runner \\
      --mode vqa --input 2d_image --format mc \\
      --output results/mymodel.json

After every full run, the script prints the user model's rank in the
paper-locked leaderboard (`metadata/leaderboard.csv`).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .eval import leaderboard


def _build_evaluator(args):
    """Construct the evaluator object based on --backend / --custom-module."""
    if args.backend == "custom":
        if not args.custom_module:
            raise ValueError("--custom-module is required when --backend custom.")
        mod = importlib.import_module(args.custom_module)
        if not hasattr(mod, "build_evaluator"):
            raise ValueError(
                f"--custom-module {args.custom_module!r} must export "
                "`build_evaluator(args) -> HFVLMEvaluator | AgentEvaluator`."
            )
        return mod.build_evaluator(args)

    if args.mode == "vqa":
        from .reference.qwen3vl_vqa import Qwen3VLEvaluator
        return Qwen3VLEvaluator(
            model_id=args.model,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )

    if args.mode == "agent":
        if args.backend != "openai":
            print(
                "[warn] Agent mode reference impl talks to a vLLM OpenAI-compatible "
                "server; --backend should be 'openai'. Using 'openai' anyway.",
                file=sys.stderr,
            )
        from .reference.qwen3vl_agent import Qwen3VLAgentEvaluator
        return Qwen3VLAgentEvaluator(
            model_id=args.model,
            api_base=args.api_base,
            mode=args.agent_mode,
            max_steps=args.max_steps,
            max_tokens=args.max_new_tokens,
        )

    raise ValueError(args.mode)


def _validate(args):
    if args.mode == "vqa" and not args.input:
        raise SystemExit("--input is required when --mode vqa "
                         "(choose 3d_volume / 2d_image / 2d_video).")
    if args.mode == "agent" and not args.agent_mode:
        raise SystemExit("--agent-mode is required when --mode agent "
                         "(choose oracle / predicted / vision).")


def main():
    p = argparse.ArgumentParser(
        prog="deeptumorvqa.evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Model / backend
    p.add_argument("--model", required=False, default="Qwen/Qwen3-VL-4B-Instruct",
                   help="HuggingFace model id or local path. Default: Qwen3-VL-4B.")
    p.add_argument("--backend", choices=["vllm", "hf", "openai", "custom"],
                   default="vllm", help="Inference backend.")
    p.add_argument("--custom-module",
                   help="Python module exporting build_evaluator(args). "
                        "Required when --backend custom.")
    p.add_argument("--api-base", default="http://localhost:8877/v1",
                   help="Base URL of the OpenAI-compatible server (agent mode).")

    # Run config
    p.add_argument("--mode", choices=["vqa", "agent"], required=True)
    p.add_argument("--input", choices=["3d_volume", "2d_image", "2d_video"],
                   help="Required when --mode vqa.")
    p.add_argument("--agent-mode", choices=["oracle", "predicted", "vision"],
                   help="Required when --mode agent.")
    p.add_argument("--format", choices=["mc", "freeform"], default="mc")
    p.add_argument("--split", choices=["benchmark"], default="benchmark",
                   help="Only 'benchmark' is evaluable. 'train' is for users to "
                        "train on (use load_dataset directly).")

    # Data location
    p.add_argument("--data-dir",
                   help="Local pre-downloaded dataset root. Skips HF if given.")
    p.add_argument("--cache-dir",
                   help="HF cache directory (default: ~/.cache/huggingface).")
    p.add_argument("--allow-hf-fallback", action="store_true",
                   help="If --data-dir is missing some artifacts, fetch from HF.")

    # Generation knobs
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=5,
                   help="Agent max steps (default 5).")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Pass enable_thinking=True (Qwen3-VL thinking mode).")

    # Eval scope
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions (for smoke tests).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing output file and re-evaluate from scratch.")
    p.add_argument("--output", required=True,
                   help="Path to write the JSON results + metrics.")

    # Leaderboard
    p.add_argument("--label", default=None,
                   help="Display name for this model in the leaderboard. "
                        "Defaults to --model.")
    p.add_argument("--no-leaderboard", action="store_true",
                   help="Skip the auto-ranking output at the end.")

    args = p.parse_args()
    _validate(args)

    print(f"[deeptumorvqa] Building {args.backend!r} evaluator for "
          f"mode={args.mode}, input={args.input or args.agent_mode}, "
          f"format={args.format}, limit={args.limit or 'full'}")
    ev = _build_evaluator(args)

    print(f"[deeptumorvqa] Running ...")
    if args.mode == "vqa":
        metrics = ev.run(
            split=args.split, input_mode=args.input, fmt=args.format,
            output_path=args.output, data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            allow_hf_fallback=args.allow_hf_fallback,
            limit=args.limit, resume=not args.no_resume,
        )
    else:
        metrics = ev.run(
            split=args.split, fmt=args.format,
            output_path=args.output, data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            allow_hf_fallback=args.allow_hf_fallback,
            limit=args.limit, resume=not args.no_resume,
        )

    print(f"\n[deeptumorvqa] Overall accuracy: "
          f"{metrics['overall']['accuracy']*100:.2f}% on N={metrics['overall']['n_total']}")
    for st, m in metrics.get("by_super_type", {}).items():
        print(f"  {st:25s}  {m['accuracy']*100:>6.2f}%  (N={m['n_total']})")

    if not args.no_leaderboard:
        print()
        user_mode = args.input if args.mode == "vqa" else args.agent_mode
        # For leaderboard, agent rows live under "oracle"/"predicted"/"vision",
        # direct VQA under "direct"
        lb_mode = args.agent_mode if args.mode == "agent" else "direct"
        label = args.label or Path(args.model).name
        print(leaderboard.report(
            metrics, user_label=label,
            user_mode=lb_mode,
            user_input=str(args.input or "2D"),
            top_k=10,
        ))


if __name__ == "__main__":
    main()
