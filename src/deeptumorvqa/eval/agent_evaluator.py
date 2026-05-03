"""Tool-using agent evaluator (ReAct + OpenAI-style function calling).

Subclass `AgentEvaluator`, implement `chat(messages, tools)`, and the
framework runs the multi-step tool loop, executes tool calls (looking
results up in the shipped tool cache), tracks the trajectory, and scores.

The contract for `chat`:
  * Input  -- `messages`: OpenAI-style list of {role, content, ...}
              `tools`:    list of OpenAI tool schemas (from `Tool.to_openai_tool`)
  * Output -- {"content": str | None, "tool_calls": list[{name, arguments: dict}]}
              `tool_calls` is empty when the model is giving its final answer.

Three modes (`mode` argument):
  * oracle    -- segment_organ + measure + lookup_medical_knowledge,
                 with GT-mask-derived cache
  * predicted -- same tools, with TotalSegmentator-derived cache
                 (noisier numbers, real-world-like)
  * vision    -- crop_organ + lookup_medical_knowledge
                 (no GT measurements; agent must visually estimate)
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Literal

from ..data.loader import (
    AgentMode,
    Split,
    iter_questions,
    resolve_root,
    load_qa,
    tool_cache_path,
)
from ..data.prompt import SYSTEM_PROMPT_AGENT, build_prompt
from .metrics import aggregate
from .tools import build_toolkit

SAVE_EVERY = 50  # agent runs are slow; save more often


class AgentEvaluator(abc.ABC):
    """ABC for tool-using agent evaluators."""

    system_prompt: str = SYSTEM_PROMPT_AGENT

    def __init__(
        self,
        model_id: str,
        mode: AgentMode = "oracle",
        tool_cache_path_override: str | Path | None = None,
        max_steps: int = 5,
        force_answer_on_last: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        self.mode = mode
        self.max_steps = max_steps
        self.force_answer_on_last = force_answer_on_last
        self.tool_cache_path_override = tool_cache_path_override
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # User-supplied
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """One turn of conversation with the model.

        Args:
            messages: OpenAI-style conversation history.
            tools:    OpenAI tool schemas. Empty list means "give final answer."

        Returns:
            {"content": str|None, "tool_calls": list[{name, arguments: dict}]}
            When the model produces a final answer, `tool_calls` is [].
        """

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def run(
        self,
        split: Split = "benchmark",
        fmt: Literal["mc", "freeform"] = "mc",
        output_path: str | Path = "results/agent.json",
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        allow_hf_fallback: bool = False,
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve dataset root + tool cache
        root = resolve_root(split, "agent", agent_mode=self.mode,
                            data_dir=data_dir, cache_dir=cache_dir,
                            allow_hf_fallback=allow_hf_fallback)
        cache_file = (Path(self.tool_cache_path_override)
                      if self.tool_cache_path_override
                      else tool_cache_path(self.mode, root) if self.mode != "vision"
                      else None)
        image_dir = (root / split / "images_2d" / "organ"
                     if self.mode == "vision" else None)
        tools = build_toolkit(self.mode, cache_file, image_dir=image_dir)
        tool_schemas = [t.to_openai_tool() for t in tools]
        tool_by_name = {t.name: t for t in tools}

        # Resume support
        done_qids: set[str] = set()
        results: list[dict] = []
        if resume and output_path.exists():
            with open(output_path) as f:
                prior = json.load(f)
            results = prior.get("results", [])
            done_qids = {r["qid"] for r in results}
            print(f"[resume] {len(done_qids)} questions already in {output_path}")

        n_processed = 0
        # vision mode also needs the 2D image to bootstrap the conversation
        for q, asset in iter_questions(
            split=split, mode="agent", agent_mode=self.mode,
            data_dir=data_dir, cache_dir=cache_dir,
            allow_hf_fallback=allow_hf_fallback, limit=limit,
        ):
            qid = q["qid"]
            if qid in done_qids:
                continue

            user_prompt = build_prompt(q, "2d_image", fmt)  # text-only for tool-mode
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            trajectory: list[dict] = []
            final_answer = ""

            for step in range(self.max_steps):
                # Last step: drop tools to force a final answer
                offered_tools = (
                    tool_schemas if not self.force_answer_on_last or step < self.max_steps - 1
                    else []
                )
                try:
                    resp = self.chat(messages, offered_tools)
                except Exception as e:
                    final_answer = f"[ERROR] {type(e).__name__}: {e}"
                    trajectory.append({"step": step + 1, "error": final_answer})
                    break

                content = resp.get("content") or ""
                tool_calls = resp.get("tool_calls") or []

                if not tool_calls:
                    final_answer = content
                    trajectory.append({"step": step + 1, "final_answer": content})
                    break

                # Append model turn
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": tool_calls})

                # Execute each tool call
                for tc in tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    args = dict(args)
                    args["image_id"] = q["image_id"]

                    tool = tool_by_name.get(name)
                    if tool is None:
                        result = {"error": f"Unknown tool '{name}'."}
                    else:
                        try:
                            result = tool.execute(**args)
                        except Exception as e:
                            result = {"error": f"{type(e).__name__}: {e}"}

                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result),
                    })
                    trajectory.append({
                        "step": step + 1,
                        "tool": name,
                        "arguments": args,
                        "result": result,
                    })
            else:
                # Loop exited without break -> max_steps reached without final answer
                final_answer = content if 'content' in locals() else ""

            results.append({
                "qid": qid,
                "image_id": q["image_id"],
                "question_subtype": q.get("question_subtype"),
                "question_type": q.get("question_type"),
                "raw_output": final_answer,
                "pred_answer": final_answer,
                "correct_option": q.get("correct_option"),
                "answer": q.get("answer"),
                "trajectory": trajectory,
            })
            n_processed += 1
            if n_processed % SAVE_EVERY == 0:
                self._save(output_path, results, n_processed)

        # Final save + score
        self._save(output_path, results, n_processed)
        qs = load_qa(split, root)
        full_index = {q["qid"]: q for q in qs}
        metrics = aggregate(results, full_index, fmt=fmt)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "n_processed": len(results),
                       "results": results}, f, indent=2)
        return metrics

    @staticmethod
    def _save(output_path: Path, results: list[dict], n_processed: int):
        with open(output_path, "w") as f:
            json.dump({"n_processed": n_processed, "results": results}, f, indent=2)
