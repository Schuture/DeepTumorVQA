"""Compare a just-finished evaluation against the paper-locked leaderboard.

`metadata/leaderboard.csv` ships in this repo (and on HF) and contains the
~38 model rows from Table 1 + Table 2 of the DeepTumorVQA paper. After every
full eval, `format_ranking()` slots the user's model into this list (sorted
by Overall) and returns a printable ASCII table.

The leaderboard is a paper-locked snapshot — PRs that update it should
include the eval log so we can cross-check the scoring pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

def _find_leaderboard(name: str = "leaderboard.csv") -> Path:
    """Locate the leaderboard CSV under either the editable-install layout
    (release/metadata/<name>) or the wheel-install layout
    (site-packages/deeptumorvqa/_data/<name>).
    """
    # 1. Editable install: release/metadata/<name>
    editable = Path(__file__).resolve().parents[3] / "metadata" / name
    if editable.exists():
        return editable
    # 2. Wheel install: bundled package data at deeptumorvqa/_data/<name>
    bundled = Path(__file__).resolve().parents[1] / "_data" / name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"Could not locate {name}; tried {editable} and {bundled}."
    )


DEFAULT_LEADERBOARD = _find_leaderboard("leaderboard.csv")


def load_leaderboard(
    csv_path: str | Path = DEFAULT_LEADERBOARD,
    mode_filter: Literal["all", "direct", "agent"] | None = None,
) -> list[dict]:
    """Load the paper leaderboard. Optional filter by `direct` vs `agent` rows."""
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["overall"] = float(r["overall"])
            r["recog"] = float(r["recog"])
            r["meas"] = float(r["meas"])
            r["visr"] = float(r["visr"])
            r["medr"] = float(r["medr"])
            r["origin"] = "paper"
            rows.append(r)
    if mode_filter == "direct":
        rows = [r for r in rows if r["mode"] == "direct"]
    elif mode_filter == "agent":
        rows = [r for r in rows if r["mode"] in ("oracle", "predicted", "vision")]
    return rows


def insert_user_model(
    leaderboard: list[dict],
    user_metrics: dict,
    user_label: str,
    user_mode: str = "direct",
    user_input: str = "2D",
    user_params: str = "?",
) -> list[dict]:
    """Add a user-submitted result to the leaderboard list (in-place safe).

    `user_metrics` is the dict returned by `eval.metrics.aggregate(...)`:
        {
            "overall": {"accuracy": 0.42, ...},
            "by_super_type": {
                "recognition": {"accuracy": ..., ...},
                "measurement":  {"accuracy": ..., ...},
                "visual reasoning": {...},
                "medical reasoning": {...},
            },
            ...
        }
    """
    bst = user_metrics.get("by_super_type", {})

    def _pct(name):
        v = bst.get(name, {}).get("accuracy", 0.0)
        return round(v * 100, 1)

    user_row = {
        "model": user_label,
        "params": user_params,
        "input": user_input,
        "mode": user_mode,
        "overall": round(user_metrics["overall"]["accuracy"] * 100, 1),
        "recog": _pct("recognition"),
        "meas": _pct("measurement"),
        "visr": _pct("visual reasoning"),
        "medr": _pct("medical reasoning"),
        "source": f"user submission (N={user_metrics['overall']['n_total']})",
        "origin": "user",
    }
    return sorted(leaderboard + [user_row], key=lambda r: -r["overall"])


def format_ranking(
    ranked: list[dict],
    highlight_origin: str = "user",
    top_k: int | None = None,
) -> str:
    """Render the ranked leaderboard as an ASCII table, marking user rows."""
    lines: list[str] = []
    lines.append("=" * 105)
    lines.append("DeepTumorVQA leaderboard  (sorted by Overall %)")
    lines.append("=" * 105)
    header = (f"{'Rank':>4}  {'Model':<40s}  {'Mode':<10s}  "
              f"{'Overall':>8s} {'Recog':>7s} {'Meas':>7s} {'VisR':>7s} {'MedR':>7s}")
    lines.append(header)
    lines.append("-" * 105)

    user_rank = None
    for i, r in enumerate(ranked, start=1):
        if r.get("origin") == highlight_origin:
            user_rank = i
            marker = ">> "
        else:
            marker = "   "
        if top_k and i > top_k and r.get("origin") != highlight_origin:
            continue
        lines.append(
            f"{marker}{i:>2}  {r['model'][:40]:<40s}  {r['mode']:<10s}  "
            f"{r['overall']:>7.1f}% {r['recog']:>6.1f}% {r['meas']:>6.1f}% "
            f"{r['visr']:>6.1f}% {r['medr']:>6.1f}%"
        )
    lines.append("=" * 105)
    if user_rank is not None:
        lines.append(f"Your model ranks #{user_rank} of {len(ranked)}.")
    return "\n".join(lines)


def report(
    user_metrics: dict,
    user_label: str,
    user_mode: str = "direct",
    user_input: str = "2D",
    user_params: str = "?",
    csv_path: str | Path = DEFAULT_LEADERBOARD,
    top_k: int | None = 10,
    mode_filter: Literal["all", "direct", "agent"] | None = None,
) -> str:
    """One-call helper: load, insert, sort, format. Returns the printable table."""
    if mode_filter is None:
        # Default: compare against same mode (direct vs agent)
        mode_filter = "agent" if user_mode in ("oracle", "predicted", "vision") else "direct"
    lb = load_leaderboard(csv_path, mode_filter=mode_filter)
    ranked = insert_user_model(lb, user_metrics, user_label,
                               user_mode=user_mode, user_input=user_input,
                               user_params=user_params)
    return format_ranking(ranked, top_k=top_k)
