"""Scoring functions for DeepTumorVQA evaluation.

Two output formats are supported:
  - mc:        single-letter exact match against `correct_option` (A/B/C/D)
  - freeform:  per-subtype answer-type-aware scoring
                * numeric    -> MRA + tolerance bands
                * binary     -> exact match + sensitivity/specificity
                * categorical-> normalized exact match + partial match

The 4 super-types (Recognition / Measurement / Visual Reasoning / Medical
Reasoning) are macro-averaged from their constituent subtypes for the final
table that the leaderboard ranking compares against.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Literal

# ---------------------------------------------------------------------------
# Subtype -> answer type mapping (frozen, drives free-form scoring)
# ---------------------------------------------------------------------------

SUBTYPE_ANSWER_TYPE: dict[str, Literal["numeric", "binary", "categorical"]] = {
    # numeric
    "organ volume measurement": "numeric",
    "organ HU measurement": "numeric",
    "lesion volume measurement": "numeric",
    "organ HU ratio": "numeric",
    "tumor burden percentage": "numeric",
    "largest lesion diameter": "numeric",
    "largest lesion slice": "numeric",
    "lesion counting": "numeric",
    "lesion count by location": "numeric",
    "organ aggregation": "numeric",
    "tumor organ HU difference": "numeric",
    # binary
    "pancreatic lesion existence": "binary",
    "liver lesion existence": "binary",
    "kidney lesion existence": "binary",
    "kidney tumor existence": "binary",
    "kidney cyst existence": "binary",
    "colon lesion existence": "binary",
    "PDAC existence": "binary",
    "PNET existence": "binary",
    "lesion outlier": "binary",
    "organ enlargement": "binary",
    "pancreatic steatosis": "binary",
    "pancreatic cyst resectability": "binary",
    "pseudocyst determination": "binary",
    "splenomegaly detection": "binary",
    "portal hypertension assessment": "binary",
    # categorical
    "liver lesion clustering": "categorical",
    "kidney volume comparison": "categorical",
    "largest lesion attenuation": "categorical",
    "largest lesion location": "categorical",
    "inter-segment comparison": "categorical",
    "adjacent organ": "categorical",
    "bilateral kidney asymmetry": "categorical",
    "multi-organ lesion burden comparison": "categorical",
    "fatty liver": "categorical",
    "hepatic steatosis grading": "categorical",
    "pancreatic tumor staging": "categorical",
    "pancreatic lesion resectability": "categorical",
    "PDAC vs PNET classification": "categorical",
    "lesion type classification": "categorical",
    "splenomegaly grading": "categorical",
    "renal mass characterization": "categorical",
}

QUESTION_TYPES = ["recognition", "measurement", "visual reasoning", "medical reasoning"]
TOLERANCE_THRESHOLDS = (0.05, 0.10, 0.20, 0.50)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
_NUMBER_RE = re.compile(r"-?[\d]+\.?\d*")
_LETTER_RE = re.compile(r"\b([A-D])\b")


def strip_think(text: str) -> str:
    """Drop `<think>...</think>` blocks, including unclosed ones."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def extract_letter(text: str) -> str:
    """Pull a single MC letter (A/B/C/D) out of free-form output. '' if none."""
    text = strip_think(text)
    if not text:
        return ""
    s = text.strip().upper()
    if s and s[0] in "ABCD":
        return s[0]
    m = _LETTER_RE.search(s)
    return m.group(1) if m else ""


def extract_number(text: str) -> float | None:
    """Return the first numeric token, or None."""
    if text is None:
        return None
    m = _NUMBER_RE.search(str(text).strip())
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s)
    return re.sub(r"\s+", " ", s)


def extract_binary(text: str) -> str:
    """Return 'yes' / 'no' / '' from free-form output."""
    s = str(text).strip().lower()
    if s in ("yes", "no"):
        return s
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return ""


# ---------------------------------------------------------------------------
# Per-subtype scorers (free-form)
# ---------------------------------------------------------------------------

def _mra(pred: float, gt: float) -> float:
    """Mean Relative Accuracy (Yang 2024). max(0, 1 - |pred-gt|/|gt|)."""
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1.0 - abs(pred - gt) / abs(gt))


def _rel_err(pred: float, gt: float) -> float:
    if gt == 0:
        return 0.0 if pred == 0 else float("inf")
    return abs(pred - gt) / abs(gt)


def score_numeric(pairs: Iterable[tuple[str, str]]) -> dict:
    """`pairs`: iterable of (pred_text, gt_text). Returns MRA + tolerance bands."""
    pairs = list(pairs)
    mras: list[float] = []
    tol_hits = {t: 0 for t in TOLERANCE_THRESHOLDS}
    n_total = len(pairs)
    n_with_gt = 0

    for pred_text, gt_text in pairs:
        gt = extract_number(gt_text)
        if gt is None:
            continue
        n_with_gt += 1
        pred = extract_number(strip_think(pred_text))
        if pred is None:
            mras.append(0.0)
            continue
        mras.append(_mra(pred, gt))
        e = _rel_err(pred, gt)
        for t in TOLERANCE_THRESHOLDS:
            if e <= t:
                tol_hits[t] += 1

    return {
        "metric": "MRA",
        "mean_mra": round(sum(mras) / max(len(mras), 1), 4),
        "tolerance_acc": {
            f"tol_{int(t * 100)}%": round(tol_hits[t] / max(n_with_gt, 1), 4)
            for t in TOLERANCE_THRESHOLDS
        },
        "n_total": n_total,
        "n_scored": n_with_gt,
    }


def score_binary(pairs: Iterable[tuple[str, str]]) -> dict:
    pairs = list(pairs)
    tp = fp = tn = fn = correct = 0
    for pred_text, gt_text in pairs:
        gt = extract_binary(gt_text)
        pred = extract_binary(strip_think(pred_text))
        if pred == gt and pred:
            correct += 1
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "yes":  # pred is no/empty
            fn += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no":   # pred is no/empty -> count as no
            tn += 1
    n = len(pairs)
    return {
        "metric": "exact_match",
        "accuracy": round(correct / max(n, 1), 4),
        "sensitivity": round(tp / max(tp + fn, 1), 4),
        "specificity": round(tn / max(tn + fp, 1), 4),
        "n_total": n,
    }


def score_categorical(pairs: Iterable[tuple[str, str]]) -> dict:
    pairs = list(pairs)
    exact = partial = 0
    for pred_text, gt_text in pairs:
        gt = normalize_text(gt_text)
        pred = normalize_text(strip_think(pred_text))
        if gt == pred:
            exact += 1
            partial += 1
        elif gt and pred and (gt in pred or pred in gt):
            partial += 1
    n = len(pairs)
    return {
        "metric": "exact_match",
        "accuracy": round(exact / max(n, 1), 4),
        "partial_accuracy": round(partial / max(n, 1), 4),
        "n_total": n,
    }


def score_mc(pairs: Iterable[tuple[str, str]]) -> dict:
    """MC scoring. `pairs`: (pred_text, correct_option_letter)."""
    pairs = list(pairs)
    correct = 0
    parsed = 0
    for pred_text, gt_letter in pairs:
        pred_letter = extract_letter(pred_text)
        if pred_letter:
            parsed += 1
        if pred_letter == gt_letter.strip().upper():
            correct += 1
    n = len(pairs)
    return {
        "metric": "exact_match",
        "accuracy": round(correct / max(n, 1), 4),
        "n_parsed": parsed,
        "n_total": n,
    }


# ---------------------------------------------------------------------------
# Aggregation: per-subtype -> per-super-type -> overall
# ---------------------------------------------------------------------------

def _primary_acc(metric: dict, atype: str) -> float:
    if atype == "numeric":
        return metric["mean_mra"]
    return metric["accuracy"]


def aggregate(
    predictions: Iterable[dict],
    benchmark_index: dict[str, dict],
    fmt: Literal["mc", "freeform"],
) -> dict:
    """Score a model's output across the full benchmark.

    Args:
        predictions: iterable of {qid, pred_answer (or raw_output)}
        benchmark_index: {qid -> question dict from benchmark_v4.json}
        fmt: scoring protocol

    Returns:
        {
          "overall": {"accuracy": .., "n_total": ..},
          "by_super_type": {recognition: {"accuracy": ..}, ...},
          "by_subtype": {subtype_name: {<scorer-dict>}},
        }
    """
    by_subtype_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_super_type_acc: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for r in predictions:
        qid = r["qid"]
        if qid not in benchmark_index:
            continue
        q = benchmark_index[qid]
        st = q["question_subtype"]
        pred_text = r.get("pred_answer") or r.get("raw_output") or ""
        gt = q["correct_option"] if fmt == "mc" else q["answer"]
        by_subtype_pairs[st].append((str(pred_text), str(gt)))

    by_subtype_metrics: dict[str, dict] = {}
    for st, pairs in by_subtype_pairs.items():
        if fmt == "mc":
            m = score_mc(pairs)
            atype = "mc"
        else:
            atype = SUBTYPE_ANSWER_TYPE.get(st, "categorical")
            if atype == "numeric":
                m = score_numeric(pairs)
            elif atype == "binary":
                m = score_binary(pairs)
            else:
                m = score_categorical(pairs)
        by_subtype_metrics[st] = m
        # group into super-type using benchmark
        super_type = next(
            (q["question_type"] for q in benchmark_index.values() if q["question_subtype"] == st),
            None,
        )
        if super_type:
            acc = _primary_acc(m, atype if fmt == "freeform" else "mc")
            by_super_type_acc[super_type].append((acc, m["n_total"]))

    by_super_type: dict[str, dict] = {}
    for super_type, accs in by_super_type_acc.items():
        weighted = sum(acc * n for acc, n in accs)
        n_total = sum(n for _, n in accs)
        by_super_type[super_type] = {
            "accuracy": round(weighted / max(n_total, 1), 4),
            "n_total": n_total,
        }

    # Overall: micro-average across all questions (matches paper convention)
    total_correct = sum(
        _primary_acc(m, SUBTYPE_ANSWER_TYPE.get(st, "categorical") if fmt == "freeform" else "mc")
        * m["n_total"]
        for st, m in by_subtype_metrics.items()
    )
    total_n = sum(m["n_total"] for m in by_subtype_metrics.values())

    return {
        "overall": {
            "accuracy": round(total_correct / max(total_n, 1), 4),
            "n_total": total_n,
            "n_subtypes": len(by_subtype_metrics),
        },
        "by_super_type": by_super_type,
        "by_subtype": by_subtype_metrics,
    }
