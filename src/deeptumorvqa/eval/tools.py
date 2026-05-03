"""Agent tools for DeepTumorVQA evaluation.

Four tools, all OpenAI function-calling compatible:
  - segment_organ:           voxel count + bounding box for an organ/lesion
  - measure:                 volume / HU / diameter / lesion count / etc.
  - lookup_medical_knowledge: search a small clinical KB by keyword
  - crop_organ:              return a cropped region of the whole-volume PNG

Cache-first design: every tool first looks up its answer in a shipped tool-cache
JSON (`tool_cache/benchmark_oracle_tool_cache.json` for oracle mode, or
`tool_cache/benchmark_totalsegmentator_cache.json` for predicted mode). Live
recomputation from raw CT/segmentation is supported but requires
`allow_recompute=True` and a path to AbdomenAtlas-format data.

The release ships the caches so the default code path needs no NIfTI dependency.
"""

from __future__ import annotations

import abc
import base64
import io
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Knowledge base (verbatim from agent/tools/knowledge_tool.py)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[dict] = [
    {"keywords": ["liver", "volume", "normal", "size", "enlargement", "enlarged"],
     "text": "Normal liver volume: 1000-1800 cm³. Liver is enlarged (hepatomegaly) when volume > 1800 cm³ or atrophic when < 600 cm³.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["spleen", "volume", "normal", "size", "enlargement", "enlarged", "splenomegaly"],
     "text": "Normal spleen volume: <314.5 cm³. Splenomegaly = spleen volume > 314.5 cm³.",
     "source": "Bezerra 2005, AJR 184(5):1510"},
    {"keywords": ["splenomegaly", "grading", "grade", "mild", "moderate", "severe", "spleen"],
     "text": "Splenomegaly grading by spleen volume (Bezerra 2005): Normal <314.5; Mild 314.5-500; Moderate 500-800; Severe >800 cm³.",
     "source": "Bezerra 2005, AJR 184(5):1510"},
    {"keywords": ["splenomegaly", "detection", "enlarged", "spleen", "threshold"],
     "text": "Splenomegaly detection: spleen enlarged when volume > 314.5 cm³.",
     "source": "Bezerra 2005"},
    {"keywords": ["pancreas", "volume", "normal", "size"],
     "text": "Normal pancreas volume: 40-100 cm³.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["kidney", "volume", "normal", "size"],
     "text": "Normal single kidney volume: 130-260 cm³. Bilateral total: 260-520 cm³.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["fatty", "liver", "steatosis", "hepatic", "L/S", "ratio"],
     "text": "Fatty liver on non-contrast CT: L/S HU ratio < 1.0 indicates fatty liver. Liver HU < 40 suggests moderate-severe.",
     "source": "PMC3377794"},
    {"keywords": ["hepatic", "steatosis", "grading", "grade", "liver", "fatty", "kodama"],
     "text": "Hepatic steatosis grading (Kodama 2007): G0 Normal Liver HU>=58 L/S>=1.0; G1 Mild HU 51-57 or L/S 0.8-1.0; G2 Moderate HU 39-50 or L/S 0.5-0.8; G3 Severe HU<39 or L/S<0.5.",
     "source": "Kodama 2007, Radiology 245(1):95"},
    {"keywords": ["pancreatic", "steatosis", "fatty", "pancreas", "P/S", "ratio", "infiltration"],
     "text": "Pancreatic steatosis: P/S HU ratio < 0.70 indicates fatty infiltration of the pancreas.",
     "source": "PMC8833002"},
    {"keywords": ["pseudocyst", "pancreatic", "cyst", "mucin", "mucinous"],
     "text": "Pseudocyst vs mucinous cyst: HU > 14.5 -> pseudocyst; HU <= 14.5 -> mucinous cyst.",
     "source": "PMID:21737901"},
    {"keywords": ["pancreatic", "cyst", "resection", "surgery", "remove"],
     "text": "Pancreatic cyst resection: recommended when volume > 3.0 cm³ or diameter > 3 cm (worrisome features per ACR).",
     "source": "Hopkins / ACR Guidelines"},
    {"keywords": ["kidney", "cyst", "tumor", "classification", "type", "renal"],
     "text": "Kidney lesion HU: mean HU > 34.8 suggests solid tumor (RCC); <= 34.8 suggests cyst.",
     "source": "AJR 2017 (10.2214/AJR.16.17119)"},
    {"keywords": ["pdac", "pnet", "pancreatic", "adenocarcinoma", "neuroendocrine", "classification", "subtype", "ductal"],
     "text": "PDAC vs PNET: PDAC = hypoattenuating, irregular margins, head 60-70%; PNET = iso/hyperattenuating, body/tail, well-circumscribed.",
     "source": "NCCN Pancreatic Cancer 2024"},
    {"keywords": ["portal", "hypertension", "splenomegaly", "liver", "cirrhosis", "caudate"],
     "text": "Portal hypertension on CT: splenomegaly (>314.5 cm³), caudate-to-right-lobe ratio > 0.65, portosystemic collaterals, ascites.",
     "source": "Harbin 1980; Radiology 2024"},
    {"keywords": ["renal", "mass", "kidney", "bosniak", "cyst", "characterization", "HU", "solid"],
     "text": "Renal mass by HU (simplified Bosniak): simple cyst -10..20; hyperattenuating cyst >70; indeterminate 20-70; enhancing solid -> RCC.",
     "source": "Silverman 2019 Radiology (Bosniak v2019)"},
    {"keywords": ["tumor", "burden", "percentage", "lesion", "volume", "ratio"],
     "text": "Tumor burden = total lesion volume / organ volume × 100%. >50% = advanced disease, poor prognosis.",
     "source": "EASL-EORTC HCC Guidelines"},
    {"keywords": ["organ", "HU", "ratio", "L/S", "P/S", "liver", "spleen", "pancreas"],
     "text": "Organ HU ratios: L/S normal >=1.0 (<1.0 = hepatic steatosis); P/S normal >=0.70 (<0.70 = pancreatic steatosis).",
     "source": "Kodama 2007; PMC8833002"},
    {"keywords": ["bilateral", "kidney", "asymmetry", "left", "right", "comparison"],
     "text": "Bilateral kidney asymmetry = one kidney has substantially more lesion burden (volume/count) than the other.",
     "source": "Clinical practice"},
    {"keywords": ["pancreatic", "tumor", "staging", "T1", "T2", "T3", "T4", "stage"],
     "text": "Pancreatic T-staging (AJCC 8): T1 <=2 cm; T2 2-4 cm; T3 >4 cm; T4 invades celiac/SMA/CHA (unresectable).",
     "source": "AJCC 8th"},
    {"keywords": ["resectability", "resectable", "borderline", "unresectable", "surgical", "pancreatic"],
     "text": "Pancreatic resectability: Resectable = no SMA/CHA contact, no SMV/portal involvement; Borderline = SMA/CHA <=180° or SMV narrowing; Unresectable = encasement >180° or SMV/portal occlusion.",
     "source": "NCCN Guidelines"},
    {"keywords": ["liver", "segment", "hepatic", "couinaud", "anatomy"],
     "text": "Couinaud segments: 1=caudate; 2,3=left lateral; 4=left medial; 5,8=right anterior; 6,7=right posterior.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["attenuation", "hypoattenuating", "hyperattenuating", "isoattenuating", "enhancement"],
     "text": "Lesion attenuation vs organ: Hypo = lower HU (cysts, some tumors); Hyper = higher HU (vascular, hemorrhagic); Iso = similar HU.",
     "source": "Radiology fundamentals"},
    {"keywords": ["LI-RADS", "liver", "lesion", "classification", "HCC"],
     "text": "LI-RADS in high-risk: LR-1 benign; LR-2 prob benign; LR-3 intermediate; LR-4 prob HCC; LR-5 def HCC.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["HU", "hounsfield", "normal", "organ", "reference", "value"],
     "text": "Normal non-contrast HU: Liver 50-65; Spleen 40-55; Kidney cortex 30-45; Pancreas 35-45.",
     "source": "Radiology reference tables"},
    {"keywords": ["bosniak", "renal", "cyst", "kidney", "classification"],
     "text": "Bosniak: I simple (HU<20, no enh); II minimally complex; IIF moderately complex (follow-up); III indeterminate (surgical); IV malignant.",
     "source": "radiologyassistant.nl"},
    {"keywords": ["outlier", "lesion", "volume", "largest"],
     "text": "Lesion outlier: largest lesion has volume >= 3× the second largest, with at least 3 lesions present.",
     "source": "DeepTumorVQA protocol"},
    {"keywords": ["clustering", "clustered", "distributed", "liver", "segment"],
     "text": "Liver lesion distribution: highly clustered = >50% in one segment; somewhat = top 2 segments >=75%; widely distributed = neither.",
     "source": "DeepTumorVQA protocol"},
]


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------

class ToolCache:
    """Thin wrapper around the shipped tool_cache JSON."""

    def __init__(self, cache_path: str | Path):
        with open(cache_path) as f:
            data = json.load(f)
        # Cache format: {"cache": {image_id: {"segment": {...}, "measure": {...}}}}
        self._cache: dict[str, dict] = data.get("cache", data)

    def has(self, image_id: str) -> bool:
        return image_id in self._cache

    def segment(self, image_id: str, target: str) -> dict | None:
        img = self._cache.get(image_id, {})
        return img.get("segment", {}).get(target)

    def measure(self, image_id: str, target: str, measurement_type: str) -> dict | None:
        img = self._cache.get(image_id, {})
        m = img.get("measure", {}).get(target, {})
        return m.get(measurement_type)

    def all_organs_with_segment(self, image_id: str) -> list[str]:
        img = self._cache.get(image_id, {})
        return list(img.get("segment", {}).keys())


# ---------------------------------------------------------------------------
# Tool ABC + four concrete tools
# ---------------------------------------------------------------------------

class Tool(abc.ABC):
    """OpenAI-function-calling-compatible tool."""

    name: str
    description: str
    parameters_schema: dict

    @abc.abstractmethod
    def execute(self, **kwargs) -> dict:
        ...

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class SegmentOrganTool(Tool):
    """Return voxel count + bounding box for an organ/lesion target."""

    name = "segment_organ"
    description = (
        "Run 3D segmentation on the CT scan. Returns voxel count and bounding "
        "box for the named organ or lesion. Use this before `measure`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Organ or lesion (e.g. 'liver', 'kidney_left', 'liver_tumor').",
            },
        },
        "required": ["target"],
    }

    def __init__(self, cache: ToolCache):
        self.cache = cache

    def execute(self, image_id: str, target: str, **kwargs) -> dict:
        result = self.cache.segment(image_id, target)
        if result is None:
            return {
                "mask_found": False,
                "error": f"No segmentation cached for target='{target}' on image='{image_id}'.",
                "available_targets": self.cache.all_organs_with_segment(image_id),
            }
        return result


class MeasureTool(Tool):
    """Return a quantitative measurement of a previously-segmented region."""

    name = "measure"
    description = (
        "Compute a quantitative measurement on a segmented organ/lesion. "
        "Supports: volume_cm3, mean_hu, max_diameter_mm, lesion_count, "
        "largest_lesion_volume_cm3, largest_lesion_diameter_cm."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "measurement_type": {
                "type": "string",
                "enum": [
                    "volume_cm3",
                    "mean_hu",
                    "max_diameter_mm",
                    "lesion_count",
                    "largest_lesion_volume_cm3",
                    "largest_lesion_diameter_cm",
                ],
            },
        },
        "required": ["target", "measurement_type"],
    }

    def __init__(self, cache: ToolCache):
        self.cache = cache

    def execute(self, image_id: str, target: str,
                measurement_type: str | None = None,
                measurement: str | None = None, **kwargs) -> dict:
        # Accept the alternate parameter name `measurement` for backward compatibility
        mt = measurement_type or measurement
        if not mt:
            return {"error": "Missing measurement_type."}
        result = self.cache.measure(image_id, target, mt)
        if result is None:
            return {
                "error": f"No '{mt}' cached for target='{target}' on image='{image_id}'.",
            }
        return result


class KnowledgeLookupTool(Tool):
    """Keyword-overlap search over a small clinical knowledge base."""

    name = "lookup_medical_knowledge"
    description = (
        "Look up clinical guidelines, diagnostic thresholds, or anatomic "
        "reference values. Returns the best-matching short text passage and "
        "its source."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A specific medical query (e.g. 'splenomegaly grading', 'L/S ratio threshold').",
            },
        },
        "required": ["query"],
    }

    def __init__(self, kb: list[dict] = KNOWLEDGE_BASE):
        self.kb = kb

    def execute(self, query: str, **kwargs) -> dict:
        q = query.lower()
        q_words = set(q.split())
        scored: list[tuple[int, dict]] = []
        for entry in self.kb:
            kws = entry["keywords"]
            score = sum(1 for k in kws if k in q) + len(q_words & set(kws))
            if score > 0:
                scored.append((score, entry))
        if not scored:
            return {"results": [{"text": "No relevant medical knowledge found.", "source": "N/A"}]}
        scored.sort(key=lambda x: -x[0])
        out = [{"text": scored[0][1]["text"], "source": scored[0][1]["source"]}]
        if len(scored) > 1 and scored[1][0] >= scored[0][0] * 0.5:
            out.append({"text": scored[1][1]["text"], "source": scored[1][1]["source"]})
        return {"results": out}


class CropOrganTool(Tool):
    """Return a pre-extracted organ-focused 2D slice PNG.

    The release ships ~7160 organ-specific PNGs at `benchmark/images_2d/organ/`
    named `{image_id}_{organ}.png`. The `organ` argument may be a single organ
    name (e.g. "liver") or a multi-organ combination needed for cross-organ
    questions (e.g. "liver_spleen", "pancreas_spleen", "left_kidney_pancreas").
    Lesion targets (e.g. "liver_tumor") fall back to the parent organ slice.
    """

    name = "crop_organ"
    description = (
        "Return a focused axial CT slice cropped around a named organ or "
        "organ combination. Returns a base64-encoded PNG. Use this to "
        "visually inspect a region without explicit measurements. "
        "Available organ names match those listed by `list_available_crops`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "organ": {
                "type": "string",
                "description": (
                    "Single organ ('liver', 'spleen', 'pancreas', 'colon', "
                    "'kidney', 'left_kidney', 'right_kidney', 'pancreatic') "
                    "or multi-organ combination needed for the question "
                    "('liver_spleen', 'pancreas_spleen', 'left_kidney_spleen', "
                    "'right_kidney_spleen', 'liver_kidney', 'liver_pancreas', "
                    "'kidney_pancreas', etc.)."
                ),
            },
        },
        "required": ["organ"],
    }

    # Lesion -> parent organ slice (no lesion-specific PNGs are shipped)
    LESION_TO_ORGAN = {
        "liver_tumor": "liver", "kidney_tumor": "kidney",
        "kidney_cyst": "kidney", "pancreas_tumor": "pancreatic",
        "colon_tumor": "colon", "hepatic_vessel_tumor": "liver",
    }
    # Common aliases users / models may produce
    ALIASES = {
        "kidney_left": "left_kidney",
        "kidney_right": "right_kidney",
    }

    def __init__(self, image_dir: str | Path):
        """`image_dir` should contain organ-specific PNGs named
        `{image_id}_{organ}.png` (the `benchmark/images_2d/organ/` folder)."""
        self.image_dir = Path(image_dir)

    def _resolve(self, image_id: str, organ: str) -> Path | None:
        # Try as-given, alias, lesion-mapped, alias-of-lesion-mapped
        candidates = [organ]
        if organ in self.ALIASES:
            candidates.append(self.ALIASES[organ])
        if organ in self.LESION_TO_ORGAN:
            candidates.append(self.LESION_TO_ORGAN[organ])
        for c in candidates:
            p = self.image_dir / f"{image_id}_{c}.png"
            if p.exists():
                return p
        return None

    def execute(self, image_id: str, organ: str, **kwargs) -> dict:
        png_path = self._resolve(image_id, organ)
        if png_path is None:
            return {
                "error": (
                    f"No organ slice found for organ='{organ}' on image='{image_id}'. "
                    f"Tried: {organ}, {self.ALIASES.get(organ, '')}, "
                    f"{self.LESION_TO_ORGAN.get(organ, '')}. "
                    "List available organs with `list_available_crops`."
                ),
            }
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prefix = f"{image_id}_"
        organ_name = png_path.stem[len(prefix):] if png_path.stem.startswith(prefix) else organ
        return {
            "image_base64": b64,
            "image_path": str(png_path),
            "organ_queried": organ,
            "organ_localized": organ_name,
            "note": "Pre-extracted organ-focused axial slice (window center 50, width 400).",
        }


class ListAvailableCropsTool(Tool):
    """Companion to CropOrganTool: tell the agent which organ slices exist."""

    name = "list_available_crops"
    description = (
        "Return the list of organ names for which a pre-extracted crop is "
        "available for the current image. Useful to call before `crop_organ` "
        "to avoid querying a non-existent slice."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)

    def execute(self, image_id: str, **kwargs) -> dict:
        prefix = f"{image_id}_"
        organs = sorted({
            p.stem[len(prefix):]
            for p in self.image_dir.glob(f"{image_id}_*.png")
            if p.stem.startswith(prefix)
        })
        return {"image_id": image_id, "available_organs": organs}


# ---------------------------------------------------------------------------
# Helper for AgentEvaluator — assemble tool kit per agent mode
# ---------------------------------------------------------------------------

def build_toolkit(
    mode: str,
    tool_cache_path: str | Path | None = None,
    image_dir: str | Path | None = None,
) -> list[Tool]:
    """Return the list of tools enabled for a given agent mode.

    mode in {"oracle", "predicted", "vision"}.
      - oracle / predicted: need `tool_cache_path` (JSON, see release docs)
      - vision:             needs `image_dir` (organ-specific PNGs); no cache
    """
    if mode in ("oracle", "predicted"):
        if tool_cache_path is None:
            raise ValueError(f"mode={mode} requires tool_cache_path.")
        cache = ToolCache(tool_cache_path)
        return [
            SegmentOrganTool(cache),
            MeasureTool(cache),
            KnowledgeLookupTool(),
        ]
    if mode == "vision":
        if image_dir is None:
            raise ValueError("vision mode requires image_dir (organ-specific PNGs).")
        return [
            CropOrganTool(image_dir),
            ListAvailableCropsTool(image_dir),
            KnowledgeLookupTool(),
        ]
    raise ValueError(f"Unknown agent mode: {mode!r}. Use oracle / predicted / vision.")
