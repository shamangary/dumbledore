"""
verl custom reward: compare model `solution_str` to DeepFace `ground_truth` (DeepFaceGTV1 JSON).

Pointwise combination (default):
  w_emb * cos_sim( facenet512 ) + w_attr * (matched analyze keys / total keys) + w_schema * 1.0

See verl: custom_reward_function.path / .name, signature:
  (data_source, solution_str, ground_truth, extra_info=None) -> float
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

# Default weights; sum is not required to be 1 (verl may normalize in recipe)
W_EMB: float = 0.4
W_ATTR: float = 0.5
W_SCHEMA: float = 0.1
AGE_TOL: float = 5.0


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    t = t.strip()
    o = t.find("{")
    c = t.rfind("}")
    if o == -1 or c == -1 or c <= o:
        return None
    try:
        return json.loads(t[o : c + 1])
    except json.JSONDecodeError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    v = dot / (na * nb)
    return max(0.0, min(1.0, (v + 1.0) / 2.0))  # map [-1,1] -> [0,1] for non-negative score mix


def _age_match(ga: Any, ra: Any) -> float:
    try:
        g = float(ga)
        r = float(ra)
    except (TypeError, ValueError):
        return 0.0
    if abs(g - r) <= AGE_TOL:
        return 1.0
    if AGE_TOL > 0:
        return max(0.0, 1.0 - abs(g - r) / (2.0 * AGE_TOL))
    return 0.0


def _str_match(ga: Any, ra: Any) -> float:
    if ga is None and ra is None:
        return 1.0
    if ga is None or ra is None:
        return 0.0
    return 1.0 if str(ga).lower().strip() == str(ra).lower().strip() else 0.0


def score_analyze(gt_an: dict[str, Any], pr_an: dict[str, Any]) -> tuple[float, int]:
    """Return (mean key score, number of keys considered)."""
    if not gt_an and not pr_an:
        return 1.0, 0
    keys = set(gt_an.keys()) | set(pr_an.keys())
    if not keys:
        return 1.0, 0
    s = 0.0
    for k in keys:
        g = gt_an.get(k)
        p = pr_an.get(k) if isinstance(pr_an, dict) else None
        if k == "age":
            s += _age_match(g, p)
        else:
            s += _str_match(g, p)
    return s / len(keys), len(keys)


def compute_score(  # noqa: PLR0911 — verl entry name
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,  # noqa: ARG001 — reserved for VLM
) -> float:
    """
    verl calls this; return a scalar reward in a sensible range (e.g. 0-1 or unbounded per recipe).
    We return a weighted score in [0, 1].
    """
    _ = data_source
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    pred = _parse_json_object(solution_str)
    if not isinstance(pred, dict) or not gt:
        return 0.0

    if str(pred.get("schema_version", "")) != str(gt.get("schema_version", "")):
        sch = 0.0
    else:
        sch = 1.0

    g_fn = gt.get("facenet512")
    p_fn = pred.get("facenet512")
    g_has_emb = isinstance(g_fn, list) and len(g_fn) == 512
    p_has_emb = isinstance(p_fn, list) and len(p_fn) == 512
    emb = 0.0
    if g_has_emb and p_has_emb:
        try:
            gf = [float(x) for x in g_fn]
            pf = [float(x) for x in p_fn]
            emb = _cosine(gf, pf)
        except (TypeError, ValueError):
            emb = 0.0
    w_emb = W_EMB if g_has_emb else 0.0

    g_an = gt.get("analyze") if isinstance(gt.get("analyze"), dict) else {}
    p_an = pred.get("analyze") if isinstance(pred.get("analyze"), dict) else {}
    if not g_an:
        attr_score = 1.0
        w_attr = 0.0
    else:
        attr_score, _n = score_analyze(g_an, p_an)
        w_attr = W_ATTR

    w_s = W_SCHEMA
    den = w_s + w_emb + w_attr
    if den <= 0:
        return 0.0
    total = w_s * sch + w_emb * emb + w_attr * attr_score
    return min(1.0, max(0.0, total / den))
