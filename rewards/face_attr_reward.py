"""
verl custom reward for RL: compare the policy output `solution_str` to the **pseudo** label in
`ground_truth` (the DeepFace JSON string: indexed multi-face `{"0": {...}, "1": {...}}`).

The training row should provide: **text** `prompt` (request), **image** context via `extra_info["image_path"]`
(full image for the VLM), and this **string** as the reference "correct" answer for scoring—never human
annotations unless you replaced the pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

W_ATTR: float = 1.0
AGE_TOL: float = 5.0
BBOX_TOL: float = 2.0


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


def _bool_match(ga: Any, ra: Any) -> float:
    if ga is None and ra is None:
        return 1.0
    if ga is None or ra is None or not isinstance(ga, (bool, int)) or not isinstance(ra, (bool, int)):
        return 0.0
    return 1.0 if bool(ga) == bool(ra) else 0.0


def _bbox_match(ga: Any, ra: Any) -> float:
    if ga is None and ra is None:
        return 1.0
    if not isinstance(ga, list) or not isinstance(ra, list) or len(ga) != 4 or len(ra) != 4:
        return 0.0
    try:
        a = [float(x) for x in ga]
        b = [float(x) for x in ra]
    except (TypeError, ValueError):
        return 0.0
    if all(abs(x - y) <= BBOX_TOL for x, y in zip(a, b, strict=True)):
        return 1.0
    return 0.0


def _indexed_face_keys(d: Any) -> list[str]:
    if not isinstance(d, dict):
        return []
    out: list[str] = []
    for k, v in d.items():
        if not isinstance(k, str) or not k.isdigit() or not isinstance(v, dict):
            continue
        out.append(k)
    return sorted(out, key=int)


def _score_one_face(gt: dict[str, Any], pr: dict[str, Any]) -> float:
    keys = [k for k in ("bbox", "age", "gender", "emotion", "race", "is_real") if k in gt]
    if not keys:
        return 0.0
    s = 0.0
    for k in keys:
        g = gt[k]
        p = pr.get(k)
        if k == "age":
            s += _age_match(g, p)
        elif k == "bbox":
            s += _bbox_match(g, p)
        elif k == "is_real":
            s += _bool_match(g, p)
        else:
            s += _str_match(g, p)
    return s / len(keys)


def _score_indexed_pairs(gt: dict[str, Any], pr: dict[str, Any]) -> float:
    gks = _indexed_face_keys(gt)
    if not gks:
        return 0.0 if _indexed_face_keys(pr) else 1.0
    scores: list[float] = []
    for k in gks:
        g = gt[k]
        if not isinstance(g, dict):
            scores.append(0.0)
            continue
        p = pr.get(k) if isinstance(pr, dict) else None
        if not isinstance(p, dict):
            scores.append(0.0)
        else:
            scores.append(_score_one_face(g, p))
    return sum(scores) / len(scores)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,  # noqa: ARG001
) -> float:
    _ = data_source
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    pred = _parse_json_object(solution_str)
    if not isinstance(pred, dict) or not isinstance(gt, dict):
        return 0.0
    s = W_ATTR * _score_indexed_pairs(gt, pred)
    return min(1.0, max(0.0, s))
