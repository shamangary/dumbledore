"""Parse and summarize `ground_truth` JSON: `{"0": { ... }, "1": { ... }}`."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dumbledore.gt_schema import is_face_index_key
from dumbledore.pipeline_config import (
    SUPPORTED_GROUND_TRUTH_KEYS,
    GroundTruthOutputConfig,
)


def _invalid_top_level_keys(o: dict[str, Any]) -> list[str]:
    return [k for k in o if not (is_face_index_key(k) and isinstance(o[k], dict))]


def _face_dicts_in_order(o: dict[str, Any]) -> list[dict[str, Any]]:
    keys = sorted((k for k in o if is_face_index_key(k) and isinstance(o[k], dict)), key=int)
    return [o[k] for k in keys if isinstance(o[k], dict)]


def _face_iter(o: dict[str, Any]) -> list[dict[str, Any]]:
    return _face_dicts_in_order(o)


def _union_face_keys(faces: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in faces:
        for k in SUPPORTED_GROUND_TRUTH_KEYS:
            if k in f and f[k] is not None and k not in seen:
                seen.add(k)
                out.append(k)
    return out


def parse_gt_keys_from_object(o: dict[str, Any]) -> list[str]:
    if _invalid_top_level_keys(o):
        return []
    faces = _face_iter(o)
    if not faces:
        return []
    return _union_face_keys(faces)


def infer_ground_truth_output_config(gt_str: str) -> GroundTruthOutputConfig:
    o = json.loads(gt_str)
    if not isinstance(o, dict) or _invalid_top_level_keys(o):
        return GroundTruthOutputConfig()
    faces = _face_iter(o)
    f0 = faces[0] if faces else {}
    if not isinstance(f0, dict):
        f0 = {}
    return GroundTruthOutputConfig(
        bbox="bbox" in f0,
        age="age" in f0,
        gender="gender" in f0,
        emotion="emotion" in f0,
        race="race" in f0,
        is_real="is_real" in f0,
    )


def parse_gt_keys(gt_str: str) -> list[str]:
    o = json.loads(gt_str)
    if not isinstance(o, dict):
        return []
    if _invalid_top_level_keys(o):
        return []
    return parse_gt_keys_from_object(o)


@dataclass
class GroundTruthRowSummary:
    ok: bool
    error: str | None = None
    face_count: int = 0
    ages: list[int] = field(default_factory=list)  # all faces, for reports
    analyze_keys: list[str] = field(default_factory=list)
    age: int | None = None  # first face (compat)
    genders: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    races: list[str] = field(default_factory=list)
    gender: str | None = None
    emotion: str | None = None
    race: str | None = None
    has_bbox: bool = False
    is_real: bool | None = None


def summarize_ground_truth_string(gt_str: str) -> GroundTruthRowSummary:
    try:
        o = json.loads(gt_str)
    except json.JSONDecodeError as e:
        return GroundTruthRowSummary(ok=False, error=str(e))
    if not isinstance(o, dict):
        return GroundTruthRowSummary(ok=False, error="not an object")
    bad = _invalid_top_level_keys(o)
    if bad:
        return GroundTruthRowSummary(
            ok=False,
            error="top-level keys must be string digits only, each value a face object",
        )
    faces = _face_iter(o)
    n = len(faces)
    if n == 0:
        return GroundTruthRowSummary(
            ok=True,
            face_count=0,
            ages=[],
            analyze_keys=[],
        )
    keys = parse_gt_keys_from_object(o)
    f0 = faces[0]
    all_ages: list[int] = []
    for fc in faces:
        ag = fc.get("age")
        try:
            if ag is not None:
                all_ages.append(int(ag))
        except (TypeError, ValueError):
            pass
    age = f0.get("age")
    try:
        age_i = int(age) if age is not None else None
    except (TypeError, ValueError):
        age_i = None
    bbox = f0.get("bbox")
    has_bbox = isinstance(bbox, list) and len(bbox) == 4
    genders = [str(f["gender"]) for f in faces if f.get("gender") is not None]
    emotions = [str(f["emotion"]) for f in faces if f.get("emotion") is not None]
    races = [str(f["race"]) for f in faces if f.get("race") is not None]
    return GroundTruthRowSummary(
        ok=True,
        face_count=n,
        ages=all_ages,
        analyze_keys=keys,
        age=age_i,
        genders=genders,
        emotions=emotions,
        races=races,
        gender=str(f0["gender"]) if f0.get("gender") is not None else None,
        emotion=str(f0["emotion"]) if f0.get("emotion") is not None else None,
        race=str(f0["race"]) if f0.get("race") is not None else None,
        has_bbox=has_bbox,
        is_real=f0.get("is_real") if "is_real" in f0 else None,
    )
