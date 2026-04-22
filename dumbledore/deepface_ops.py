"""Normalize DeepFace `represent` / `analyze` outputs into :class:`DeepFaceGTV1`."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from dumbledore.gt_schema import DeepFaceGTV1, DEFAULT_ANALYZE_ACTIONS, SCHEMA_VERSION


def _first_face(result: list | Any) -> dict[str, Any] | None:
    if isinstance(result, list) and result:
        r0 = result[0]
        if isinstance(r0, dict):
            return r0
    if isinstance(result, dict):
        return result
    return None


def _dominant_from_probs(d: Any) -> str | None:
    if not isinstance(d, dict):
        return None
    try:
        return max(d, key=lambda k: float(d[k]) if d[k] is not None else 0.0)  # type: ignore[no-untyped-def]
    except (TypeError, ValueError):
        return None


def normalize_analyze_dict(face: Mapping[str, Any], actions: Sequence[str]) -> dict[str, Any]:
    """
    Map DeepFace's first face dict to a flat `analyze` object usable in GT and rewards.

    Only includes keys in `actions` (when present in source). Uses dominant_* when needed.
    """
    out: dict[str, Any] = {}
    for a in actions:
        if a == "age" and "age" in face:
            out["age"] = int(face["age"])
        elif a == "gender":
            g = face.get("dominant_gender")
            if g is not None:
                out["gender"] = str(g)
            elif "gender" in face and isinstance(face["gender"], dict):
                d = _dominant_from_probs(face["gender"])
                if d is not None:
                    out["gender"] = d
        elif a == "emotion":
            e = face.get("dominant_emotion")
            if e is not None:
                out["emotion"] = str(e)
            elif "emotion" in face and isinstance(face["emotion"], dict):
                d = _dominant_from_probs(face["emotion"])
                if d is not None:
                    out["emotion"] = d
        elif a == "race":
            r = face.get("dominant_race")
            if r is not None:
                out["race"] = str(r)
            elif "race" in face and isinstance(face["race"], dict):
                d = _dominant_from_probs(face["race"])
                if d is not None:
                    out["race"] = d
    return out


def build_gt_from_deepface(
    image_path: str,
    represent_embedding: list[float] | None,
    analyze_result: list | Any | None,
    *,
    actions: Sequence[str] = DEFAULT_ANALYZE_ACTIONS,
    detector_backend: str | None = None,
    model_name: str = "Facenet512",
) -> DeepFaceGTV1:
    """Build GT after calling DeepFace in the caller (keeps this module importable without deepface in tests)."""
    fn: list[float] = []
    if represent_embedding is not None:
        if len(represent_embedding) != 512:
            raise ValueError(f"Facenet512 must have length 512, got {len(represent_embedding)}")
        fn = [float(x) for x in represent_embedding]

    an: dict[str, Any] = {}
    action_list = list(actions)
    face = _first_face(analyze_result) if analyze_result is not None and action_list else None
    if face is not None:
        an = normalize_analyze_dict(face, action_list)

    meta: dict[str, Any] = {"model_name": model_name, "actions": list(actions)}
    if detector_backend is not None:
        meta["detector_backend"] = detector_backend

    return DeepFaceGTV1(
        schema_version=SCHEMA_VERSION,
        image_path=image_path,
        facenet512=fn,
        analyze=an,
        deepface_metadata=meta,
    )
