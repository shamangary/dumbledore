"""Build multi-face `ground_truth` from DeepFace `extract_faces` + per-crop `represent` / `analyze`."""
from __future__ import annotations

import inspect
import logging
import os
import tempfile
from typing import Any, Mapping, Sequence

from dumbledore.gt_schema import (
    DEFAULT_ANALYZE_ACTIONS,
    build_indexed_ground_truth_string,
    build_per_face_object,
)
from dumbledore.pipeline_config import GroundTruthOutputConfig

logger = logging.getLogger(__name__)


def _write_face_array_png(path: str, face_img: Any) -> None:
    """Persist a crop (numpy) so DeepFace can `analyze` by path; BGR/uint8 as OpenCV uses."""
    try:
        import cv2  # type: ignore[import-not-found]

        cv2.imwrite(path, face_img)
    except Exception:  # noqa: BLE001
        import numpy as np
        from PIL import Image

        a = np.asarray(face_img)
        if a.ndim == 2:
            Image.fromarray(a.astype("uint8")).save(path)
        else:
            Image.fromarray(a[:, :, ::-1].astype("uint8")).save(path)


def _analyze_on_image_path(
    deepface: Any,
    p: str,
    *,
    action_list: list[str],
    detector_backend: str,
    anti_spoof: bool,
) -> Any:
    an_kw: dict[str, Any] = {
        "img_path": p,
        "actions": list(action_list),
        "enforce_detection": True,
        "detector_backend": detector_backend,
    }
    if anti_spoof:
        an_kw["anti_spoofing"] = True
    return deepface.analyze(**an_kw)  # type: ignore[no-untyped-call]


def _kwargs_for_callable(fn: Any, kw: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the installed DeepFace build does not accept (e.g. `target_size` removed in some versions)."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except (TypeError, ValueError):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in params}


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


def _bbox_from_represent(represent_block: list | Any) -> list[int] | None:
    face = _first_face(represent_block)
    if not face:
        return None
    fa = face.get("facial_area")
    if not isinstance(fa, dict):
        return None
    x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")
    if None in (x, y, w, h):
        return None
    return [int(x), int(y), int(w), int(h)]


def _validate_embedding(emb: Any, *, need_length: bool) -> list[float] | None:
    if emb is None:
        return None
    if not isinstance(emb, list) or (need_length and len(emb) != 512):
        return None
    return [float(x) for x in emb]


def _bbox_from_facial_area(fa: Mapping[str, Any] | None) -> list[int] | None:
    if not fa:
        return None
    x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")
    if None in (x, y, w, h):
        return None
    return [int(x), int(y), int(w), int(h)]


def _coerce_is_real(v: Any) -> bool | None:
    if v is None:
        return None
    return bool(v)


def build_one_face_from_deepface(
    represent_result: list | Any | None,
    analyze_result: list | Any | None,
    *,
    output: GroundTruthOutputConfig,
    include_facenet512: bool,
    actions: Sequence[str] = DEFAULT_ANALYZE_ACTIONS,
    bbox_from_detector: list[int] | None = None,
    is_real_value: bool | None = None,
) -> dict[str, Any]:
    action_list = list(actions)
    bbox: list[int] | None = None
    if output.bbox:
        if bbox_from_detector is not None:
            bbox = bbox_from_detector
        elif represent_result is not None:
            bbox = _bbox_from_represent(represent_result)
    an: dict[str, Any] = {}
    if action_list and analyze_result is not None:
        face = _first_face(analyze_result)
        if face is not None:
            an = normalize_analyze_dict(face, action_list)
    if include_facenet512 and represent_result is not None:
        face = _first_face(represent_result)
        emb = face.get("embedding") if face else None
        if _validate_embedding(emb, need_length=True) is None:
            raise ValueError("expected 512-d embedding in represent() when include_facenet512 is true")
    age = int(an["age"]) if output.age and "age" in an else None
    gender = str(an["gender"]) if output.gender and "gender" in an else None
    emotion = str(an["emotion"]) if output.emotion and "emotion" in an else None
    race = str(an["race"]) if output.race and "race" in an else None
    is_real: bool | None = None
    if output.is_real:
        is_real = is_real_value
        if is_real is None and analyze_result is not None:
            face_an = _first_face(analyze_result)
            if face_an is not None and face_an.get("is_real") is not None:
                is_real = _coerce_is_real(face_an.get("is_real"))
    return build_per_face_object(
        output,
        bbox=bbox,
        age=age,
        gender=gender,
        emotion=emotion,
        race=race,
        is_real=is_real,
    )


def _normalize_extracted_items(result: Any) -> list[dict[str, Any]]:
    if not result or not isinstance(result, list):
        return []
    out: list[dict[str, Any]] = []
    for it in result:
        if not isinstance(it, dict):
            continue
        if "face" in it and isinstance(it.get("facial_area"), dict):
            out.append(it)
    return out


def _sort_facial_index(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key_fn(it: dict[str, Any]) -> tuple[float, float]:
        fa = it.get("facial_area")
        if not isinstance(fa, dict):
            return (0.0, 0.0)
        return (float(fa.get("y", 0) or 0), float(fa.get("x", 0) or 0))

    return sorted(items, key=key_fn)


def _ground_truth_full_image(
    pstr: str,
    deepface: Any,
    *,
    output: GroundTruthOutputConfig,
    include_facenet512: bool,
    action_list: list[str],
    detector_backend: str,
    model_name: str,
) -> str:
    need_represent = bool(include_facenet512 or output.bbox or output.is_real)
    rep_out: list | object | None = None
    if need_represent:
        rep_out = deepface.represent(  # type: ignore[no-untyped-call]
            img_path=pstr,
            model_name=model_name,
            enforce_detection=True,
            detector_backend=detector_backend,
        )
        if include_facenet512:
            r0 = _first_face(rep_out)
            emb = r0.get("embedding") if r0 else None
            if not isinstance(emb, list) or len(emb) != 512:
                raise ValueError("embedding not length 512 when include_facenet512 is true")
    is_real_spoof: bool | None = None
    ex_for_spoof = getattr(deepface, "extract_faces", None)
    if output.is_real and callable(ex_for_spoof):
        try:
            r_s = ex_for_spoof(  # type: ignore[operator]
                **_kwargs_for_callable(
                    ex_for_spoof,
                    {
                        "img_path": pstr,
                        "target_size": (224, 224),
                        "detector_backend": detector_backend,
                        "enforce_detection": True,
                        "anti_spoofing": True,
                    },
                )
            )
            its = _normalize_extracted_items(r_s)
            if its and its[0].get("is_real") is not None:
                is_real_spoof = _coerce_is_real(its[0].get("is_real"))
        except Exception:  # noqa: BLE001
            logger.debug("anti_spoofing extract_faces in fallback failed for %s", pstr, exc_info=True)
    an_out = None
    if action_list:
        kw: dict[str, Any] = {
            "img_path": pstr,
            "actions": list(action_list),
            "enforce_detection": True,
            "detector_backend": detector_backend,
        }
        if output.is_real:
            kw["anti_spoofing"] = True
        an_out = deepface.analyze(**kw)  # type: ignore[no-untyped-call]
    fa = _bbox_from_represent(rep_out) if rep_out is not None else None
    one = build_one_face_from_deepface(
        rep_out,
        an_out,
        output=output,
        include_facenet512=include_facenet512,
        actions=tuple(action_list),
        bbox_from_detector=fa,
        is_real_value=is_real_spoof,
    )
    return build_indexed_ground_truth_string([one])


def extract_ground_truth_string(
    image_path: str,
    deepface: Any,
    *,
    output: GroundTruthOutputConfig,
    include_facenet512: bool,
    actions: tuple[str, ...],
    detector_backend: str,
    model_name: str,
    max_faces: int,
) -> str:
    """
    Run detection + per-crop `represent` / `analyze` and return `ground_truth` JSON
    (only the indexed `{"0": {...}, "1": {...}}` string).
    """
    pstr = str(image_path)
    action_list = list(actions) if actions else []
    need_represent = bool(include_facenet512 or output.bbox or output.is_real)
    if not need_represent and not action_list:
        raise ValueError("need represent and/or at least one analyze action")
    ex_fn = getattr(deepface, "extract_faces", None)
    items: list[dict[str, Any]] = []
    if callable(ex_fn):
        try:
            ex_kw: dict[str, Any] = {
                "img_path": pstr,
                "target_size": (224, 224),
                "detector_backend": detector_backend,
                "enforce_detection": True,
            }
            if output.is_real:
                ex_kw["anti_spoofing"] = True
            raw = ex_fn(**_kwargs_for_callable(ex_fn, ex_kw))  # type: ignore[operator]
            items = _sort_facial_index(_normalize_extracted_items(raw))[: max(0, max_faces)]
        except Exception as e:  # noqa: BLE001
            logger.warning("extract_faces failed for %s, trying full-image path: %s", pstr, e)
            items = []
    if not items:
        return _ground_truth_full_image(
            pstr,
            deepface,
            output=output,
            include_facenet512=include_facenet512,
            action_list=action_list,
            detector_backend=detector_backend,
            model_name=model_name,
        )
    aspoof = bool(output.is_real)
    # Per-crop numpy + detector_backend=skip is unreliable: attributes collapse to the same
    # pseudo-labels across images. Prefer `analyze` on the full file path, or a temp file path
    # for a crop, with a real detector.
    an_single: Any | None = None
    an_multi: list[dict[str, Any]] | None = None
    if action_list and len(items) == 1:
        an_single = _analyze_on_image_path(
            deepface,
            pstr,
            action_list=action_list,
            detector_backend=detector_backend,
            anti_spoof=aspoof,
        )
    elif action_list and len(items) > 1:
        am = _analyze_on_image_path(
            deepface,
            pstr,
            action_list=action_list,
            detector_backend=detector_backend,
            anti_spoof=aspoof,
        )
        if isinstance(am, list) and len(am) == len(items):
            an_multi = [x for x in am if isinstance(x, dict)]
            if len(an_multi) != len(items):
                an_multi = None
        else:
            an_multi = None

    faces: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        fa = item.get("facial_area")
        bbox_d = _bbox_from_facial_area(fa) if isinstance(fa, dict) else None
        face_img = item.get("face")
        if face_img is None:
            continue
        rep_c: list | object | None = None
        if need_represent and face_img is not None:
            rep_c = deepface.represent(  # type: ignore[no-untyped-call]
                img_path=face_img,
                model_name=model_name,
                enforce_detection=False,
                detector_backend="skip",
            )
        an_c: Any = None
        if action_list and face_img is not None:
            if an_single is not None and len(items) == 1:
                an_c = an_single
            elif an_multi is not None:
                an_c = [an_multi[idx]]
            else:
                fd, tmp = tempfile.mkstemp(suffix=".png")
                try:
                    os.close(fd)
                    _write_face_array_png(tmp, face_img)
                    an_c = _analyze_on_image_path(
                        deepface,
                        tmp,
                        action_list=action_list,
                        detector_backend=detector_backend,
                        anti_spoof=aspoof,
                    )
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        ir = _coerce_is_real(item.get("is_real")) if output.is_real else None
        faces.append(
            build_one_face_from_deepface(
                rep_c,
                an_c,
                output=output,
                include_facenet512=include_facenet512,
                actions=tuple(action_list),
                bbox_from_detector=bbox_d,
                is_real_value=ir,
            )
        )
    if not faces:
        return _ground_truth_full_image(
            pstr,
            deepface,
            output=output,
            include_facenet512=include_facenet512,
            action_list=action_list,
            detector_backend=detector_backend,
            model_name=model_name,
        )
    return build_indexed_ground_truth_string(faces)
