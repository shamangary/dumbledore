"""
Multi-face ground truth: one JSON object per `ground_truth` string with **string indices**
`"0"`, `"1"`, ... mapping to per-face field objects. No file path, embeddings, or schema
version in the string.
"""
from __future__ import annotations

import json
from typing import Any

from dumbledore.face_attr_domains import user_prompt_line_for_ground_truth_key
from dumbledore.pipeline_config import GroundTruthOutputConfig

# Per-face object field order in prompts
PER_FACE_KEY_ORDER: tuple[str, ...] = ("bbox", "age", "emotion", "gender", "race", "is_real")


def is_face_index_key(k: str) -> bool:
    """True for JSON top-level face slots (`\"0\"`, `\"1\"`, ...)."""
    return isinstance(k, str) and k.isdigit()


SYSTEM_INSTRUCTION = (
    "You are a face analysis assistant. Your inputs are: (1) the **full image**, provided to the "
    "vision encoder from the dataset (this text is only the request; it does not contain image pixels), "
    "and (2) the instructions below. For RL or SFT, the reference answer is the pseudo–ground truth "
    "produced by DeepFace (Parquet column `ground_truth`), not human labels. "
    "Respond with exactly one JSON object, no markdown, no code fences, no text before or after. "
    "Top-level keys must be the string indices "
    '"0", "1", ... — one object per face — as described in the user instructions.'
)


def build_per_face_object(
    output: GroundTruthOutputConfig,
    *,
    bbox: list[int] | None = None,
    age: int | None = None,
    gender: str | None = None,
    emotion: str | None = None,
    race: str | None = None,
    is_real: bool | None = None,
) -> dict[str, Any]:
    """Build one face record (only keys turned on in `output`)."""
    m = output.enabled_map()
    values: dict[str, Any] = {
        "bbox": bbox,
        "age": age,
        "emotion": emotion,
        "gender": gender,
        "race": race,
        "is_real": is_real,
    }
    d: dict[str, Any] = {}
    for k in PER_FACE_KEY_ORDER:
        if m.get(k, False):
            d[k] = values[k]
    return d


def build_indexed_ground_truth_string(ordered_face_dicts: list[dict[str, Any]]) -> str:
    """
    Stringify to `{"0": {...}, "1": {...}, ...}`. Use `[]` input → `{}`.
    """
    indexed: dict[str, Any] = {str(i): f for i, f in enumerate(ordered_face_dicts)}
    return json.dumps(indexed, ensure_ascii=False, separators=(",", ": "))


def _dataset_user_context(*, dataset_key: str | None, output: GroundTruthOutputConfig) -> str:
    """Short dataset note so LFW vs WIDER prompts differ (bbox semantics)."""
    if not dataset_key:
        return ""
    k = dataset_key.lower()
    m = output.enabled_map()
    bbox_on = bool(m.get("bbox", False))
    if k in ("wider_face", "wider"):
        return (
            "Dataset context: WIDER FACE — full in-the-wild images. "
            + (
                "Bounding boxes are in full-image pixel coordinates [x, y, width, height]. "
                if bbox_on
                else ""
            )
            + "Multiple faces per image are common; use indices \"0\", \"1\", … in the usual order "
            "(box top, then x).\n\n"
        )
    if k in ("lfw", "lfw_people", "default"):
        return (
            "Dataset context: LFW — images are small, pre-cropped face pictures (not full scenes). "
            + (
                "Bboxes are still [x, y, width, height] in image pixels, but the frame is tight; "
                "face attributes are often more informative than box geometry.\n\n"
                if bbox_on
                else "Prefer face attributes over spatial detail when uncertain.\n\n"
            )
        )
    return ""


def build_user_prompt(
    *,
    image_id: str,
    output: GroundTruthOutputConfig,
    dataset_key: str | None = None,
) -> str:
    """User turn: describe indexed face objects; no file path in the text."""
    m = output.enabled_map()
    ctx = _dataset_user_context(dataset_key=dataset_key, output=output)
    field_lines: list[str] = []
    for k in PER_FACE_KEY_ORDER:
        if not m.get(k, False):
            continue
        field_lines.append(user_prompt_line_for_ground_truth_key(k))
    if not field_lines:
        example_slot = "Each value under \"0\", \"1\", ... may be {} (this config has no per-face fields)."
    else:
        example_slot = (
            "The object for each index may include only these fields (and may use null for unknown):\n"
            + "\n".join(field_lines)
        )
    return (
        f"{ctx}"
        f"For image id {image_id!r}, output one JSON object. Each **top-level key** must be a string "
        f'of digits only: "0" for the first face (reading order: sort by the top of the bounding box, '
        f"then by **x**), \"1\" for the second, and so on. The value for each key is one object with "
        f"the per-face fields below.\n"
        f"Shape example: {{ \"0\": {{ ... }}, \"1\": {{ ... }} }}.\n"
        f"{example_slot}\n"
        f"If a field is unknown, use null. If there are no faces, output {{}} (empty object). "
        f"Begin the response with {{ and end with }}."
    )


def _full_prompt_text(
    image_id: str,
    output: GroundTruthOutputConfig,
    prompt_config: "PromptConfig | None" = None,
    dataset_key: str | None = None,
) -> str:
    """Full training prompt; ``prompt_config`` comes from pipeline YAML (see ``dumbledore.prompts``)."""
    from dumbledore.pipeline_config import PromptConfig
    from dumbledore.prompts import build_training_prompt

    if prompt_config is not None and not isinstance(prompt_config, PromptConfig):
        raise TypeError("prompt_config must be PromptConfig or None")
    return build_training_prompt(image_id, output, prompt_config, dataset_key=dataset_key)


DEFAULT_ANALYZE_ACTIONS: tuple[str, ...] = ("age", "gender", "emotion", "race")

__all__ = [
    "PER_FACE_KEY_ORDER",
    "SYSTEM_INSTRUCTION",
    "is_face_index_key",
    "build_per_face_object",
    "build_indexed_ground_truth_string",
    "build_user_prompt",
    "DEFAULT_ANALYZE_ACTIONS",
    "_full_prompt_text",
]
