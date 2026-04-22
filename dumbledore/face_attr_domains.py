"""
Allowed string sets for **DeepFace** `analyze()` `dominant_*` outputs and bbox/age shape.

The teacher (extract) and reward compare the model to these same conventions. Values follow
[DeepFace's analyze API](https://github.com/serengil/deepface): `dominant_gender` is "Man" or
"Woman"; `dominant_emotion` is one of seven labels; `dominant_race` includes
``\"latino hispanic\"`` (two words) as a single string.
"""
from __future__ import annotations

# DeepFace.analyze (demography) — dominant_race
DEEPFACE_DOMINANT_RACES: tuple[str, ...] = (
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
)

# DeepFace.analyze — dominant_emotion
DEEPFACE_DOMINANT_EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# DeepFace.analyze — dominant_gender
DEEPFACE_DOMINANT_GENDERS: tuple[str, ...] = ("Man", "Woman")


def _fmt_list(choices: tuple[str, ...], *, quoted: bool = True) -> str:
    if quoted:
        return ", ".join(f'"{c}"' for c in choices)
    return ", ".join(choices)


def user_prompt_line_for_ground_truth_key(key: str) -> str:
    """Single bullet line (leading spaces) for `build_user_prompt` for an enabled per-face key."""
    if key == "bbox":
        return (
            '    - "bbox": [x, y, width, height] with non-negative numbers (integers). '
            "Coordinates are in the **full image** in pixels, origin top-left; "
            "[x, y] is the top-left corner of the face box, width/height the box size, or null"
        )
    if key == "age":
        return '    - "age": non-negative number (use an integer, typical range 0–120; DeepFace may estimate a float; round to int), or null'
    if key == "gender":
        return f'    - "gender": one of: {_fmt_list(DEEPFACE_DOMINANT_GENDERS)} (DeepFace `dominant_gender`), or null'
    if key == "emotion":
        return f'    - "emotion": one of: {_fmt_list(DEEPFACE_DOMINANT_EMOTIONS)} (DeepFace `dominant_emotion`), or null'
    if key == "race":
        return f'    - "race": one of: {_fmt_list(DEEPFACE_DOMINANT_RACES)} (DeepFace `dominant_race`; note two words in \"latino hispanic\"), or null'
    if key == "is_real":
        return '    - "is_real": boolean `true` (likely live) or `false` (likely spoof), or null if unknown'
    return f"    - {key!r}: (see `face_attr_domains`)"


__all__ = [
    "DEEPFACE_DOMINANT_EMOTIONS",
    "DEEPFACE_DOMINANT_GENDERS",
    "DEEPFACE_DOMINANT_RACES",
    "user_prompt_line_for_ground_truth_key",
]
