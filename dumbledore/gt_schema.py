"""
Ground-truth JSON schema (v1) for DeepFace outputs used as verl `ground_truth` strings.

- `facenet512`: 512 floats from `DeepFace.represent(..., model_name="Facenet512")`
- `analyze`: optional subset of `DeepFace.analyze` (first face). Keys depend on `actions=`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"

DEFAULT_ANALYZE_ACTIONS: tuple[str, ...] = ("age", "gender", "emotion", "race")


@dataclass
class DeepFaceGTV1:
    """Canonical ground-truth object; serialize with `to_json` / `to_ground_truth_string`."""

    schema_version: str = SCHEMA_VERSION
    image_path: str = ""
    facenet512: list[float] = field(default_factory=list)
    analyze: dict[str, Any] = field(default_factory=dict)
    deepface_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.facenet512) not in (0, 512):
            raise ValueError("facenet512 must be length 0 or 512")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {self.schema_version!r}")

    def to_ground_truth_string(self) -> str:
        return json.dumps(self.to_flat_dict(), sort_keys=True, ensure_ascii=False)

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "facenet512": self.facenet512,
            "analyze": self.analyze,
        }
        if self.deepface_metadata:
            d["deepface_metadata"] = self.deepface_metadata
        return d

    @classmethod
    def from_ground_truth_string(cls, s: str) -> DeepFaceGTV1:
        data = json.loads(s)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeepFaceGTV1:
        v = str(data.get("schema_version", SCHEMA_VERSION))
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {v!r}")
        fn = data.get("facenet512") or []
        if not isinstance(fn, list):
            raise TypeError("facenet512 must be a list")
        fn_f = [float(x) for x in fn]
        an = data.get("analyze")
        if an is not None and not isinstance(an, dict):
            raise TypeError("analyze must be a dict or null")
        meta = data.get("deepface_metadata")
        if meta is not None and not isinstance(meta, dict):
            raise TypeError("deepface_metadata must be a dict or null")
        return cls(
            schema_version=v,
            image_path=str(data.get("image_path", "")),
            facenet512=fn_f,
            analyze=dict(an or {}),
            deepface_metadata=dict(meta or {}),
        )


# Prompt / response contract: model must emit a single JSON object (no markdown fences).
SYSTEM_INSTRUCTION = (
    "You are a face analysis assistant. Respond with exactly one JSON object, no markdown, "
    "no code fences, no text before or after. Use the same keys as in the user instructions."
)

def build_user_prompt(
    *,
    image_id: str,
    image_path: str,
    analyze_keys: Sequence[str] | None = None,
    include_facenet512: bool = True,
) -> str:
    """Build the user turn for RL/SFT. `image_path` may contain any characters; it is not used as a format string."""
    keys = list(analyze_keys) if analyze_keys is not None else list(DEFAULT_ANALYZE_ACTIONS)
    keys_s = json.dumps(keys)
    fac_lines = ""
    if include_facenet512:
        fac_lines = (
            f'- "facenet512": array of 512 numbers (floats) for the face embedding\n'
        )
        fac_note = 'The "facenet512" array must have length 512. '
    else:
        fac_lines = '- "facenet512": [] (empty array; embedding not used)\n'
        fac_note = ""
    return (
        f"The face image to analyze is identified as: {image_id}\n\n"
        f"Output a JSON object with this exact structure:\n"
        f'- "schema_version": "{SCHEMA_VERSION}"\n'
        f"{fac_lines}"
        f'- "analyze": object with only these keys if present: {keys_s}\n'
        f'- "image_path": {json.dumps(image_path)} (string)\n\n'
        "If you cannot determine a value, use null for that key inside \"analyze\" only; "
        "always include the keys listed for analyze with null when unknown. "
        f"{fac_note}\n\n"
        "Begin the response with { and end with }."
    )
