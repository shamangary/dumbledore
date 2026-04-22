"""
Training-time text prompt: built from `prompt` in the pipeline YAML plus `deepface.ground_truth` flags.

Used for JSONL `prompt`, Parquet `prompt`, and must stay aligned with the pseudo-`ground_truth` schema.
"""
from __future__ import annotations

import hashlib
import re

from dumbledore.gt_schema import SYSTEM_INSTRUCTION, build_user_prompt
from dumbledore.pipeline_config import (
    SUPPORTED_GROUND_TRUTH_KEYS,
    GroundTruthOutputConfig,
    PipelineConfig,
    PromptConfig,
)

_USER_PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Prose should mention a field when it is on and there is no JSON example with that key
_PROSE_MENTION: dict[str, re.Pattern[str]] = {
    "bbox": re.compile(r"\bbbox\b", re.IGNORECASE),
    "age": re.compile(r"\bage\b", re.IGNORECASE),
    "gender": re.compile(r"\bgender\b", re.IGNORECASE),
    "emotion": re.compile(r"\bemotion\b", re.IGNORECASE),
    "race": re.compile(r"\brace\b", re.IGNORECASE),
    "is_real": re.compile(r"\bis_real\b", re.IGNORECASE),
}


def dataset_prompt_key(dataset_name: str) -> str:
    """
    Normalize ``dataset.name`` from YAML to a key for ``prompt.by_dataset`` and auto user text.

    Returns ``lfw`` | ``wider_face`` | a lowercased/stripped name for custom datasets.
    """
    n = (dataset_name or "").strip().lower().replace("-", "_")
    if n in ("", "lfw", "lfw_people", "default"):
        return "lfw"
    if n in ("wider", "wider_face"):
        return "wider_face"
    return n


def get_effective_prompt_config(cfg: PipelineConfig) -> PromptConfig:
    """Merge top-level ``prompt`` with ``prompt.by_dataset`` for ``cfg.dataset.name``."""
    p = cfg.prompt
    k = dataset_prompt_key(cfg.dataset.name)
    sub = p.by_dataset.get(k) if p.by_dataset else None
    if sub is None:
        return PromptConfig(
            system_instruction=p.system_instruction,
            user=p.user,
        )
    return PromptConfig(
        system_instruction=sub.system_instruction if sub.system_instruction is not None else p.system_instruction,
        user=sub.user if sub.user is not None else p.user,
    )


def list_prompt_ground_truth_mismatches(cfg: PipelineConfig) -> list[str]:
    """
    If ``prompt.user`` is **empty** (after `get_effective_prompt_config`), the user turn is
    **auto-generated** from ``deepface.ground_truth`` — it always matches; returns ``[]``.

    If ``prompt.user`` is **non-empty** (fully custom, including ``by_dataset``), check for obvious
    contradictions: a JSON field name in the template that is disabled in the YAML, or a field
    enabled in the YAML with no mention and no example ``\"key\":`` in the text.
    """
    eff = get_effective_prompt_config(cfg)
    user = (eff.user or "").strip()
    if not user:
        return []
    m = cfg.deepface.ground_truth.enabled_map()
    out: list[str] = []
    u = user
    # `prompt.user` may double braces `{{` / `}}` for Python str.format; treat like JSON when scanning keys
    u_json_scan = u.replace("{{", "{").replace("}}", "}")
    for key in SUPPORTED_GROUND_TRUTH_KEYS:
        json_key = re.compile(rf'"\s*{re.escape(key)}\s*"\s*:', re.IGNORECASE)
        in_json = bool(json_key.search(u_json_scan))
        on = bool(m.get(key, False))
        if not on and in_json:
            out.append(
                f'prompt.user contains JSON key "{key}" but deepface.ground_truth.{key} is false; '
                "remove the key from the template or set it true."
            )
            continue
        if not on:
            continue
        if in_json:
            continue
        pat = _PROSE_MENTION.get(key)
        if pat is not None and not pat.search(u):
            out.append(
                f"deepface.ground_truth.{key} is true but prompt.user has no {key!r} in prose and no "
                f'\\"{key}\\" JSON key in examples; align the two.'
            )
    return out


def image_id_for_path(path: str) -> str:
    """Stable per-file id used in prompts (Parquet/JSONL), not a label."""
    h = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"img_{h}"


def build_training_prompt(
    image_id: str,
    output: GroundTruthOutputConfig,
    prompt_cfg: PromptConfig | None,
    *,
    dataset_key: str | None = None,
) -> str:
    """
    Full text `prompt` column: system block + user block.

    YAML `prompt.user` (if set and non-empty) may use `{image_id}`. No other `{name}` is allowed
    (KeyError) so people do not depend on undefined placeholders.

    When `prompt.user` is auto-generated, ``dataset_key`` (from ``dataset.name``) tweaks the user
    text (LFW vs WIDER bbox semantics). Custom `user` strings ignore ``dataset_key``.
    """
    if prompt_cfg and (prompt_cfg.system_instruction or "").strip():
        system = (prompt_cfg.system_instruction or "").strip()
    else:
        system = SYSTEM_INSTRUCTION

    if prompt_cfg and (prompt_cfg.user or "").strip():
        u = (prompt_cfg.user or "").strip()
        user_text = _format_user_template(u, image_id)
    else:
        user_text = build_user_prompt(image_id=image_id, output=output, dataset_key=dataset_key)
    return f"{system}\n\n{user_text}"


def _format_user_template(user_raw: str, image_id: str) -> str:
    """Return user text; only ``{image_id}`` is a valid template field."""
    if not _USER_PLACEHOLDER.search(user_raw):
        return user_raw
    allowed = {"image_id"}
    for m in _USER_PLACEHOLDER.finditer(user_raw):
        if m.group(1) not in allowed:
            raise KeyError(
                f"Invalid placeholder in prompt.user: {m.group(0)!r}; only {{image_id}} is supported"
            )
    try:
        return user_raw.format(image_id=image_id)
    except KeyError as e:
        raise KeyError("prompt.user may only use {image_id} as placeholder") from e
