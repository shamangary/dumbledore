"""Load master pipeline YAML (model, DeepFace, flat ground-truth keys, data paths, verl)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# DeepFace `analyze` actions (subset of per-face fields in each indexed slot)
SUPPORTED_ANALYZE_ATTRIBUTES: tuple[str, ...] = ("age", "gender", "emotion", "race")

# Per-face keys inside `ground_truth` → `{"0": { ... }, "1": { ... }, ...}` (string indices; order for prompts)
SUPPORTED_GROUND_TRUTH_KEYS: tuple[str, ...] = ("bbox", "age", "emotion", "gender", "race", "is_real")


@dataclass
class GroundTruthOutputConfig:
    """Which per-face fields appear under each top-level index \"0\", \"1\", ... (see `gt_schema.py`)."""

    bbox: bool = True
    age: bool = True
    gender: bool = True
    emotion: bool = True
    race: bool = True
    is_real: bool = False

    def enabled_map(self) -> dict[str, bool]:
        return {
            "bbox": self.bbox,
            "age": self.age,
            "emotion": self.emotion,
            "gender": self.gender,
            "race": self.race,
            "is_real": self.is_real,
        }

    def enabled_list(self) -> list[str]:
        m = self.enabled_map()
        return [k for k in SUPPORTED_GROUND_TRUTH_KEYS if m.get(k, False)]

    def any_attribute_for_analyze(self) -> bool:
        """If false, we may only need `represent` for bbox."""
        return any(
            {
                "age": self.age,
                "gender": self.gender,
                "emotion": self.emotion,
                "race": self.race,
            }.values()
        )


@dataclass
class DeepFaceConfig:
    model_name: str = "Facenet512"
    detector_backend: str = "opencv"
    # Run Facenet `represent()` (512-d is never stored in `ground_truth`; use for quality / detector).
    include_facenet512: bool = True
    # Cap how many face records per image (after sort by y,x).
    max_faces: int = 5
    ground_truth: GroundTruthOutputConfig = field(default_factory=GroundTruthOutputConfig)

    def enabled_analyze_actions(self) -> list[str]:
        g = self.ground_truth
        return [a for a in SUPPORTED_ANALYZE_ATTRIBUTES if getattr(g, a, False)]


@dataclass
class DataConfig:
    num_images: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    raw_dir: str = "data/raw_images"
    jsonl: str = "data/gt.jsonl"
    parquet_dir: str = "data/verl"


@dataclass
class DatasetHubConfig:
    name: str = "lfw"
    config_name: str | None = None
    split: str = "train"
    image_column: str = "image"
    # WIDER (HuggingFace): default Hub id; override if you mirror the dataset
    hf_id: str = "CUHK-CSE/wider_face"


@dataclass
class VerlConfig:
    method: str = "grpo"
    example_config: str | None = None
    verl_root: str | None = None


@dataclass
class PromptConfig:
    """
    Optional override for the training `prompt` (system + user) string.

    - ``system_instruction``: if null/omitted, use the default in `gt_schema.SYSTEM_INSTRUCTION`.
    - ``user``: if null/omitted, build the user block from `deepface.ground_truth` (field list).
            If set, it becomes the user turn; you may use ``{image_id}`` once or more in the string.
    - ``by_dataset``: optional per-dataset keys ``lfw`` and ``wider_face``; non-null fields override
            the top-level ``system_instruction`` / ``user`` when ``dataset.name`` matches (see
            `prompts.get_effective_prompt_config`).
    """

    system_instruction: str | None = None
    user: str | None = None
    by_dataset: dict[str, "PromptConfig"] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    hf_model_id: str = "google/gemma-4-E2B-it"
    deepface: DeepFaceConfig = field(default_factory=DeepFaceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetHubConfig = field(default_factory=DatasetHubConfig)
    verl: VerlConfig = field(default_factory=VerlConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PipelineConfig:
        if not isinstance(d, Mapping):
            raise TypeError("config root must be a mapping")
        df_raw: dict[str, Any] = dict(d.get("deepface") or {}) if d.get("deepface") else {}
        data_raw = d.get("data") or {}
        ds_raw = d.get("dataset") or {}
        v_raw = d.get("verl") or {}

        gt_in = df_raw.get("ground_truth")
        default_gt = {k: True for k in SUPPORTED_GROUND_TRUTH_KEYS} | {"is_real": False}
        if isinstance(gt_in, Mapping):
            ground = GroundTruthOutputConfig(
                bbox=bool(gt_in.get("bbox", default_gt["bbox"])),
                age=bool(gt_in.get("age", default_gt["age"])),
                gender=bool(gt_in.get("gender", default_gt["gender"])),
                emotion=bool(gt_in.get("emotion", default_gt["emotion"])),
                race=bool(gt_in.get("race", default_gt["race"])),
                is_real=bool(gt_in.get("is_real", default_gt["is_real"])),
            )
        else:
            ground = GroundTruthOutputConfig()

        deepface = DeepFaceConfig(
            model_name=str(df_raw.get("model_name", "Facenet512")),
            detector_backend=str(df_raw.get("detector_backend", "opencv")),
            include_facenet512=bool(df_raw.get("include_facenet512", True)),
            max_faces=int(df_raw.get("max_faces", 5)),
            ground_truth=ground,
        )
        data = DataConfig(
            num_images=int(data_raw.get("num_images", 100)),
            train_ratio=float(data_raw.get("train_ratio", 0.8)),
            val_ratio=float(data_raw.get("val_ratio", 0.1)),
            seed=int(data_raw.get("seed", 42)),
            raw_dir=str(data_raw.get("raw_dir", "data/raw_images")),
            jsonl=str(data_raw.get("jsonl", "data/gt.jsonl")),
            parquet_dir=str(data_raw.get("parquet_dir", "data/verl")),
        )
        _hfr = ds_raw.get("hf_id")
        dataset = DatasetHubConfig(
            name=str(ds_raw.get("name", "lfw")),
            config_name=ds_raw.get("config_name"),
            split=str(ds_raw.get("split", "train")),
            image_column=str(ds_raw.get("image_column", "image")),
            hf_id="CUHK-CSE/wider_face" if _hfr is None else str(_hfr),
        )
        vcfg = VerlConfig(
            method=str(v_raw.get("method", "grpo")).lower(),
            example_config=v_raw.get("example_config"),
            verl_root=v_raw.get("verl_root"),
        )
        p_raw = d.get("prompt")
        if isinstance(p_raw, Mapping):
            s_inst = p_raw.get("system_instruction")
            p_user = p_raw.get("user")
            by_sub: dict[str, PromptConfig] = {}
            raw_by = p_raw.get("by_dataset")
            if isinstance(raw_by, Mapping):
                for k, v in raw_by.items():
                    if not isinstance(v, Mapping):
                        continue
                    s2 = v.get("system_instruction")
                    u2 = v.get("user")
                    by_sub[str(k).strip().lower()] = PromptConfig(
                        system_instruction=None if s2 is None else str(s2),
                        user=None if u2 is None else str(u2),
                    )
            prompt = PromptConfig(
                system_instruction=None if s_inst is None else str(s_inst),
                user=None if p_user is None else str(p_user),
                by_dataset=by_sub,
            )
        else:
            prompt = PromptConfig()
        return cls(
            hf_model_id=str(d.get("hf_model_id", "google/gemma-4-E2B-it")),
            deepface=deepface,
            data=data,
            dataset=dataset,
            verl=vcfg,
            prompt=prompt,
        )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Install PyYAML: pip install pyyaml") from e
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not raw:
        return PipelineConfig()
    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping")
    return PipelineConfig.from_dict(raw)
