"""Load master pipeline YAML (model, DeepFace attribute toggles, data paths, verl)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keys supported for DeepFace `analyze` in this project (and YAML toggles)
SUPPORTED_ANALYZE_ATTRIBUTES: tuple[str, ...] = ("age", "gender", "emotion", "race")


@dataclass
class DeepFaceConfig:
    model_name: str = "Facenet512"
    detector_backend: str = "opencv"
    include_facenet512: bool = True
    attributes: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in SUPPORTED_ANALYZE_ATTRIBUTES}
    )

    def enabled_actions(self) -> list[str]:
        return [k for k in SUPPORTED_ANALYZE_ATTRIBUTES if self.attributes.get(k, False)]


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
    """HF datasets load params for `download_face_subset`."""
    name: str = "imagefolder"  # placeholder; script uses a built-in default
    config_name: str | None = None
    split: str = "train"
    image_column: str = "image"  # column with PIL/bytes; depends on dataset


@dataclass
class VerlConfig:
    method: str = "grpo"  # grpo, ppo, etc. — used by launcher docs / helper
    example_config: str | None = None  # path inside verl clone, if any
    verl_root: str | None = None  # path to verl project for examples


@dataclass
class PipelineConfig:
    hf_model_id: str = "google/gemma-4-E2B-it"
    deepface: DeepFaceConfig = field(default_factory=DeepFaceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetHubConfig = field(default_factory=DatasetHubConfig)
    verl: VerlConfig = field(default_factory=VerlConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PipelineConfig:
        if not isinstance(d, Mapping):
            raise TypeError("config root must be a mapping")
        df_raw = d.get("deepface") or {}
        data_raw = d.get("data") or {}
        ds_raw = d.get("dataset") or {}
        v_raw = d.get("verl") or {}
        # Merge attribute defaults: YAML must list all keys; missing -> False
        attr_in = df_raw.get("attributes") or {}
        if not isinstance(attr_in, Mapping):
            attr_in = {}
        attributes: dict[str, bool] = {}
        for k in SUPPORTED_ANALYZE_ATTRIBUTES:
            if k in attr_in:
                attributes[k] = bool(attr_in[k])
            else:
                attributes[k] = False
        deepface = DeepFaceConfig(
            model_name=str(df_raw.get("model_name", "Facenet512")),
            detector_backend=str(df_raw.get("detector_backend", "opencv")),
            include_facenet512=bool(df_raw.get("include_facenet512", True)),
            attributes=attributes,
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
        dataset = DatasetHubConfig(
            name=str(ds_raw.get("name", "imagefolder")),
            config_name=ds_raw.get("config_name"),
            split=str(ds_raw.get("split", "train")),
            image_column=str(ds_raw.get("image_column", "image")),
        )
        vcfg = VerlConfig(
            method=str(v_raw.get("method", "grpo")).lower(),
            example_config=v_raw.get("example_config"),
            verl_root=v_raw.get("verl_root"),
        )
        return cls(
            hf_model_id=str(d.get("hf_model_id", "google/gemma-4-E2B-it")),
            deepface=deepface,
            data=data,
            dataset=dataset,
            verl=vcfg,
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
