import tempfile
from pathlib import Path

from dumbledore.pipeline_config import PipelineConfig, load_pipeline_config


def test_load_example_roundtrip() -> None:
    text = """
hf_model_id: google/gemma-4-4B-it
deepface:
  include_facenet512: false
  ground_truth:
    bbox: true
    age: true
    gender: false
    emotion: false
    race: false
    is_real: false
data:
  num_images: 50
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        f.flush()
        p = Path(f.name)
    try:
        c = load_pipeline_config(p)
        assert c.hf_model_id == "google/gemma-4-4B-it"
        assert c.deepface.include_facenet512 is False
        assert c.deepface.enabled_analyze_actions() == ["age"]
    finally:
        p.unlink()


def test_from_dict_ground_truth() -> None:
    c = PipelineConfig.from_dict(
        {
            "hf_model_id": "m",
            "deepface": {
                "model_name": "Facenet512",
                "include_facenet512": True,
                "ground_truth": {
                    "age": True,
                    "gender": False,
                    "bbox": True,
                    "emotion": False,
                    "race": False,
                },
            },
        }
    )
    assert c.deepface.ground_truth.gender is False
    assert c.deepface.enabled_analyze_actions() == ["age"]
    assert c.deepface.max_faces == 5


def test_prompt_config_in_yaml() -> None:
    c = PipelineConfig.from_dict(
        {
            "hf_model_id": "m",
            "deepface": {"ground_truth": {"age": True, "gender": False, "bbox": False, "emotion": False, "race": False}},
            "prompt": {
                "system_instruction": "Custom system.",
                "user": "For {image_id}, output JSON.",
            },
        }
    )
    assert c.prompt.system_instruction == "Custom system."
    assert c.prompt.user == "For {image_id}, output JSON."


def test_prompt_by_dataset_and_dataset_hf_id() -> None:
    c = PipelineConfig.from_dict(
        {
            "dataset": {"name": "wider_face", "split": "validation", "hf_id": "CUHK-CSE/wider_face"},
            "prompt": {
                "system_instruction": "S",
                "user": "U",
                "by_dataset": {
                    "wider_face": {"system_instruction": "W", "user": None},
                },
            },
        }
    )
    assert c.dataset.name == "wider_face"
    assert c.dataset.split == "validation"
    assert c.dataset.hf_id == "CUHK-CSE/wider_face"
    assert "wider_face" in c.prompt.by_dataset
    assert c.prompt.by_dataset["wider_face"].system_instruction == "W"
    assert c.prompt.by_dataset["wider_face"].user is None
