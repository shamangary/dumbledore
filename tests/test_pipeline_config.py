import tempfile
from pathlib import Path

from dumbledore.pipeline_config import PipelineConfig, load_pipeline_config


def test_load_example_roundtrip() -> None:
    text = """
hf_model_id: google/gemma-4-4B-it
deepface:
  include_facenet512: false
  attributes:
    age: true
    gender: false
    emotion: false
    race: false
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
        assert c.deepface.enabled_actions() == ["age"]
    finally:
        p.unlink()


def test_from_dict_defaults_attributes_false() -> None:
    c = PipelineConfig.from_dict(
        {
            "hf_model_id": "m",
            "deepface": {"model_name": "Facenet512", "include_facenet512": True, "attributes": {"age": True}},
        }
    )
    assert c.deepface.attributes["gender"] is False
    assert c.deepface.enabled_actions() == ["age"]
