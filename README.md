# DeepFace → verl → Gemma 4 (RL)

Face images → **DeepFace** (optional Facenet512 + selected `analyze` fields) as **pseudo–ground truth** → Parquet for **[verl](https://github.com/verl-project/verl)** → `compute_score` reward → (optional) SFT + RL. Training uses a **Hugging Face** checkpoint (e.g. `google/gemma-4-E2B-it`); LiteRT [`.litertlm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) is for inference export after training.

**Ethics:** use only data you may process; minimize sensitive attributes in YAML when you can.

## Master config (single YAML)

1. Copy [configs/pipeline.example.yaml](configs/pipeline.example.yaml) to `configs/pipeline.yaml` (gitignored) and edit:
   - **`hf_model_id`**: e.g. `google/gemma-4-E2B-it`, `google/gemma-4-4B-it`, or any Hub id you train in verl.
   - **`deepface.attributes`**: per-field `true` / `false` for what goes into GT and prompts (`age`, `gender`, `emotion`, `race`).
   - **`deepface.include_facenet512`**: include embedding in GT or not.
   - **`data.*`**: `num_images`, paths, `train_ratio` / `val_ratio` / `seed`.
   - **`verl.method`**: e.g. `grpo` or `ppo` (for documentation; actual command comes from the verl version you install).

2. Run scripts with `--config configs/pipeline.yaml` so CLI args are optional overrides.

## Tiny run (≈100 LFW images)

**Dataset:** [scripts/download_face_subset.py](scripts/download_face_subset.py) uses scikit-learn’s LFW fetch (citation required in publications; see [LFW](http://vis-www.cs.umass.edu/lfw/)).

```bash
cp configs/pipeline.example.yaml configs/pipeline.yaml
# edit configs/pipeline.yaml (hf_model, attributes, data.num_images, …)

python scripts/download_face_subset.py --config configs/pipeline.yaml
python scripts/extract_deepface_gt.py --config configs/pipeline.yaml
python scripts/build_verl_parquet.py --config configs/pipeline.yaml --max-rows 100
./scripts/run_verl_rl.sh
# or: CONFIG=configs/pipeline.yaml ./scripts/run_verl_rl.sh
```

Then install **verl** and merge the printed Hydra overrides with the matching official example. See [scripts/verl_config_helper.py](scripts/verl_config_helper.py) and [configs/verl/README.md](configs/verl/README.md).

## Without download (own images)

```bash
python scripts/extract_deepface_gt.py --image-dir /path/to/faces --out data/gt.jsonl --max-images 100
python scripts/build_verl_parquet.py --jsonl data/gt.jsonl --out-dir data/verl
```

**Flags:** `extract` supports `--no-facenet512`, `--no-analyze` (skips all attributes), `--max-images`. `build_verl_parquet` supports `--max-rows`, split ratios, `--config`.

## Reward

- [rewards/face_attr_reward.py](rewards/face_attr_reward.py) — `compute_score`. Weights apply only to terms present in GT (no attribute block → attribute weight off; no 512-d embedding → embedding weight off).
- [configs/grpo_gemma4_e2b.example.yaml](configs/grpo_gemma4_e2b.example.yaml) — small merge fragment; prefer **pipeline** YAML for project-wide settings.

## Optional: SFT cold-start (text)

```bash
pip install trl transformers peft torch accelerate datasets
python scripts/sft_coldstart.py --config configs/pipeline.yaml
# or: --parquet data/verl/train.parquet --out checkpoints/sft-lora --base-model google/gemma-2-2b-it
```

## Optional: LiteRT export

[docs/EXPORT_LITERT.md](docs/EXPORT_LITERT.md)

## Setup

```bash
cd dumbledore
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Tests

```bash
pytest
# or: PYTHONPATH=. python -m pytest
```

## Dataset (for VLM + RL)

See [Data.md](Data.md) for the **pseudo–GT schema**, **Parquet columns**, and how **images** (`extra_info` / `image_path`) line up with **verl** rollouts and rewards.

## Layout

- [dumbledore/pipeline_config.py](dumbledore/pipeline_config.py) — load master YAML
- [dumbledore/gt_schema.py](dumbledore/gt_schema.py) — `DeepFaceGTV1` + `build_user_prompt`
- [dumbledore/deepface_ops.py](dumbledore/deepface_ops.py) — normalize DeepFace outputs
- [scripts/extract_deepface_gt.py](scripts/extract_deepface_gt.py) — DeepFace → JSONL
- [scripts/build_verl_parquet.py](scripts/build_verl_parquet.py) — Parquet for verl
- [scripts/download_face_subset.py](scripts/download_face_subset.py) — LFW subset
- [rewards/face_attr_reward.py](rewards/face_attr_reward.py) — verl reward
- [schema/gt_v1.json](schema/gt_v1.json) — JSON schema reference
