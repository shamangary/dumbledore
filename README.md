<p align="center">
  <img src="https://raw.githubusercontent.com/shamangary/dumbledore/main/icon.png" alt="Dumbledore — raw face imagery alchemized into structured, verl-ready training data" width="520" />
</p>

<h1 align="center">Dumbledore</h1>

<p align="center">
  <strong>Face images → pseudo–ground truth → JSONL → QC → Parquet → verl + Gemma 4 (SFT & RL).</strong>
</p>

<p align="center">
  <em>Turn messy pixels into disciplined training rows—without losing the story of what each image actually contained.</em>
</p>

---

## Brand & positioning

**Dumbledore** is a small, opinionated pipeline for **vision–language model (VLM)** post-training: it takes a folder of **face images**, runs **DeepFace** as an automated “teacher” (optional Facenet512 + selected `analyze` fields), and materializes **pseudo–ground truth** as **JSONL**, then **Parquet** formatted for **[verl](https://github.com/verl-project/verl)** with a **`compute_score`** reward hook. Training targets a **Hugging Face** checkpoint (e.g. `google/gemma-4-E2B-it`); LiteRT [`.litertlm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) is for inference export after training.

The mascot illustration (`icon.png`) is intentional: **input** is everyday imagery (a face in the wild); **output** is something **boxed, repeatable, and ready for your stack**—structured rows your trainer can ingest, not ad-hoc logs.

**Ethics:** use only data you may process; minimize sensitive attributes in YAML when you can.

---

## What you get

| Artifact | Role |
| --- | --- |
| **JSONL** | One line per sample: path + stringified `ground_truth` from DeepFace |
| **Dataset report** | Stats and QC before you commit to Parquet |
| **Parquet splits** | `prompt`, `ground_truth`, `extra_info` (incl. full-frame `image_path`) for verl |
| **Reward** | [`rewards/face_attr_reward.py`](rewards/face_attr_reward.py) — `compute_score` for RL |

Data semantics and schema details: **[DATA.md](DATA.md)**.

---

## Pipeline stages (one script per step)

**`scripts/`** contains **only** shell (`.sh`). Python CLIs live under [`dumbledore/cli/`](dumbledore/cli/) and are invoked with `python -m dumbledore.cli.<module>`; the `run_stage*.sh` helpers do that for you.

From the **repo root**, with your venv activated, run the numbered `scripts/run_stageN_*.sh` scripts in order. They `cd` to the repo, set `PYTHONPATH`, and point at a pipeline config:

- If **`configs/pipeline.yaml` exists**, it is used.
- Otherwise the scripts fall back to **`configs/pipeline.example.yaml`**.
- To pick a file explicitly: `CONFIG=/path/to/pipeline.yaml ./scripts/run_stage2_extract.sh`  
  (or a path relative to the repo, e.g. `CONFIG=configs/pipeline.yaml`).
- **Example templates** in `configs/`: `pipeline.example.yaml` / `pipeline.lfw.example.yaml` (LFW, **`deepface.ground_truth.bbox: false`**), `pipeline.wider.example.yaml` (WIDER FACE, **`bbox: true`**, full-scene-oriented `prompt`).

| Stage | Script (exact entry point) | What it runs |
| --- | --- | --- |
| **0. Config** | `./scripts/run_stage0_config.sh` | Creates `configs/pipeline.yaml` from the example if missing (`--force` to overwrite) |
| **1. Images** *(skip if you already have a folder)* | `./scripts/run_stage1_images.sh` | [`dumbledore/cli/download_face_subset.py`](dumbledore/cli/download_face_subset.py) — **LFW** (scikit-learn) or **WIDER FACE** subset (HuggingFace streaming; set `dataset.name` in the pipeline YAML) → `data.raw_dir` |
| **2. Extract** | `./scripts/run_stage2_extract.sh` | [`dumbledore/cli/extract_deepface_gt.py`](dumbledore/cli/extract_deepface_gt.py) — DeepFace → JSONL |
| **3. Report** | `./scripts/run_stage3_report.sh` | [`dumbledore/cli/dataset_report.py`](dumbledore/cli/dataset_report.py) — `report.json` + optional PNGs (JSONL first; re-run after stage 4 to add Parquet stats) |
| **4. Parquet** | `./scripts/run_stage4_parquet.sh` | [`dumbledore/cli/build_verl_parquet.py`](dumbledore/cli/build_verl_parquet.py) — JSONL → train/val/test Parquet |
| **5a. SFT** *(optional)* | `./scripts/run_stage5_sft.sh` | [`dumbledore/cli/sft_coldstart.py`](dumbledore/cli/sft_coldstart.py) — LoRA on `train.parquet` |
| **5b. verl hints** | `./scripts/run_stage5_verl.sh` | [`dumbledore/cli/verl_config.py`](dumbledore/cli/verl_config.py) — prints exports + Hydra lines (verl is **not** installed here) |

**Order:** after **JSONL** exists, run **stage 3 (report)** to catch GT issues, then **stage 4 (Parquet)**, then start SFT/verl. You can **run `run_stage3_report.sh` again** after stage 4 so the report includes split Parquet stats. The intended flow is **JSONL → QC report → Parquet → train**.

Shared resolution logic lives in [scripts/pipeline_env.sh](scripts/pipeline_env.sh).

Data flow: **images** → `data/gt.jsonl` → **report (stage 3)** → `data/verl/*.parquet` (stage 4) → **training**. Paths default from `data.*` in the YAML ([DATA.md](DATA.md)). **VLM+RL rows:** `prompt` = text request, `extra_info["image_path"]` = full-image input for the vision stack, `ground_truth` = pseudo label string for the reward.

```mermaid
flowchart LR
  subgraph s1[Stage 1]
    I[images]
  end
  subgraph s2[Stage 2]
    J[gt.jsonl]
  end
  subgraph s3[Stage 3]
    R[Dataset report]
  end
  subgraph s4[Stage 4]
    P[Parquet splits]
  end
  subgraph s5[Stage 5]
    V[Train SFT or verl]
  end
  I --> J --> R --> P --> V
```

### Quick reference: flags you might pass

- **Stage 1–4:** all forward extra args to the Python script (e.g. `--max-images`, `--max-rows 100`, `--no-viz`).
- **Stage 3 (report):** default output is `reports/latest`; set `REPORT_DIR` or add `--out /other/dir` (if you use `--out` on the command line, it overrides `REPORT_DIR`).

**Low-level (same as the shell wrappers):** e.g. `python -m dumbledore.cli.extract_deepface_gt --config configs/pipeline.yaml` from the repo root after `pip install -e .`; the `run_stage*.sh` scripts are the supported interface.

**Requires DeepFace (and a TF backend) for stage 2** — see [Setup](#setup). For stage 3 report figures: `pip install -e ".[report]"` (**pandas** + **pyarrow** are in the main deps).

---

## End-to-end (≈100 LFW images, config-driven)

```bash
cd dumbledore
# . .venv/bin/activate   # if you use a venv

./scripts/run_stage0_config.sh
# edit configs/pipeline.yaml (hf_model_id, deepface, data.*)

./scripts/run_stage1_images.sh
./scripts/run_stage2_extract.sh
./scripts/run_stage3_report.sh
./scripts/run_stage4_parquet.sh --max-rows 100
# optional: ./scripts/run_stage3_report.sh  # again, to add train/val/test Parquet stats to report.json
./scripts/run_stage5_verl.sh
# then start verl with the printed Hydra overrides; optionally ./scripts/run_stage5_sft.sh first
```

## Without the downloader (your own image tree)

```bash
./scripts/run_stage2_extract.sh --image-dir /path/to/faces --out data/gt.jsonl --max-images 100
./scripts/run_stage3_report.sh --jsonl data/gt.jsonl
./scripts/run_stage4_parquet.sh --jsonl data/gt.jsonl --out-dir data/verl
# optional: ./scripts/run_stage3_report.sh --jsonl data/gt.jsonl --parquet-dir data/verl
```

(When you use `--config`, `run_stage4_parquet.sh` and `run_stage2_extract.sh` also load splits and model settings from the YAML.)

## Config fragments

- [configs/grpo_gemma4_e2b.example.yaml](configs/grpo_gemma4_e2b.example.yaml) — small verl merge fragment; **pipeline YAML** is the source of truth for this project.

## Optional: LiteRT export

[docs/EXPORT_LITERT.md](docs/EXPORT_LITERT.md)

## Setup

```bash
cd dumbledore
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install -e ".[lfw]"   # LFW image download (stage 1; sklearn + SciPy)
pip install -e ".[wider]" # WIDER FACE subset (Hub streaming; `datasets` 2.x; non-commercial license)
pip install -r requirements.txt
# DeepFace: pip install -e ".[face]"   # includes TensorFlow + **torch** (torch is required if `is_real` is true in YAML)
# Report figures: pip install -e ".[report]"
```

With `.venv` present, `run_stage1_images.sh` / `run_stage2_*.sh` **prepend** `.venv/bin` to `PATH` (see [scripts/pipeline_env.sh](scripts/pipeline_env.sh)) so the download uses the venv’s Python, not a broken system Anaconda.

**Stages (order):** `run_stage0_config.sh` → `run_stage1_images.sh` (images) → `run_stage2_extract.sh` (DeepFace JSONL) → `run_stage3_report.sh` → `run_stage4_parquet.sh` → `verl_config` / `run_stage5_verl.sh` (verl is external). `PIPELINE_CONFIG` / `CONFIG` default to `configs/pipeline.yaml` or the example; override with `CONFIG=path ./scripts/run_stage2_extract.sh …`.

**Environment issues:** If `pandas` / `sklearn` / `numexpr` print `_ARRAY_API` or fail to import with **NumPy 2.x**, upgrade `pyarrow`, `numexpr`, and `bottleneck` to current wheels (`pip install -U pyarrow numexpr bottleneck`) or use a venv and `pip install -r requirements.txt` so binaries match. Stage 1 (LFW) needs **scikit-learn** with a matching **NumPy/SciPy** stack; if the downloader fails, point stage 2 at an existing folder: `./scripts/run_stage2_extract.sh --image-dir /path/to/faces --out data/gt.jsonl`. Stage 2 needs **TensorFlow** + `deepface` (see `pip install -e ".[face]"` and `requirements.txt`).

## Tests

```bash
pytest
# or: PYTHONPATH=. python -m pytest
```

## Layout

- [DATA.md](DATA.md) — dataset format, teacher vs VLM, Parquet / `extra_info`
- [dumbledore/pipeline_config.py](dumbledore/pipeline_config.py) — load master YAML
- [scripts/pipeline_env.sh](scripts/pipeline_env.sh) — `PIPELINE_CONFIG` / `CONFIG` for all `run_stage*.sh` scripts
- [dumbledore/paths.py](dumbledore/paths.py) — `REPO_ROOT` for CLIs
- [dumbledore/cli/](dumbledore/cli/) — pipeline CLIs (`python -m dumbledore.cli.…`)
- [dumbledore/dataset_report.py](dumbledore/dataset_report.py) — `analyze_jsonl` / `analyze_parquet_dir` (library; CLI is `dumbledore.cli.dataset_report`)
- [dumbledore/face_attr_domains.py](dumbledore/face_attr_domains.py) — allowed strings/ranges for bbox/age/gender/emotion/race in prompts and pseudo-GT (DeepFace `dominant_*`)
- [dumbledore/prompts.py](dumbledore/prompts.py) — `build_training_prompt` (YAML `prompt` + `deepface.ground_truth` → `prompt` column / JSONL)
- [dumbledore/gt_schema.py](dumbledore/gt_schema.py) — `build_user_prompt` + `build_indexed_ground_truth_string`
- [dumbledore/deepface_ops.py](dumbledore/deepface_ops.py) — normalize DeepFace outputs
- [dumbledore/gt_inspect.py](dumbledore/gt_inspect.py) — parse / summarize ground-truth JSON
- [rewards/face_attr_reward.py](rewards/face_attr_reward.py) — `compute_score` for verl
- [schema/gt_sample.json](schema/gt_sample.json) — example `ground_truth` object (stringified in JSONL/Parquet rows)
