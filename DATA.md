# Dataset for VLM + RL (this repo)

This document describes the **artificial training data** produced by the Dumbledore pipeline: face images, **pseudo–ground truth** from [DeepFace](https://github.com/serengil/deepface), and **verl**-ready Parquet used when you post-train a **vision–language model** (VLM) such as Gemma 4 with reinforcement learning. The “labels” are **not** human annotations; they are **whatever DeepFace returns** for each image, under the settings in the pipeline YAML (default template [`configs/pipeline.example.yaml`](configs/pipeline.example.yaml); see also [`configs/pipeline.lfw.example.yaml`](configs/pipeline.lfw.example.yaml) and [`configs/pipeline.wider.example.yaml`](configs/pipeline.wider.example.yaml)).

## Ground truth vs VLM input (read this first)

**Building GT (offline, teacher stack):** each **sample** is one image file. DeepFace runs an internal **face detector** (e.g. OpenCV / MTCNN per your `detector_backend`), then **recognition** on the crop (Facenet512 embedding) and **attribute** heads (`analyze`). Conceptually:

```text
dataset  -->  one sample  -->  [ detector  -->  face box / align  ]  -->  [ recognition: embed + attributes ]
   |              |                        |
   +-- image file on disk (full frame)      +-- GT string encodes only recognition outputs, not the box
```

**Training the VLM (student):** the model does **not** use DeepFace’s detector. The VLM is fed the **entire image file** at `image_path` (as loaded by your verl / processor, typically resize/patch the **whole** frame, not a DeepFace export crop). The **reward** still compares the model’s text output to the same pseudo-GT JSON. So:

- The dataset is **suitable for whole-image VLM** because `extra_info` includes `image_path` to that **full** file and `whole_image: true` (see Parquet below).
- **Alignment** between GT and vision is: “this file produced this DeepFace output.” The student must learn from full context; scenes with tiny/occluded faces are harder and may not match the teacher’s assumptions.

**Practical tips:** prefer images where a single face is clearly present if you want attribute GT to be meaningful; the VLM still sees the **full** image, not a detector crop. If **`gt.jsonl` pseudo-labels are almost identical** across many different files, check (1) **image resolution** — LFW with a low sklearn `resize` and tiny PNGs; the downloader now uses `resize=1.0` and **upscales** so the short side is at least **160px** before saving. (2) **Extraction** — older logic ran `DeepFace.analyze` on a numpy crop with `detector_backend="skip"`, which often collapses attributes; the pipeline now calls **`analyze` on the full image path** (single face) or on a **temp file** of the crop (multi-face) with a real detector.

## Stages: ASCII overview

**Pipeline stages (files on disk, up to verl):** JSONL is checked with **dataset_report** (stage 3) before **build_verl_parquet** (stage 4); see [README](README.md).

```text
+-------------+     +------------------+     +-----------+     +----------------+     +----------------------+     +------------------+
|   Images    | --> |  extract_        | --> |  gt.jsonl | --> | dataset_report | --> |  build_verl_         | --> | train/val/       |
|  directory  |     |  deepface_gt.py  |     | (per line) |     |   (JSONL+?)    |     |  parquet.py          |     | test .parquet    |
+-------------+     +------------------+     +-----------+     +----------------+     +----------------------+     +------------------+
  raw_image.*          DeepFace per file        path +            stats / QC               prompt + ground_truth         ready for
                       writes GT string          ground_truth                            + extra_info (path)         verl + VLM loader
```

**What happens inside GT extraction (teacher only, not the VLM):**

```text
  sample = one image file (path stored in row)
         |
         v
  +------------------+
  | DeepFace: detect |  (internal detector_backend; not repeated in VLM)
  +------------------+
         |
         v
  +------------------+     +-------------------------+
  | represent()      |     | analyze()  (if enabled) |
  | (e.g. Facenet512) |   | age, gender, ...        |
  +------------------+     +-------------------------+
         |                            |
         +---------> serialize multi-face JSON `{"0":{...},"1":{...}}` into ground_truth (no path, no embedding)
```

**VLM + RL (student):**

```text
  train.parquet row
         |
         v
  load image_path  --------->  [ VLM: full image tensor  +  tokenized prompt  ]
  (whole file; NOT                     |
   DeepFace crop)                     v
                               generated JSON string
                                     |
                                     v
                         compute_score(solution, ground_truth)
```

## End-to-end flow (narrative)

1. **Images** — Local folder of image files. **Stage 1** can pull a subset via [`dumbledore/cli/download_face_subset.py`](dumbledore/cli/download_face_subset.py): set `dataset.name: lfw` (install `.[lfw]`, scikit-learn) or `dataset.name: wider_face` (install `.[wider]`, `datasets` 2.x; WIDER is **CC BY-NC-ND 4.0**, non-commercial). The downloader streams **WIDER** so you do not need the full Hub archive. You can also use your own image tree. When `prompt.user` is omitted, the built-in user text includes a **dataset line** (LFW ≈ tight face crops, bbox usually off in the example YAMLs; WIDER FACE = full frames with **`ground_truth.bbox: true`**).
2. **DeepFace** — For each file: internal **detector** + **recognition** / `analyze` per config; output serialized as one GT string.
3. **JSONL** — One line per success with `image_path`, `ground_truth`, and **`prompt`** (full text, from `pipeline.yaml` → `prompt` + `deepface.ground_truth` when using `--config`).
4. **Parquet** — Splits: each row has **text** `prompt`, string `ground_truth`, and `extra_info` with **`image_path` (full file)** and **`whole_image: true`** so dataloaders load the same file as the full-frame VLM input.
5. **verl** — Rollout + reward; VLM conditions on **whole image** + text, not on DeepFace’s internal crop.

## Parquet row (VLM + RL / verl)

Each `train|val|test*.parquet` row is the contract for post-training and RL:

| Column | Role |
|--------|--------|
| **`prompt`** | **Text** instruction only: the user request and JSON shape (the **request**; no pixels). |
| **`ground_truth`** | **String** of pseudo–labels: the indexed JSON the student should match (from DeepFace, not hand labels). This is the **target** for SFT/RL; [`compute_score`](rewards/face_attr_reward.py) compares the model’s answer string to it. |
| **`extra_info`** (JSON) | **Image context** for the VLM: `image_path` = absolute path to the **full** frame, `whole_image: true`, plus `verl_rl` — a small machine-readable summary that `text` = column `prompt`, `image` = that path, `label` = column `ground_truth` (pseudo-GT). |

**Rollout:** the trainer loads the image from `extra_info["image_path"]` (or your recipe’s field mapping), tokenizes `prompt` for the text side, and conditions the VLM on **both**. **Do not** put the file path only inside `prompt` to avoid leaking paths; the path lives in `extra_info` for the dataloader as designed.

## Ground truth: multi-face JSON (indexed)

The `ground_truth` column is **one JSON object per sample** (stringified). **Top-level keys** are string indices only: `"0"`, `"1"`, `"2"`, … — each mapping to one face object. There is **no** `schema_version`, `image_path`, `facenet512`, or nested `analyze`—only per-face fields the VLM should predict. The image path lives in the JSONL / Parquet row and in `extra_info`, not in `ground_truth`. With **no** faces, use **`{}`**.

Each face object (under any index) may include a subset of these fields (whichever are enabled in `deepface.ground_truth` in the pipeline YAML):

| Field | Meaning |
|--------|--------|
| `bbox` | `[x, y, width, height]` in **full-image** pixels, or `null`. |
| `age` | Integer, or `null`. |
| `gender` / `emotion` / `race` | Strings from DeepFace `analyze` (`dominant_*`). Allowed values (for matching pseudo-GT) are listed in [dumbledore/face_attr_domains.py](dumbledore/face_attr_domains.py) — e.g. gender `Man`/`Woman`; seven emotions; six race labels including `latino hispanic`. Training prompts (default or YAML `prompt.user`) state these explicitly. Or `null`. |
| `is_real` | `true` = live face, `false` = likely spoof, from DeepFace [anti-spoofing](https://github.com/serengil/deepface) (enable `is_real: true` under `deepface.ground_truth` so extract uses `anti_spoofing=True`); or `null` if the run failed to produce a score. |

Faces are **sorted** by detector output (order used for teach/reward: top-left, **y** then **x**). `max_faces` caps how many entries appear per image. The **512-d** Facenet vector is **never** in `ground_truth`; `include_facenet512` only enforces a valid embedding during extract. See [schema/gt_sample.json](schema/gt_sample.json) for a concrete example of the parsed object.

**Pseudo-GT** means: the “correct answer” in RL is this JSON. [`compute_score`](rewards/face_attr_reward.py) compares the object for each index `"0"`, `"1"`, … (same string key in `ground_truth` and in the model output) and **averages** per-face, per-key scores. Missing or wrong-typed values under an index are scored 0 for that part.

**Ethics and quality:** face attributes and demographics are sensitive; DeepFace is imperfect and biased. Use data you are allowed to process and treat metrics as **best-effort** with respect to this automated supervisor.

## Intermediate format: `gt.jsonl`

Each line is a JSON object, for example:

```json
{
  "image_path": "/abs/path/to/face_0000.png",
  "ground_truth": "{\"0\":{\"bbox\":[...],\"age\":30,...}}",
  "prompt": "…system…\\n\\n…user (request + JSON shape)…",
  "ok": true
}
```

`prompt` is the same text as in Parquet’s `prompt` column: built from **`prompt.system_instruction` / `prompt.user` in the pipeline YAML** (placeholders: `{image_id}` only) plus defaults that follow **`deepface.ground_truth`**. Re-run `build_verl_parquet` with `--config` to refresh Parquet if you only change the YAML. Produced by [`dumbledore/cli/extract_deepface_gt.py`](dumbledore/cli/extract_deepface_gt.py). Failures are optional lines in a log file, not in this JSONL.

## Verl / VLM format: `train.parquet`, `val.parquet`, `test.parquet`

Built by [`dumbledore/cli/build_verl_parquet.py`](dumbledore/cli/build_verl_parquet.py) under `data.parquet_dir` (default `data/verl/`). **Columns** (per row, one per image / sample):

| Column | Type (conceptual) | Role |
|--------|-------------------|------|
| `data_source` | string | E.g. `deepface_face_attr`; for filtering in multi-task setups. |
| `prompt` | string | System + user: output **one** JSON object with top-level keys `"0"`, `"1"`, … and per-face fields enabled in `deepface.ground_truth`. No file path in the prompt. This is **text**; the **image** is not bytes in this column. |
| `ability` | string | E.g. `face_attr_json`. |
| `ground_truth` | string | The same canonical JSON as in JSONL—used as the **verifiable target** in `compute_score`. |
| `extra_info` | string (JSON) | Metadata: `image_id`, **`image_path`** (full frame for the VLM), `modality`, **`whole_image: true`**, `ground_truth_keys`, `include_facenet512`, and **`verl_rl`** — documents that inputs are `prompt` + `image_path`, label is `ground_truth` (pseudo-GT). |

**Important for VLM:** **multimodal training** = **(full image from `image_path`, `prompt` text) → model generates a string** compared to `ground_truth` by the custom reward. The **prompt** does not repeat the file path; use `image_path` in `extra_info` to load pixels. The model may predict one **bbox** per face per index; the vision encoder should still use the full frame unless you change the recipe.

**Splits** — By default, rows are shuffled (fixed `seed`) and split by `train_ratio` / `val_ratio`; the remainder is `test`. You may set `train_ratio + val_ratio` to `1.0` so the test split is empty. If the train split would be empty, the build script can put all rows in train and warn.

## How this connects to RL

- **Policy** — Typically a VLM (e.g. `hf_model_id` in pipeline YAML) produces **one continuation** (string) per **(prompt [+ image])**.
- **Reward** — [`compute_score(data_source, solution_str, ground_truth, extra_info)`](rewards/face_attr_reward.py) parses `solution_str` as JSON, compares to `ground_truth` (same string indices `"0"`, `"1"`, …, per-key inside each object).
- **verl** — Algorithm (GRPO, PPO, …) and trainer config are **not** in this dataset; they come from your installed [verl](https://github.com/verl-project/verl) and are merged with paths from [`dumbledore/cli/verl_config.py`](dumbledore/cli/verl_config.py) and your `configs/pipeline.yaml` (see the `verl` section in the example files under [`configs/`](configs/)).

## Controlling the dataset: pipeline YAMLs

Copy one of the examples to `configs/pipeline.yaml` (often gitignored): minimal [`configs/pipeline.example.yaml`](configs/pipeline.example.yaml) (LFW, **no bbox** in pseudo-GT), [`configs/pipeline.lfw.example.yaml`](configs/pipeline.lfw.example.yaml) (same), or WIDER with bbox on in [`configs/pipeline.wider.example.yaml`](configs/pipeline.wider.example.yaml). The important knobs for *what* gets generated:

- `hf_model_id` — Trained model id (independent of the Parquet schema; used for launch/SFT).
- `deepface.ground_truth.*` — Bools for which per-face keys appear under each index and in the prompt.
- `deepface.max_faces` — Maximum faces per image after sorting (detector + cap).
- `deepface.include_facenet512` — Run Facenet `represent()` during extract and require a 512-d embedding; the **embedding is not** written into `ground_truth`.
- `data.*` — Paths, `num_images` cap, split ratios, `seed`.
- `prompt` — If you set a **custom** `user` string, keep it consistent with `deepface.ground_truth` (e.g. do not list `"bbox":` in sample JSON if `ground_truth.bbox` is `false`). With `user: null`, the user turn is **auto-generated** and always matches. [`extract`](dumbledore/cli/extract_deepface_gt.py) and [`build_verl_parquet`](dumbledore/cli/build_verl_parquet.py) run a **heuristic check** and log warnings; [`dumbledore/prompts.py`](dumbledore/prompts.py) exposes `list_prompt_ground_truth_mismatches` for the same rules.

## Quality check in one run

[`dumbledore/cli/dataset_report.py`](dumbledore/cli/dataset_report.py) loads **JSONL** and/or a **Parquet** directory, prints a text summary, writes **`report.json`**, and (if `matplotlib` and `pillow` are installed and image paths are readable) saves **`image_grid.png`** and **`age_histogram.png`**. **Run the report after JSONL exists (and again after Parquet for split stats) and before SFT or verl training** so you catch bad paths or label skew early. Use the same paths as in your pipeline config. Preferred (see [README](README.md)):

```bash
./scripts/run_stage3_report.sh
```

Or: `python -m dumbledore.cli.dataset_report --config configs/pipeline.yaml --out reports/latest`

## Quick reference: files

| Artifact | Path (defaults) | Purpose |
|----------|-----------------|--------|
| Config | `configs/pipeline*.yaml` | What to extract and where to write. |
| Raw images | `data/raw_images/` (typical) | Input to DeepFace. |
| JSONL | `data/gt.jsonl` | One sample per line, GT strings. |
| Parquet | `data/verl/train|val|test.parquet` | verl + VLM integration. |
| Sample `ground_truth` (parsed) | `schema/gt_sample.json` | Example with two faces; omit keys your pipeline disables; use `{}` when there are no faces. |

## Further reading

- [README](README.md) — Commands and setup.
- [configs/verl/README.md](configs/verl/README.md) — Merging with verl examples.
- [docs/EXPORT_LITERT.md](docs/EXPORT_LITERT.md) — After RL, export to LiteRT (separate from this dataset story).
