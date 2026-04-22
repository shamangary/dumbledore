#!/usr/bin/env python3
"""
Optional SFT cold-start: Parquet rows -> TRL SFT on chat-formatted text.
Requires: pip install transformers trl peft torch accelerate datasets pandas pyarrow
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Gemma 4 (incl. E2B) uses Gemma4ClippableLinear wrapping nn.Linear; PEFT only injects on Linear/Conv/etc.,
# so we match the inner submodules, e.g. `.../self_attn/q_proj/linear` (name suffix `.q_proj.linear`).
def _lora_target_modules_for_model(model) -> list[str]:
    for m in model.modules():
        if type(m).__name__ == "Gemma4ClippableLinear":
            return [
                "q_proj.linear",
                "k_proj.linear",
                "v_proj.linear",
                "o_proj.linear",
                "gate_proj.linear",
                "up_proj.linear",
                "down_proj.linear",
            ]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    import argparse

    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None, help="Master pipeline YAML (sets base model + paths)")
    ap.add_argument("--parquet", type=Path, default=None, help="Train parquet; default from config `data.parquet_dir/train.parquet`")
    ap.add_argument("--out", type=Path, default=None, help="Output LoRA dir; default checkpoints/sft-lora if using config")
    ap.add_argument("--base-model", type=str, default=None, help="HF model id; overrides config hf_model_id")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    if a.config is not None:
        from dumbledore.pipeline_config import load_pipeline_config

        cfg = load_pipeline_config(a.config)
        base = a.base_model or cfg.hf_model_id
        pq = a.parquet or (Path(cfg.data.parquet_dir) / "train.parquet")
        outd = a.out or Path("checkpoints") / "sft-lora"
    else:
        if a.parquet is None or a.out is None:
            print("Need --parquet and --out, or use --config", file=sys.stderr)
            return 1
        cfg = None
        pq = a.parquet
        outd = a.out
        base = a.base_model or "google/gemma-2-2b-it"
    a = argparse.Namespace(
        **{**a.__dict__, "parquet": pq, "out": outd, "base_model": base}
    )

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_parquet(a.parquet, engine="pyarrow")
    if a.max_samples is not None:
        df = df.head(a.max_samples)

    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def row_to_text(row) -> str:  # type: ignore[no-untyped-def]
        p, g = str(row["prompt"]), str(row["ground_truth"])
        # system + user are separated by a blank line (from `build_training_prompt`); SFT only needs the user turn
        # here so the chat template can add system if the model card expects a single user message.
        user = p.split("\n\n", 1)[1].lstrip() if "\n\n" in p else p
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": g}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)  # type: ignore[no-untyped-call]

    ds = Dataset.from_dict({"text": [row_to_text(r) for _, r in df.iterrows()]})

    model = AutoModelForCausalLM.from_pretrained(
        a.base_model,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=_lora_target_modules_for_model(model),
        task_type="CAUSAL_LM",
    )
    targs = SFTConfig(
        output_dir=str(a.out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=a.epochs,
        learning_rate=1e-4,
        max_length=2048,
        logging_steps=5,
        dataset_text_field="text",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    params = list(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in params:
        trainer = SFTTrainer(
            model=model,
            args=targs,
            train_dataset=ds,
            peft_config=lora,
            processing_class=tok,
        )
    else:
        trainer = SFTTrainer(
            model=model, args=targs, train_dataset=ds, peft_config=lora, tokenizer=tok
        )
    trainer.train()
    trainer.save_model(str(a.out))
    print("Saved to", a.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print("Install: pip install trl transformers peft torch accelerate datasets", file=sys.stderr)
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
