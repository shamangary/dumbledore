# Optional: export a fine-tuned Hugging Face checkpoint to LiteRT-LM (`.litertlm`)

`verl` produces **PyTorch / Hugging Face** weights. The Hugging Face asset
[litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) is a **pre-built inference** bundle, not a training format.

1. **Finish RL** and save a merged HF checkpoint (LoRA merge if applicable).
2. **Follow Google** [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [litert-torch](https://github.com/google-ai-edge/litert-torch) documentation for `hf_export` / `text_generation` export to LiteRT-LM. Pin tool versions; Gemma-4 support moves quickly—check the latest issues for Gemma-4 and `task="text_generation"`.
3. **Validate** the exported `.litertlm` in the same CLI or sample app you use for deployment (Android, Web, or desktop from the model card’s links).

Re-exporting multimodal (vision) heads may require the exact same encoder wiring as
`google/gemma-4-E2B-it`; treat on-device tests as the source of truth.
