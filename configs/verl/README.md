# verl merge fragments

This folder holds **project-local** hints. Full PPO/GRPO/DAPO jobs are **defined by your installed
[verl](https://github.com/verl-project/verl) version** (see `examples/` in that repo).

- [`../grpo_gemma4_e2b.example.yaml`](../grpo_gemma4_e2b.example.yaml) – reward + data key names only; merge with an official `examples/.../config.yaml`.

**Method choice:** set `verl.method` in your pipeline YAML (e.g. [`../pipeline.example.yaml`](../pipeline.example.yaml) or the LFW / WIDER examples next to it) and follow verl’s guide for that algorithm (PPO, GRPO, etc.). The repo does not ship a runnable verl config because APIs change by release.

Run (same thing):

```bash
./scripts/run_stage5_verl.sh
```

Then paste the printed Hydra overrides onto the `python -m verl...` line from the matching verl example.
