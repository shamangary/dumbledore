# verl merge fragments

This folder holds **project-local** hints. Full PPO/GRPO/DAPO jobs are **defined by your installed
[verl](https://github.com/verl-project/verl) version** (see `examples/` in that repo).

- [`../grpo_gemma4_e2b.example.yaml`](../grpo_gemma4_e2b.example.yaml) – reward + data key names only; merge with an official `examples/.../config.yaml`.

**Method choice:** set `verl.method` in [`../pipeline.example.yaml`](../pipeline.example.yaml) and follow verl’s guide for that algorithm (PPO, GRPO, etc.). The repo does not ship a runnable verl config because APIs change by release.

Run:

```bash
./scripts/run_verl_rl.sh
# or
python scripts/verl_config_helper.py --config configs/pipeline.yaml
```

Then paste the printed Hydra overrides onto the `python -m verl...` line from the matching verl example.
