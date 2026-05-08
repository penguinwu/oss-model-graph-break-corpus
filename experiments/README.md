# Experiments

Full documentation has moved to [docs/running-experiments.md](../docs/running-experiments.md).

## Quick start

```bash
python3 tools/run_experiment.py template > experiments/configs/my-test.json
python3 tools/run_experiment.py validate experiments/configs/my-test.json
python3 tools/run_experiment.py run experiments/configs/my-test.json --dry-run
python3 tools/run_experiment.py run experiments/configs/my-test.json

# For multi-stage launches with reproducibility guardrails (gate → sample → full):
# Requires settings.python_bin + settings.modellib_pins in the config.
python3 tools/derive_sweep_commands.py experiments/configs/my-test.json --stage gate --run
python3 tools/derive_sweep_commands.py experiments/configs/my-test.json --stage sample --run
python3 tools/derive_sweep_commands.py experiments/configs/my-test.json --stage full --run
```

Canonical multi-stage example: [`experiments/configs/ngb-verify-2026-05-07.json`](configs/ngb-verify-2026-05-07.json).
