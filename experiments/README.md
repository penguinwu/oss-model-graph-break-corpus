# Experiments

Full documentation has moved to [docs/running-experiments.md](../docs/running-experiments.md).

## Quick start

```bash
python3 tools/run_experiment.py template > experiments/configs/my-test.json
python3 tools/run_experiment.py validate experiments/configs/my-test.json
python3 tools/run_experiment.py run experiments/configs/my-test.json --dry-run
python3 tools/run_experiment.py run experiments/configs/my-test.json
```
