# scripts/

Build and CI-style scripts for the corpus repo.

## Files

- **`pre-push`** — Git pre-push hook. Runs the test suite when a push touches
  Python files under `sweep/`, `tools/`, `corpora/`, `scripts/`, `corpus/`, or
  any top-level `*.py`. Refuses the push on any failure. Bypass with
  `--no-verify`.

  **Install (one-line, per repo clone):**
  ```bash
  git config core.hooksPath scripts
  ```
  After this, `git push` will invoke `scripts/pre-push` automatically.
  Override the Python interpreter via `PYTHON=…` env var (default:
  `~/envs/torch211/bin/python`).

  **Why this exists:** approved by Peng 2026-05-07 to replace
  discipline-only test-running. Test failures had no mechanical block before
  this hook — a commit could push to main with broken tests and nothing
  would catch it pre-push.

- **`build-nightly-from-source.sh`** — builds PyTorch nightly from source.
  See script header for usage.

- **`setup_env.sh`** — environment setup helper.
