"""Driver: run discovery trials for one case.

Usage:
  # Single skill arm, single variant, one trial.
  python -m discovery.run_case --case dbrx_moe_data_dep --variants V0 --n 1

  # Cross-product: 2 skill arms × 2 constraint variants × 3 trials = 12 trials.
  python -m discovery.run_case --case dbrx_moe_data_dep \
      --skills none,/path/to/debug_graph_breaks_skill.md \
      --variants V0,V2 --n 3

`--skills` is a comma-separated list. The literal string `none` means
"no skill loaded" (the bare baseline). Anything else is treated as a path to
a markdown file whose contents get appended to the trial agent's system
prompt via `--append-system-prompt-file`. Trial labels are
`<skill-tag>_<variant>_<idx>`, where `<skill-tag>` is `noskill` for `none` or
the file's stem (e.g. `debug_graph_breaks_skill`).
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from discovery.runner import run_trial
from discovery.variants import ALL_VARIANTS

DISCOVERY_RESULTS = Path("/tmp/discovery-runs")


def _parse_skills(skills_arg: str) -> list[tuple[str, Path | None]]:
    """Parse the comma-separated --skills argument.

    Returns list of (tag, path_or_None) pairs. The literal token `none` is
    converted to (tag="noskill", path=None). Each non-`none` token is
    validated as an existing file.
    """
    skills: list[tuple[str, Path | None]] = []
    for raw in skills_arg.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "none":
            skills.append(("noskill", None))
            continue
        path = Path(token).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--skills: file does not exist: {path}")
        if not path.is_file():
            raise SystemExit(f"--skills: not a file: {path}")
        # Tag = filename stem, sanitized. If the filename is generic
        # (SKILL.md / skill.md), use the parent dir name instead — that's the
        # actual skill identity (e.g. debug-graph-breaks/SKILL.md → "debug_graph_breaks").
        if path.stem.lower() == "skill":
            tag = path.parent.name.replace("-", "_")
        else:
            tag = path.stem.replace("-", "_")
        skills.append((tag, path))
    if not skills:
        raise SystemExit("--skills must be non-empty (use 'none' for the baseline arm)")
    return skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="case id, e.g. dbrx_moe_data_dep")
    parser.add_argument(
        "--variants", required=True,
        help="comma-separated variant ids, e.g. V0,V2,V4,V6",
    )
    parser.add_argument(
        "--skills", default="none",
        help=(
            "comma-separated skill arms. 'none' = no skill loaded; otherwise a "
            "path to a markdown file injected via --append-system-prompt-file. "
            "Default: 'none' (single bare arm)."
        ),
    )
    parser.add_argument("--n", type=int, default=1, help="trials per (skill, variant) cell")
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="per-trial agent timeout in seconds (default 1800 = 30 min)",
    )
    args = parser.parse_args()

    case_mod = importlib.import_module(f"discovery.cases.{args.case}")
    case = case_mod.get_case_spec()

    variants = [ALL_VARIANTS[vid.strip()] for vid in args.variants.split(",")]
    skills = _parse_skills(args.skills)

    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = DISCOVERY_RESULTS / args.case / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir = {run_dir}", flush=True)
    print(
        f"matrix: {len(skills)} skill arms × {len(variants)} variants × {args.n} trials "
        f"= {len(skills) * len(variants) * args.n} trials total",
        flush=True,
    )

    summary: list[dict] = []
    for skill_tag, skill_path in skills:
        for variant in variants:
            for trial_idx in range(args.n):
                trial_label = f"{skill_tag}_{variant.id}_{trial_idx + 1}"
                trial_dir = run_dir / trial_label
                print(f"\n--- {trial_label} ---", flush=True)
                t0 = time.time()
                result = run_trial(
                    case, variant, trial_label, trial_dir,
                    timeout_s=args.timeout,
                    skill_prompt_file=skill_path,
                )
                dt = time.time() - t0
                print(
                    f"  exit={result.agent_exit_code} elapsed={dt:.1f}s flags={result.flags}",
                    flush=True,
                )
                if result.validation:
                    gb = result.validation.get("graph_break_count")
                    md = result.validation.get("max_diff_compiled_vs_eager_now")
                    print(f"  graph_break_count={gb} max_diff={md}", flush=True)
                if result.perf:
                    print(
                        f"  perf: eager={result.perf.get('eager_ms'):.2f}ms "
                        f"compiled={result.perf.get('compiled_ms'):.2f}ms "
                        f"speedup={result.perf.get('speedup'):.2f}x",
                        flush=True,
                    )
                # Tag the result with skill arm so downstream synthesis can group by it.
                result_dict = result.to_dict()
                result_dict["skill_arm"] = skill_tag
                summary.append(result_dict)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary at {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
