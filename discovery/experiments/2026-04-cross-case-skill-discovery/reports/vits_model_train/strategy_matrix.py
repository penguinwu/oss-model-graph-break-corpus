"""Build strategy-adoption matrix from per-trial agent_diff.patch files.

For each of S1-S11 (defined in findings.md), apply a regex/substring detector
to the trial's diff. Output: matrix of (trial × strategy) → applied?
"""
import json
import re
from pathlib import Path

# Trial dir → canonical label (15 trials; 3 contaminated rogues excluded;
# replaced by parallel relaunch trials).
TRIALS = [
    ("/tmp/runs/vits-r5v9-smoke/noskill_V0_1", "V0/noskill/smoke"),
    ("/tmp/runs/vits-r5v9-batch/wave1a/SKILL_V0_1", "V0/SKILL/wave1a"),
    ("/tmp/runs/vits-r5v9-batch2/waveA_noskill/noskill_V2_1", "V2/noskill/waveA"),
    ("/tmp/runs/vits-r5v9-batch/wave1a/SKILL_V2_1", "V2/SKILL/wave1a"),
    ("/tmp/runs/vits-r5v9-batch/wave1a/SKILL_V4_1", "V4/SKILL/wave1a"),
    ("/tmp/runs/vits-r5v9-parallel-relaunch-2026-04-28/noskill_V4_parallel", "V4/noskill/parallel"),
    ("/tmp/runs/vits-r5v9-batch2/waveA_SKILL/SKILL_V6_1", "V6/SKILL/waveA"),
    ("/tmp/runs/vits-r5v9-parallel-relaunch-2026-04-28/noskill_V6_parallel", "V6/noskill/parallel"),
    # V8 trials live in the parallel-runner experiment (Apr 27-28)
    ("/tmp/runs/parallel-vits-validation/step3/SKILL_V8_1", "V8/SKILL/step3"),
    ("/tmp/runs/parallel-vits-validation/step2/noskill_V8_1", "V8/noskill/step2"),
    ("/tmp/runs/parallel-vits-validation/step3/noskill_V8_1", "V8/noskill/step3"),
    ("/tmp/runs/vits-r5v9-smoke/noskill_V9_1", "V9/noskill/smoke s1"),
    ("/tmp/runs/vits-r5v9-batch2/waveA_SKILL/SKILL_V9_1", "V9/SKILL/waveA s1"),
    ("/tmp/runs/vits-r5v9-batch2/waveB_v9_seed2/noskill_V9_1", "V9/noskill/waveB s2"),
    ("/tmp/runs/vits-r5v9-parallel-relaunch-2026-04-28/SKILL_V9_parallel", "V9/SKILL/parallel"),
]


# Strategy detectors: each returns True if the strategy was applied in the diff.
def detect_S1(diff: str) -> bool:
    """Remove @torch.jit.script — should appear as a deletion line."""
    return bool(re.search(r"^-\s*@torch\.jit\.script", diff, re.MULTILINE))


def detect_S2(diff: str) -> bool:
    """Replace boolean-indexed scatter with torch.where over clamped inputs.
    Detector accepts either `torch.clamp(... tail_bound)` or `<tensor>.clamp(... tail_bound)`."""
    has_where_added = bool(re.search(r"^\+.*torch\.where\(", diff, re.MULTILINE))
    has_clamp = bool(re.search(r"^\+.*\.clamp\([^)]*tail_bound", diff, re.MULTILINE)) or \
                bool(re.search(r"^\+.*torch\.clamp\([^)]*tail_bound", diff, re.MULTILINE))
    return has_where_added and has_clamp


def detect_S3(diff: str) -> bool:
    """Replace np.log/exp with math.log/exp."""
    np_removed = bool(re.search(r"^-.*np\.(log|exp)\(", diff, re.MULTILINE))
    math_added = bool(re.search(r"^\+.*math\.(log|exp)\(", diff, re.MULTILINE))
    return np_removed and math_added


def detect_S4(diff: str) -> bool:
    """Drop torch_compilable_check calls."""
    return bool(re.search(r"^-\s*torch_compilable_check", diff, re.MULTILINE))


def detect_S5(diff: str) -> bool:
    """clamp(discriminant, min=0) or discriminant.clamp(min=0) — replaces sqrt-of-negative assert."""
    return bool(re.search(r"^\+.*torch\.clamp\(\s*discriminant", diff, re.MULTILINE)) or \
           bool(re.search(r"^\+.*discriminant.*\.clamp\(\s*min\s*=\s*0", diff, re.MULTILINE))


def detect_S6(diff: str) -> bool:
    """Declare-and-flip capture_scalar_outputs OR similar dynamo config."""
    return bool(re.search(
        r"^\+.*(_dynamo\.config\.|torch\._dynamo\.config\.|capture_scalar_outputs|capture_dynamic_output_shape_ops)",
        diff, re.MULTILINE))


def detect_S7(diff: str) -> bool:
    """Static cap on a data-dep arange/sequence size. Captures variations:
    _MAX_FRAMES_PER_TOKEN, _MAX_TOKEN_DURATION, MAX_PREDICTED_LENGTH, etc.
    Also: any `<var>.clamp(max=<UPPER>)` pattern that's an upstream of an arange."""
    constant_def = bool(re.search(
        r"^\+.*\b(_?MAX_[A-Z_]+|MAX_[a-z_]+_[a-z_]+).*=\s*\d+",
        diff, re.MULTILINE))
    cap_pattern = bool(re.search(
        r"^\+.*(predicted_lengths|duration|frames).*\.clamp\(\s*max\s*=",
        diff, re.MULTILINE))
    return constant_def or cap_pattern


def detect_S8(diff: str) -> bool:
    """torch.compiler.is_compiling() guard."""
    return bool(re.search(r"^\+.*torch\.compiler\.is_compiling\(\)", diff, re.MULTILINE))


def detect_S9(diff: str) -> bool:
    """Pre-compute as __init__ cached attribute (e.g. self.num_channels_tensor = ...).
    Heuristic: look for new attribute assignments in __init__."""
    return bool(re.search(
        r"^\+.*self\.(num_channels|hidden_size|_max).*=\s*(torch\.|self\.)",
        diff, re.MULTILINE))


def detect_S10(diff: str) -> bool:
    """torch._check / torch._check_is_size to constrain unbacked symints."""
    return bool(re.search(r"^\+.*torch\._check(_is_size)?\(", diff, re.MULTILINE))


def detect_S11(diff: str) -> bool:
    """random.random() instead of np.random.uniform — Python's random module."""
    np_removed = bool(re.search(r"^-.*np\.random\.uniform", diff, re.MULTILINE))
    py_random = bool(re.search(r"^\+.*\brandom\.(random|uniform)\(", diff, re.MULTILINE))
    torch_rand = bool(re.search(r"^\+.*torch\.rand\(", diff, re.MULTILINE))
    return np_removed and (py_random or torch_rand)


DETECTORS = {
    "S1": ("jit-script-remove", detect_S1),
    "S2": ("scatter→where+clamp", detect_S2),
    "S3": ("np→math", detect_S3),
    "S4": ("drop torch_compilable_check", detect_S4),
    "S5": ("clamp discriminant", detect_S5),
    "S6": ("declared dynamo config flip", detect_S6),
    "S7": ("static-cap arange size", detect_S7),
    "S8": ("torch.compiler.is_compiling guard", detect_S8),
    "S9": ("__init__ cached attr", detect_S9),
    "S10": ("torch._check unbacked", detect_S10),
    "S11": ("random.random not np.random.uniform", detect_S11),
}


def analyze_trial(trial_dir: str, label: str) -> dict:
    diff_path = Path(trial_dir) / "agent_diff.patch"
    result_path = Path(trial_dir) / "result.json"
    diff = diff_path.read_text() if diff_path.exists() else ""
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    val = result.get("validation") or {}
    fix_status = val.get("fix_status", "?")
    gb = (val.get("details") or {}).get("gb_under_canonical_inputs")
    perf = result.get("perf") or {}
    speedup = perf.get("speedup")
    strategies = {sid: detector(diff) for sid, (_, detector) in DETECTORS.items()}
    return {
        "label": label,
        "fix_status": fix_status,
        "gb": gb,
        "speedup": speedup,
        "diff_lines": diff.count("\n"),
        "strategies": strategies,
    }


def main():
    rows = [analyze_trial(d, l) for d, l in TRIALS]
    print(f"{'Trial':<28}  {'fix':<7}  {'gb':<3}  {'sp':<5}  {'diff':<5}  " + "  ".join(DETECTORS.keys()))
    print("-" * 120)
    for r in rows:
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "—"
        gb = str(r["gb"]) if r["gb"] is not None else "?"
        marks = "  ".join(" ✓" if r["strategies"][s] else " ·" for s in DETECTORS)
        print(f"{r['label']:<28}  {r['fix_status']:<7}  {gb:<3}  {sp:<5}  {r['diff_lines']:<5}  {marks}")
    print()
    # Per-strategy adoption count
    print("\nStrategy adoption summary:")
    for sid, (name, _) in DETECTORS.items():
        count = sum(1 for r in rows if r["strategies"][sid])
        general_count = sum(1 for r in rows if r["strategies"][sid] and r["fix_status"] == "general")
        print(f"  {sid} ({name}): {count}/15 trials, of which {general_count} general")
    # Save JSON for downstream
    Path("/tmp/strategy_matrix.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
