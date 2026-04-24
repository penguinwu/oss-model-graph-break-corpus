# discovery/

Per-case discovery agent. Sibling to `sweep/`.

`sweep/` is breadth-first across the corpus — "did each model break, how badly."
`discovery/` is depth-first per case — "what strategies exist for this case, what do they cost."

## Status

Scaffolding. See `design.md` (mirror of `/tmp/discovery_agent_design.md` v0.2) for the full design.

## Layout

```
discovery/
├── README.md          # this file
├── design.md          # design doc v0.2
├── perf.py            # measure_perf primitive (eager_ms, compiled_ms, peak_mem, compile_s)
├── runner.py          # config-driven trial runner (M variants × N trials per case) — TODO
├── variants.py        # constraint variant catalog (V0..V6) — TODO
├── fingerprint.py     # strategy fingerprint extractor — TODO
├── assessor.py        # 4-axis scoring (compute / accuracy / complexity / perf) — TODO
├── synthesizer.py     # per-case report generator — TODO
└── cases/             # per-case config files
```

## Running

Not yet wired. First milestone: `python -m discovery.perf` smoke test on a tiny model.
