#!/usr/bin/env python3
"""Render sweep_results/version_summary.json as a self-contained HTML dashboard.

Input is the file written by tools/version_summary.py (one key per torch
version). Output is a single HTML page with inline SVG charts, hover tooltips,
light/dark themes, and a full table view. No network or external assets.

Usage:
    python3 tools/version_summary_viz.py [--in sweep_results/version_summary.json]
                                         [--out sweep_results/version_summary.html]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>torch.compile corpus — results by PyTorch version</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --good-text: #006300; --bad-text: #d03b3b;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --on-fill: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --good-text: #0ca30c; --bad-text: #e66767;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --good-text: #0ca30c; --bad-text: #e66767;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  background: var(--page); color: var(--text-primary); min-height: 100vh;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 28px 32px 48px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0; }
button.theme {
  font: inherit; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; cursor: pointer;
}
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px; }
.tile, .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; }
.tile { padding: 16px 18px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 30px; font-weight: 600; margin: 2px 0; }
.tile .delta { font-size: 13px; color: var(--text-secondary); }
.tile .delta b { font-weight: 600; }
.tile .delta .good { color: var(--good-text); }
.tile .delta .bad { color: var(--bad-text); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.card { padding: 18px 20px 14px; position: relative; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.card .takeaway { color: var(--text-secondary); font-size: 13px; margin: 0 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; }
.legend i.rect { width: 10px; height: 10px; border-radius: 2px; }
.legend i.line { width: 14px; height: 2px; border-radius: 1px; }
.chart { width: 100%; }
.chart svg { display: block; overflow: visible; }
.chart text { font-family: inherit; }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.dlabel { fill: var(--text-secondary); font-size: 12px; }
.dlabel.strong { fill: var(--text-primary); font-weight: 600; }
.mark { cursor: default; outline: none; transition: opacity .12s; }
.mark:hover, .mark:focus-visible { opacity: .82; }
.mark:focus-visible { stroke: var(--text-primary); stroke-width: 1.5; }
.tablelink { font-size: 12px; color: var(--text-muted); text-decoration: none; position: absolute; top: 18px; right: 20px; }
.tablelink:hover { color: var(--text-secondary); text-decoration: underline; }
#tip {
  position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
  min-width: 150px;
}
#tip .h { color: var(--text-secondary); margin-bottom: 4px; }
#tip .r { display: flex; align-items: center; gap: 8px; justify-content: space-between; }
#tip .r .k { display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); }
#tip .r .k i { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
#tip .r b { font-variant-numeric: tabular-nums; }
.notes { color: var(--text-secondary); font-size: 12.5px; margin: 4px 0 18px; padding-left: 18px; }
.notes li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
tr.total td { font-weight: 600; }
.tablecard { padding: 18px 20px; }
.tablecard h2 { margin-bottom: 10px; }
@media (max-width: 900px) { .tiles { grid-template-columns: 1fr 1fr; } .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">
  <header>
    <div>
      <h1>torch.compile corpus — results by PyTorch version</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <button class="theme" id="themeBtn" type="button">Toggle theme</button>
  </header>

  <section class="tiles" id="tiles" aria-label="Headline numbers"></section>

  <div class="grid2">
    <div class="card">
      <h2>Graph-break rows by version</h2>
      <p class="takeaway" id="takeA"></p>
      <a class="tablelink" href="#table">Table view</a>
      <div class="legend" id="legA"></div>
      <div class="chart" id="chartA" role="img"></div>
    </div>
    <div class="card">
      <h2>Model outcomes by source</h2>
      <p class="takeaway" id="takeB"></p>
      <a class="tablelink" href="#table">Table view</a>
      <div class="legend" id="legB"></div>
      <div class="chart" id="chartB" role="img"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Dynamo recompiles in graph-break rows</h2>
      <p class="takeaway" id="takeC"></p>
      <a class="tablelink" href="#table">Table view</a>
      <div class="legend" id="legC"></div>
      <div class="chart" id="chartC" role="img"></div>
    </div>
    <div class="card">
      <h2>Average recompiles per graph-break model</h2>
      <p class="takeaway" id="takeD"></p>
      <a class="tablelink" href="#table">Table view</a>
      <div class="chart" id="chartD" role="img"></div>
    </div>
  </div>

  <ul class="notes" id="notes"></ul>

  <div class="card tablecard" id="table">
    <h2>Table view</h2>
    <div style="overflow-x:auto"><table id="tbl"></table></div>
  </div>
</div>
<div id="tip" role="tooltip"></div>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(() => {
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const VERSIONS = Object.keys(DATA).sort((a, b) => {
  const pa = a.split(/[.+]/).map(Number), pb = b.split(/[.+]/).map(Number);
  for (let i = 0; i < 3; i++) if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) - (pb[i] || 0);
  return 0;
});
const short = v => v.split("+")[0];
const FIRST = VERSIONS[0], LAST = VERSIONS[VERSIONS.length - 1];
const SRC_ORDER = ["hf", "timm", "diffusers", "custom"];
const SRC_NAME = { hf: "HF Transformers", timm: "TIMM", diffusers: "Diffusers", custom: "Custom" };
const sources = SRC_ORDER.filter(s => DATA[LAST].models.by_source[s])
  .concat(Object.keys(DATA[LAST].models.by_source).filter(s => !SRC_ORDER.includes(s)));
const fmt = n => n == null ? "—" : Number(n).toLocaleString("en-US");
const fmt2 = n => n == null ? "—" : Number(n).toFixed(2);
const pct = (a, b) => b ? (100 * a / b) : 0;
const SVGNS = "http://www.w3.org/2000/svg";
const css = name => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, attrs = {}, text, parent) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
// Column with 4px rounded data-end (top), square at the baseline.
function colPath(x, y, w, y0) {
  const hgt = y0 - y, r = Math.max(0, Math.min(4, hgt, w / 2));
  return `M${x},${y0}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y0}Z`;
}
function roundedRect(x, y, w, hh, r) {
  r = Math.max(0, Math.min(r, w / 2, hh / 2));
  return `M${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hh - r}Q${x + w},${y + hh} ${x + w - r},${y + hh}H${x + r}Q${x},${y + hh} ${x},${y + hh - r}V${y + r}Q${x},${y} ${x + r},${y}Z`;
}

// ---------- tooltip ----------
const tip = document.getElementById("tip");
function showTip(evtOrRect, head, rows) {
  tip.replaceChildren();
  h("div", { class: "h" }, head, tip);
  for (const [color, name, val] of rows) {
    const r = h("div", { class: "r" }, null, tip);
    const k = h("span", { class: "k" }, null, r);
    if (color) { const i = h("i", {}, null, k); i.style.background = color; }
    k.appendChild(document.createTextNode(name));
    h("b", {}, val, r);
  }
  tip.style.display = "block";
  let cx, cy;
  if (evtOrRect.clientX != null) { cx = evtOrRect.clientX; cy = evtOrRect.clientY; }
  else { cx = evtOrRect.left + evtOrRect.width / 2; cy = evtOrRect.top; }
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = cx + 14, top = cy - th - 10;
  if (left + tw > window.innerWidth - 8) left = cx - tw - 14;
  if (top < 8) top = cy + 16;
  tip.style.left = left + "px"; tip.style.top = top + "px";
}
const hideTip = () => { tip.style.display = "none"; };
function bindMark(node, label, getTip) {
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", label);
  node.classList.add("mark");
  node.addEventListener("pointermove", e => { const [hd, rows] = getTip(); showTip(e, hd, rows); });
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => { const [hd, rows] = getTip(); showTip(node.getBoundingClientRect(), hd, rows); });
  node.addEventListener("blur", hideTip);
}

function legend(id, items) {
  const box = document.getElementById(id); box.replaceChildren();
  for (const [colorVar, name, kind] of items) {
    const s = h("span", {}, null, box);
    const i = h("i", { class: kind }, null, s); i.style.background = `var(${colorVar})`;
    s.appendChild(document.createTextNode(name));
  }
}

// ---------- axes helpers ----------
function yAxis(svg, m, W, y, ymax, ticks, fmtTick) {
  for (let i = 0; i <= ticks; i++) {
    const v = ymax * i / ticks, yy = y(v);
    el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: i === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1, "shape-rendering": "crispEdges" }, svg);
    const t = el("text", { x: m.l - 8, y: yy + 4, "text-anchor": "end", class: "tick" }, svg);
    t.textContent = fmtTick(v);
  }
}

// ---------- line chart (shared by A and D) ----------
function lineChart(id, series, opts) {
  const box = document.getElementById(id); box.replaceChildren();
  const W = box.clientWidth, H = 250, m = { l: 44, r: opts.endLabelW || 70, t: 14, b: 30 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box);
  const n = VERSIONS.length;
  const x = i => m.l + 16 + i * (W - m.l - m.r - 32) / (n - 1);
  const vmax = Math.max(...series.flatMap(s => s.values));
  const ymax = opts.ymax || niceMax(vmax * 1.08);
  const y = v => H - m.b - (v / ymax) * (H - m.t - m.b);
  yAxis(svg, m, W, y, ymax, opts.ticks || 5, opts.tickFmt || fmt);
  VERSIONS.forEach((v, i) => {
    const t = el("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg);
    t.textContent = short(v);
  });
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  for (const s of series) {
    el("path", { d: s.values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(""), fill: "none",
      stroke: `var(${s.color})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  }
  for (const s of series) {
    s.values.forEach((v, i) => el("circle", { cx: x(i), cy: y(v), r: 4, fill: `var(${s.color})`,
      stroke: "var(--surface-1)", "stroke-width": 2 }, svg));
    const last = s.values.length - 1;
    const t = el("text", { x: x(last) + 10, y: y(s.values[last]) + 4 + (s.labelDy || 0), class: "dlabel" }, svg);
    t.textContent = s.endLabel;
  }
  // Crosshair hit layer: one column per version, full height; keyboard-focusable.
  const step = n > 1 ? (x(1) - x(0)) : (W - m.l - m.r);
  VERSIONS.forEach((v, i) => {
    const hit = el("rect", { x: x(i) - step / 2, y: m.t, width: step, height: H - m.t - m.b, fill: "transparent" }, svg);
    const getTip = () => [`PyTorch ${v}`, opts.tipRows(i)];
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("aria-label", `PyTorch ${short(v)}: ` + opts.tipRows(i).map(r => `${r[1]} ${r[2]}`).join(", "));
    hit.style.outline = "none";
    const on = (e) => { cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      const [hd, rows] = getTip(); showTip(e || hit.getBoundingClientRect(), hd, rows); };
    const off = () => { cross.setAttribute("visibility", "hidden"); hideTip(); };
    hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", off);
    hit.addEventListener("focus", () => on(null)); hit.addEventListener("blur", off);
  });
  box.setAttribute("aria-label", opts.aria);
}

// ---------- A: graph-break rows, eval vs train ----------
function chartA() {
  const ev = VERSIONS.map(v => DATA[v].rows.by_mode.eval.graph_break);
  const tr = VERSIONS.map(v => DATA[v].rows.by_mode.train.graph_break);
  legend("legA", [["--series-1", "Eval", "line"], ["--series-2", "Train", "line"]]);
  lineChart("chartA", [
    { values: tr, color: "--series-2", endLabel: `Train ${fmt(tr.at(-1))}`, labelDy: -2 },
    { values: ev, color: "--series-1", endLabel: `Eval ${fmt(ev.at(-1))}`, labelDy: 4 },
  ], {
    endLabelW: 78,
    aria: "Line chart of graph-break rows per PyTorch version, eval and train modes",
    tipRows: i => [
      [css("--series-2"), "Train", fmt(tr[i])],
      [css("--series-1"), "Eval", fmt(ev[i])],
      [null, "All rows", fmt(DATA[VERSIONS[i]].rows.total.graph_break)],
      [null, "of which HF", fmt(DATA[VERSIONS[i]].rows.by_source.hf?.graph_break)],
    ],
  });
  const a = DATA[FIRST].rows.total.graph_break, b = DATA[LAST].rows.total.graph_break;
  const hfA = DATA[FIRST].rows.by_source.hf.graph_break, hfB = DATA[LAST].rows.by_source.hf.graph_break;
  document.getElementById("takeA").textContent =
    `${fmt(a)} → ${fmt(b)} rows (${fmt(b - a)}) from ${short(FIRST)} to ${short(LAST)}; ` +
    `${Math.round(pct(hfA - hfB, a - b))}% of the drop is HF. Almost all of it lands in ${short(VERSIONS[1])}; later versions are flat.`;
}

// ---------- B: model outcome mix by source (latest version) ----------
function chartB() {
  const cats = [["full_graph", "Fullgraph", "--series-1"], ["graph_break", "Graph break", "--series-2"], ["other", "Error / skipped", "--series-3"]];
  legend("legB", cats.map(([, n, c]) => [c, n, "rect"]));
  const box = document.getElementById("chartB"); box.replaceChildren();
  const W = box.clientWidth, labelW = 128, rowH = 24, gap = 26, top = 8;
  const H = top + sources.length * (rowH + gap) + 4;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box);
  const plotW = W - labelW - 8;
  sources.forEach((s, ri) => {
    const st = DATA[LAST].models.by_source[s];
    const yy = top + ri * (rowH + gap) + 10;
    const name = el("text", { x: 0, y: yy + rowH / 2 - 2, class: "dlabel strong" }, svg); name.textContent = SRC_NAME[s] || s;
    const cnt = el("text", { x: 0, y: yy + rowH / 2 + 13, class: "tick" }, svg); cnt.textContent = `${fmt(st.n_models)} models`;
    const parts = cats.filter(([k]) => st[k] > 0);
    const totalGap = 2 * (parts.length - 1);
    let xx = labelW;
    parts.forEach(([k, label, color], pi) => {
      const w = Math.max(2, (plotW - totalGap) * st[k] / st.n_models);
      const p = el("path", { d: roundedRect(xx, yy, w, rowH, 4), fill: `var(${color})` }, svg);
      const share = pct(st[k], st.n_models);
      const shareTxt = share < 1 ? share.toFixed(1) + "%" : Math.round(share) + "%";
      const t = el("text", { x: xx + 6, y: yy + rowH / 2 + 4, "font-size": 12, "font-weight": 600, fill: "var(--on-fill)", "pointer-events": "none" }, svg);
      t.textContent = shareTxt;
      if (t.getComputedTextLength() + 12 > w) t.remove();
      bindMark(p, `${SRC_NAME[s] || s}, ${label}: ${st[k]} of ${st.n_models} models (${shareTxt})`,
        () => [`${SRC_NAME[s] || s} · PyTorch ${short(LAST)}`, [[css(color), label, `${fmt(st[k])} (${shareTxt})`], [null, "Models", fmt(st.n_models)]]]);
      xx += w + 2;
    });
  });
  box.setAttribute("aria-label", `Segmented bars of model outcomes per source at PyTorch ${short(LAST)}`);
  const hf = DATA[LAST].models.by_source.hf, ti = DATA[LAST].models.by_source.timm, df = DATA[LAST].models.by_source.diffusers;
  document.getElementById("takeB").textContent =
    `PyTorch ${short(LAST)}. HF holds ${fmt(hf.graph_break)} of ${fmt(DATA[LAST].models.total.graph_break)} graph-break models` +
    (ti ? `; TIMM is ${Math.round(pct(ti.full_graph, ti.n_models))}% fullgraph` : "") +
    (df ? `; ${Math.round(pct(df.other, df.n_models))}% of Diffusers models never reach a compile verdict.` : ".");
}

// ---------- C: recompile totals, first vs second call ----------
function chartC() {
  const c1 = VERSIONS.map(v => DATA[v].rows.total.recompile.recompile_count.total);
  const c2 = VERSIONS.map(v => DATA[v].rows.total.recompile.recompile_count_steady.total);
  legend("legC", [["--series-1", "First call", "rect"], ["--series-2", "Second call (same inputs)", "rect"]]);
  const box = document.getElementById("chartC"); box.replaceChildren();
  const W = box.clientWidth, H = 250, m = { l: 44, r: 8, t: 18, b: 30 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box);
  const ymax = niceMax(Math.max(...c1) * 1.1);
  const y = v => H - m.b - (v / ymax) * (H - m.t - m.b);
  yAxis(svg, m, W, y, ymax, 4, fmt);
  const band = (W - m.l - m.r) / VERSIONS.length, bw = Math.min(24, band / 4);
  VERSIONS.forEach((v, i) => {
    const cx = m.l + band * (i + 0.5);
    const t = el("text", { x: cx, y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg); t.textContent = short(v);
    const rc = DATA[v].rows.total.recompile;
    const tipFn = () => [`PyTorch ${v}`, [
      [css("--series-1"), "First call", fmt(c1[i])],
      [css("--series-2"), "Second call", fmt(c2[i])],
      [null, "Graph-break rows with data", `${fmt(rc.n_with_data)} (${rc.n_missing} missing)`]]];
    [[c1[i], "--series-1", "first call", cx - bw - 1], [c2[i], "--series-2", "second call", cx + 1]].forEach(([val, color, nm, x0]) => {
      const p = el("path", { d: colPath(x0, y(val), bw, y(0)), fill: `var(${color})` }, svg);
      bindMark(p, `PyTorch ${short(v)} ${nm}: ${val} recompiles`, tipFn);
    });
    const lab = el("text", { x: cx - bw / 2 - 1, y: y(c1[i]) - 6, "text-anchor": "middle", class: "dlabel" }, svg);
    lab.textContent = fmt(c1[i]);
  });
  box.setAttribute("aria-label", "Grouped columns of total Dynamo recompiles per PyTorch version, first and second call");
  document.getElementById("takeC").textContent =
    `First-call recompiles fall ${fmt(c1[0])} → ${fmt(c1.at(-1))} with the graph breaks. ` +
    `Second-call recompiles (guards that don't settle) stay at ${Math.min(...c2)}–${Math.max(...c2)} across all versions.`;
}

// ---------- D: avg recompiles per graph-break model ----------
function chartD() {
  const avg = VERSIONS.map(v => DATA[v].models.total.recompile.recompile_count.avg);
  lineChart("chartD", [{ values: avg, color: "--series-1", endLabel: fmt2(avg.at(-1)) }], {
    ymax: 5, ticks: 5, tickFmt: v => v.toFixed(0), endLabelW: 44,
    aria: "Line chart of average Dynamo recompiles per graph-break model, per PyTorch version",
    tipRows: i => {
      const mt = DATA[VERSIONS[i]].models.total;
      return [[css("--series-1"), "Avg per model", fmt2(avg[i])],
              [null, "Graph-break models", fmt(mt.graph_break)],
              [null, "Second-call avg", fmt2(mt.recompile.recompile_count_steady.avg)]];
    },
  });
  const gbA = DATA[FIRST].models.total.graph_break, gbB = DATA[LAST].models.total.graph_break;
  document.getElementById("takeD").textContent =
    `${fmt2(avg[0])} → ${fmt2(avg.at(-1))} while graph-break models fell ${fmt(gbA)} → ${fmt(gbB)}: ` +
    `the models that were fixed recompiled less than the ones still breaking.`;
}

// ---------- tiles ----------
function tiles() {
  const box = document.getElementById("tiles"); box.replaceChildren();
  const A = DATA[FIRST], B = DATA[LAST];
  const items = [
    ["Fullgraph models", A.models.total.full_graph, B.models.total.full_graph, true, fmt],
    ["Graph-break models", A.models.total.graph_break, B.models.total.graph_break, false, fmt],
    ["Recompiles in graph-break rows", A.rows.total.recompile.recompile_count.total, B.rows.total.recompile.recompile_count.total, false, fmt],
    ["Avg recompiles per graph-break model", A.models.total.recompile.recompile_count.avg, B.models.total.recompile.recompile_count.avg, false, fmt2],
  ];
  for (const [label, a, b, upGood, f] of items) {
    const t = h("div", { class: "tile" }, null, box);
    h("div", { class: "label" }, label, t);
    h("div", { class: "value" }, f(b), t);
    const d = h("div", { class: "delta" }, null, t);
    const diff = b - a, good = diff === 0 ? null : (diff > 0) === upGood;
    const s = h("b", { class: good == null ? "" : good ? "good" : "bad" },
      `${diff > 0 ? "▲ +" : diff < 0 ? "▼ −" : ""}${f(Math.abs(diff))}${good == null ? "" : good ? " better" : " worse"}`, d);
    d.appendChild(document.createTextNode(` vs ${short(FIRST)}`));
  }
}

// ---------- table ----------
function table() {
  const tbl = document.getElementById("tbl"); tbl.replaceChildren();
  const head = ["PyTorch", "Source", "Fullgraph rows", "Graph-break rows", "Recompiles (1st)", "Avg / gb row", "Recompiles (2nd)",
                "Models", "Fullgraph models", "Graph-break models", "Error / skipped models", "Avg recompiles / gb model"];
  const tr0 = h("tr", {}, null, h("thead", {}, null, tbl));
  head.forEach(x => h("th", { scope: "col" }, x, tr0));
  const tb = h("tbody", {}, null, tbl);
  for (const v of VERSIONS) {
    const keys = sources.concat(["__total"]);
    for (const s of keys) {
      const r = s === "__total" ? DATA[v].rows.total : DATA[v].rows.by_source[s];
      const mm = s === "__total" ? DATA[v].models.total : DATA[v].models.by_source[s];
      const tr = h("tr", { class: s === "__total" ? "total" : "" }, null, tb);
      [short(v), s === "__total" ? "All" : (SRC_NAME[s] || s), fmt(r.full_graph), fmt(r.graph_break),
       fmt(r.recompile.recompile_count.total), fmt2(r.recompile.recompile_count.avg), fmt(r.recompile.recompile_count_steady.total),
       fmt(mm.n_models), fmt(mm.full_graph), fmt(mm.graph_break), fmt(mm.other), fmt2(mm.recompile.recompile_count.avg)]
        .forEach(x => h("td", {}, x, tr));
    }
  }
}

function notes() {
  const box = document.getElementById("notes"); box.replaceChildren();
  const miss = VERSIONS.map(v => DATA[v].rows.total.recompile.n_missing);
  [
    "Rows = one (model, mode) pair; each model is tested in eval and train. A model counts as fullgraph only if every mode is fullgraph, and as graph-break if any mode breaks.",
    "Recompiles come from the explain pass, which only runs on graph-break rows. “Second call” reruns the compiled model with identical inputs, so a recompile there means guards that don't settle.",
    `${Math.min(...miss)}–${Math.max(...miss)} graph-break rows per version have no recompile data (explain error or skip) and are excluded from totals and averages.`,
    "Diffusers models skipped for size (size_routing.json) don't appear in any count. All versions use transformers 5.23.0.dev0 and diffusers 0.42.0.dev0.",
  ].forEach(t => h("li", {}, t, box));
}

function renderAll() { chartA(); chartB(); chartC(); chartD(); }

document.getElementById("subtitle").textContent =
  `${VERSIONS.length} full sweeps (${VERSIONS.map(short).join(", ")}) · backend eager · ${fmt(DATA[LAST].models.total.n_models)} models × eval/train`;
tiles(); table(); notes(); renderAll();

document.getElementById("themeBtn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
});
let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(renderAll, 120); });
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=REPO / "sweep_results" / "version_summary.json")
    ap.add_argument("--out", type=Path, default=REPO / "sweep_results" / "version_summary.html")
    args = ap.parse_args()
    data = json.loads(args.inp.read_text())
    # Escape "</" so embedded JSON can't close the <script> element.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    args.out.write_text(TEMPLATE.replace("__DATA__", blob))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
