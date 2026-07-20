// G4 — final presentation deck (DESIGN §16). pptxgenjs, LAYOUT_WIDE (13.3x7.5in).
// Wildlife + edge-tech palette; figures from results/analysis/figures/. Run: node build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.resolve(__dirname, "..", "results", "analysis", "figures");
const ASSET = path.resolve(__dirname, "assets");
const fig = (n) => path.join(FIG, n + ".png");

// palette
const INK = "143126";       // deep forest (dark bg / headings)
const FOREST = "2C5F2D";    // forest green
const MOSS = "6E9B5B";      // moss
const AMBER = "E0912F";     // sharp accent (the "shutter fires")
const BLUE = "1F77B4";      // data highlight (matches figures / M2)
const CREAM = "F5F7F3";     // light content tint
const GREY = "5B665C";      // muted text
const WHITE = "FFFFFF";

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.defineSlideMaster({ title: "LIGHT", background: { color: WHITE } });
p.defineSlideMaster({ title: "DARK", background: { color: INK } });

const HEAD = "Cambria";     // safe-list serif header
const BODY = "Calibri";     // safe-list sans body
const W = 13.3, H = 7.5, M = 0.6;

// helpers ------------------------------------------------------------------
function title(s, t, opts = {}) {
  s.addText(t, {
    x: M, y: opts.y || 0.45, w: W - 2 * M, h: 0.9, fontFace: HEAD, fontSize: opts.size || 32,
    bold: true, color: opts.color || INK, align: "left", margin: 0,
  });
}
function kicker(s, t, color = AMBER) {
  s.addText(t.toUpperCase(), { x: M, y: 0.28, w: W - 2 * M, h: 0.3, fontFace: BODY,
    fontSize: 12, bold: true, color, charSpacing: 2, margin: 0 });
}
function bullets(s, items, o = {}) {
  s.addText(items.map((it, i) => ({ text: it, options: {
    bullet: { code: "2022", indent: 14 }, color: o.color || "222222", fontSize: o.size || 15,
    fontFace: BODY, breakLine: true, paraSpaceAfter: 8 } })),
    { x: o.x || M, y: o.y, w: o.w || 6.0, h: o.h || 4.5, valign: "top", margin: 0 });
}
function figure(s, name, o) {
  s.addImage({ path: fig(name), x: o.x, y: o.y, w: o.w, h: o.h, sizing: { type: "contain", w: o.w, h: o.h } });
}
function statCard(s, x, y, w, big, label, color = FOREST) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h: 1.55, fill: { color: CREAM }, rectRadius: 0.1,
    line: { color: "E3E8E1", width: 1 } });
  s.addText(big, { x, y: y + 0.12, w, h: 0.9, align: "center", fontFace: HEAD, fontSize: 34, bold: true, color, margin: 0 });
  s.addText(label, { x: x + 0.15, y: y + 1.02, w: w - 0.3, h: 0.45, align: "center", fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0 });
}
function chip(s, x, y, w, head, body, hcolor = FOREST) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h: 3.7, fill: { color: WHITE }, rectRadius: 0.08,
    line: { color: "E1E6E0", width: 1 }, shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "D8DDD7", opacity: 0.5 } });
  s.addText(head, { x: x + 0.22, y: y + 0.2, w: w - 0.44, h: 0.5, fontFace: HEAD, fontSize: 16, bold: true, color: hcolor, margin: 0 });
  s.addText(body.map((b, i) => ({ text: b, options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 7, fontSize: 12.5, color: "333333", fontFace: BODY } })),
    { x: x + 0.22, y: y + 0.8, w: w - 0.44, h: 2.7, valign: "top", margin: 0 });
}

// S1 — title (dark) --------------------------------------------------------
let s = p.addSlide({ masterName: "DARK" });
s.addText("WILDLIFE TRIGGER", { x: M, y: 2.2, w: 9.6, h: 1.1, fontFace: HEAD, fontSize: 52, bold: true, color: WHITE, margin: 0 });
s.addText("Bobcat shutter trigger on a Raspberry Pi — MobileNetV2 quantized to INT8, deployed in C++",
  { x: M, y: 3.35, w: 10.2, h: 0.9, fontFace: BODY, fontSize: 19, color: "CFE0C8", margin: 0 });
s.addText([
  { text: "2.27× faster on-device", options: { color: AMBER, bold: true } },
  { text: "  ·  20.4 → 46.3 FPS  ·  3.5× smaller  ·  accuracy-equivalent", options: { color: "9FB79A" } },
], { x: M, y: 4.35, w: 10, h: 0.5, fontFace: BODY, fontSize: 16, margin: 0 });
s.addText("Vadym (imagic9)   ·   Efficient ML final project   ·   SET University   ·   2026-07-20",
  { x: M, y: 6.55, w: 9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: "8AA285", margin: 0 });
s.addImage({ path: path.join(ASSET, "repo_qr.png"), x: 11.15, y: 5.35, w: 1.55, h: 1.55 });
s.addText("github.com/imagic9/\nefficient-ml-set", { x: 10.7, y: 6.9, w: 2.4, h: 0.5, align: "center", fontFace: BODY, fontSize: 9.5, color: "9FB79A", margin: 0 });

// S2 — why edge AI ---------------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "The product");
title(s, "Why Edge AI: the shutter must decide locally");
const why = [["Offline", "No cloud, no signal in the field — the model runs on the device itself."],
  ["Battery / CPU-only", "No GPU. A small INT8 model that sips power and fits on an SD card."],
  ["Real-time", "Keep up with a camera-trap burst — a decision every few tens of ms."]];
why.forEach((c, i) => {
  const x = M + i * 4.05;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 3.8, h: 3.6, fill: { color: CREAM }, rectRadius: 0.1, line: { color: "E3E8E1", width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.3, y: 2.35, w: 0.9, h: 0.9, fill: { color: FOREST } });
  s.addText(["📶", "🔋", "⚡"][i], { x: x + 0.3, y: 2.4, w: 0.9, h: 0.8, align: "center", fontSize: 26, color: WHITE, margin: 0 });
  s.addText(c[0], { x: x + 0.3, y: 3.45, w: 3.2, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK, margin: 0 });
  s.addText(c[1], { x: x + 0.3, y: 4.0, w: 3.2, h: 1.4, fontFace: BODY, fontSize: 13.5, color: "333333", valign: "top", margin: 0 });
});
s.addText("The core artifact is a CPU-only C++ inference engine that fires SHUTTER_TRIGGER only when a bobcat is present.",
  { x: M, y: 6.1, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 14, italic: true, color: GREY, margin: 0 });

// S3 — engineering question ------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Assignment mapping");
title(s, "The engineering question");
s.addShape(p.ShapeType.roundRect, { x: M, y: 1.7, w: W - 2 * M, h: 1.7, fill: { color: INK }, rectRadius: 0.1 });
s.addText([{ text: "How much on-device speedup can INT8 quantization + structured pruning buy for a MobileNetV2 wildlife classifier — ", options: { color: WHITE } },
  { text: "and does it cost accuracy?", options: { color: AMBER, bold: true } }],
  { x: M + 0.4, y: 1.85, w: W - 2 * M - 0.8, h: 1.4, fontFace: HEAD, fontSize: 21, valign: "middle", margin: 0 });
const map = [["Model training & optimization", "PTQ · QAT · structured pruning, justified"],
  ["C++ inference implementation", "ONNX Runtime CPU EP, parity-tested"],
  ["Benchmarking & metrics", "baseline vs optimized on the real Pi"],
  ["Results analysis & presentation", "what worked / what didn't / next"]];
map.forEach((r, i) => {
  const x = M + (i % 2) * 6.15, y = 3.75 + Math.floor(i / 2) * 1.15;
  s.addShape(p.ShapeType.roundRect, { x, y, w: 5.9, h: 1.0, fill: { color: CREAM }, rectRadius: 0.08, line: { color: "E3E8E1", width: 1 } });
  s.addText(r[0], { x: x + 0.25, y: y + 0.13, w: 5.4, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: FOREST, margin: 0 });
  s.addText(r[1], { x: x + 0.25, y: y + 0.52, w: 5.4, h: 0.4, fontFace: BODY, fontSize: 12, color: "444444", margin: 0 });
});
s.addText([{ text: "Answer: ", options: { bold: true, color: FOREST } }, { text: "2.27× faster, 3.5× smaller, no accuracy cost (better in-distribution).", options: { color: "333333" } }],
  { x: M, y: 6.35, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 14.5, italic: true, margin: 0 });

// S4 — dataset -------------------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Data");
title(s, "CCT-20 · bobcat target · leakage controls");
statCard(s, M, 1.75, 2.75, "57,864", "images · 16 classes", FOREST);
statCard(s, M + 2.95, 1.75, 2.75, "bobcat", "graded target (idx 3)", BLUE);
statCard(s, M + 5.9, 1.75, 2.75, "0.5775", "shortcut probe (≈0.50)", MOSS);
statCard(s, M + 8.85, 1.75, 2.75, "256×192", "input · +31% real pixels", AMBER);
bullets(s, [
  "cis / trans split — same cameras vs new locations (the hard generalization test).",
  "Leakage fingerprinted: 224 seqs / 270 imgs / 10 bobcat overlap → all decisions use cis_val_clean (3,214 imgs / 144 bobcat).",
  "Empty supplement: 5,000 frames, location-disjoint, downsized ≤1024 px so resolution is not a shortcut for 'empty'.",
  "Multi-label frames excluded from cross-entropy, kept for target-presence. Gate B passed 43/43.",
], { y: 3.7, w: W - 2 * M, size: 14.5 });

// S5 — baseline + C++ pipeline --------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Baseline & deployment");
title(s, "MobileNetV2 baseline → C++ inference pipeline");
bullets(s, [
  "ImageNet-pretrained MobileNetV2 (width 1.0), 16 outputs, input 256×192.",
  "Effective-number weighted CE; two-phase head→full fine-tuning (seed 42).",
  "Threshold calibrated on validation inside a 5% false-fire budget (§6.3).",
  "Python only for training/export — inference is C++ + ONNX Runtime CPU EP.",
], { y: 1.8, w: 6.2, size: 14.5 });
const steps = ["JPEG", "decode", "letterbox\n256×192", "ORT\nCPU EP", "policy\n(threshold)", "SHUTTER"];
steps.forEach((t, i) => {
  const y = 2.0 + i * 0.78, last = i === steps.length - 1;
  s.addShape(p.ShapeType.roundRect, { x: 7.4, y, w: 4.5, h: 0.62, fill: { color: last ? AMBER : FOREST }, rectRadius: 0.08 });
  s.addText(t, { x: 7.4, y, w: 4.5, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: last ? 15 : 13.5, bold: last, color: WHITE, margin: 0 });
  if (!last) s.addText("▼", { x: 9.5, y: y + 0.6, w: 0.3, h: 0.2, align: "center", fontSize: 9, color: MOSS, margin: 0 });
});

// S6 — optimization ladder -------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Optimization ladder");
title(s, "PTQ · QAT · structured pruning");
const rowsL = [["Model", "transform", "score", "MB", "MACs"],
  ["M0", "FP32 baseline", "0.3663", "8.95", "293M"],
  ["M1", "INT8 PTQ", "0.3527", "2.62", "293M"],
  ["M2", "INT8 QAT ✓final", "0.3832", "2.54", "293M"],
  ["M3", "pruned FP32", "0.3583", "7.04", "206M"],
  ["M4", "pruned+QAT", "0.3730", "2.01", "206M"]];
s.addTable(rowsL.map((r, ri) => r.map((c) => ({ text: c, options: {
  fontFace: BODY, fontSize: ri === 0 ? 12.5 : 12, bold: ri === 0 || (ri === 3), align: ri === 0 ? "left" : "left",
  color: ri === 3 ? BLUE : (ri === 0 ? WHITE : "333333"), fill: { color: ri === 0 ? FOREST : (ri === 3 ? "EAF2F8" : WHITE) } } }))),
  { x: M, y: 1.8, w: 6.1, colW: [0.8, 2.5, 1.1, 0.9, 0.9], border: { type: "solid", color: "DDE3DD", pt: 1 }, rowH: 0.42 });
bullets(s, [
  "M1 (PTQ) — pre-registered negative: depthwise MobileNetV2 loses accuracy.",
  "M2 (QAT) — most accurate; recovers the PTQ loss.",
  "M3/M4 pruning — 30% fewer MACs (widths rounded to ×8 first).",
  "Final = M2 by rule §8.4: a more complex stack must be more accurate to win — M4 isn't.",
], { x: M, y: 4.6, w: 6.1, size: 12.5 });
figure(s, "ladder_accuracy_vs_size", { x: 7.0, y: 1.85, w: 5.7, h: 4.9 });

// S7 — correctness gates ---------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Trust the numbers");
title(s, "Correctness & reproducibility before performance");
const gates = [["P1 preprocessing", "C++ ↔ Python golden tensors: 0.0 / 7e-7"],
  ["P2 · P3 · P4 parity", "C++ ORT = Python ORT; run-dataset confusion identical"],
  ["QEMU cortex-a76", "native vs emulated bit-identical (M0/M2/M4)"],
  ["Pi ↔ gx10 parity", "bit-identical on the CM5 — target equivalence"],
  ["Reproducibility", "F4 vs F5 within ±3.5%; seeds 17/42/73"],
  ["Fail-closed preflight", "refuses non-aarch64 / non-Ubuntu-24.04 / no asimddp"]];
gates.forEach((g, i) => {
  const x = M + (i % 2) * 6.15, y = 1.85 + Math.floor(i / 2) * 1.5;
  s.addShape(p.ShapeType.roundRect, { x, y, w: 5.9, h: 1.3, fill: { color: CREAM }, rectRadius: 0.08, line: { color: "E3E8E1", width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.25, y: y + 0.35, w: 0.6, h: 0.6, fill: { color: FOREST } });
  s.addText("✓", { x: x + 0.25, y: y + 0.35, w: 0.6, h: 0.6, align: "center", valign: "middle", fontSize: 20, bold: true, color: WHITE, margin: 0 });
  s.addText(g[0], { x: x + 1.05, y: y + 0.2, w: 4.7, h: 0.45, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText(g[1], { x: x + 1.05, y: y + 0.65, w: 4.7, h: 0.55, fontFace: BODY, fontSize: 12, color: "444444", valign: "top", margin: 0 });
});
s.addText("Every performance number below sits on a passed correctness gate.",
  { x: M, y: 6.55, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 13.5, italic: true, color: GREY, margin: 0 });

// S8 — accuracy at operating point ----------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Accuracy");
title(s, "At the bobcat operating point (frozen test, opened once)");
const rowsA = [["", "cis-test F2", "capture", "trans-test F2", "capture"],
  ["M0 FP32", "0.5812", "0.767", "0.2517", "0.395"],
  ["M2 QAT", "0.6387", "0.858", "0.2209", "0.347"]];
s.addTable(rowsA.map((r, ri) => r.map((c, ci) => ({ text: c, options: {
  fontFace: BODY, fontSize: ri === 0 ? 11.5 : 13, bold: ri === 0 || ci === 0, align: ci === 0 ? "left" : "center",
  color: ri === 2 && (ci === 1 || ci === 2) ? BLUE : (ri === 0 ? WHITE : "333333"),
  fill: { color: ri === 0 ? FOREST : WHITE } } }))),
  { x: M, y: 1.85, w: 6.1, colW: [1.5, 1.2, 1.0, 1.2, 1.0], border: { type: "solid", color: "DDE3DD", pt: 1 }, rowH: 0.5 });
bullets(s, [
  "In-distribution (cis-test): M2 BEATS the FP32 baseline — captures 85.8% of bobcat visits vs 76.7%, at equal 5.5% false-fire.",
  "New locations (trans-test): both drop to ~35–40% capture — the honest domain-shift limitation, reported not hidden.",
  "Both stay recall_floor_infeasible: ships the best admissible threshold, never claims the 90% floor is met.",
], { x: M, y: 3.9, w: 6.2, size: 12.5 });
figure(s, "confusion_frozen_test", { x: 7.2, y: 1.85, w: 5.5, h: 5.0 });

// S9 — the on-device result (dark-ish highlight) ---------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "The result that counts (DESIGN §12.4)");
title(s, "On-device: baseline vs optimized — real Raspberry Pi CM5");
statCard(s, M, 1.75, 3.7, "2.27×", "faster end-to-end (M0 → M2)", AMBER);
statCard(s, M + 3.9, 1.75, 3.7, "20.4 → 46.3", "FPS (frozen, threads=1)", BLUE);
statCard(s, M + 7.8, 1.75, 3.7, "3.5×", "smaller: 8.95 → 2.54 MB", FOREST);
figure(s, "pi_latency_fps", { x: 2.4, y: 3.5, w: 8.5, h: 3.6 });
s.addText("CM5 · Cortex-A76 @ 2.4 GHz · performance governor · no throttling · 3 reps ×≥1000 iters · Pi↔gx10 bit-identical",
  { x: M, y: 7.0, w: W - 2 * M, h: 0.35, align: "center", fontFace: BODY, fontSize: 10.5, color: GREY, margin: 0 });

// S10 — pareto + bottleneck ------------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Analysis");
title(s, "Pareto front & where the time goes");
figure(s, "pareto_accuracy_latency", { x: M, y: 1.8, w: 5.9, h: 4.2 });
figure(s, "pi_stage_breakdown", { x: 6.9, y: 1.8, w: 5.8, h: 4.2 });
s.addText([
  { text: "Inference is 85% of the FP32 pipeline and collapses 4× under INT8 — the bottleneck shifts to the fixed ~6 ms JPEG decode. ", options: { color: "333333" } },
  { text: "On the 4-core A76 threads=3 is optimal — the opposite of the 20-core gx10.", options: { color: FOREST, bold: true } },
], { x: M, y: 6.15, w: W - 2 * M, h: 0.9, fontFace: BODY, fontSize: 13.5, margin: 0 });

// S11 — worked / didn't / limits -------------------------------------------
s = p.addSlide({ masterName: "LIGHT" });
kicker(s, "Critical evaluation");
title(s, "What worked · what didn't · limitations");
chip(s, M, 1.85, 3.85, "What worked", [
  "INT8 QAT: 2.27× on-device, 3.5× smaller, no accuracy cost",
  "C++/ORT parity exact — Pi↔gx10 bit-identical",
  "QEMU rehearsal de-risked the one-shot rental",
  "Threading on Pi: +1.25× (target-specific)",
], FOREST);
chip(s, M + 4.05, 1.85, 3.85, "What didn't", [
  "Trans-domain recall poor (~35–40% capture)",
  "90% recall floor infeasible in the 5% budget",
  "PTQ (M1) — pre-registered negative result",
  "Pruning (M4) not more accurate → not selected",
], AMBER);
chip(s, M + 8.1, 1.85, 3.85, "Limitations", [
  "Remote Pi: no camera, no GPIO, no power meter",
  "Shutter is an emulated JSON signal",
  "CCT domain; weak on unseen sites",
  "badger/deer/fox: null thresholds (no support)",
], BLUE);
s.addText("Negatives are measured and reported — nothing is hidden (DESIGN §18 decision rules).",
  { x: M, y: 5.95, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 13.5, italic: true, color: GREY, margin: 0 });

// S12 — closing (dark) -----------------------------------------------------
s = p.addSlide({ masterName: "DARK" });
s.addText("Result & next steps", { x: M, y: 0.7, w: 10, h: 0.9, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0 });
s.addText([{ text: "2.27× faster  ·  46.3 FPS  ·  3.5× smaller  ·  ", options: { color: AMBER, bold: true } },
  { text: "accuracy-equivalent on the real Raspberry Pi CM5", options: { color: "CFE0C8" } }],
  { x: M, y: 1.75, w: 11.5, h: 0.6, fontFace: BODY, fontSize: 19, margin: 0 });
const nexts = [["Close the decode bottleneck", "inference no longer dominates — attack the ~6 ms JPEG decode"],
  ["Improve OOD generalization", "more locations, TTA, or crop-teacher KD (Phase S)"],
  ["Physical integration", "wire GPIO shutter + on-device power (Joules/decision)"]];
nexts.forEach((n, i) => {
  const y = 2.7 + i * 1.05;
  s.addShape(p.ShapeType.ellipse, { x: M, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: FOREST } });
  s.addText(String(i + 1), { x: M, y: y + 0.05, w: 0.55, h: 0.55, align: "center", valign: "middle", fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
  s.addText([{ text: n[0] + "  ", options: { bold: true, color: WHITE } }, { text: "— " + n[1], options: { color: "9FB79A" } }],
    { x: M + 0.8, y: y, w: 8.5, h: 0.7, valign: "middle", fontFace: BODY, fontSize: 15, margin: 0 });
});
s.addImage({ path: path.join(ASSET, "repo_qr.png"), x: 10.6, y: 2.8, w: 2.0, h: 2.0 });
s.addText("github.com/imagic9/efficient-ml-set", { x: 9.9, y: 4.85, w: 3.4, h: 0.4, align: "center", fontFace: BODY, fontSize: 12, bold: true, color: AMBER, margin: 0 });
s.addText("Vadym (imagic9) · Efficient ML · SET University", { x: M, y: 6.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 12, color: "8AA285", margin: 0 });

p.writeFile({ fileName: path.resolve(__dirname, "final_presentation.pptx") }).then((f) => console.log("wrote", f));
