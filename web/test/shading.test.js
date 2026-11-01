// Shading correction vs truth: spread of (recovered brightness / true brightness) across the label.
const fs = require("fs"), LF = require("../engine.js"), G = require("../labelgeo.js");
const buf = fs.readFileSync(process.argv[2] || "/tmp/claude-0/label_75.f32");
const base = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4), Lw = 600, Lh = 880;
function variant(kind) {
  const d = Float32Array.from(base);
  for (let y = 0; y < Lh; y++) for (let x = 0; x < Lw; x++) {
    const i = (y * Lw + x) * 3;
    if (kind === "band" && x > 0.55 * Lw && x < 0.85 * Lw && y > 0.1 * Lh && y < 0.9 * Lh) { d[i] = 0.35; d[i + 1] = 0.08; d[i + 2] = 0.12; }
    if (kind === "dark" && (y < 0.3 * Lh || y > 0.62 * Lh)) { d[i] *= 0.25; d[i + 1] *= 0.22; d[i + 2] *= 0.2; }
    if (kind === "twotone" && x > 0.5 * Lw) { d[i] *= 1.0; d[i + 1] *= 0.93; d[i + 2] *= 0.68; }
    if (kind === "black" && !(y > 0.12 * Lh && y < 0.3 * Lh)) { d[i] = 0.1 + 0.05 * d[i]; d[i + 1] = 0.1 + 0.05 * d[i + 1]; d[i + 2] = 0.1 + 0.05 * d[i + 2]; }
  }
  return { w: Lw, h: Lh, data: d };
}
function spread(sc, label, rgb, map, W, H) {
  const nb = 20, bins = Array.from({ length: nb }, () => []), o = [0, 0];
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) {
    const k = y * W + x; if (!sc.pixelToLabelMM(map[2 * k], map[2 * k + 1], o)) continue;
    const lx = Math.round((o[0] / sc.lw) * (Lw - 1)), ly = Math.round((o[1] / sc.lh) * (Lh - 1));
    if (lx < 0 || ly < 0 || lx >= Lw || ly >= Lh) continue;
    const j = (ly * Lw + lx) * 3, t = (0.299 * label.data[j] + 0.587 * label.data[j + 1] + 0.114 * label.data[j + 2]) * 255;
    if (t < 150) continue;                                    // paper only, where the ratio is meaningful
    const r = 0.299 * rgb[3 * k] + 0.587 * rgb[3 * k + 1] + 0.114 * rgb[3 * k + 2];
    bins[Math.min(nb - 1, Math.floor((x / W) * nb))].push(r / t);
  }
  const m = bins.filter((b) => b.length > 10).map((b) => b.sort((a, c) => a - c)[b.length >> 1]);
  return 1 - Math.min(...m) / Math.max(...m);
}
console.log("label   light | shading left: before → after (darkest column vs brightest, after removing the artwork)");
for (const kind of (process.argv[3] || "plain,band,dark,twotone,black").split(",")) for (const light of [0, 30, 60]) {
  const label = variant(kind), sc = LF.Scene.fromView({ labelW: 75, light, W: 960, H: 720 });
  const img = sc.render(label, 2, 1.5);
  let r; try { r = LF.flatten(img, sc.W, sc.H, sc.f, { outW: 360 }); } catch (e) { console.log(kind, light, "flatten failed:", e.message); continue; }
  const before = spread(sc, label, r.cyl.rgb, r.cyl.map, r.outW, r.outH);
  const fixed = Uint8ClampedArray.from(r.cyl.rgb);
  const info = G.evenLight(fixed, r.outW, r.outH, 3, r.model.thL, r.model.thR);
  const after = spread(sc, label, fixed, r.cyl.map, r.outW, r.outH);
  console.log(`${kind.padEnd(6)} ${String(light).padStart(4)}° | ${(100 * before).toFixed(0).padStart(3)}% → ${(100 * after).toFixed(0).padStart(3)}%   ${info.applied ? "" : "(not applied)"}`);
}
