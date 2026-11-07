// Does the flattened label come out as the label and nothing else? Rendered photos, six dots with
// errors like a quick tap or the label reader: node web/test/rectangle.test.js [error % of label height] [trials] [outward bias %]
// Columns: worst distance (mm) between the flattened rectangle's edges and the label's real edges (the
// 0.6 mm inset included), share of the edge lying on the bottle, RMS error inside (mm), snapped count.
// harness: how well does the label map to a rectangle? (edge error, bottle included)
const fs = require("fs"), LF = require("../engine.js");
const G = require(process.env.LG || "../labelgeo.js");
const W = 1200, H = 1600, f = (26 / 43.27) * Math.hypot(W, H);
let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
const labels = {}; for (const w of [75, 90]) { const b = fs.readFileSync(`/tmp/claude-0/lab_${w}x100.f32`); labels[w] = { w: w * 8, h: 800, data: new Float32Array(b.buffer, b.byteOffset, b.length / 4) }; }
function score(sc, at) {
  const o = [0, 0]; let edge = 0, glass = 0, nb = 0, s2 = 0, n = 0;
  for (let i = 0; i <= 40; i++) for (const [side, fx, fy] of [["t", i / 40, 0], ["b", i / 40, 1], ["l", 0, i / 40], ["r", 1, i / 40]]) {
    const p = at(fx, fy); nb++;
    if (!sc.pixelToLabelMM(p[0], p[1], o)) { glass++; edge = Math.max(edge, 99); continue; }
    const e = side === "t" ? o[1] : side === "b" ? sc.lh - o[1] : side === "l" ? o[0] : sc.lw - o[0];
    if (e < -0.3) glass++;                       // outside the label: bottle in the output
    edge = Math.max(edge, Math.abs(e));
  }
  for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) { const p = at(i / 20, j / 20); n++; if (!sc.pixelToLabelMM(p[0], p[1], o)) { s2 += 400; continue; } s2 += (o[0] - i / 20 * sc.lw) ** 2 + (o[1] - j / 20 * sc.lh) ** 2; }
  return { edge, glass: glass / nb, rms: Math.sqrt(s2 / n) };
}
const sigPct = +(process.argv[2] ?? 3), trials = +(process.argv[3] ?? 5), bias = +(process.argv[4] ?? 0);
const med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
console.log(`dots off by ${sigPct}% of label height (+${bias}% outward bias); medians over ${trials}: max edge error mm | share of edge on bottle | rms mm | snapped`);
const views = [{ elevation: 5 }, { elevation: 15 }, { elevation: 30 }, { elevation: 15, rotation: 15 }, { elevation: 15, labelW: 90 }, { elevation: 35, rotation: -12, lean: 12 }, { elevation: -10, rotation: 8 }, { elevation: 20, light: 50, ambient: 0.35 }];
const tot = { e: [], g: [], r: [] }; let F = 0, N = 0;
for (const v of views) {
  const sc = LF.Scene.fromView({ W, H, f, distance: 300, labelH: 100, labelW: 75, lean: 3, pan: 2, ...v });
  const rgb = sc.render(labels[Math.round(sc.lw)], 1, 2), S = new Uint8ClampedArray(W * H * 4);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) { S[i] = rgb[j]; S[i + 1] = rgb[j + 1]; S[i + 2] = rgb[j + 2]; S[i + 3] = 255; }
  const gray = G.grayOf(S, W, H), cam = { f, cx: W / 2, cy: H / 2 };
  const P = (th, y) => sc.project(sc.cylPoint(th, y)), tm = (y) => P((sc.th0 + sc.th1) / 2, y);
  const truth = { tl: P(sc.th0, sc.y1), tr: P(sc.th1, sc.y1), bl: P(sc.th0, sc.y0), br: P(sc.th1, sc.y0), tm: tm(sc.y1), bm: tm(sc.y0) };
  const hpx = Math.hypot(truth.tm[0] - truth.bm[0], truth.tm[1] - truth.bm[1]), sig = sigPct / 100 * hpx;
  const cx = (truth.tl[0] + truth.br[0]) / 2, cy = (truth.tm[1] + truth.bm[1]) / 2;
  const E = [], Gl = [], Rm = []; let ok = 0;
  for (let t = 0; t < trials; t++) {
    const o = {}; for (const k in truth) { const dx = truth[k][0] - cx, dy = truth[k][1] - cy; o[k] = [truth[k][0] + dx * bias / 100 + sig * gauss(), truth[k][1] + dy * bias / 100 + sig * gauss()]; }
    const r = G.fitAndSnap(gray, W, H, o, cam);
    if (!r) { E.push(99); Gl.push(1); Rm.push(99); continue; }
    const at = (r.final || r.fit).at, s = score(sc, at); E.push(s.edge); Gl.push(s.glass); Rm.push(s.rms); ok += r.snapped ? 1 : 0;
  }
  F += E.filter(x=>x>2).length; N += trials; tot.e.push(med(E)); tot.g.push(med(Gl)); tot.r.push(med(Rm));
  console.log(JSON.stringify(v).padEnd(48), 'fails', E.filter(x=>x>2).length + '/' + trials, med(E).toFixed(2).padStart(6), (100 * med(Gl)).toFixed(0).padStart(4) + "%", med(Rm).toFixed(2).padStart(6), `${ok}/${trials}`);
}
console.log("MEAN".padEnd(48), "fails " + F + "/" + N, (tot.e.reduce((a, b) => a + b) / tot.e.length).toFixed(2).padStart(6), (100 * tot.g.reduce((a, b) => a + b) / tot.g.length).toFixed(0).padStart(4) + "%", (tot.r.reduce((a, b) => a + b) / tot.r.length).toFixed(2).padStart(6));
