// Rendered photos: app's current method vs six-dot fit vs fit + edge snapping.
const fs = require("fs"), LF = require("../engine.js"), G = require("../labelgeo.js");
const W = 1200, H = 1600, f = (26 / 43.27) * Math.hypot(W, H);
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
const labels = {}; for (const w of [75, 90]) { const b = fs.readFileSync(`/tmp/claude-0/lab_${w}x100.f32`); labels[w] = { w: w * 8, h: 800, data: new Float32Array(b.buffer, b.byteOffset, b.length / 4) }; }
function oldMap(o, curveDeg) {
  const quad = (a, m, b) => { const c = [2 * m[0] - (a[0] + b[0]) / 2, 2 * m[1] - (a[1] + b[1]) / 2]; return (t) => { const s = 1 - t; return [s * s * a[0] + 2 * s * t * c[0] + t * t * b[0], s * s * a[1] + 2 * s * t * c[1] + t * t * b[1]]; }; };
  const T = quad(o.tl, o.tm, o.tr), B = quad(o.bl, o.bm, o.br), th = curveDeg * Math.PI / 180, sh = Math.sin(th / 2);
  const mapU = th > .01 ? (u) => .5 + Math.sin((u - .5) * th) / (2 * sh) : (u) => u;
  return (fx, fy) => { const t = mapU(fx), a = T(t), b = B(t); return [a[0] + (b[0] - a[0]) * fy, a[1] + (b[1] - a[1]) * fy]; };
}
function err(sc, map) {
  let s2 = 0, n = 0; const o = [0, 0];
  for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) {
    const p = map(i / 20, j / 20); n++;
    if (!sc.pixelToLabelMM(p[0], p[1], o)) { s2 += 400; continue; }
    s2 += (o[0] - i / 20 * sc.lw) ** 2 + (o[1] - j / 20 * sc.lh) ** 2;
  }
  return Math.sqrt(s2 / n);
}
const sig = +(process.argv[2] ?? 8), trials = +(process.argv[3] ?? 6), med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
console.log(`dots off by ${sig} px (rms per coordinate); median RMS error on the label, mm, over ${trials} trials`);
console.log("view                      | app 90° | 6-dot fit | fit + snap | snapped | time ms");
for (const v of [{ elevation: 5 }, { elevation: 15 }, { elevation: 30 }, { elevation: 15, rotation: 15 }, { elevation: 15, labelW: 90 }, { elevation: 35, rotation: -12, lean: 12 }, { elevation: -10, rotation: 8 }]) {
  const sc = LF.Scene.fromView({ W, H, f, distance: 300, labelH: 100, labelW: 75, lean: 3, pan: 2, ...v });
  if (!sc.labelVisible()) { console.log(JSON.stringify(v), "hidden"); continue; }
  const rgb = sc.render(labels[Math.round(sc.lw)], 2, 2), S = new Uint8ClampedArray(W * H * 4);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) { S[i] = rgb[j]; S[i + 1] = rgb[j + 1]; S[i + 2] = rgb[j + 2]; S[i + 3] = 255; }
  const gray = G.grayOf(S, W, H), cam = { f, cx: W / 2, cy: H / 2 };
  const P = (th, y) => sc.project(sc.cylPoint(th, y)), tm = (y) => P((sc.th0 + sc.th1) / 2, y);
  const truth = { tl: P(sc.th0, sc.y1), tr: P(sc.th1, sc.y1), bl: P(sc.th0, sc.y0), br: P(sc.th1, sc.y0), tm: tm(sc.y1), bm: tm(sc.y0) };
  const R = { old: [], six: [], snap: [], ok: 0, t: [] };
  for (let t = 0; t < trials; t++) {
    const o = {}; for (const k in truth) o[k] = [truth[k][0] + sig * gauss(), truth[k][1] + sig * gauss()];
    R.old.push(err(sc, oldMap(o, 90)));
    const t0 = Date.now(), r = G.fitAndSnap(gray, W, H, o, cam); R.t.push(Date.now() - t0);
    R.six.push(r ? err(sc, r.dotsOnly.at) : 99); R.snap.push(r ? err(sc, r.fit.at) : 99); R.ok += r && r.snapped ? 1 : 0;
  }
  const lab = Object.entries(v).map(([k, x]) => `${k.slice(0, 4)} ${x}`).join(" ");
  console.log(`${lab.padEnd(25)} | ${med(R.old).toFixed(2).padStart(7)} | ${med(R.six).toFixed(2).padStart(9)} | ${med(R.snap).toFixed(2).padStart(10)} | ${String(R.ok).padStart(4)}/${trials} | ${med(R.t)}`);
}
