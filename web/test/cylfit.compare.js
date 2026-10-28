const LF = require("../engine.js"), { fitLabel } = require("../cylfit.js");
const W = 1200, H = 1600, f = (26 / 43.27) * Math.hypot(W, H);
function gauss(rnd) { return Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd()); }
let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

function scene(v) { return LF.Scene.fromView({ W, H, f, distance: 300, labelH: 100, lean: 3, pan: 2, ...v }); }
function sixPoints(sc) {
  const P = (th, y) => sc.project(sc.cylPoint(th, y));
  const mid = (y) => { // halfway along the edge as it appears in the photo
    const pts = []; for (let i = 0; i <= 200; i++) pts.push(P(sc.th0 + (sc.th1 - sc.th0) * i / 200, y));
    let L = [0]; for (let i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const half = L[L.length - 1] / 2; const i = L.findIndex((v) => v >= half); return pts[i];
  };
  return { tl: P(sc.th0, sc.y1), tr: P(sc.th1, sc.y1), bl: P(sc.th0, sc.y0), br: P(sc.th1, sc.y0), tm: mid(sc.y1), bm: mid(sc.y0) };
}
function outlinePts(sc) {
  const [lo, hi] = sc.limbs(), P = (th, y) => sc.project(sc.cylPoint(th, y));
  return { left: [P(lo, sc.y1 + 20), P(lo, sc.y0 - 20)], right: [P(hi, sc.y1 + 20), P(hi, sc.y0 - 20)] };
}
const jitter = (p, s) => [p[0] + s * gauss(rnd), p[1] + s * gauss(rnd)];
// the app's current mapping (v1.18 flattenLabel)
function oldMap(o, curveDeg) {
  const quad = (a, m, b) => { const c = [2 * m[0] - (a[0] + b[0]) / 2, 2 * m[1] - (a[1] + b[1]) / 2]; return (t) => { const s = 1 - t; return [s * s * a[0] + 2 * s * t * c[0] + t * t * b[0], s * s * a[1] + 2 * s * t * c[1] + t * t * b[1]]; }; };
  const T = quad(o.tl, o.tm, o.tr), B = quad(o.bl, o.bm, o.br);
  const th = curveDeg * Math.PI / 180, sh = th > .01 ? Math.sin(th / 2) : 0;
  const mapU = th > .01 ? (u) => .5 + Math.sin((u - .5) * th) / (2 * sh) : (u) => u;
  return (fx, fy) => { const t = mapU(fx), a = T(t), b = B(t); return [a[0] + (b[0] - a[0]) * fy, a[1] + (b[1] - a[1]) * fy]; };
}
function err(sc, map) {
  let s2 = 0, n = 0, mx = 0; const o = [0, 0];
  for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) {
    const fx = i / 20, fy = j / 20, p = map(fx, fy);
    if (!sc.pixelToLabelMM(p[0], p[1], o)) { s2 += 400; n++; continue; }   // missed the bottle: count 20 mm
    const e = Math.hypot(o[0] - fx * sc.lw, o[1] - fy * sc.lh); s2 += e * e; n++; mx = Math.max(mx, e);
  }
  return Math.sqrt(s2 / n);
}
const cfgs = [];
for (const elevation of [5, 15, 30]) for (const rotation of [0, 20]) for (const labelW of [75, 90]) cfgs.push({ elevation, rotation, labelW });
const sig = +(process.argv[2] ?? 3), trials = 12;
console.log(`noise ${sig} px on each dot, ${trials} trials per view; RMS error on the label, mm (median over trials)`);
console.log("view                     | app 90°  | app best° | new 6 dots | new + outline | + snapped edges");
const med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
for (const v of cfgs) {
  const sc = scene(v);
  if (!sc.labelVisible()) { console.log(JSON.stringify(v), "label hidden"); continue; }
  const wrapTrue = (sc.th1 - sc.th0) * 180 / Math.PI;
  const R = { a90: [], abest: [], n6: [], no: [], ne: [], wrap: [] };
  for (let t = 0; t < trials; t++) {
    const six = sixPoints(sc), o = {}; for (const k in six) o[k] = jitter(six[k], sig);
    const ol = outlinePts(sc); for (const s of ["left", "right"]) ol[s] = ol[s].map((p) => jitter(p, sig));
    R.a90.push(err(sc, oldMap(o, 90)));
    R.abest.push(Math.min(...[0, 30, 60, 90, 120, 150].map((c) => err(sc, oldMap(o, c)))));
    const cam = { f, cx: W / 2, cy: H / 2 };
    const m6 = fitLabel(o, cam); R.n6.push(m6 ? err(sc, (x, y) => m6.labelToImage(x, y)) : 99);
    const mo = fitLabel(o, cam, { outline: ol }); R.no.push(mo ? err(sc, (x, y) => mo.labelToImage(x, y)) : 99);
    // simulated edge snapping: 12 points along each edge, 1.5 px noise
    const P = (th, y) => sc.project(sc.cylPoint(th, y)), lin = (k) => Array.from({ length: k }, (_, i) => (i + 0.5) / k);
    const ed = { top: lin(12).map((u) => jitter(P(sc.th0 + u * (sc.th1 - sc.th0), sc.y1), 1.5)),
      bottom: lin(12).map((u) => jitter(P(sc.th0 + u * (sc.th1 - sc.th0), sc.y0), 1.5)),
      left: lin(12).map((u) => jitter(P(sc.th0, sc.y0 + u * (sc.y1 - sc.y0)), 1.5)),
      right: lin(12).map((u) => jitter(P(sc.th1, sc.y0 + u * (sc.y1 - sc.y0)), 1.5)) };
    const me = fitLabel(o, cam, { edges: ed, outline: ol }); R.ne.push(me ? err(sc, (x, y) => me.labelToImage(x, y)) : 99);
    R.wrap.push(m6 ? m6.wrapDeg : NaN);
  }
  const lab = `elev ${String(v.elevation).padStart(2)} rot ${String(v.rotation).padStart(2)} w ${v.labelW}`;
  console.log(`${lab.padEnd(24)} | ${med(R.a90).toFixed(2).padStart(8)} | ${med(R.abest).toFixed(2).padStart(9)} | ${med(R.n6).toFixed(2).padStart(10)} | ${med(R.no).toFixed(2).padStart(13)} | ${med(R.ne).toFixed(2).padStart(15)}`);
}
