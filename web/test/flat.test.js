// Flat labels: corners from four dots with errors, then snapped to the straight edges. node web/test/flat.test.js
const fs = require("fs"), G = require("../labelgeo.js");
const b = fs.readFileSync(`/tmp/claude-0/lab_75x100.f32`), L = new Float32Array(b.buffer, b.byteOffset, b.length / 4), Lw = 600, Lh = 800;
const W = 1000, H = 1200; let seed = 3; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
let fails = 0, n = 0, worstAll = [];
for (let t = 0; t < 12; t++) {
  const c = { tl: [250 + 60 * gauss(), 250 + 40 * gauss()], tr: [750 + 60 * gauss(), 230 + 40 * gauss()], br: [770 + 50 * gauss(), 950 + 40 * gauss()], bl: [230 + 50 * gauss(), 970 + 40 * gauss()] };
  // inverse map pixel -> label via homography from image quad to unit square
  const Rq = G.fitFlat ? null : null;
  const hq = (() => { const A = [], bb = []; const src = [c.tl, c.tr, c.br, c.bl], dst = [[0, 0], [1, 0], [1, 1], [0, 1]]; return { src, dst }; })();
  // build forward map via planeMap trick: use G internals? replicate homography solve
  const solve = (A, bv) => { const n = bv.length, M = A.map((r, i) => r.concat([bv[i]])); for (let k = 0; k < n; k++) { let p = k; for (let r = k + 1; r < n; r++) if (Math.abs(M[r][k]) > Math.abs(M[p][k])) p = r; [M[k], M[p]] = [M[p], M[k]]; for (let r = k + 1; r < n; r++) { const f = M[r][k] / M[k][k]; for (let j = k; j <= n; j++) M[r][j] -= f * M[k][j]; } } const x = new Array(n); for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j]; x[r] = s / M[r][r]; } return x; };
  const homog = (src, dst) => { const A = [], bv = []; for (let k = 0; k < 4; k++) { const [x, y] = src[k], [u, v] = dst[k]; A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); bv.push(u); A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); bv.push(v); } return solve(A, bv).concat([1]); };
  const hi = homog(hq.src, hq.dst), ap = (h, x, y) => { const w = h[6] * x + h[7] * y + 1; return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w]; };
  const S = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const [u, v] = ap(hi, x + 0.5, y + 0.5), i = (y * W + x) * 4; let r = 60, g = 50, bl = 45;
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) { const j = (Math.min(Lh - 1, Math.floor(v * Lh)) * Lw + Math.min(Lw - 1, Math.floor(u * Lw))) * 3; r = 255 * L[j]; g = 255 * L[j + 1]; bl = 255 * L[j + 2]; }
    S[i] = r + 2 * gauss(); S[i + 1] = g + 2 * gauss(); S[i + 2] = bl + 2 * gauss(); S[i + 3] = 255;
  }
  const hpx = 720, dots = {}; for (const k of ["tl", "tr", "br", "bl"]) dots[k] = [c[k][0] + 0.04 * hpx * gauss(), c[k][1] + 0.04 * hpx * gauss()];
  dots.tm = [(dots.tl[0] + dots.tr[0]) / 2, (dots.tl[1] + dots.tr[1]) / 2]; dots.bm = [(dots.bl[0] + dots.br[0]) / 2, (dots.bl[1] + dots.br[1]) / 2];
  const r = G.fitFlat(G.grayOf(S, W, H), W, H, dots);
  // edge error in label mm (75x100): map final boundary to label coords
  let worst = 0;
  for (let i = 0; i <= 40; i++) for (const [sd, fx, fy] of [["t", i / 40, 0], ["b", i / 40, 1], ["l", 0, i / 40], ["r", 1, i / 40]]) {
    const p = r.final.at(fx, fy), [u, v] = ap(hi, p[0], p[1]);
    const e = sd === "t" ? v * 100 : sd === "b" ? (1 - v) * 100 : sd === "l" ? u * 75 : (1 - u) * 75; worst = Math.max(worst, Math.abs(e));
  }
  n++; if (worst > 2) fails++; worstAll.push(worst.toFixed(2) + (r.snapped ? '' : '*'));
}
console.log('flat: fails', fails + '/' + n, worstAll.join(' '));
