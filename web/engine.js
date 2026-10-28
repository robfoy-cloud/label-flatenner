/* Label flattening engine: a JavaScript port of the Python package `labelflat`.
 * Synthetic bottle renderer, edge detection, cylinder geometry, grid of
 * homographies, and ground-truth scoring. Units are millimetres.
 * No dependencies; works in browsers and in Node (module.exports). */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ small maths
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const unit = (a) => scale(a, 1 / norm(a));
  const rad = (d) => (d * Math.PI) / 180;
  const deg = (r) => (r * 180) / Math.PI;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  function median(arr) {
    const a = Float64Array.from(arr).sort();
    const n = a.length;
    if (!n) return NaN;
    return n % 2 ? a[(n - 1) >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
  }
  function percentile(arr, p) {                    // numpy-style linear interpolation
    const a = Float64Array.from(arr).sort();
    const x = (p / 100) * (a.length - 1), i = Math.floor(x), w = x - i;
    return i + 1 < a.length ? a[i] * (1 - w) + a[i + 1] * w : a[i];
  }
  function interp(x, xs, ys) {                     // xs ascending
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const w = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] * (1 - w) + ys[hi] * w;
  }

  // Symmetric 3x3 eigen-decomposition (Jacobi). Returns {values, vectors[col]}.
  function eigSym3(A) {
    const a = [A[0].slice(), A[1].slice(), A[2].slice()];
    const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
      if (off < 1e-300) break;
      for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 3; k++) {            // A = J^T A J
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: [a[0][0], a[1][1], a[2][2]], vectors: [0, 1, 2].map((j) => [v[0][j], v[1][j], v[2][j]]) };
  }
  // Unit vector x minimising |L x| for rows L (the SVD null direction).
  function nullVector(rows) {
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const r of rows) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i][j] += r[i] * r[j];
    const { values, vectors } = eigSym3(M);
    let k = 0; for (let i = 1; i < 3; i++) if (values[i] < values[k]) k = i;
    return unit(vectors[k]);
  }
  // Principal / minor direction of a 2x2 scatter matrix.
  function eig2(sxx, sxy, syy) {
    const tr = sxx + syy, det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc;                      // largest eigenvalue
    let vx, vy;
    if (Math.abs(sxy) > 1e-300) { vx = l1 - syy; vy = sxy; }
    else if (sxx >= syy) { vx = 1; vy = 0; } else { vx = 0; vy = 1; }
    const n = Math.hypot(vx, vy);
    return { major: [vx / n, vy / n], minor: [-vy / n, vx / n] };
  }

  // Seeded RNG and Gaussian noise
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------------------------------------------------------------ scene
  function lookAt(C, target, rollDeg) {
    const fwd = unit(sub(target, C));
    const right = unit(cross(fwd, [0, 1, 0]));
    const down = cross(fwd, right);
    const a = rad(rollDeg), c = Math.cos(a), s = Math.sin(a);
    // Rr @ [right; down; fwd]
    return [
      sub(scale(right, c), scale(down, s)),
      add(scale(right, s), scale(down, c)),
      fwd,
    ];
  }

  class Scene {
    constructor(o) {
      this.R = o.R ?? 37; this.lw = o.labelW ?? 90; this.lh = o.labelH ?? 110;
      const half = this.lw / this.R / 2, c = rad(o.labelCentreDeg ?? 0);
      this.th0 = c - half; this.th1 = c + half;
      const ly = o.labelY ?? 0;
      this.y0 = ly - this.lh / 2; this.y1 = ly + this.lh / 2;
      this.C = [o.camSide, o.camHeight, o.camDist];
      this.Rw = lookAt(this.C, o.target, o.roll ?? 0);
      this.W = o.W; this.H = o.H; this.f = o.f;
      this.cx = this.W / 2; this.cy = this.H / 2;
    }
    static fromView(v) {
      const W = v.W ?? 720, H = v.H ?? 540;
      const e = rad(v.elevation ?? 17), D = v.distance ?? 260;
      return new Scene({
        R: v.R ?? 37, labelW: v.labelW ?? 75, labelH: v.labelH ?? 110, labelCentreDeg: v.rotation ?? 0,
        camSide: 0, camHeight: D * Math.sin(e), camDist: D * Math.cos(e),
        target: [D * Math.tan(rad(v.pan ?? 3)), 0, 0], roll: v.lean ?? 4,
        W, H, f: v.f ?? 0.75 * W,
      });
    }
    project(P) {
      const d = sub(P, this.C);
      const x = dot(this.Rw[0], d), y = dot(this.Rw[1], d), z = dot(this.Rw[2], d);
      return [(this.f * x) / z + this.cx, (this.f * y) / z + this.cy];
    }
    cylPoint(th, y) { return [this.R * Math.sin(th), y, this.R * Math.cos(th)]; }
    ray(u, v) {
      const x = (u - this.cx) / this.f, y = (v - this.cy) / this.f, R = this.Rw;
      return [x * R[0][0] + y * R[1][0] + R[2][0], x * R[0][1] + y * R[1][1] + R[2][1], x * R[0][2] + y * R[1][2] + R[2][2]];
    }
    // Front hit of the ray through pixel (u,v): writes [theta, y, shade] into out, returns ok.
    hit(u, v, out) {
      const d = this.ray(u, v), C = this.C;
      const a = d[0] * d[0] + d[2] * d[2];
      const b = 2 * (C[0] * d[0] + C[2] * d[2]);
      const c = C[0] * C[0] + C[2] * C[2] - this.R * this.R;
      const disc = b * b - 4 * a * c;
      if (disc <= 0) return false;
      const t = (-b - Math.sqrt(disc)) / (2 * a);
      const px = C[0] + t * d[0], py = C[1] + t * d[1], pz = C[2] + t * d[2];
      out[0] = Math.atan2(px, pz); out[1] = py;
      const vx = -d[0], vz = -d[2], vn = Math.max(Math.hypot(vx, vz), 1e-9);
      out[2] = clamp((px * vx + pz * vz) / this.R / vn, 0, 1);
      return true;
    }
    limbs() {
      const Dh = Math.hypot(this.C[0], this.C[2]), az = Math.atan2(this.C[0], this.C[2]);
      const a = Math.acos(this.R / Dh);
      return [az - a, az + a];
    }
    labelVisible() { const [lo, hi] = this.limbs(); return lo < this.th0 && this.th1 < hi; }
    // Perpendicular image distance (px) from each label side edge to the outline beside it.
    edgeGaps() {
      const [lo, hi] = this.limbs(), ym = 0.5 * (this.y0 + this.y1);
      const p = (t, y) => this.project(this.cylPoint(t, y));
      const gap = (edge, limb) => {
        const a = p(limb, ym - 20), b = p(limb, ym + 20), q = p(edge, ym);
        const tx = b[0] - a[0], ty = b[1] - a[1], n = Math.hypot(tx, ty);
        return Math.abs(((q[0] - a[0]) * ty - (q[1] - a[1]) * tx) / n);
      };
      return [this.th0 > lo ? gap(this.th0, lo) : 0, this.th1 < hi ? gap(this.th1, hi) : 0];
    }

    /* Ray-trace the photo. label = {w, h, data: Float32Array RGB in [0,1]}.
       Returns Uint8ClampedArray RGB (W*H*3). */
    render(label, ss = 2, noise = 1.5, seed = 1) {
      const { W, H } = this, Lw = label.w, Lh = label.h, L = label.data;
      const acc = new Float32Array(W * H * 3);
      const bg = [200 / 255, 205 / 255, 215 / 255];
      const out3 = [0, 0, 0];
      const span = this.th1 - this.th0, ysp = this.y1 - this.y0;
      for (let oy = 0; oy < ss; oy++) for (let ox = 0; ox < ss; ox++) {
        const dx = (ox + 0.5) / ss - 0.5, dy = (oy + 0.5) / ss - 0.5;
        for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
          const i = (v * W + u) * 3;
          let r = bg[0], g = bg[1], b = bg[2];
          if (this.hit(u + dx, v + dy, out3)) {
            const th = out3[0], y = out3[1], sh = out3[2];
            const k = 0.5 + 0.8 * sh, hl = 0.25 * Math.exp(-(((sh - 0.93) / 0.03) ** 2));
            r = 0.05 * k + hl; g = 0.16 * k + hl; b = 0.08 * k + hl;
            if (th >= this.th0 && th <= this.th1 && y >= this.y0 && y <= this.y1) {
              const lx = clamp(((th - this.th0) / span) * (Lw - 1), 0, Lw - 1);
              const ly = clamp(((this.y1 - y) / ysp) * (Lh - 1), 0, Lh - 1);
              const x0 = Math.floor(lx), y0 = Math.floor(ly);
              const x1 = Math.min(x0 + 1, Lw - 1), y1 = Math.min(y0 + 1, Lh - 1);
              const fx = lx - x0, fy = ly - y0, m = 0.55 + 0.45 * sh;
              const i00 = (y0 * Lw + x0) * 3, i01 = (y0 * Lw + x1) * 3, i10 = (y1 * Lw + x0) * 3, i11 = (y1 * Lw + x1) * 3;
              const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
              r = m * (L[i00] * w00 + L[i01] * w01 + L[i10] * w10 + L[i11] * w11);
              g = m * (L[i00 + 1] * w00 + L[i01 + 1] * w01 + L[i10 + 1] * w10 + L[i11 + 1] * w11);
              b = m * (L[i00 + 2] * w00 + L[i01 + 2] * w01 + L[i10 + 2] * w10 + L[i11 + 2] * w11);
            }
          }
          acc[i] += r; acc[i + 1] += g; acc[i + 2] += b;
        }
      }
      const rnd = mulberry32(seed), out = new Uint8ClampedArray(W * H * 3), inv = 255 / (ss * ss);
      for (let i = 0; i < out.length; i += 2) {
        const u1 = Math.max(rnd(), 1e-12), u2 = rnd(), m = Math.sqrt(-2 * Math.log(u1)) * noise;
        out[i] = acc[i] * inv + m * Math.cos(2 * Math.PI * u2);
        if (i + 1 < out.length) out[i + 1] = acc[i + 1] * inv + m * Math.sin(2 * Math.PI * u2);
      }
      return out;
    }
    // Ground truth: label mm (u across from the left edge, v down from the top) seen at pixel (x,y).
    pixelToLabelMM(x, y, out) {
      const h = [0, 0, 0];
      if (!this.hit(x, y, h)) return false;
      out[0] = (h[0] - this.th0) * this.R; out[1] = this.y1 - h[1];
      return true;
    }
    exactFeatures(n = 400) {
      const [lo, hi] = this.limbs();
      const lin = (a, b, k) => Array.from({ length: k }, (_, i) => a + ((b - a) * i) / (k - 1));
      const ruling = (t, ys) => ys.map((y) => this.project(this.cylPoint(t, y)));
      const ys = lin(-400, 400, 4 * n), yl = lin(this.y0, this.y1, n), th = lin(this.th0, this.th1, n);
      const inFrame = (p) => p[0] > 0 && p[0] < this.W && p[1] > 0 && p[1] < this.H;
      return {
        sil_l: ruling(lo, ys).filter(inFrame), sil_r: ruling(hi, ys).filter(inFrame),
        edge_l: ruling(this.th0, yl), edge_r: ruling(this.th1, yl),
        top: th.map((t) => this.project(this.cylPoint(t, this.y1))),
        bot: th.map((t) => this.project(this.cylPoint(t, this.y0))),
      };
    }
  }

  // ------------------------------------------------------------ detection
  function subpix(p, idx, thr) {
    const a = p(idx - 1), b = p(idx);
    return b !== a ? idx - 1 + (thr - a) / (b - a) : idx;
  }
  // Binary erode/dilate with a k x k square (separable), mask as Uint8Array 0/1.
  function morph(mask, W, H, k, isDilate) {
    const r = k >> 1, tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      let acc = isDilate ? 0 : 1;
      for (let d = -r; d <= r; d++) {
        const uu = u + d;
        const val = uu < 0 || uu >= W ? (isDilate ? 0 : 1) : mask[v * W + uu];
        if (isDilate) { if (val) { acc = 1; break; } } else if (!val) { acc = 0; break; }
      }
      tmp[v * W + u] = acc;
    }
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      let acc = isDilate ? 0 : 1;
      for (let d = -r; d <= r; d++) {
        const vv = v + d;
        const val = vv < 0 || vv >= H ? (isDilate ? 0 : 1) : tmp[vv * W + u];
        if (isDilate) { if (val) { acc = 1; break; } } else if (!val) { acc = 0; break; }
      }
      out[v * W + u] = acc;
    }
    return out;
  }

  function detect(img, W, H) {
    const N = W * H;
    // background colour = median of a 5 px border
    const bc = [[], [], []];
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      if (v < 5 || v >= H - 5 || u < 5 || u >= W - 5) for (let c = 0; c < 3; c++) bc[c].push(img[(v * W + u) * 3 + c]);
    }
    const bg = bc.map(median);
    const dist = new Float32Array(N), Rch = new Float32Array(N);
    let raw = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const r = img[3 * i], g = img[3 * i + 1], b = img[3 * i + 2];
      dist[i] = Math.max(Math.abs(r - bg[0]), Math.abs(g - bg[1]), Math.abs(b - bg[2]));
      Rch[i] = r;
      raw[i] = dist[i] > 25 || r - b > 5 ? 1 : 0;
    }
    const notbg = morph(morph(raw, W, H, 3, false), W, H, 3, true);   // open 3x3

    // silhouettes
    const silL = [], silR = [];
    for (let v = 2; v < H - 2; v++) {
      let i0 = -1, i1 = -1, cnt = 0;
      for (let u = 0; u < W; u++) if (notbg[v * W + u]) { if (i0 < 0) i0 = u; i1 = u; cnt++; }
      if (cnt < 20 || i0 < 4 || i1 > W - 5) continue;
      const row = (k) => dist[v * W + k];
      let lvl = (row(i0 + 4) + row(i0 - 3)) / 2;
      silL.push([row(i0) !== row(i0 - 1) ? subpix(row, i0, lvl) : i0, v]);
      const rr = (k) => dist[v * W + (W - 1 - k)], j = W - 1 - i1;
      lvl = (rr(j + 4) + rr(j - 3)) / 2;
      silR.push([W - 1 - subpix(rr, j, lvl), v]);
    }

    // label mask: cream paper is red-dominant, glass green-dominant
    let lab = new Uint8Array(N);
    for (let i = 0; i < N; i++) lab[i] = Rch[i] > 110 && notbg[i] ? 1 : 0;
    lab = morph(morph(lab, W, H, 5, true), W, H, 5, false);           // close 5x5
    // largest 8-connected component
    const comp = new Int32Array(N).fill(-1), stack = new Int32Array(N);
    let best = -1, bestSize = 0, id = 0;
    for (let s = 0; s < N; s++) {
      if (!lab[s] || comp[s] >= 0) continue;
      let sp = 0, size = 0; stack[sp++] = s; comp[s] = id;
      while (sp) {
        const p = stack[--sp]; size++;
        const pu = p % W, pv = (p - pu) / W;
        for (let dv = -1; dv <= 1; dv++) for (let du = -1; du <= 1; du++) {
          const qu = pu + du, qv = pv + dv;
          if (qu < 0 || qv < 0 || qu >= W || qv >= H) continue;
          const q = qv * W + qu;
          if (lab[q] && comp[q] < 0) { comp[q] = id; stack[sp++] = q; }
        }
      }
      if (size > bestSize) { bestSize = size; best = id; }
      id++;
    }
    if (best < 0) throw new Error("no label found in the photo");
    for (let i = 0; i < N; i++) lab[i] = comp[i] === best ? 1 : 0;
    // fill holes: flood the outside (4-connected) from the corner
    const outside = new Uint8Array(N);
    { let sp = 0; stack[sp++] = 0; outside[0] = 1;
      while (sp) {
        const p = stack[--sp], pu = p % W, pv = (p - pu) / W;
        const nb = [pu > 0 ? p - 1 : -1, pu < W - 1 ? p + 1 : -1, pv > 0 ? p - W : -1, pv < H - 1 ? p + W : -1];
        for (const q of nb) if (q >= 0 && !lab[q] && !outside[q]) { outside[q] = 1; stack[sp++] = q; }
      } }
    for (let i = 0; i < N; i++) if (!outside[i]) lab[i] = 1;

    let c0 = W, c1 = -1;
    for (let i = 0; i < N; i++) if (lab[i]) { const u = i % W; if (u < c0) c0 = u; if (u > c1) c1 = u; }
    let top = [], bot = [];
    for (let u = c0 + 3; u < c1 - 2; u++) {
      let f0 = -1, f1 = -1, cnt = 0;
      for (let v = 0; v < H; v++) if (lab[v * W + u]) { if (f0 < 0) f0 = v; f1 = v; cnt++; }
      if (cnt < 10 || f0 < 3 || f1 > H - 4) continue;
      const p = (k) => Rch[k * W + u];
      let thr = (p(f0 - 3) + p(f0 + 1)) / 2;
      top.push([u, subpix(p, f0, thr)]);
      const pr = (k) => Rch[(H - 1 - k) * W + u], j = H - 1 - f1;
      thr = (pr(j - 3) + pr(j + 1)) / 2;
      bot.push([u, H - 1 - subpix(pr, j, thr)]);
    }
    if (top.length < 10) throw new Error("could not trace the label's top and bottom edges");

    const sideEdges = (vmin, vmax) => {
      const el = [], er = [];
      for (let v = Math.max(3, Math.trunc(vmin)); v < Math.min(H - 3, Math.trunc(vmax)); v++) {
        let x0 = -1, x1 = -1, cnt = 0;
        for (let u = 0; u < W; u++) if (lab[v * W + u]) { if (x0 < 0) x0 = u; x1 = u; cnt++; }
        if (cnt < 5 || x0 < 3 || x1 > W - 4) continue;
        const p = (k) => Rch[v * W + k];
        let thr = (p(x0 - 3) + Math.max(p(x0), p(x0 + 1), p(x0 + 2))) / 2;
        el.push([subpix(p, x0, thr), v]);
        const pr = (k) => Rch[v * W + (W - 1 - k)], j = W - 1 - x1;
        thr = (pr(j - 3) + Math.max(pr(j), pr(j + 1), pr(j + 2))) / 2;
        er.push([W - 1 - subpix(pr, j, thr), v]);
      }
      return [el, er];
    };
    const fitUofV = (e) => {                       // least-squares u = a v + b
      let sv = 0, su = 0, svv = 0, svu = 0; const n = e.length;
      for (const [u, v] of e) { sv += v; su += u; svv += v * v; svu += v * u; }
      const a = (n * svu - sv * su) / (n * svv - sv * sv), b = (su - a * sv) / n;
      return (v) => a * v + b;
    };
    let el, er;
    for (let it = 0; it < 2; it++) {
      [el, er] = sideEdges(percentile(top.map((p) => p[1]), 90) + 3, percentile(bot.map((p) => p[1]), 10) - 3);
      if (el.length < 5 || er.length < 5) throw new Error("could not find the label's side edges");
      const fl = fitUofV(el), fr = fitUofV(er);
      const keep = (c) => c.filter(([u, v]) => u > fl(v) + 3 && u < fr(v) - 3);
      top = keep(top); bot = keep(bot);
    }
    [el, er] = sideEdges(Math.max(...top.map((p) => p[1])) + 3, Math.min(...bot.map((p) => p[1])) - 3);
    if (el.length < 5 || er.length < 5) throw new Error("could not find the label's side edges");
    return { sil_l: silL, sil_r: silR, edge_l: el, edge_r: er, top, bot };
  }

  // ------------------------------------------------------------ geometry
  function fitLine(q) {                             // [a,b,c], a^2+b^2=1
    let mx = 0, my = 0; for (const p of q) { mx += p[0]; my += p[1]; }
    mx /= q.length; my /= q.length;
    let sxx = 0, sxy = 0, syy = 0;
    for (const p of q) { const dx = p[0] - mx, dy = p[1] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const n = eig2(sxx, sxy, syy).minor;
    return [n[0], n[1], -(n[0] * mx + n[1] * my)];
  }
  function fitLineThrough(q, V) {
    if (Math.abs(V[2]) > 1e-9) {
      const p0 = [V[0] / V[2], V[1] / V[2]];
      let sxx = 0, sxy = 0, syy = 0;
      for (const p of q) { const dx = p[0] - p0[0], dy = p[1] - p0[1]; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
      const d = eig2(sxx, sxy, syy).major, n = [-d[1], d[0]];
      return [n[0], n[1], -(n[0] * p0[0] + n[1] * p0[1])];
    }
    const h = Math.hypot(V[0], V[1]), n = [-V[1] / h, V[0] / h];
    return [n[0], n[1], -median(q.map((p) => p[0] * n[0] + p[1] * n[1]))];
  }
  function trimLine(q, f) {                         // drop rows that missed the edge
    for (let it = 0; it < 3; it++) {
      const l = fitLine(q), r = q.map((p) => p[0] * l[0] + p[1] * l[1] + l[2]);
      const mr = median(r), mad = median(r.map((x) => Math.abs(x - mr))) * 1.4826 + 1e-12;
      const thr = Math.max(2.5 * mad, 0.3 / f);
      q = q.filter((_, i) => Math.abs(r[i] - mr) < thr);
    }
    return q;
  }
  function cleanCurve(q, f, win = 9) {              // free curve: drop isolated spikes only
    q = q.slice().sort((a, b) => a[0] - b[0]);
    const h = win >> 1, n = q.length, r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const w = [];
      for (let k = i - h; k <= i + h; k++) w.push(q[clamp(k, 0, n - 1)][1]);
      r[i] = q[i][1] - median(w);
    }
    const mad = median(Array.from(r, Math.abs)) * 1.4826;
    const thr = Math.max(4 * mad, 1 / f);
    return q.filter((_, i) => Math.abs(r[i]) < thr);
  }

  class Model {
    constructor(feats, f, centre) {
      this.f = f; this.c = centre;
      const norm2 = (pts) => pts.map((p) => [(p[0] - centre[0]) / f, (p[1] - centre[1]) / f]);
      const nq = {};
      const keys = ["sil_l", "sil_r", "edge_l", "edge_r"];
      for (const k of keys) { nq[k] = trimLine(norm2(feats[k]), f); if (nq[k].length < 5) throw new Error(`too few points on ${k}`); }
      for (const k of ["top", "bot"]) nq[k] = cleanCurve(norm2(feats[k]), f);
      this.nq = nq;
      const normL = (l) => { const h = Math.hypot(l[0], l[1]); return [l[0] / h, l[1] / h, l[2] / h]; };
      let V = nullVector(keys.map((k) => normL(fitLine(nq[k]))));
      this.lines = {};
      for (const k of keys) this.lines[k] = fitLineThrough(nq[k], V);
      this.V = V = nullVector(keys.map((k) => this.lines[k]));
      let d = unit(V); if (d[1] > 0) d = scale(d, -1);
      this.d = d;
      const t = ["sil_l", "sil_r"].map((k) => { const ti = unit(cross(d, unit(this.lines[k]))); return ti[2] > 0 ? ti : scale(ti, -1); });
      const b = unit(add(t[0], t[1]));
      this.alpha = 0.5 * Math.acos(clamp(dot(t[0], t[1]), -1, 1));
      this.c0 = scale(b, 1 / Math.sin(this.alpha));  // axis foot point, radius = 1
      this.e1 = scale(b, -1);
      this.e2 = cross(d, this.e1);
      this.thL = this.edgeTheta(this.lines.edge_l);
      this.thR = this.edgeTheta(this.lines.edge_r);
      if (this.thL > this.thR) { this.e2 = scale(this.e2, -1); this.thL = -this.thL; this.thR = -this.thR; }
    }
    edgeTheta(l) {
      const n = sub(l, scale(this.d, dot(l, this.d)));
      const A = dot(n, this.e1), B = dot(n, this.e2), C0 = dot(n, this.c0);
      const rho = Math.hypot(A, B), phi = Math.atan2(B, A), a = Math.acos(clamp(-C0 / rho, -1, 1));
      const c = [phi + a, phi - a];
      return norm(this.surface(c[0], 0)) <= norm(this.surface(c[1], 0)) ? c[0] : c[1];
    }
    surface(th, s) {
      const c = Math.cos(th), sn = Math.sin(th);
      return [0, 1, 2].map((i) => this.c0[i] + c * this.e1[i] + sn * this.e2[i] + s * this.d[i]);
    }
    toPixels(P) { return [(P[0] / P[2]) * this.f + this.c[0], (P[1] / P[2]) * this.f + this.c[1]]; }
    sOnCurve(th, curve) {
      const Q = this.surface(th, 0);
      let l = cross(this.V, Q); const h = Math.hypot(l[0], l[1]); l = scale(l, 1 / h);
      const sd = curve.map((p) => p[0] * l[0] + p[1] * l[1] + l[2]);
      let i = -1;
      for (let k = 0; k + 1 < sd.length; k++) if (Math.sign(sd[k]) !== Math.sign(sd[k + 1])) { i = k; break; }
      if (i < 0) i = Math.abs(sd[0]) < Math.abs(sd[sd.length - 1]) ? 0 : sd.length - 2;
      const w = sd[i] / (sd[i] - sd[i + 1]);
      const p = [curve[i][0] * (1 - w) + curve[i + 1][0] * w, curve[i][1] * (1 - w) + curve[i + 1][1] * w];
      const ray = [p[0], p[1], 1], md = scale(this.d, -1);
      // least squares [ray, -d] [t, s]^T = Q
      const a11 = dot(ray, ray), a12 = dot(ray, md), a22 = dot(md, md);
      const b1 = dot(ray, Q), b2 = dot(md, Q), det = a11 * a22 - a12 * a12;
      return (a11 * b2 - a12 * b1) / det;
    }
    fitHeights(n = 240) {
      this.thS = new Float64Array(n); this.sTop = new Float64Array(n); this.sBot = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const th = this.thL + ((this.thR - this.thL) * i) / (n - 1);
        this.thS[i] = th; this.sTop[i] = this.sOnCurve(th, this.nq.top); this.sBot[i] = this.sOnCurve(th, this.nq.bot);
      }
      this.width = this.thR - this.thL;
      this.height = median(Array.from(this.sBot, (b, i) => Math.abs(b - this.sTop[i])));
      return this;
    }
    labelToImage(fx, fy) {
      const th = this.thL + fx * (this.thR - this.thL);
      const st = interp(th, this.thS, this.sTop), sb = interp(th, this.thS, this.sBot);
      return this.toPixels(this.surface(th, st + fy * (sb - st)));
    }
  }

  // ------------------------------------------------------------ grids & warping
  function modelGrid(m, nc, nr) {
    const nodes = [];
    for (let j = 0; j <= nr; j++) { const row = []; for (let i = 0; i <= nc; i++) row.push(m.labelToImage(i / nc, j / nr)); nodes.push(row); }
    return nodes;
  }
  function naiveGrid(feats, m, nc, nr) {
    const Vp = [(m.V[0] / m.V[2]) * m.f + m.c[0], (m.V[1] / m.V[2]) * m.f + m.c[1]];
    const ang = (k) => {
      let mx = 0, my = 0; for (const p of feats[k]) { mx += p[0]; my += p[1]; }
      return Math.atan2(my / feats[k].length - Vp[1], mx / feats[k].length - Vp[0]);
    };
    const a0 = ang("edge_l"), a1 = ang("edge_r");
    const cols = [];
    for (let i = 0; i <= nc; i++) {
      const a = a0 + ((a1 - a0) * i) / nc, nv = [-Math.sin(a), Math.cos(a)];
      const pts = ["top", "bot"].map((k) => {
        const cur = feats[k], sd = cur.map((p) => (p[0] - Vp[0]) * nv[0] + (p[1] - Vp[1]) * nv[1]);
        let idx = -1;
        for (let q = 0; q + 1 < sd.length; q++) if (Math.sign(sd[q]) !== Math.sign(sd[q + 1])) { idx = q; break; }
        if (idx < 0) idx = Math.abs(sd[0]) < Math.abs(sd[sd.length - 1]) ? 0 : sd.length - 2;
        const w = sd[idx] / (sd[idx] - sd[idx + 1]);
        return [cur[idx][0] * (1 - w) + cur[idx + 1][0] * w, cur[idx][1] * (1 - w) + cur[idx + 1][1] * w];
      });
      cols.push(pts);
    }
    const nodes = [];
    for (let j = 0; j <= nr; j++) {
      const t = j / nr, row = [];
      for (let i = 0; i <= nc; i++) { const [p, q] = cols[i]; row.push([p[0] * (1 - t) + q[0] * t, p[1] * (1 - t) + q[1] * t]); }
      nodes.push(row);
    }
    return nodes;
  }
  // Homography mapping 4 src points to 4 dst points (solve 8x8).
  function homography(src, dst) {
    const A = [], b = [];
    for (let k = 0; k < 4; k++) {
      const [x, y] = src[k], [u, v] = dst[k];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    for (let c = 0; c < 8; c++) {                  // Gaussian elimination, partial pivoting
      let p = c; for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
      for (let r = c + 1; r < 8; r++) {
        const m = A[r][c] / A[c][c];
        for (let k = c; k < 8; k++) A[r][k] -= m * A[c][k];
        b[r] -= m * b[c];
      }
    }
    const h = new Array(8);
    for (let r = 7; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < 8; k++) s -= A[r][k] * h[k]; h[r] = s / A[r][r]; }
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }
  /* Piecewise-homography warp. Returns {rgb: Uint8ClampedArray, map: Float32Array (x,y per pixel)}. */
  function gridWarp(img, W, H, nodes, outW, outH) {
    const nr = nodes.length - 1, nc = nodes[0].length - 1, cw = outW / nc, ch = outH / nr;
    const Hs = [];
    for (let j = 0; j < nr; j++) for (let i = 0; i < nc; i++) {
      Hs.push(homography(
        [[i * cw, j * ch], [(i + 1) * cw, j * ch], [(i + 1) * cw, (j + 1) * ch], [i * cw, (j + 1) * ch]],
        [nodes[j][i], nodes[j][i + 1], nodes[j + 1][i + 1], nodes[j + 1][i]]));
    }
    const rgb = new Uint8ClampedArray(outW * outH * 3), map = new Float32Array(outW * outH * 2);
    for (let y = 0; y < outH; y++) {
      const cj = Math.min(nr - 1, Math.floor((y + 0.5) / ch));
      for (let x = 0; x < outW; x++) {
        const ci = Math.min(nc - 1, Math.floor((x + 0.5) / cw)), h = Hs[cj * nc + ci];
        const X = x + 0.5, Y = y + 0.5, w = h[6] * X + h[7] * Y + 1;
        const sx = (h[0] * X + h[1] * Y + h[2]) / w, sy = (h[3] * X + h[4] * Y + h[5]) / w;
        const o = y * outW + x; map[2 * o] = sx; map[2 * o + 1] = sy;
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        if (x0 < 0 || y0 < 0 || x0 >= W - 1 || y0 >= H - 1) continue;
        const fx = sx - x0, fy = sy - y0, i00 = (y0 * W + x0) * 3;
        for (let c = 0; c < 3; c++) {
          rgb[3 * o + c] = (img[i00 + c] * (1 - fx) + img[i00 + 3 + c] * fx) * (1 - fy)
            + (img[i00 + 3 * W + c] * (1 - fx) + img[i00 + 3 * W + 3 + c] * fx) * fy;
        }
      }
    }
    return { rgb, map };
  }

  // ------------------------------------------------------------ pipeline
  function flatten(img, W, H, f, opts = {}) {
    const nc = opts.nc ?? 48, nr = opts.nr ?? 32, outW = opts.outW ?? 360;
    const feats = opts.feats ?? detect(img, W, H);
    const model = new Model(feats, f, opts.centre ?? [W / 2, H / 2]).fitHeights();
    const aspect = model.width / model.height;
    const outH = Math.max(8, Math.round(outW / aspect));
    const nodes = modelGrid(model, nc, nr), naiveNodes = naiveGrid(feats, model, nc, nr);
    return {
      feats, model, aspect, outW, outH, nodes, naiveNodes,
      cyl: gridWarp(img, W, H, nodes, outW, outH),
      naive: gridWarp(img, W, H, naiveNodes, outW, outH),
    };
  }
  // Error in label mm: trace output pixels back through the true geometry.
  function score(scene, map, outW, outH, step = 4) {
    let s2 = 0, mx = 0, n = 0; const o = [0, 0];
    for (let y = 0; y < outH; y += step) for (let x = 0; x < outW; x += step) {
      const k = y * outW + x;
      if (!scene.pixelToLabelMM(map[2 * k], map[2 * k + 1], o)) continue;
      const e = Math.hypot(o[0] - ((x + 0.5) / outW) * scene.lw, o[1] - ((y + 0.5) / outH) * scene.lh);
      s2 += e * e; if (e > mx) mx = e; n++;
    }
    return n ? { rms: Math.sqrt(s2 / n), max: mx } : { rms: Infinity, max: Infinity };
  }

  const api = { Scene, detect, Model, flatten, score, modelGrid, naiveGrid, gridWarp, homography, deg, rad };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelFlat = api;
})(typeof self !== "undefined" ? self : this);
