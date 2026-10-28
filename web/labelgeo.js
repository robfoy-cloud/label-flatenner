/* LabelGeo: flatten a wine label photographed on a round bottle.
 * From github.com/robfoy-cloud/label-flatenner (web/labelgeo.js).
 *
 * 1. fit:   a cylinder and label to the six-dot outline (corners plus a point on
 *           the top and bottom edges), using the side edges' vanishing point and
 *           the fact that the top and bottom edges are circles on the bottle.
 * 2. snap:  look for the label's real edges in the photo near the fitted ones,
 *           and refit with those points (the dots then only guide the search).
 * 3. warp:  map the label flat with a grid of small homographies.
 * Coordinates are image pixels; cam = {f, cx, cy} (focal length, principal point).
 */
(function (root) {
  "use strict";
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const unit = (a) => scl(a, 1 / Math.hypot(a[0], a[1], a[2]));
  const median = (v) => { const a = Float64Array.from(v).sort(); const n = a.length; return n ? (n % 2 ? a[(n - 1) >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2) : NaN; };
  const KEYS = ["tl", "tm", "tr", "br", "bm", "bl"];

  /* Default camera for a phone photo: 26 mm-equivalent lens, centred. */
  function defaultCam(W, H) { return { f: (26 / 43.27) * Math.hypot(W, H), cx: W / 2, cy: H / 2 }; }

  // ---------------------------------------------------------------- model
  // x = [axis tilt a, axis tilt b, axis foot p, axis foot q, theta left, theta right, s top, s bottom]
  function geom(x, B) {
    const d = unit(add(B.d0, add(scl(B.u1, x[0]), scl(B.u2, x[1]))));
    const p1 = unit(cross(d, B.w)), p2 = cross(d, p1);
    const c0 = add(scl(p1, x[2]), scl(p2, x[3])), D = Math.hypot(c0[0], c0[1], c0[2]);
    const e1 = scl(c0, -1 / D);
    return { d, c0, D, e1, e2: cross(d, e1), thl: x[4], thr: x[5], st: x[6], sb: x[7] };
  }
  const surf = (g, th, s) => { const c = Math.cos(th), n = Math.sin(th); return [g.c0[0] + c * g.e1[0] + n * g.e2[0] + s * g.d[0], g.c0[1] + c * g.e1[1] + n * g.e2[1] + s * g.d[1], g.c0[2] + c * g.e1[2] + n * g.e2[2] + s * g.d[2]]; };
  const proj = (P) => [P[0] / P[2], P[1] / P[2]];
  function curvePts(g, s, n) { const a = []; for (let i = 0; i <= n; i++) a.push(proj(surf(g, g.thl + ((g.thr - g.thl) * i) / n, s))); return a; }
  function distPoly(q, pts) {
    let best = Infinity;
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i], vx = b[0] - a[0], vy = b[1] - a[1], L2 = vx * vx + vy * vy || 1e-30;
      const t = Math.max(0, Math.min(1, ((q[0] - a[0]) * vx + (q[1] - a[1]) * vy) / L2));
      best = Math.min(best, Math.hypot(q[0] - a[0] - t * vx, q[1] - a[1] - t * vy));
    }
    return best;
  }
  function residuals(x, B, O) {
    const g = geom(x, B); if (!(g.D > 1.0001)) return null;
    const r = [], w = O.dotW;
    const corner = (q, th, s) => { const p = proj(surf(g, th, s)); r.push(w * (p[0] - q[0]), w * (p[1] - q[1])); };
    corner(O.tl, g.thl, g.st); corner(O.tr, g.thr, g.st); corner(O.bl, g.thl, g.sb); corner(O.br, g.thr, g.sb);
    const T = curvePts(g, g.st, 32), Bt = curvePts(g, g.sb, 32);
    r.push(w * distPoly(O.tm, T), w * distPoly(O.bm, Bt));
    if (O.edges) {
      for (const q of O.edges.top) r.push(distPoly(q, T));
      for (const q of O.edges.bottom) r.push(distPoly(q, Bt));
      for (const [k, th] of [["left", g.thl], ["right", g.thr]]) {
        const a = proj(surf(g, th, g.st)), b = proj(surf(g, th, g.sb)), vx = b[0] - a[0], vy = b[1] - a[1], L = Math.hypot(vx, vy);
        for (const q of O.edges[k]) r.push(((q[0] - a[0]) * vy - (q[1] - a[1]) * vx) / L);
      }
    }
    const lim = Math.acos(1 / g.D);               // the label must stay on the visible side
    for (const th of [g.thl, g.thr]) r.push(Math.max(0, Math.abs(th) - lim));
    return r;
  }
  function solve(A, b) {
    const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (!(Math.abs(M[p][c]) > 1e-300)) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  function lm(x, B, O, iters = 80) {
    const cost = (v) => v.reduce((s, e) => s + e * e, 0);
    let r = residuals(x, B, O); if (!r) return null;
    let c = cost(r), lambda = 1e-3;
    for (let it = 0; it < iters; it++) {
      const n = x.length, m = r.length, J = [];
      for (let j = 0; j < n; j++) {
        const h = 1e-6 * Math.max(1, Math.abs(x[j])), xp = x.slice(); xp[j] += h;
        const rp = residuals(xp, B, O); J.push(rp ? rp.map((v, i) => (v - r[i]) / h) : new Array(m).fill(0));
      }
      const A = [], gr = [];
      for (let i = 0; i < n; i++) {
        let s = 0; for (let k = 0; k < m; k++) s += J[i][k] * r[k]; gr.push(-s); A.push([]);
        for (let j = 0; j < n; j++) { let t = 0; for (let k = 0; k < m; k++) t += J[i][k] * J[j][k]; A[i].push(t); }
      }
      let ok = false;
      for (let tries = 0; tries < 8; tries++) {
        const step = solve(A.map((row, i) => row.map((v, j) => (i === j ? v * (1 + lambda) + 1e-12 : v))), gr);
        if (step) {
          const xn = x.map((v, i) => v + step[i]), rn = residuals(xn, B, O);
          if (rn) { const cn = cost(rn); if (cn < c) { const tiny = c - cn < 1e-10 * (1 + c); x = xn; r = rn; c = cn; lambda = Math.max(1e-9, lambda / 3); ok = !tiny; break; } }
        }
        lambda *= 10;
      }
      if (!ok) break;
    }
    return { x, cost: c, n: r.length };
  }

  // Normalised observations (divide by f); edges optional.
  function observe(dots, cam, edges, dotW) {
    const n = (p) => [(p[0] - cam.cx) / cam.f, (p[1] - cam.cy) / cam.f];
    const O = { dotW: dotW ?? 1 };
    for (const k of KEYS) O[k] = n(dots[k]);
    if (edges) O.edges = { top: edges.top.map(n), bottom: edges.bottom.map(n), left: edges.left.map(n), right: edges.right.map(n) };
    return O;
  }
  function basis(O) {
    const h = (p) => [p[0], p[1], 1];
    let d0 = unit(cross(cross(h(O.tl), h(O.bl)), cross(h(O.tr), h(O.br))));
    if (d0[1] > 0) d0 = scl(d0, -1);
    const w = Math.abs(d0[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const u1 = unit(cross(d0, w));
    return { d0, u1, u2: cross(d0, u1), w };
  }
  function starts(O, B) {                         // a starting guess for a range of bottle distances
    const h = (p) => [p[0], p[1], 1], d0 = B.d0;
    const mid = h([(O.tm[0] + O.bm[0]) / 2, (O.tm[1] + O.bm[1]) / 2]);
    const b0 = unit(add(mid, scl(d0, -dot(mid, d0))));
    const p1 = unit(cross(d0, B.w)), p2 = cross(d0, p1), out = [];
    for (const D of [1.6, 2.2, 3, 4.5, 7, 11, 18, 30]) {
      const c0 = scl(b0, D), e1 = scl(c0, -1 / D), g = { d: d0, c0, D, e1, e2: cross(d0, e1) }, lim = Math.acos(1 / D), init = {};
      for (const k of ["tl", "tr", "bl", "br"]) {
        const ray = h(O[k]), rp = add(ray, scl(d0, -dot(ray, d0))), cp = add(c0, scl(d0, -dot(c0, d0)));
        const A = dot(rp, rp), Bq = -2 * dot(rp, cp), C = dot(cp, cp) - 1, disc = Bq * Bq - 4 * A * C;
        if (disc > 0) {
          const t = (-Bq - Math.sqrt(disc)) / (2 * A), u = add(scl(ray, t), scl(c0, -1)), up = add(u, scl(d0, -dot(u, d0)));
          init[k] = [Math.atan2(dot(up, g.e2), dot(up, g.e1)), dot(u, d0)];
        } else {
          const th = (k[1] === "l" ? -1 : 1) * 0.95 * lim, P = surf({ ...g, thl: 0, thr: 0 }, th, 0);
          init[k] = [th, dot(scl(ray, dot(P, ray) / dot(ray, ray)), d0) - dot(c0, d0)];
        }
      }
      let thl = (init.tl[0] + init.bl[0]) / 2, thr = (init.tr[0] + init.br[0]) / 2; if (thl > thr) [thl, thr] = [thr, thl];
      out.push([0, 0, dot(c0, p1), dot(c0, p2), thl, thr, (init.tl[1] + init.tr[1]) / 2, (init.bl[1] + init.br[1]) / 2]);
    }
    return out;
  }

  /* Wrap a solution so callers can sample it in pixels. */
  function result(sol, B, cam) {
    const g = geom(sol.x, B), px = (P) => { const p = proj(P); return [p[0] * cam.f + cam.cx, p[1] * cam.f + cam.cy]; };
    // the flattened label's left is the side of the photo's tl dot
    const flip = proj(surf(g, g.thl, g.st))[0] > proj(surf(g, g.thr, g.st))[0];
    const tA = flip ? g.thr : g.thl, tB = flip ? g.thl : g.thr;
    return {
      x: sol.x, B, cam, cost: sol.cost,
      wrapDeg: (Math.abs(g.thr - g.thl) * 180) / Math.PI,
      aspect: Math.abs(g.thr - g.thl) / Math.abs(g.sb - g.st),
      at(fx, fy) { return px(surf(g, tA + fx * (tB - tA), g.st + fy * (g.sb - g.st))); },
    };
  }

  /* Fit to the six dots alone. */
  function fit(dots, cam) {
    const O = observe(dots, cam), B = basis(O);
    let best = null;
    for (const x0 of starts(O, B)) { const s = lm(x0, B, O); if (s && (!best || s.cost < best.cost)) best = s; }
    return best ? result(best, B, cam) : null;
  }

  // ---------------------------------------------------------------- snapping
  function sampler(gray, W, H) {
    return (x, y) => {
      if (!(x >= 0 && y >= 0 && x < W - 1 && y < H - 1)) return NaN;
      const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * W + x0;
      return (gray[i] * (1 - fx) + gray[i + 1] * fx) * (1 - fy) + (gray[i + W] * (1 - fx) + gray[i + W + 1] * fx) * fy;
    };
  }
  /* Search across each fitted edge for where the glass ends and the label begins.
     Scanning inwards from the glass side, the edge is the first place the brightness
     moves from the glass's level to the label's. This ignores printed borders and
     text inside the label, and the bottle's outline beyond it. */
  function levels(I, R) {
    const lab = [], glass = [];
    for (let i = 1; i < 10; i++) for (let j = 1; j < 10; j++) { const p = R.at(i / 10, j / 10); lab.push(I(p[0], p[1])); }
    for (let i = 2; i < 9; i++) for (const fy of [-0.14, -0.1, 1.1, 1.14]) { const p = R.at(i / 10, fy); glass.push(I(p[0], p[1])); }
    return { L: median(lab.filter((v) => v === v)), G: median(glass.filter((v) => v === v)) };
  }
  function findEdges(gray, W, H, R, reach) {
    const I = sampler(gray, W, H), out = { top: [], bottom: [], left: [], right: [] };
    const { L, G } = levels(I, R);
    if (!(Math.abs(L - G) >= 25)) return out;      // too little contrast between label and glass
    const spec = { top: (u) => [u, 0], bottom: (u) => [u, 1], left: (u) => [0, u], right: (u) => [1, u] };
    const c = R.at(0.5, 0.5);
    for (const side of Object.keys(spec)) {
      for (let k = 0; k < 28; k++) {
        const u = 0.06 + (0.88 * k) / 27, [fx, fy] = spec[side](u), p = R.at(fx, fy), e = 0.01, along = side === "top" || side === "bottom";
        const a = along ? R.at(fx - e, fy) : R.at(fx, fy - e), b = along ? R.at(fx + e, fy) : R.at(fx, fy + e);
        let nx = -(b[1] - a[1]), ny = b[0] - a[0]; const Ln = Math.hypot(nx, ny); if (!Ln) continue; nx /= Ln; ny /= Ln;
        if ((p[0] - c[0]) * nx + (p[1] - c[1]) * ny < 0) { nx = -nx; ny = -ny; }   // outwards
        // labelness along the normal, 0 = glass, 1 = label; also averaged 1.5 px either side
        const tx = -ny, ty = nx, step = 0.5, n = Math.round(reach / step), lv = [];
        for (let j = -n; j <= n; j++) {
          const x = p[0] + j * step * nx, y = p[1] + j * step * ny;
          const v = (I(x, y) + I(x + 1.5 * tx, y + 1.5 * ty) + I(x - 1.5 * tx, y - 1.5 * ty)) / 3;
          lv.push((v - G) / (L - G));
        }
        if (lv.some((v) => !(v === v))) continue;
        let seenGlass = false, hit = -1;
        for (let j = lv.length - 1; j > 0; j--) {
          if (lv[j] < 0.3) seenGlass = true;
          else if (seenGlass && lv[j - 1] >= 0.5 && lv[j] < 0.5) { hit = j - 1; break; }
          if (seenGlass && lv[j] >= 0.5) { hit = j; break; }
        }
        if (hit < 0 || hit === 0) continue;
        // sub-sample crossing of 0.5 between hit (label side) and hit+1 (glass side)
        const y0 = lv[hit], y1 = lv[hit + 1] ?? y0, fr = y0 !== y1 ? (y0 - 0.5) / (y0 - y1) : 0;
        const t = (hit + fr - n) * step;
        out[side].push([p[0] + t * nx, p[1] + t * ny]);
      }
    }
    return out;
  }
  /* Six-dot fit, then refine on the label's edges in the photo. Returns {fit, snapped, edges}. */
  function fitAndSnap(gray, W, H, dots, cam) {
    const R0 = fit(dots, cam); if (!R0) return null;
    const size = Math.hypot(R0.at(1, 0.5)[0] - R0.at(0, 0.5)[0], R0.at(1, 0.5)[1] - R0.at(0, 0.5)[1]);
    let R = R0, edges = null, reach = Math.max(12, Math.min(80, 0.15 * size));
    for (let pass = 0; pass < 3; pass++) {
      const e = findEdges(gray, W, H, R, reach);
      const enough = ["top", "bottom", "left", "right"].filter((k) => e[k].length >= 8).length;
      if (enough < 3) break;
      for (const k of Object.keys(e)) if (e[k].length < 8) e[k] = [];
      let O = observe(dots, cam, e, 0.25), s = lm(R.x.slice(), R.B, O);
      if (!s) break;
      // drop points that disagree with the refit, then fit again
      const Rt = result(s, R.B, cam), keep = {};
      for (const k of Object.keys(e)) {
        if (!e[k].length) { keep[k] = []; continue; }
        const fx = k === "left" ? 0 : k === "right" ? 1 : null, fy = k === "top" ? 0 : k === "bottom" ? 1 : null, pts = [];
        for (let i = 0; i <= 48; i++) pts.push(fx === null ? Rt.at(i / 48, fy) : Rt.at(fx, i / 48));
        const dd = e[k].map((q) => distPoly(q, pts)), m = median(dd), mad = median(dd.map((v) => Math.abs(v - m))) * 1.4826;
        keep[k] = e[k].filter((_, i) => dd[i] < Math.max(1.5, m + 3 * mad));
      }
      O = observe(dots, cam, keep, 0.25); s = lm(s.x, R.B, O) || s;
      R = result(s, R.B, cam); edges = keep; reach = Math.max(4, reach / 2);
    }
    // trust the snapped fit only if it stays near what the dots describe
    if (edges) {
      let worst = 0;
      for (const [fx, fy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const a = R.at(fx, fy), b = R0.at(fx, fy); worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1])); }
      if (worst > 0.25 * size) { R = R0; edges = null; }
    }
    return { fit: R, snapped: !!edges, edges, dotsOnly: R0 };
  }

  // ---------------------------------------------------------------- warping
  function homography(src, dst) {
    const A = [], b = [];
    for (let k = 0; k < 4; k++) {
      const [x, y] = src[k], [u, v] = dst[k];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u); A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const h = solve(A, b); return h ? h.concat([1]) : null;
  }
  /* Flatten RGBA pixels S (W x H) into an ImageData-like {data, width, height} of size ow x oh. */
  function warp(S, W, H, R, ow, oh, nc = 32, nr = 24) {
    const nodes = []; for (let j = 0; j <= nr; j++) { const row = []; for (let i = 0; i <= nc; i++) row.push(R.at(i / nc, j / nr)); nodes.push(row); }
    const cw = ow / nc, ch = oh / nr, Hs = [];
    for (let j = 0; j < nr; j++) for (let i = 0; i < nc; i++) {
      Hs.push(homography([[i * cw, j * ch], [(i + 1) * cw, j * ch], [(i + 1) * cw, (j + 1) * ch], [i * cw, (j + 1) * ch]],
        [nodes[j][i], nodes[j][i + 1], nodes[j + 1][i + 1], nodes[j + 1][i]]) || [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }
    const d = new Uint8ClampedArray(ow * oh * 4);
    for (let y = 0; y < oh; y++) {
      const cj = Math.min(nr - 1, Math.floor((y + 0.5) / ch));
      for (let x = 0; x < ow; x++) {
        const h = Hs[cj * nc + Math.min(nc - 1, Math.floor((x + 0.5) / cw))], X = x + 0.5, Y = y + 0.5, w = h[6] * X + h[7] * Y + 1;
        let fx = (h[0] * X + h[1] * Y + h[2]) / w - 0.5, fy = (h[3] * X + h[4] * Y + h[5]) / w - 0.5;
        fx = Math.min(W - 1.001, Math.max(0, fx)); fy = Math.min(H - 1.001, Math.max(0, fy));
        const x0 = fx | 0, y0 = fy | 0, ax = fx - x0, ay = fy - y0, i = (y0 * W + x0) * 4, j = (y * ow + x) * 4;
        for (let c = 0; c < 3; c++) d[j + c] = (S[i + c] * (1 - ax) + S[i + 4 + c] * ax) * (1 - ay) + (S[i + 4 * W + c] * (1 - ax) + S[i + 4 * W + 4 + c] * ax) * ay;
        d[j + 3] = 255;
      }
    }
    return { data: d, width: ow, height: oh };
  }
  /* Output size: the label's own proportions, about as many pixels as it has in the photo. */
  function outSize(R, max) {
    const len = (pts) => pts.reduce((s, p, i) => (i ? s + Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) : 0), 0);
    const side = (len([R.at(0, 0), R.at(0, 1)]) + len([R.at(1, 0), R.at(1, 1)]) + len([R.at(0.5, 0), R.at(0.5, 1)])) / 3;
    let oh = side, ow = side * R.aspect; const k = Math.min(1, max / Math.max(ow, oh));
    return [Math.max(40, Math.round(ow * k)), Math.max(40, Math.round(oh * k))];
  }
  /* Outline of the fitted label as image points (for drawing). */
  function outline(R, n = 24) {
    const P = []; for (let i = 0; i <= n; i++) P.push(R.at(i / n, 0)); for (let i = 0; i <= n; i++) P.push(R.at(1, i / n));
    for (let i = n; i >= 0; i--) P.push(R.at(i / n, 1)); for (let i = n; i >= 0; i--) P.push(R.at(0, i / n));
    return P;
  }
  function grayOf(S, W, H) { const g = new Float32Array(W * H); for (let i = 0, j = 0; j < g.length; i += 4, j++) g[j] = 0.299 * S[i] + 0.587 * S[i + 1] + 0.114 * S[i + 2]; return g; }

  const api = { defaultCam, fit, fitAndSnap, findEdges, warp, outSize, outline, grayOf };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelGeo = api;
})(typeof self !== "undefined" ? self : this);
