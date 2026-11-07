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
    let best = Infinity, i0 = 1, i1 = pts.length;
    // q[2], if given, is roughly where along the curve (0 to 1) the point lies: search only near it
    if (q.length > 2) { const c = Math.round(q[2] * (pts.length - 1)); i0 = Math.max(1, c - 8); i1 = Math.min(pts.length, c + 9); }
    for (let i = i0; i < i1; i++) {
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
      // the bottle's outline beside the label: the line round the bottle where the view grazes it
      const lim = Math.acos(Math.min(1, 1 / g.D)), lo = g.thl < g.thr ? -lim : lim;
      for (const [k, th] of [["oleft", lo], ["oright", -lo]]) {
        if (!O.edges[k] || !O.edges[k].length) continue;
        const a = proj(surf(g, th, g.st)), b = proj(surf(g, th, g.sb)), vx = b[0] - a[0], vy = b[1] - a[1], L = Math.hypot(vx, vy);
        for (const q of O.edges[k]) r.push(0.5 * ((q[0] - a[0]) * vy - (q[1] - a[1]) * vx) / L);
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
    const n = (p) => (p.length > 2 ? [(p[0] - cam.cx) / cam.f, (p[1] - cam.cy) / cam.f, p[2]] : [(p[0] - cam.cx) / cam.f, (p[1] - cam.cy) / cam.f]);
    const O = { dotW: dotW ?? 1 };
    for (const k of KEYS) O[k] = n(dots[k]);
    if (edges) { O.edges = {}; for (const k in edges) O.edges[k] = edges[k].map(n); }
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
      x: sol.x, B, cam, cost: sol.cost, theta: [tA, tB],
      wrapDeg: (Math.abs(g.thr - g.thl) * 180) / Math.PI,
      aspect: Math.abs(g.thr - g.thl) / Math.abs(g.sb - g.st),
      lim: Math.acos(Math.min(1, 1 / g.D)),          // the bottle's outline, as an angle round it
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
  /* Search for the label's edges along lines laid on the bottle itself: across the top and
     bottom edges the lines run along the bottle, across the side edges they run round it (and
     stop short of the bottle's outline, so the background beyond it is never mistaken for
     label). On each line the glass level is taken from its outer end and the label level from
     its inner end, so uneven light across the label doesn't matter; scanning in from the
     glass, the edge is the first place the brightness reaches the label's level and stays
     there. reach is in label widths (sides) or heights (top and bottom).
     Returns image points per side, and for each the offset from the fitted edge (label units). */
  function findEdges2(gray, W, H, R, reach, outline) {
    const I = sampler(gray, W, H), out = { top: [], bottom: [], left: [], right: [] }, off = { top: [], bottom: [], left: [], right: [] };
    const span = R.theta[1] - R.theta[0], lim = 0.985 * R.lim;
    // how far past a side edge (in label widths) before the bottle's outline
    const fxMax = (dir) => { const th = dir > 0 ? R.theta[1] : R.theta[0], go = dir * Math.sign(span); return (go > 0 ? lim - th : th + lim) / Math.abs(span); };
    const plen = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    for (const side of ["top", "bottom", "left", "right"]) {
      const horiz = side === "top" || side === "bottom", N = 36;
      for (let k = 0; k < N; k++) {
        const s = 0.05 + (0.9 * k) / (N - 1);
        // outward offset t (label units) -> label coordinates
        const pt = horiz ? (t) => [s, side === "top" ? -t : 1 + t] : (t) => [side === "left" ? -t : 1 + t, s];
        // beside the label, the search runs round the bottle up to its outline as fitted, then
        // straight on in the photo (the fitted outline may be off while the fit is rough)
        const tOut = reach, tIn = Math.min(reach, 0.3);
        const tc = horiz || R.planar ? Infinity : Math.max(0, Math.min(reach, side === "left" ? fxMax(-1) : fxMax(1)));
        const A0 = R.at(...pt(0)), Ai = R.at(...pt(-0.02)), dA = [(A0[0] - Ai[0]) / 0.02, (A0[1] - Ai[1]) / 0.02], Ac = tc < Infinity ? R.at(...pt(tc)) : null;
        const at = (t) => (t <= tc ? R.at(...pt(t)) : [Ac[0] + (t - tc) * dA[0], Ac[1] + (t - tc) * dA[1]]);
        const L = plen(at(-tIn), A0) + Math.hypot(dA[0], dA[1]) * tOut; if (!(L > 6 && L < 6000)) continue;
        const n = Math.ceil(L / 0.5), ts = [], v = [], e = 0.004;
        const tang = (() => { const a = horiz ? R.at(s - e, pt(0)[1]) : R.at(pt(0)[0], s - e), c = horiz ? R.at(s + e, pt(0)[1]) : R.at(pt(0)[0], s + e); return [(c[0] - a[0]) / 2, (c[1] - a[1]) / 2]; })();
        // the bottle's outline, if found, ends the search
        let ol = null;
        if (!horiz && outline && outline[side] && outline[side].length >= 5) {
          const p = outline[side], mx = p.reduce((a, q) => a + q[0], 0) / p.length, my = p.reduce((a, q) => a + q[1], 0) / p.length;
          let sxx = 0, syy = 0, sxy = 0; for (const [x, y] of p) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
          const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), nx = -Math.sin(ang), ny = Math.cos(ang), sd = (q) => (q[0] - mx) * nx + (q[1] - my) * ny, si = Math.sign(sd(R.at(0.5, 0.5)));
          ol = (q) => si * sd(q);                    // > 0 on the bottle
        }
        for (let j = 0; j <= n; j++) {
          const t = -tIn + ((tOut + tIn) * j) / n, a = at(t);
          if (ol && j > 0 && ol(a) < 0) break;
          ts.push(t); v.push((I(a[0], a[1]) + I(a[0] - tang[0], a[1] - tang[1]) + I(a[0] + tang[0], a[1] + tang[1])) / 3);
        }
        if (v.length < 8) continue;
        if (v.some((x) => !(x === x))) continue;
        const m = v.length, q = Math.max(3, Math.round(m * 0.15)), need = Math.max(4, Math.round(m * 0.03));
        const lab = median(v.slice(0, q));
        let hit = -1;
        if (horiz || R.planar) {
          // glass (or whatever is behind a flat label) from the line's outer end; scan in from it
          const gl = median(v.slice(m - q)); if (!(Math.abs(lab - gl) >= 14)) continue;
          const lv = v.map((x) => (x - gl) / (lab - gl));
          for (let j = m - 2; j >= need; j--) {
            if (lv[j] >= 0.5 && lv[j + 1] < 0.5) {
              let ok = true; for (let i = 1; i <= need; i++) if (lv[j - i] < 0.3) { ok = false; break; }
              if (ok) { hit = j; break; }
            }
          }
          if (hit < 0) continue;
          const y0 = lv[hit], y1 = lv[hit + 1], fr = (y0 - 0.5) / (y0 - y1), t = ts[hit] + fr * (ts[hit + 1] - ts[hit]);
          out[side].push([...at(t), s]); off[side].push([s, t]); continue;
        }
        // beside the label the line may run off the bottle onto the background, so take the
        // glass's level from just above and below the label at the same angle, and scan out
        // from the label to the first place that stays at the glass's level
        const gs = [];
        // (just inside the label's span round the bottle, so it is glass even if the fit has
        // put the side edge out at the bottle's outline)
        for (const fy of [-0.08, -0.05, 1.05, 1.08]) for (const dx of [0.03, 0.07, 0.11]) { const a = R.at(side === "left" ? dx : 1 - dx, fy); gs.push(I(a[0], a[1])); }
        const gl = median(gs.filter((x) => x === x));
        if (!(Math.abs(lab - gl) >= 14)) continue;
        const lv = v.map((x) => (x - gl) / (lab - gl));
        // a printed line near the edge is short; the glass beside the label runs on (to the
        // bottle's outline, or at least 4% of the label's width)
        for (let j = Math.max(1, q >> 1); j < m - 1; j++) {
          if (lv[j] >= 0.5 && lv[j + 1] < 0.5) {
            const run = Math.max(need, Math.min(Math.ceil((0.04 / (tOut + tIn)) * (m - 1)), Math.floor(0.6 * (m - 1 - j))));
            let ok = true; for (let i = 1; i <= run && j + i < m; i++) if (lv[j + i] > 0.5) { ok = false; break; }
            if (ok) { hit = j; break; }
          }
        }
        if (hit < 0) {
          // no glass before the bottle's outline: the label runs right round to it
          if (ol && ts[m - 1] < tOut && lv.slice(Math.round(m * 0.8)).every((x) => x >= 0.5)) { out[side].push(at(ts[m - 1])); off[side].push([s, ts[m - 1]]); }
          continue;
        }
        const y0 = lv[hit], y1 = lv[hit + 1], fr = (y0 - 0.5) / (y0 - y1), t = ts[hit] + fr * (ts[hit + 1] - ts[hit]);
        out[side].push(at(t)); off[side].push([s, t]);
      }
    }
    return { pts: out, off };
  }
  /* The bottle's outline beside the label, found just above and below it: scanning in from
     the background to the glass. Returns image points {left, right} (may be empty). */
  function findOutline(gray, W, H, R) {
    const I = sampler(gray, W, H), out = { left: [], right: [] }, span = R.theta[1] - R.theta[0], lim = 0.99 * R.lim;
    const reachPx = 0.3 * Math.hypot(R.at(1, 0.5)[0] - R.at(0, 0.5)[0], R.at(1, 0.5)[1] - R.at(0, 0.5)[1]);
    for (const side of ["left", "right"]) {
      const dir = side === "left" ? -1 : 1, th = dir > 0 ? R.theta[1] : R.theta[0], go = dir * Math.sign(span);
      const fxl = (side === "left" ? 0 : 1) + dir * Math.max(0, (go > 0 ? lim - th : th + lim) / Math.abs(span));
      for (const fy of [-0.2, -0.16, -0.12, -0.08, -0.05, 1.05, 1.08, 1.12, 1.16, 1.2]) {
        const A = R.at(fxl, fy), Bp = R.at(fxl - dir * 0.03, fy), d = [A[0] - Bp[0], A[1] - Bp[1]], dl = Math.hypot(d[0], d[1]);
        if (!(dl > 0) || !(reachPx < 4000)) continue;
        const u = [d[0] / dl, d[1] / dl], n1 = Math.min(4000, Math.ceil(Math.hypot(A[0] - R.at(0.5, fy)[0], A[1] - R.at(0.5, fy)[1]) / 0.5) || 0), n2 = Math.ceil(reachPx / 0.5), v = [], P = [];
        for (let j = 0; j <= n1; j++) { const p = R.at(0.5 + (fxl - 0.5) * (j / n1), fy); P.push(p); v.push(I(p[0], p[1])); }
        for (let j = 1; j <= n2; j++) { const p = [A[0] + u[0] * j * 0.5, A[1] + u[1] * j * 0.5]; P.push(p); v.push(I(p[0], p[1])); }
        if (v.some((x) => !(x === x))) continue;
        const m = v.length, q = Math.max(3, Math.round(n2 * 0.25)), gl = median(v.slice(Math.round(n1 * 0.4), Math.round(n1 * 0.8))), bg = median(v.slice(m - q));
        if (!(Math.abs(bg - gl) >= 14)) continue;
        const lv = v.map((x) => (x - bg) / (gl - bg)), need = 6;
        for (let j = m - 2; j >= need; j--) {
          if (lv[j] >= 0.5 && lv[j + 1] < 0.5) {
            let ok = true; for (let i = 1; i <= need; i++) if (lv[j - i] < 0.5) { ok = false; break; }
            if (ok) { const fr = (lv[j] - 0.5) / (lv[j] - lv[j + 1]); out[side].push([P[j][0] + fr * (P[j + 1][0] - P[j][0]), P[j][1] + fr * (P[j + 1][1] - P[j][1])]); }
            break;
          }
        }
      }
    }
    // keep each side only if its points lie on a line
    for (const k of ["left", "right"]) {
      const p = out[k]; if (p.length < 5) { out[k] = []; continue; }
      const mx = p.reduce((a, q) => a + q[0], 0) / p.length, my = p.reduce((a, q) => a + q[1], 0) / p.length;
      let sxx = 0, syy = 0, sxy = 0; for (const [x, y] of p) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
      const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), nx = -Math.sin(ang), ny = Math.cos(ang);
      const dd = p.map(([x, y]) => Math.abs((x - mx) * nx + (y - my) * ny)), sc = 1.4826 * median(dd) + 0.7;
      out[k] = p.filter((_, i) => dd[i] < 3 * sc); if (out[k].length < 5) out[k] = [];
    }
    return out;
  }
  /* A smooth curve through (s, t) offsets: a robust quadratic, or null if too few agree. */
  function smoothOffsets(st) {
    if (st.length < 8) return null;
    let use = st.slice(), c = null;
    for (let it = 0; it < 3; it++) {
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
      for (const [s, t] of use) { const f = [1, s - 0.5, (s - 0.5) ** 2]; for (let i = 0; i < 3; i++) { r[i] += f[i] * t; for (let j = 0; j < 3; j++) A[i][j] += f[i] * f[j]; } }
      for (let i = 0; i < 3; i++) A[i][i] += 1e-6;
      c = solve(A, r); if (!c) return null;
      const res = st.map(([s, t]) => Math.abs(t - (c[0] + c[1] * (s - 0.5) + c[2] * (s - 0.5) ** 2))), sc = 1.4826 * median(res) + 0.002;
      use = st.filter((_, i) => res[i] < 2.5 * sc);
      if (use.length < 8) return null;
    }
    return (s) => c[0] + c[1] * (s - 0.5) + c[2] * (s - 0.5) ** 2;
  }
  /* The final map: the fitted bottle, with its four edges moved onto the label's edges as found
     in the photo, and pulled in by a hair, so the flattened label is bounded exactly by the
     label's own edges: straight, square and with no glass showing. */
  function boundaryMap(R, off, inset) {
    const cap = (f) => (f ? (s) => Math.max(-0.06, Math.min(0.06, f(s))) : () => 0);
    const tt = cap(smoothOffsets(off.top)), tb = cap(smoothOffsets(off.bottom)), tl = cap(smoothOffsets(off.left)), tr = cap(smoothOffsets(off.right));
    const corrected = [off.top, off.bottom, off.left, off.right].map((o) => !!smoothOffsets(o));
    const at0 = (u, v) => {
      const top = -tt(u), bot = 1 + tb(u), lef = -tl(v), rig = 1 + tr(v);
      return R.at(lef + (rig - lef) * u, top + (bot - top) * v);
    };
    // where the label runs round to (or behind) the bottle's outline, the photo has almost nothing
    // of it: leave out the strip seen at less than a tenth of the detail at the front
    const g = [], n = 40; for (let i = 0; i <= n; i++) { const u = i / n, a = at0(Math.max(0, u - 0.01), 0.5), b = at0(Math.min(1, u + 0.01), 0.5); g.push(Math.hypot(a[0] - b[0], a[1] - b[1])); }
    const gmax = Math.max(...g); let i0 = 0, i1 = n; while (i0 < n && g[i0] < 0.1 * gmax) i0++; while (i1 > 0 && g[i1] < 0.1 * gmax) i1--;
    const u0 = Math.max(inset, i0 / n), u1 = Math.min(1 - inset, i1 / n);
    const at = (u, v) => at0(u0 + (u1 - u0) * u, inset + (1 - 2 * inset) * v);
    return { ...R, at, corrected, aspect: R.aspect * (u1 - u0) / (1 - 2 * inset), trimmed: [u0 > inset, u1 < 1 - inset] };
  }
  /* Six-dot fit, then refine on the label's edges in the photo.
     Returns {fit, final, snapped, edges}: draw and warp with `final`. */
  function fitAndSnap(gray, W, H, dots, cam, opts = {}) {
    const R0 = fit(dots, cam); if (!R0) return null;
    const size = Math.hypot(R0.at(1, 0.5)[0] - R0.at(0, 0.5)[0], R0.at(1, 0.5)[1] - R0.at(0, 0.5)[1]);
    let R = R0, edges = null, first = true;
    // search wide first, then closer in as the fit improves
    for (const reach of [0.22, 0.12, 0.06, 0.035, 0.025]) {
      const oln = findOutline(gray, W, H, R), { pts: e } = findEdges2(gray, W, H, R, reach, oln);
      e.oleft = oln.left; e.oright = oln.right;
      // a side that wasn't found within reach may be further out than the rough fit says
      if (reach < 0.3 && ["top", "bottom", "left", "right"].some((k) => e[k].length < 10)) {
        const { pts: w } = findEdges2(gray, W, H, R, 0.3, oln);
        for (const k of ["top", "bottom", "left", "right"]) if (e[k].length < 10 && w[k].length >= 10) e[k] = w[k];
      }
      const enough = ["top", "bottom", "left", "right"].filter((k) => e[k].length >= 10).length;
      // three sides are needed, except that the top and bottom alone may start a rough fit off
      const tb = e.top.length >= 10 && e.bottom.length >= 10;
      if (enough < 3 && !(tb && !edges)) continue;
      for (const k of ["top", "bottom", "left", "right"]) if (e[k].length < 10) e[k] = [];
      const dw = enough >= 3 ? 0.05 : 0.25;
      let O = observe(dots, cam, e, dw), s = lm(R.x.slice(), R.B, O);
      // on the first pass, also start from the usual range of bottle distances: the dots may have
      // misled the first fit
      if (first) {
        first = false; let alt = null;
        for (const x0 of starts(observe(dots, cam), R.B)) { const t = lm(x0, R.B, O, 40); if (t && (!alt || t.cost < alt.cost)) alt = t; }
        if (alt && (!s || alt.cost < s.cost)) { const t = lm(alt.x, R.B, O); s = t && (!s || t.cost < s.cost) ? t : s; }
      }
      if (!s) continue;
      // drop points that disagree with the refit, then fit again
      const Rt = result(s, R.B, cam), keep = {};
      for (const k of Object.keys(e)) {
        if (!e[k].length || k[0] === "o") { keep[k] = e[k]; continue; }
        const fx = k === "left" ? 0 : k === "right" ? 1 : null, fy = k === "top" ? 0 : k === "bottom" ? 1 : null, pts = [];
        for (let i = 0; i <= 48; i++) pts.push(fx === null ? Rt.at(i / 48, fy) : Rt.at(fx, i / 48));
        const dd = e[k].map((q) => distPoly(q, pts)), m = median(dd), mad = median(dd.map((v) => Math.abs(v - m))) * 1.4826;
        keep[k] = e[k].filter((_, i) => dd[i] < Math.max(1.5, m + 3 * mad));
      }
      O = observe(dots, cam, keep, dw); s = lm(s.x, R.B, O) || s;
      R = result(s, R.B, cam); edges = keep;
      if (opts.trace) opts.trace(R, reach, keep);
    }
    // trust the snapped fit only if the label's edges were found all round (or on three sides),
    // they fit the bottle closely, and it stays roughly where the dots are
    if (edges) {
      const sides = ["top", "bottom", "left", "right"].filter((k) => edges[k].length >= 10), dd = [];
      for (const k of sides) {
        const fx = k === "left" ? 0 : k === "right" ? 1 : null, fy = k === "top" ? 0 : k === "bottom" ? 1 : null, pts = [];
        for (let i = 0; i <= 48; i++) pts.push(fx === null ? R.at(i / 48, fy) : R.at(fx, i / 48));
        for (const q of edges[k]) dd.push(distPoly(q, pts));
      }
      let worst = 0;
      for (const [fx, fy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const a = R.at(fx, fy), b = R0.at(fx, fy); worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1])); }
      if (sides.length < 3 || !(median(dd) < Math.max(1.5, 0.004 * size)) || worst > 0.6 * size || !(R.aspect > 0.1 && R.aspect < 10)) { R = R0; edges = null; }
    }
    const final = edges ? boundaryMap(R, findEdges2(gray, W, H, R, 0.04, findOutline(gray, W, H, R)).off, opts.inset ?? 0.006) : R;
    return { fit: R, final, snapped: !!edges, edges, dotsOnly: R0 };
  }

  // ---------------------------------------------------------------- flat labels
  function lineFit(p) {                           // robust straight line: {mx, my, nx, ny}
    let use = p.slice(), L = null;
    for (let it = 0; it < 3 && use.length >= 5; it++) {
      const mx = use.reduce((a, q) => a + q[0], 0) / use.length, my = use.reduce((a, q) => a + q[1], 0) / use.length;
      let sxx = 0, syy = 0, sxy = 0; for (const [x, y] of use) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
      const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy); L = { mx, my, nx: -Math.sin(ang), ny: Math.cos(ang) };
      const dd = p.map(([x, y]) => Math.abs((x - L.mx) * L.nx + (y - L.my) * L.ny)), sc = 1.4826 * median(dd) + 0.7;
      use = p.filter((_, i) => dd[i] < 3 * sc);
    }
    return use.length >= 5 ? L : null;
  }
  const meet = (a, b) => { const d = a.nx * b.ny - a.ny * b.nx; if (Math.abs(d) < 1e-9) return null; const ca = a.nx * a.mx + a.ny * a.my, cb = b.nx * b.mx + b.ny * b.my; return [(ca * b.ny - a.ny * cb) / d, (a.nx * cb - ca * b.nx) / d]; };
  function planeMap(c) {
    const h = homography([[0, 0], [1, 0], [1, 1], [0, 1]], [c.tl, c.tr, c.br, c.bl]); if (!h) return null;
    const at = (fx, fy) => { const w = h[6] * fx + h[7] * fy + 1; return [(h[0] * fx + h[1] * fy + h[2]) / w, (h[3] * fx + h[4] * fy + h[5]) / w]; };
    const len = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    return { planar: true, corners: c, at, theta: [0, 0], wrapDeg: 0, aspect: (len(c.tl, c.tr) + len(c.bl, c.br)) / (len(c.tl, c.bl) + len(c.tr, c.br)) };
  }
  /* A flat label: the four corner dots give a first quadrilateral; its sides are then moved onto
     the label's straight edges in the photo. Returns {fit, final, snapped, edges}. */
  function fitFlat(gray, W, H, dots, opts = {}) {
    const R0 = planeMap({ tl: dots.tl, tr: dots.tr, br: dots.br, bl: dots.bl }); if (!R0) return null;
    let R = R0, edges = null;
    for (const reach of [0.2, 0.1, 0.05, 0.03]) {
      const { pts: e } = findEdges2(gray, W, H, R, reach), L = {};
      for (const k of ["top", "bottom", "left", "right"]) L[k] = e[k].length >= 10 ? lineFit(e[k]) : null;
      if (!(L.top && L.bottom && L.left && L.right)) continue;
      const c = { tl: meet(L.top, L.left), tr: meet(L.top, L.right), br: meet(L.bottom, L.right), bl: meet(L.bottom, L.left) };
      if (!Object.values(c).every((p) => p && p.every(Number.isFinite))) continue;
      const Rn = planeMap(c); if (!Rn || !(Rn.aspect > 0.1 && Rn.aspect < 10)) continue;
      R = Rn; edges = e;
    }
    const inset = opts.inset ?? 0.006;
    const final = edges ? { ...R, at: (u, v) => R.at(inset + (1 - 2 * inset) * u, inset + (1 - 2 * inset) * v) } : R;
    return { fit: R, final, snapped: !!edges, edges, dotsOnly: R0 };
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
  // ---------------------------------------------------------------- shading
  /* Even out the light across a flattened label.
     On a cylinder the surface faces one way all the way down each column, so under a
     distant light the brightness depends only on the angle round the bottle. Lambert's
     law makes it  a + b·cos θ + c·sin θ  (plus a little curvature for soft or clipped
     light). Fit that to the paper's brightness in each column, ignoring columns where
     artwork dominates, and divide it out.
       pix: Uint8ClampedArray, ch = 3 (RGB) or 4 (RGBA), W x H, edited in place
       th0, th1: angle round the bottle (radians) at the left and right edges
     Returns { applied, before, after, curve } where before/after are the spread of the
     fitted light across the label (0.3 = the darkest part had 30% less light). */
  function evenLight(pix, W, H, ch, th0, th1, opts = {}) {
    const span = Math.abs(th1 - th0);
    if (!(span > (12 * Math.PI) / 180) || W < 16 || H < 16) return { applied: false, before: 0, after: 0 };
    const nb = Math.min(96, W), q = opts.quantile ?? 0.8, maxGain = opts.maxGain ?? 3;
    const y0 = Math.floor(H * 0.04), y1 = Math.ceil(H * 0.96), ystep = Math.max(1, Math.floor((y1 - y0) / 240));
    const lum = (i) => 0.299 * pix[i] + 0.587 * pix[i + 1] + 0.114 * pix[i + 2];
    // the paper's colour: shading changes how bright it is, not its hue, so find the most common
    // hue among the brighter pixels and use only pixels of that hue
    // "bright" is judged within each column, so a shaded part of the label still counts
    const hist = new Float64Array(64 * 64), nbx = Math.min(48, W), p90 = new Float64Array(nbx);
    for (let b = 0; b < nbx; b++) {
      const xa = Math.floor((b * W) / nbx), xb = Math.max(xa + 1, Math.floor(((b + 1) * W) / nbx)), v = [];
      for (let y = y0; y < y1; y += ystep) for (let x = xa; x < xb; x += 2) v.push(lum((y * W + x) * ch));
      v.sort((a, c) => a - c); p90[b] = v[Math.floor(0.9 * (v.length - 1))];
    }
    const chroma = (i) => { const t = pix[i] + pix[i + 1] + pix[i + 2] + 1e-6; return [pix[i] / t, pix[i + 1] / t]; };
    for (let y = y0; y < y1; y += ystep) for (let x = 0; x < W; x += 2) {
      const i = (y * W + x) * ch, L = lum(i); if (L < 25 || L < 0.6 * p90[Math.min(nbx - 1, Math.floor((x * nbx) / W))]) continue;
      const [a, b] = chroma(i), ia = Math.min(63, Math.max(0, Math.round(a * 128 - 10))), ib = Math.min(63, Math.max(0, Math.round(b * 128 - 10)));
      hist[ia * 64 + ib] += 1;
    }
    // up to three paper colours (labels often have panels in a second colour)
    const peaks = [], hh = Float64Array.from(hist);
    for (let n = 0; n < 3; n++) {
      let pk = 0; for (let k = 1; k < hh.length; k++) if (hh[k] > hh[pk]) pk = k;
      if (!(hh[pk] > 0) || (peaks.length && hh[pk] < 0.15 * peaks[0].mass)) break;
      const ia = Math.floor(pk / 64), ib = pk % 64;
      peaks.push({ a: (ia + 10) / 128, b: (ib + 10) / 128, mass: hh[pk] });
      for (let da = -3; da <= 3; da++) for (let db = -3; db <= 3; db++) { const ja = ia + da, jb = ib + db; if (ja >= 0 && jb >= 0 && ja < 64 && jb < 64) hh[ja * 64 + jb] = 0; }
    }
    const tol = 0.016, NP = peaks.length;
    const th = new Float64Array(nb), val = new Float64Array(nb), cnt = new Float64Array(nb), grp = new Int32Array(nb);
    let rows = 0; for (let y = y0; y < y1; y += ystep) rows++;
    for (let b = 0; b < nb; b++) {
      const xa = Math.floor((b * W) / nb), xb = Math.max(xa + 1, Math.floor(((b + 1) * W) / nb));
      const xs = Math.max(1, Math.floor((xb - xa) / 3)), v = peaks.map(() => []);
      for (let y = y0; y < y1; y += ystep) for (let x = xa; x < xb; x += xs) {
        const i = (y * W + x) * ch, L = lum(i); if (L < 25) continue;
        const [a, c2] = chroma(i);
        for (let k = 0; k < NP; k++) if (Math.abs(a - peaks[k].a) < tol && Math.abs(c2 - peaks[k].b) < tol) { v[k].push(L); break; }
      }
      th[b] = th0 + ((th1 - th0) * ((xa + xb) / 2)) / W;
      let k = 0; for (let j = 1; j < NP; j++) if (v[j].length > v[k].length) k = j;
      grp[b] = k; cnt[b] = v[k].length / (Math.ceil((xb - xa) / xs) * rows);
      if (v[k].length) { v[k].sort((a, c) => a - c); val[b] = v[k][Math.min(v[k].length - 1, Math.floor(q * v[k].length))]; }
    }
    // robust fit (Tukey biweight) with a brightness factor per paper colour; the second
    // harmonic is damped so artwork can't bend the curve much
    const basis = (t) => [1, Math.cos(t), Math.sin(t), Math.cos(2 * t), Math.sin(2 * t)], K = 5, ridge = [0, 0, 0, 1, 1];
    let w0 = Float64Array.from(cnt, (f) => (f < 0.08 ? 0 : Math.min(1, f / 0.3)));
    if (w0.filter((v) => v > 0).length < nb * 0.4) return { applied: false, before: 0, after: 0 };
    // the reference colour is the one covering the most columns
    const cover = new Array(NP).fill(0); for (let b = 0; b < nb; b++) if (w0[b] > 0) cover[grp[b]]++;
    const ref = cover.indexOf(Math.max(...cover)), sc_ = new Array(NP).fill(1);
    let w = Float64Array.from(w0), c = null;
    const vmax = Math.max(...val) || 1;
    // where the paper colour changes from one column to the next, the light doesn't: the jump
    // in brightness there is the ratio of the two papers' brightness
    if (NP > 1) {
      const good = []; for (let b = 0; b < nb; b++) if (w0[b] > 0) good.push(b);
      const side = (idx, dir) => {              // up to 3 same-colour columns going away from the boundary
        const k = grp[good[idx]], pts = [];
        for (let j = idx; j >= 0 && j < good.length && pts.length < 3; j += dir) { if (grp[good[j]] !== k) break; pts.push([good[j], val[good[j]]]); }
        return pts;
      };
      const at = (pts, x) => {                   // straight-line extrapolation to x
        if (pts.length < 2) return pts[0][1];
        const n = pts.length, mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
        let sxy = 0, sxx = 0; for (const [x0, y0] of pts) { sxy += (x0 - mx) * (y0 - my); sxx += (x0 - mx) ** 2; }
        return my + (sxx ? sxy / sxx : 0) * (x - mx);
      };
      const pairs = {};
      for (let i = 0; i + 1 < good.length; i++) {
        const a = good[i], b = good[i + 1]; if (grp[a] === grp[b] || b - a > 4) continue;
        const L = side(i, -1), R = side(i + 1, 1), x = (a + b) / 2, la = at(L, x), rb = at(R, x);
        if (!(la > 0 && rb > 0)) continue;
        const key = grp[a] + ">" + grp[b]; (pairs[key] = pairs[key] || []).push(rb / la);
      }
      const known = { [ref]: 1 };
      for (let pass = 0; pass < NP; pass++) for (const key in pairs) {
        const [p0, p1] = key.split(">").map(Number), r = median(pairs[key]);
        if (known[p0] !== undefined && known[p1] === undefined) known[p1] = known[p0] * r;
        else if (known[p1] !== undefined && known[p0] === undefined) known[p0] = known[p1] / r;
      }
      for (let k = 0; k < NP; k++) sc_[k] = known[k] ?? NaN;
      // a colour that never meets another: leave it out of the fit rather than guess its brightness
      for (let b = 0; b < nb; b++) if (!(sc_[grp[b]] > 0)) { w0[b] = 0; }
      for (let k = 0; k < NP; k++) if (!(sc_[k] > 0)) sc_[k] = 1;
    }
    {
      w = Float64Array.from(w0);
      const vs = Array.from(val, (v, b) => v / sc_[grp[b]]);
      for (let it = 0; it < 12; it++) {
        const A = Array.from({ length: K }, () => new Array(K).fill(0)), r = new Array(K).fill(0);
        for (let b = 0; b < nb; b++) { const f = basis(th[b]); for (let i = 0; i < K; i++) { r[i] += w[b] * f[i] * vs[b]; for (let j = 0; j < K; j++) A[i][j] += w[b] * f[i] * f[j]; } }
        const sw = w.reduce((a, v) => a + v, 0);
        if (!(sw > 0)) return { applied: false, before: 0, after: 0 };
        for (let i = 0; i < K; i++) A[i][i] += ridge[i] * 0.05 * sw;
        c = solve(A, r); if (!c) return { applied: false, before: 0, after: 0 };
        const res = Array.from(vs, (v, b) => v - basis(th[b]).reduce((a, f, i) => a + f * c[i], 0));
        const sc = 1.4826 * median(res.filter((_, b) => w0[b] > 0).map(Math.abs)) + 1e-6 + 0.004 * vmax;
        w = Float64Array.from(res, (e, b) => { const u = e / (4.685 * sc); return Math.abs(u) < 1 ? w0[b] * (1 - u * u) ** 2 : 0; });
      }
    }
    const gp = (t) => basis(t).reduce((a, f, i) => a + f * c[i], 0);
    const ratio = Array.from(val, (v, b) => (w[b] > 0.2 ? v / sc_[grp[b]] / Math.max(1e-6, gp(th[b])) : NaN));
    const half = Math.max(2, Math.round(nb / 14)), corr = new Float64Array(nb);
    for (let b = 0; b < nb; b++) {
      const win = []; for (let k = b - half; k <= b + half; k++) if (k >= 0 && k < nb && ratio[k] === ratio[k]) win.push(ratio[k]);
      corr[b] = win.length >= half ? Math.min(1.25, Math.max(0.8, median(win))) : 1;
    }
    const g = (t) => { const x = ((t - th0) / (th1 - th0)) * nb - 0.5, i = Math.max(0, Math.min(nb - 2, Math.floor(x))), f = Math.max(0, Math.min(1, x - i)); return gp(t) * (corr[i] * (1 - f) + corr[i + 1] * f); };
    const curve = Array.from(th, g), gmax = Math.max(...curve), gmin = Math.min(...curve);
    if (!(gmax > 20) || gmin <= 0.15 * gmax) return { applied: false, before: 0, after: 0, curve };
    const before = 1 - gmin / gmax;
    if (before < 0.02) return { applied: false, before, after: before, curve };
    const gain = new Float32Array(W);
    for (let x = 0; x < W; x++) gain[x] = Math.min(maxGain, gmax / Math.max(1e-6, g(th0 + ((th1 - th0) * (x + 0.5)) / W)));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * ch, k = gain[x]; pix[i] *= k; pix[i + 1] *= k; pix[i + 2] *= k; }
    return { applied: true, before, after: 0, curve, gain };
  }

  function grayOf(S, W, H) { const g = new Float32Array(W * H); for (let i = 0, j = 0; j < g.length; i += 4, j++) g[j] = 0.299 * S[i] + 0.587 * S[i + 1] + 0.114 * S[i + 2]; return g; }

  const api = { defaultCam, fit, fitAndSnap, findEdges, findEdges2, findOutline, fitFlat, warp, outSize, outline, grayOf, evenLight };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelGeo = api;
})(typeof self !== "undefined" ? self : this);
