/* Fit a label on a cylindrical bottle to the six-point outline used by the
 * Terroir app (tl, tm, tr, br, bm, bl: the four corners and a point on the top
 * and bottom edges), optionally with points on the bottle's two outlines.
 *
 * Unknowns (camera coordinates, bottle radius = 1): the axis direction (2),
 * the axis position (2), the angles of the two label side edges and the
 * heights of the top and bottom edges (4). Solved with Levenberg-Marquardt.
 *
 * fitLabel(points, cam, opts) -> { labelToImage(fx, fy), aspect, wrapDeg, rmsPx, ... } or null
 *   points: {tl:[x,y], tm, tr, br, bm, bl} in image pixels
 *   cam:    {f, cx, cy} focal length and principal point in the same pixels
 *   opts.outline: {left: [[x,y],...], right: [[x,y],...]} optional bottle-outline points
 *   opts.edges:   {top, bottom, left, right: [[x,y],...]} optional extra points along the label's edges
 */
(function (root) {
  "use strict";
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const nrm = (a) => Math.hypot(a[0], a[1], a[2]);
  const unit = (a) => scl(a, 1 / nrm(a));

  function geometry(x, base) {
    const d = unit(add(base.d0, add(scl(base.u1, x[0]), scl(base.u2, x[1]))));
    let p1 = cross(d, base.w); p1 = unit(p1); const p2 = cross(d, p1);
    const c0 = add(scl(p1, x[2]), scl(p2, x[3])), D = nrm(c0);
    const e1 = scl(c0, -1 / D), e2 = cross(d, e1);
    return { d, c0, D, e1, e2, thl: x[4], thr: x[5], st: x[6], sb: x[7] };
  }
  const surface = (g, th, s) => {
    const c = Math.cos(th), sn = Math.sin(th);
    return [0, 1, 2].map((i) => g.c0[i] + c * g.e1[i] + sn * g.e2[i] + s * g.d[i]);
  };
  const proj = (P) => [P[0] / P[2], P[1] / P[2]];
  function distToCurve(q, g, s, n = 40) {
    let best = Infinity, prev = proj(surface(g, g.thl, s));
    for (let i = 1; i <= n; i++) {
      const cur = proj(surface(g, g.thl + ((g.thr - g.thl) * i) / n, s));
      const vx = cur[0] - prev[0], vy = cur[1] - prev[1], L2 = vx * vx + vy * vy || 1e-30;
      const t = Math.max(0, Math.min(1, ((q[0] - prev[0]) * vx + (q[1] - prev[1]) * vy) / L2));
      const dx = q[0] - prev[0] - t * vx, dy = q[1] - prev[1] - t * vy;
      best = Math.min(best, Math.hypot(dx, dy)); prev = cur;
    }
    return best;
  }
  function limbLine(g, sign) {                     // image line of a bottle outline
    const th = sign * Math.acos(Math.min(1, 1 / g.D));
    const a = surface(g, th, 0), b = surface(g, th, 1);
    const l = cross([a[0] / a[2], a[1] / a[2], 1], [b[0] / b[2], b[1] / b[2], 1]);
    const h = Math.hypot(l[0], l[1]); return [l[0] / h, l[1] / h, l[2] / h];
  }

  function residuals(x, base, obs) {
    const g = geometry(x, base), r = [];
    if (!(g.D > 1.0001)) return null;
    const corner = (q, th, s) => { const p = proj(surface(g, th, s)); r.push(p[0] - q[0], p[1] - q[1]); };
    corner(obs.tl, g.thl, g.st); corner(obs.tr, g.thr, g.st);
    corner(obs.bl, g.thl, g.sb); corner(obs.br, g.thr, g.sb);
    r.push(distToCurve(obs.tm, g, g.st), distToCurve(obs.bm, g, g.sb));
    if (obs.edges) {                               // extra points traced along the label's edges
      for (const q of obs.edges.top || []) r.push(distToCurve(q, g, g.st));
      for (const q of obs.edges.bottom || []) r.push(distToCurve(q, g, g.sb));
      for (const [side, th] of [["left", g.thl], ["right", g.thr]]) {
        const a = proj(surface(g, th, g.st)), b = proj(surface(g, th, g.sb));
        const vx = b[0] - a[0], vy = b[1] - a[1], L = Math.hypot(vx, vy);
        for (const q of obs.edges[side] || []) r.push(((q[0] - a[0]) * vy - (q[1] - a[1]) * vx) / L);
      }
    }
    if (obs.outline) {
      const L = [limbLine(g, -1), limbLine(g, 1)];
      const midY = (obs.tl[1] + obs.bl[1]) / 2, xAt = (l) => -(l[1] * midY + l[2]) / l[0];
      if (xAt(L[0]) > xAt(L[1])) L.reverse();
      for (const [side, l] of [["left", L[0]], ["right", L[1]]])
        for (const q of obs.outline[side] || []) r.push(l[0] * q[0] + l[1] * q[1] + l[2]);
    }
    // the label must stay on the visible side of the bottle
    const lim = Math.acos(1 / g.D);
    for (const th of [g.thl, g.thr]) r.push(Math.max(0, Math.abs(th) - lim) * 0.5);
    return r;
  }

  function lm(x, base, obs, iters = 120) {
    const cost = (rr) => rr.reduce((s, v) => s + v * v, 0);
    let r = residuals(x, base, obs); if (!r) return null;
    let c = cost(r), lambda = 1e-3;
    for (let it = 0; it < iters; it++) {
      const n = x.length, m = r.length, J = [];
      for (let j = 0; j < n; j++) {
        const h = 1e-6 * Math.max(1, Math.abs(x[j])), xp = x.slice(); xp[j] += h;
        const rp = residuals(xp, base, obs); if (!rp) { J.push(new Array(m).fill(0)); continue; }
        J.push(rp.map((v, i) => (v - r[i]) / h));
      }
      const A = [], g = [];
      for (let i = 0; i < n; i++) {
        A.push([]); let s = 0; for (let k = 0; k < m; k++) s += J[i][k] * r[k]; g.push(s);
        for (let j = 0; j < n; j++) { let t = 0; for (let k = 0; k < m; k++) t += J[i][k] * J[j][k]; A[i].push(t); }
      }
      let improved = false;
      for (let tries = 0; tries < 10; tries++) {
        const M = A.map((row, i) => row.map((v, j) => v + (i === j ? lambda * (A[i][i] + 1e-12) : 0)));
        const step = solve(M, g.map((v) => -v)); if (!step) { lambda *= 10; continue; }
        const xn = x.map((v, i) => v + step[i]), rn = residuals(xn, base, obs);
        if (rn && cost(rn) < c) { x = xn; r = rn; const cn = cost(rn); const done = c - cn < 1e-12 * (1 + c); c = cn; lambda = Math.max(1e-9, lambda / 3); improved = true; if (done) it = iters; break; }
        lambda *= 10;
      }
      if (!improved) break;
    }
    return { x, cost: c, r };
  }
  function solve(A, b) {
    const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-300) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }

  function fitLabel(points, cam, opts = {}) {
    const f = cam.f, n = (p) => [(p[0] - cam.cx) / f, (p[1] - cam.cy) / f];
    const obs = {};
    for (const k of ["tl", "tm", "tr", "br", "bm", "bl"]) obs[k] = n(points[k]);
    if (opts.outline) obs.outline = { left: (opts.outline.left || []).map(n), right: (opts.outline.right || []).map(n) };
    if (opts.edges) obs.edges = Object.fromEntries(Object.entries(opts.edges).map(([k, v]) => [k, v.map(n)]));
    const h = (p) => [p[0], p[1], 1];
    // vertical vanishing direction from the two side edges
    let V = cross(cross(h(obs.tl), h(obs.bl)), cross(h(obs.tr), h(obs.br)));
    let d0 = unit(V); if (d0[1] > 0) d0 = scl(d0, -1);
    const w = Math.abs(d0[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const u1 = unit(cross(d0, w)), u2 = cross(d0, u1);
    const base = { d0, u1, u2, w };
    // initial bottle centre: towards the middle of the label, perpendicular to the axis
    const mid = h([(obs.tm[0] + obs.bm[0]) / 2, (obs.tm[1] + obs.bm[1]) / 2]);
    let b0 = unit(add(mid, scl(d0, -dot(mid, d0))));
    const p1 = unit(cross(d0, w)), p2 = cross(d0, p1);
    let best = null;
    for (const D of [1.6, 2.2, 3, 4.5, 7, 11, 18, 30]) {
      const c0 = scl(b0, D);
      const g = { d: d0, c0, D, e1: scl(c0, -1 / D) };
      g.e2 = cross(d0, g.e1);
      // corner rays onto the cylinder for a starting guess
      const lim = Math.acos(1 / D), init = {};
      for (const k of ["tl", "tr", "bl", "br"]) {
        const ray = h(obs[k]);
        // solve |ray*t - c0 - s d|_perp = 1, front hit
        const rp = add(ray, scl(d0, -dot(ray, d0))), cp = add(c0, scl(d0, -dot(c0, d0)));
        const A = dot(rp, rp), B = -2 * dot(rp, cp), C = dot(cp, cp) - 1, disc = B * B - 4 * A * C;
        let th, s;
        if (disc > 0) {
          const t = (-B - Math.sqrt(disc)) / (2 * A), P = scl(ray, t);
          const u = add(P, scl(c0, -1)), up = add(u, scl(d0, -dot(u, d0)));
          th = Math.atan2(dot(up, g.e2), dot(up, g.e1)); s = dot(u, d0);
        } else {
          th = (k[1] === "l" ? -1 : 1) * 0.95 * lim;
          const P = surface(g, th, 0);  s = dot(scl(ray, dot(P, ray) / dot(ray, ray)), d0) - dot(c0, d0);
        }
        init[k] = [th, s];
      }
      let thl = (init.tl[0] + init.bl[0]) / 2, thr = (init.tr[0] + init.br[0]) / 2;
      if (thl > thr) { [thl, thr] = [thr, thl]; }
      const x0 = [0, 0, dot(c0, p1), dot(c0, p2), thl, thr, (init.tl[1] + init.tr[1]) / 2, (init.bl[1] + init.br[1]) / 2];
      const res = lm(x0, base, obs);
      if (res && (!best || res.cost < best.cost)) best = res;
    }
    if (!best) return null;
    let g = geometry(best.x, base);
    // make "left" in the flattened label the label's own left: the tl corner's side
    const flip = proj(surface(g, g.thl, g.st))[0] > proj(surface(g, g.thr, g.st))[0];
    const thA = flip ? g.thr : g.thl, thB = flip ? g.thl : g.thr;
    const toPx = (P) => { const p = proj(P); return [p[0] * f + cam.cx, p[1] * f + cam.cy]; };
    return {
      geometry: g,
      wrapDeg: (Math.abs(g.thr - g.thl) * 180) / Math.PI,
      aspect: Math.abs(g.thr - g.thl) / Math.abs(g.sb - g.st),
      rmsPx: Math.sqrt(best.cost / best.r.length) * f,
      distanceInRadii: g.D,
      labelToImage(fx, fy) { return toPx(surface(g, thA + fx * (thB - thA), g.st + fy * (g.sb - g.st))); },
    };
  }

  const api = { fitLabel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CylFit = api;
})(typeof self !== "undefined" ? self : this);
