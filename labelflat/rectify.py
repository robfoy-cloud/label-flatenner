"""Flatten a cylindrical label from one photo.

Inputs: the photo, the focal length in pixels (from EXIF) and the principal
point (assumed to be the image centre). Nothing else about the scene is used.

Steps
 1. detect the two bottle silhouettes, the two label side edges, and the
    label's top and bottom edges (traced as free curves, no model fitted)
 2. vertical vanishing point V from the four straight edges
 3. cylinder pose from the silhouettes (top-down cross-section trick)
 4. angle theta of each label side edge
 5. grid: columns at equal theta, rows by interpolation along each ruling
 6. piecewise homography per grid cell -> flat label
"""
import numpy as np
import cv2


# ---------------------------------------------------------------- detection
def _subpix(profile, idx, thr):
    """Sub-pixel crossing of thr between profile[idx-1] and profile[idx]."""
    a, b = profile[idx - 1], profile[idx]
    return idx - 1 + (thr - a) / (b - a) if b != a else float(idx)


def detect(img):
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    border = np.concatenate([f[:5].reshape(-1, 3), f[-5:].reshape(-1, 3),
                             f[:, :5].reshape(-1, 3), f[:, -5:].reshape(-1, 3)])
    bg = np.median(border, 0)
    dist = np.abs(f - bg).max(-1)                      # how un-background-like
    notbg = cv2.morphologyEx(((dist > 25) | (f[..., 0] - f[..., 2] > 5)).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)

    # silhouettes: first / last bottle pixel on each row (sub-pixel on `dist`)
    sil_l, sil_r = [], []
    for v in range(2, H - 2):
        xs = np.flatnonzero(notbg[v])
        if len(xs) < 20:
            continue
        row = dist[v]
        i0, i1 = xs[0], xs[-1]
        if i0 < 4 or i1 > W - 5:
            continue
        lvl = (row[i0 + 4] + row[i0 - 3]) / 2
        sil_l.append((_subpix(row, i0, lvl) if row[i0] != row[i0 - 1] else i0, v))
        rr = row[::-1]; j = W - 1 - i1
        lvl = (rr[j + 4] + rr[j - 3]) / 2
        sil_r.append((W - 1 - _subpix(rr, j, lvl), v))

    # label: cream paper is red-dominant, glass is green-dominant
    R = f[..., 0]
    lab = ((R > 110) & notbg).astype(np.uint8)
    lab = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(lab)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    lab = (cc == k).astype(np.uint8)
    flood = lab.copy(); mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    lab = lab | (1 - flood)                            # fill holes (text etc.)

    cols = np.flatnonzero(lab.any(0)); rows = np.flatnonzero(lab.any(1))
    top, bot = [], []
    for u in range(cols[0] + 3, cols[-1] - 2):
        ys = np.flatnonzero(lab[:, u])
        if len(ys) < 10:
            continue
        p = R[:, u]
        i0 = ys[0]; thr = (p[i0 - 3] + p[i0 + 1]) / 2
        top.append((u, _subpix(p, i0, thr)))
        pr = p[::-1]; j = H - 1 - ys[-1]; thr = (pr[j - 3] + pr[j + 1]) / 2
        bot.append((u, H - 1 - _subpix(pr, j, thr)))
    top, bot = np.array(top), np.array(bot)

    def side_edges(vmin, vmax):
        el, er = [], []
        for v in range(int(vmin), int(vmax)):
            xs = np.flatnonzero(lab[v])
            if len(xs) < 5:
                continue
            p = R[v]
            i0 = xs[0]; thr = (p[i0 - 3] + p[i0:i0 + 3].max()) / 2
            el.append((_subpix(p, i0, thr), v))
            pr = p[::-1]; j = W - 1 - xs[-1]; thr = (pr[j - 3] + pr[j:j + 3].max()) / 2
            er.append((W - 1 - _subpix(pr, j, thr), v))
        return np.array(el), np.array(er)

    # Columns near the corners hit the slanted side edges, not the top/bottom,
    # so: get side edges from a safe middle band, then keep only top/bottom
    # samples that lie between them, then redo the side edges on the full band.
    for _ in range(2):
        el, er = side_edges(np.percentile(top[:, 1], 90) + 3, np.percentile(bot[:, 1], 10) - 3)
        def u_at(e, v):
            A = np.polyfit(e[:, 1], e[:, 0], 1); return np.polyval(A, v)
        keep = lambda c: c[(c[:, 0] > u_at(el, c[:, 1]) + 3) & (c[:, 0] < u_at(er, c[:, 1]) - 3)]
        top, bot = keep(top), keep(bot)
    el, er = side_edges(top[:, 1].max() + 3, bot[:, 1].min() - 3)
    return dict(sil_l=np.array(sil_l), sil_r=np.array(sil_r),
                edge_l=el, edge_r=er, top=top, bot=bot)


# ---------------------------------------------------------------- geometry
def _norm_pts(p, f, c):
    return (p - c) / f


def _fit_line(q):
    m = q.mean(0)
    _, _, vt = np.linalg.svd(q - m)
    n = vt[1]                                       # unit normal
    return np.array([n[0], n[1], -n @ m])


def _fit_line_through(q, V):
    """Line through homogeneous point V minimising perpendicular distances."""
    if abs(V[2]) > 1e-9:
        p0 = V[:2] / V[2]
        _, _, vt = np.linalg.svd(q - p0)            # uncentred about V
        dvec = vt[0]
        n = np.array([-dvec[1], dvec[0]])
        return np.array([n[0], n[1], -n @ p0])
    n = np.array([-V[1], V[0]]); n /= np.linalg.norm(n)   # V at infinity
    return np.array([n[0], n[1], -np.median(q @ n)])


def _clean_curve(q, f, win=9):
    """Drop isolated spikes from a traced curve without assuming its shape:
    compare each point with the running median of its neighbours."""
    q = q[np.argsort(q[:, 0])]
    v = np.pad(q[:, 1], win // 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(v, win), axis=1)
    r = q[:, 1] - med
    mad = np.median(np.abs(r)) * 1.4826
    return q[np.abs(r) < max(4 * mad, 1.0 / f)]


def _unit(x):
    return x / np.linalg.norm(x)


class Model:
    """Cylinder (radius 1) in calibrated camera coordinates."""

    def __init__(self, feats, f, c):
        self.f, self.c = f, np.asarray(c, float)
        nq = {k: _norm_pts(v, f, self.c) for k, v in feats.items()}
        keys = ["sil_l", "sil_r", "edge_l", "edge_r"]
        for k in keys:                              # robust: drop rows that missed the edge
            q = nq[k]
            for _ in range(3):
                l = _fit_line(q); r = q @ l[:2] + l[2]
                mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-12
                q = q[np.abs(r - np.median(r)) < max(2.5 * mad, 0.3 / f)]   # floor ~0.3 px
            nq[k] = q
        for k in ["top", "bot"]:                    # free curves: only remove stray points
            nq[k] = _clean_curve(nq[k], f)
        self.nq = nq
        L = np.array([_fit_line(nq[k]) for k in keys])
        L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(L)
        V = vt[-1]
        self.lines = {k: _fit_line_through(nq[k], V) for k in keys}
        L = np.array([self.lines[k] for k in keys])
        self.V = np.linalg.svd(L)[2][-1]
        d = _unit(self.V)
        if d[1] > 0:                                # point d "up" in the image
            d = -d
        self.d = d

        # cross-section looking along the axis: two tangents from the camera
        t = []
        for k in ["sil_l", "sil_r"]:
            ti = _unit(np.cross(d, _unit(self.lines[k])))
            t.append(ti if ti[2] > 0 else -ti)
        b = _unit(t[0] + t[1])
        alpha = 0.5 * np.arccos(np.clip(t[0] @ t[1], -1, 1))
        self.alpha = alpha
        self.c0 = b / np.sin(alpha)                 # axis foot point (R = 1)
        self.e1 = -b                                # towards the camera
        self.e2 = np.cross(d, self.e1)

        self.th_l = self._edge_theta(self.lines["edge_l"])
        self.th_r = self._edge_theta(self.lines["edge_r"])
        if self.th_l > self.th_r:                   # make theta increase left->right
            self.e2 = -self.e2
            self.th_l, self.th_r = -self.th_l, -self.th_r

    def _edge_theta(self, l):
        n = l - (l @ self.d) * self.d
        A, B, C0 = n @ self.e1, n @ self.e2, n @ self.c0
        rho, phi = np.hypot(A, B), np.arctan2(B, A)
        a = np.arccos(np.clip(-C0 / rho, -1, 1))
        cands = [phi + a, phi - a]
        return min(cands, key=lambda th: np.linalg.norm(self.surface(th, 0.0)))

    def surface(self, th, s):
        th, s = np.broadcast_arrays(np.asarray(th, float), np.asarray(s, float))
        return (self.c0 + np.cos(th)[..., None] * self.e1
                + np.sin(th)[..., None] * self.e2 + s[..., None] * self.d)

    def to_pixels(self, P):
        return P[..., :2] / P[..., 2:3] * self.f + self.c

    def _s_on_curve(self, th, curve):
        """Where the ruling at theta meets a traced curve, as axial position s."""
        Q = self.surface(th, 0.0)
        l = np.cross(self.V, Q); l /= np.hypot(l[0], l[1])
        sd = curve @ l[:2] + l[2]
        i = np.flatnonzero(np.sign(sd[:-1]) != np.sign(sd[1:]))
        if len(i) == 0:                             # beyond the traced curve: extrapolate
            i = [0] if abs(sd[0]) < abs(sd[-1]) else [len(sd) - 2]
        i = i[0]
        w = sd[i] / (sd[i] - sd[i + 1])
        p = curve[i] * (1 - w) + curve[i + 1] * w
        ray = np.array([p[0], p[1], 1.0])
        A = np.stack([ray, -self.d], 1)
        ts, *_ = np.linalg.lstsq(A, Q, rcond=None)
        return ts[1]

    def fit_heights(self, n=400):
        self.th_s = np.linspace(self.th_l, self.th_r, n)
        self.s_top = np.array([self._s_on_curve(t, self.nq["top"]) for t in self.th_s])
        self.s_bot = np.array([self._s_on_curve(t, self.nq["bot"]) for t in self.th_s])
        self.width = self.th_r - self.th_l           # in units of R
        self.height = np.median(np.abs(self.s_bot - self.s_top))
        return self

    def label_to_image(self, fx, fy):
        """Fractions across (fx) and down (fy) the label -> image pixels."""
        th = self.th_l + fx * (self.th_r - self.th_l)
        st = np.interp(th, self.th_s, self.s_top)
        sb = np.interp(th, self.th_s, self.s_bot)
        return self.to_pixels(self.surface(th, st + fy * (sb - st)))


# ---------------------------------------------------------------- warping
def grid_warp(img, nodes, out_w, out_h):
    """Piecewise homography: nodes[j, i] = image xy of grid node (row j, col i)."""
    nr, nc = nodes.shape[0] - 1, nodes.shape[1] - 1
    cw, ch = out_w / nc, out_h / nr
    Hs = np.zeros((nr, nc, 3, 3))
    for j in range(nr):
        for i in range(nc):
            dst = np.float32([[i * cw, j * ch], [(i + 1) * cw, j * ch],
                              [(i + 1) * cw, (j + 1) * ch], [i * cw, (j + 1) * ch]])
            src = np.float32([nodes[j, i], nodes[j, i + 1], nodes[j + 1, i + 1], nodes[j + 1, i]])
            Hs[j, i] = cv2.getPerspectiveTransform(dst, src)
    y, x = np.mgrid[0:out_h, 0:out_w].astype(np.float64) + 0.5
    ci = np.clip((x // cw).astype(int), 0, nc - 1)
    cj = np.clip((y // ch).astype(int), 0, nr - 1)
    Hp = Hs[cj, ci]
    p = np.einsum("hwab,hwb->hwa", Hp, np.stack([x, y, np.ones_like(x)], -1))
    mx, my = (p[..., 0] / p[..., 2]).astype(np.float32), (p[..., 1] / p[..., 2]).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return out, np.stack([mx, my], -1)


def model_grid(model, nc, nr):
    fx, fy = np.meshgrid(np.linspace(0, 1, nc + 1), np.linspace(0, 1, nr + 1))
    return model.label_to_image(fx, fy)


def naive_grid(feats, model, nc, nr):
    """The straightforward version: column lines through V at equal image angles
    between the label edges, rows by straight linear interpolation in the image."""
    f, c = model.f, model.c
    Vp = model.V[:2] / model.V[2] * f + c
    def ang(line_k):
        q = feats[line_k].mean(0) - Vp
        return np.arctan2(q[1], q[0])
    a0, a1 = ang("edge_l"), ang("edge_r")
    nodes = np.zeros((nr + 1, nc + 1, 2))
    for i, a in enumerate(np.linspace(a0, a1, nc + 1)):
        dvec = np.array([np.cos(a), np.sin(a)]); nvec = np.array([-dvec[1], dvec[0]])
        pts = []
        for k in ["top", "bot"]:
            cur = feats[k]; sd = (cur - Vp) @ nvec
            idx = np.flatnonzero(np.sign(sd[:-1]) != np.sign(sd[1:]))
            idx = idx[0] if len(idx) else (0 if abs(sd[0]) < abs(sd[-1]) else len(sd) - 2)
            w = sd[idx] / (sd[idx] - sd[idx + 1])
            pts.append(cur[idx] * (1 - w) + cur[idx + 1] * w)
        for j, t in enumerate(np.linspace(0, 1, nr + 1)):
            nodes[j, i] = pts[0] * (1 - t) + pts[1] * t
    return nodes
