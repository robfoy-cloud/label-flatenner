"""Synthetic wine-bottle scene: a flat label wrapped on a perfect cylinder,
photographed by a pinhole camera. Units are millimetres.

World frame: the bottle axis is the y axis (up), the bottle surface is
x^2 + z^2 = R^2, and theta = atan2(x, z) is the angle around the bottle.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

FONT_BOLD = font_manager.findfont(FontProperties(family="DejaVu Serif", weight="bold"))
FONT_REG = font_manager.findfont(FontProperties(family="DejaVu Sans"))


def make_label(w_mm=90, h_mm=110, ppm=10):
    """Test label: 10 mm grid, a border, circles and text (ppm = pixels per mm).
    Returns float RGB in [0, 1]."""
    w_mm, h_mm = int(round(w_mm)), int(round(h_mm))
    W, H = w_mm * ppm, h_mm * ppm
    img = Image.new("RGB", (W, H), (240, 230, 205))
    d = ImageDraw.Draw(img)
    for x in range(0, w_mm + 1, 10):
        d.line([(x * ppm, 0), (x * ppm, H)], fill=(185, 175, 160), width=2)
    for y in range(0, h_mm + 1, 10):
        d.line([(0, y * ppm), (W, y * ppm)], fill=(185, 175, 160), width=2)
    m = 4 * ppm
    d.rectangle([m, m, W - m, H - m], outline=(120, 30, 40), width=6)
    r_big = min(18, w_mm / 4)
    for cx, cy, r in [(w_mm / 2, 62, r_big), (20, 95, 7), (w_mm - 20, 95, 7)]:
        d.ellipse([(cx - r) * ppm, (cy - r) * ppm, (cx + r) * ppm, (cy + r) * ppm],
                  outline=(30, 60, 110), width=5)
    scale = min(1.0, w_mm / 90)
    f1 = ImageFont.truetype(FONT_BOLD, int(9 * ppm * scale))
    f2 = ImageFont.truetype(FONT_REG, int(5 * ppm * scale))
    for text, font, y in [("CHÂTEAU", f1, 14), ("FLATLAND", f1, 26),
                          ("GRAND VIN · 2026", f2, 38)]:
        tw = d.textlength(text, font=font)
        d.text(((W - tw) / 2, y * ppm), text, font=font, fill=(40, 30, 30))
    return np.asarray(img).astype(np.float32) / 255.0


def look_at(C, target, roll_deg=0.0):
    """World->camera rotation (camera x right, y down, z forward)."""
    fwd = target - C; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rw = np.stack([right, down, fwd])
    a = np.radians(roll_deg)
    Rr = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rr @ Rw


class Scene:
    def __init__(self, R=37.0, label_w=90.0, label_h=110.0, label_centre_deg=8.0,
                 label_y=0.0, cam_dist=260.0, cam_height=80.0, cam_side=25.0,
                 target=(12.0, 0.0, 0.0), roll=4.0, f=1200.0, size=(1600, 1200)):
        self.R, self.lw, self.lh = R, label_w, label_h
        half = label_w / R / 2
        c = np.radians(label_centre_deg)
        self.th0, self.th1 = c - half, c + half          # label angular span
        self.y0, self.y1 = label_y - label_h / 2, label_y + label_h / 2
        self.C = np.array([cam_side, cam_height, cam_dist], float)
        self.Rw = look_at(self.C, np.array(target, float), roll)
        self.f, self.size = f, tuple(size)
        W, H = size
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    @classmethod
    def from_view(cls, elevation=17.0, rotation=0.0, lean=4.0, pan=3.0, distance=260.0,
                  label_w=90.0, label_h=110.0, R=37.0, size=(960, 720), f=None):
        """Scene described by viewing angles (degrees) and distance (mm).

        elevation: camera height angle above the label centre (negative = below)
        rotation:  how far the bottle is spun, i.e. label centre vs line of sight
        lean:      camera roll, so the bottle leans in the frame
        pan:       bottle sits off the image centre by this angle
        f:         focal length in pixels (default 0.75 x image width, phone-like)
        """
        e = np.radians(elevation)
        C = (0.0, distance * np.sin(e), distance * np.cos(e))
        target = (distance * np.tan(np.radians(pan)), 0.0, 0.0)
        return cls(R=R, label_w=label_w, label_h=label_h, label_centre_deg=rotation,
                   cam_side=C[0], cam_height=C[1], cam_dist=C[2], target=target,
                   roll=lean, f=f or 0.75 * size[0], size=size)

    # --- geometry helpers -------------------------------------------------
    def project(self, Pw):
        Pc = (Pw - self.C) @ self.Rw.T
        uvw = Pc @ self.K.T
        return uvw[..., :2] / uvw[..., 2:3]

    def cyl_point(self, th, y):
        th, y = np.broadcast_arrays(th, y)
        return np.stack([self.R * np.sin(th), y, self.R * np.cos(th)], -1)

    def rays(self, uv):
        h = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1)
        d = (h @ np.linalg.inv(self.K).T) @ self.Rw   # camera->world
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def limbs(self):
        """Angles theta of the two visible outline (silhouette) lines."""
        Dh = np.hypot(self.C[0], self.C[2]); az = np.arctan2(self.C[0], self.C[2])
        a = np.arccos(self.R / Dh)
        return az - a, az + a

    def label_visible(self):
        lo, hi = self.limbs()
        return lo < self.th0 and self.th1 < hi

    def edge_gaps_px(self):
        """Image distance (pixels) between each label side edge and the bottle
        outline beside it, at mid-label height. Below ~3 px little glass shows
        between label and background, so the edge angle is poorly determined."""
        lo, hi = self.limbs()
        ym = 0.5 * (self.y0 + self.y1)
        p = lambda t, y: self.project(self.cyl_point(t, y))

        def gap(edge, limb):
            # perpendicular distance from the label edge to the outline line
            a, b = p(limb, ym - 20.0), p(limb, ym + 20.0)
            q = p(edge, ym)
            t = (b - a) / np.linalg.norm(b - a)
            return float(abs((q - a)[0] * t[1] - (q - a)[1] * t[0]))

        gl = gap(self.th0, lo) if self.th0 > lo else 0.0
        gr = gap(self.th1, hi) if self.th1 < hi else 0.0
        return gl, gr

    def hit(self, uv):
        """Front intersection of pixel rays with the cylinder -> (theta, y, valid, shade)."""
        d = self.rays(uv)
        C = self.C
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (C[0] * d[..., 0] + C[2] * d[..., 2])
        c = C[0] ** 2 + C[2] ** 2 - self.R ** 2
        disc = b * b - 4 * a * c
        ok = disc > 0
        t = (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a)
        P = C + t[..., None] * d
        th = np.arctan2(P[..., 0], P[..., 2])
        nz = np.stack([P[..., 0], P[..., 2]], -1) / self.R          # surface normal (x,z)
        vz = -np.stack([d[..., 0], d[..., 2]], -1)
        shade = np.clip((nz * vz).sum(-1) / np.maximum(np.linalg.norm(vz, axis=-1), 1e-9), 0, 1)
        return th, P[..., 1], ok, shade

    # --- rendering ----------------------------------------------------------
    def render(self, label, ss=3, noise=1.5, seed=0):
        """Ray-trace the photo with ss x ss supersampling and Gaussian sensor noise."""
        W, H = self.size
        Lh, Lw = label.shape[:2]
        out = np.zeros((H, W, 3), np.float32)
        offs = (np.arange(ss) + 0.5) / ss - 0.5
        bg = np.array([200, 205, 215], np.float32) / 255
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        for oy in offs:
            for ox in offs:
                th, y, ok, shade = self.hit(np.stack([u + ox, v + oy], -1))
                col = np.broadcast_to(bg, (H, W, 3)).copy()
                glass = np.array([0.05, 0.16, 0.08]) * (0.5 + 0.8 * shade[..., None])
                glass += 0.25 * np.exp(-((shade - 0.93) / 0.03) ** 2)[..., None]  # highlight
                col[ok] = glass[ok]
                inl = ok & (th >= self.th0) & (th <= self.th1) & (y >= self.y0) & (y <= self.y1)
                # label texture coords: left edge = th0, top edge = y1
                lx = (th - self.th0) / (self.th1 - self.th0) * (Lw - 1)
                ly = (self.y1 - y) / (self.y1 - self.y0) * (Lh - 1)
                lx, ly = np.clip(lx, 0, Lw - 1), np.clip(ly, 0, Lh - 1)
                x0, y0 = np.floor(lx).astype(int), np.floor(ly).astype(int)
                x1, y1 = np.minimum(x0 + 1, Lw - 1), np.minimum(y0 + 1, Lh - 1)
                fx, fy = (lx - x0)[..., None], (ly - y0)[..., None]
                tex = (label[y0, x0] * (1 - fx) * (1 - fy) + label[y0, x1] * fx * (1 - fy)
                       + label[y1, x0] * (1 - fx) * fy + label[y1, x1] * fx * fy)
                tex = tex * (0.55 + 0.45 * shade[..., None])
                col[inl] = tex[inl]
                out += col
        out /= ss * ss
        rng = np.random.default_rng(seed)
        out += rng.normal(0, noise / 255, out.shape)
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    # --- ground truth for evaluation ----------------------------------------
    def uv_to_label_mm(self, uv):
        th, y, ok, _ = self.hit(uv)
        return (th - self.th0) * self.R, (self.y1 - y), ok

    def exact_features(self, n=400):
        """Noise-free edge features straight from the geometry (for tests)."""
        lo, hi = self.limbs()
        ys = np.linspace(-400, 400, 4 * n)
        yl = np.linspace(self.y0, self.y1, n)
        th = np.linspace(self.th0, self.th1, n)
        ruling = lambda t, y: self.project(self.cyl_point(np.full_like(y, t), y))
        f = dict(sil_l=ruling(lo, ys), sil_r=ruling(hi, ys),
                 edge_l=ruling(self.th0, yl), edge_r=ruling(self.th1, yl),
                 top=self.project(self.cyl_point(th, np.full_like(th, self.y1))),
                 bot=self.project(self.cyl_point(th, np.full_like(th, self.y0))))
        W, H = self.size
        for k in ["sil_l", "sil_r"]:
            p = f[k]; f[k] = p[(p[:, 1] > 0) & (p[:, 1] < H) & (p[:, 0] > 0) & (p[:, 0] < W)]
        return f
