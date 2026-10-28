"""One-call flattening plus ground-truth scoring for synthetic scenes."""
from dataclasses import dataclass
import numpy as np
from .rectify import detect, Model, grid_warp, model_grid, naive_grid


@dataclass
class Flattened:
    model: Model
    feats: dict
    aspect: float          # estimated label width / height
    nodes: np.ndarray      # grid nodes in the photo (cylinder model)
    image: np.ndarray      # flattened label (cylinder model)
    mapxy: np.ndarray      # photo xy for every output pixel
    naive_nodes: np.ndarray
    naive_image: np.ndarray
    naive_mapxy: np.ndarray


def flatten(img, f, centre=None, nc=48, nr=32, out_w=600, feats=None):
    """Flatten the label in `img` given the focal length `f` in pixels.
    `centre` is the principal point (default: image centre)."""
    H, W = img.shape[:2]
    centre = centre if centre is not None else (W / 2, H / 2)
    feats = feats if feats is not None else detect(img)
    mdl = Model(feats, f, centre).fit_heights()
    aspect = mdl.width / mdl.height
    out_h = int(round(out_w / aspect))
    nodes = model_grid(mdl, nc, nr)
    flat, mp = grid_warp(img, nodes, out_w, out_h)
    nn = naive_grid(feats, mdl, nc, nr)
    flat_n, mp_n = grid_warp(img, nn, out_w, out_h)
    return Flattened(mdl, feats, aspect, nodes, flat, mp, nn, flat_n, mp_n)


def score(scene, mapxy, step=4):
    """Error of a flattening against the true label, in label millimetres.
    Every output pixel is traced back through the true geometry to the point of
    the label it actually shows, and compared with where it should be."""
    out_h, out_w = mapxy.shape[:2]
    m = mapxy[::step, ::step].astype(np.float64)
    y, x = np.mgrid[0:out_h:step, 0:out_w:step] + 0.5
    u_t, v_t, ok = scene.uv_to_label_mm(m)
    u_i, v_i = x / out_w * scene.lw, y / out_h * scene.lh
    if not ok.any():
        return dict(rms_mm=float("inf"), max_mm=float("inf"))
    e = np.hypot(u_t - u_i, v_t - v_i)[ok]     # pixels that missed the bottle are ignored
    return dict(rms_mm=float(np.sqrt((e ** 2).mean())), max_mm=float(e.max()))
