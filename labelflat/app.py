"""Interactive explorer: move the camera around a synthetic bottle and compare
the recovered label with the true one.

    python -m labelflat.app                 # interactive window
    python -m labelflat.app --save out.png --elevation 30 --rotation -20
"""
import argparse
import sys
import time

PARAMS = [
    # key,         slider label,          min,  max,  default, format
    ("elevation", "Camera elevation (°)", -30.0, 60.0, 17.0, "%.0f"),
    ("rotation",  "Bottle rotation (°)",  -60.0, 60.0, 0.0,  "%.0f"),
    ("lean",      "Bottle lean (°)",      -30.0, 30.0, 4.0,  "%.0f"),
    ("pan",       "Off-centre (°)",       -15.0, 15.0, 3.0,  "%.0f"),
    ("distance",  "Distance (mm)",        150.0, 600.0, 260.0, "%.0f"),
    ("label_w",   "Label width (mm)",     40.0, 115.0, 75.0, "%.0f"),
    ("f_err",     "Focal length error (%)", -30.0, 30.0, 0.0, "%+.0f"),
    ("noise",     "Sensor noise",         0.0, 15.0, 1.5,  "%.1f"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, label, lo, hi, d, _ in PARAMS:
        p.add_argument("--" + key.replace("_", "-"), dest=key, type=float, default=d,
                       help=f"{label}, {lo:g} to {hi:g} (default {d:g})")
    p.add_argument("--save", metavar="PNG", help="render once to a file instead of opening a window")
    p.add_argument("--size", default="960x720", help="photo size in pixels, WxH (default 960x720)")
    p.add_argument("--supersample", type=int, default=2, help="anti-aliasing samples per axis (default 2)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, CheckButtons
    from .scene import Scene, make_label
    from .pipeline import flatten, score

    size = tuple(int(v) for v in args.size.lower().split("x"))

    class App:
        def __init__(self):
            self.fig = fig = plt.figure(figsize=(13, 9))
            fig.canvas.manager.set_window_title("Wine label flattening") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
            self.ax_photo = fig.add_axes([0.02, 0.34, 0.53, 0.61])
            self.ax_truth = fig.add_axes([0.58, 0.66, 0.40, 0.29])
            self.ax_rec = fig.add_axes([0.58, 0.34, 0.40, 0.29])
            for a in (self.ax_photo, self.ax_truth, self.ax_rec):
                a.set_xticks([]); a.set_yticks([])
            self.stats = fig.text(0.58, 0.322, "", va="top", fontsize=9, family="monospace")

            self.sliders = {}
            for i, (key, label, lo, hi, d, fmt) in enumerate(PARAMS):
                col, row = divmod(i, 4)
                ax = fig.add_axes([0.16 + col * 0.48, 0.19 - row * 0.045, 0.27, 0.025])
                s = Slider(ax, label, lo, hi, valinit=getattr(args, key), valfmt=fmt)
                s.on_changed(self._mark_dirty)
                self.sliders[key] = s

            ax_m = fig.add_axes([0.02, 0.215, 0.12, 0.07]); ax_m.set_title("Recovered with", fontsize=9, loc="left")
            self.method = RadioButtons(ax_m, ["Cylinder model", "Naive"], active=0)
            self.method.on_clicked(lambda _: self._show())
            ax_g = fig.add_axes([0.15, 0.215, 0.14, 0.07])
            self.grid_on = CheckButtons(ax_g, ["Grid on photo"], [True])
            self.grid_on.on_clicked(lambda _: self._show())
            fig.text(0.62, 0.012, "Sliders apply when you release the mouse.", fontsize=8, color="0.45")

            self.dirty = True
            self.labels = {}
            self.res = None
            fig.canvas.mpl_connect("button_release_event", self._on_release)

        def _mark_dirty(self, _):
            self.dirty = True

        def _on_release(self, event):
            if self.dirty:
                self.recompute()

        def values(self):
            return {k: s.val for k, s in self.sliders.items()}

        def recompute(self):
            self.dirty = False
            v = self.values()
            self.ax_photo.set_title("Rendering…", fontsize=11)
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except NotImplementedError:
                pass
            t0 = time.time()
            sc = Scene.from_view(elevation=v["elevation"], rotation=v["rotation"], lean=v["lean"],
                                 pan=v["pan"], distance=v["distance"], label_w=v["label_w"], size=size)
            w = int(round(v["label_w"]))
            if w not in self.labels:
                self.labels[w] = make_label(w, 110)
            label = self.labels[w]
            img = sc.render(label, ss=args.supersample, noise=v["noise"])
            t1 = time.time()
            self.scene, self.img, self.label = sc, img, label
            self.res, self.err, self.scores = None, None, None
            try:
                self.res = flatten(img, sc.f * (1 + v["f_err"] / 100), out_w=500)
                self.scores = (score(sc, self.res.mapxy), score(sc, self.res.naive_mapxy))
            except Exception as e:                  # detection or geometry gave up
                self.err = f"{type(e).__name__}: {e}"
            self.timing = (t1 - t0, time.time() - t1)
            self._show()

        def _show(self):
            sc, v = self.scene, self.values()
            ax = self.ax_photo; ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(self.img)
            ax.set_title("Synthetic photo", fontsize=11)
            naive = self.method.value_selected == "Naive"
            if self.res is not None and self.grid_on.get_status()[0]:
                nodes = self.res.naive_nodes if naive else self.res.nodes
                col = "#2f7de1" if naive else "#e8423c"
                step = max(1, (nodes.shape[1] - 1) // 24)      # keep the overlay readable
                for j in range(0, nodes.shape[0], step):
                    ax.plot(nodes[j, :, 0], nodes[j, :, 1], c=col, lw=0.6)
                for i in range(0, nodes.shape[1], step):
                    ax.plot(nodes[:, i, 0], nodes[:, i, 1], c=col, lw=0.6)
                ax.set_xlim(0, self.img.shape[1]); ax.set_ylim(self.img.shape[0], 0)

            a = self.ax_truth; a.clear(); a.set_xticks([]); a.set_yticks([])
            a.imshow(self.label); a.set_title(f"True label  ({sc.lw:.0f} × {sc.lh:.0f} mm)", fontsize=10)

            a = self.ax_rec; a.clear(); a.set_xticks([]); a.set_yticks([])
            warn = []
            gaps = sc.edge_gaps_px()
            if not sc.label_visible():
                warn.append("A label edge is behind the bottle's outline from this\n"
                            "viewpoint, so its corner can't be seen or recovered.")
            else:
                for side, g in zip(("left", "right"), gaps):
                    if g < 3.0:
                        warn.append(f"The {side} label edge is only {g:.1f} px from the bottle outline, so\n"
                                    f"no glass shows beside it and its angle is poorly determined.")
            if self.res is None:
                a.text(0.5, 0.5, "Could not recover the label\nfrom this view", ha="center", va="center",
                       transform=a.transAxes, fontsize=11, color="#c0392b")
                a.set_title("Recovered label", fontsize=10)
                warn.append(self.err or "")
                self.stats.set_text(f"render {self.timing[0]:.1f}s")
            else:
                r = self.res
                a.imshow(r.naive_image if naive else r.image)
                a.set_title(f"Recovered label ({'naive' if naive else 'cylinder model'})", fontsize=10)
                m, n = self.scores
                span_t = np.degrees(sc.th1 - sc.th0); span_e = np.degrees(r.model.width)
                self.stats.set_text(
                    f"Error vs truth   cylinder: RMS {m['rms_mm']:5.2f} mm  max {m['max_mm']:5.2f} mm\n"
                    f"                 naive:    RMS {n['rms_mm']:5.2f} mm  max {n['max_mm']:5.2f} mm\n"
                    f"Aspect (w/h)     est {r.aspect:.3f}   true {sc.lw / sc.lh:.3f}\n"
                    f"Label wraps      est {span_e:5.1f}°  true {span_t:5.1f}°\n"
                    f"Edge-outline gap left {gaps[0]:5.1f} px   right {gaps[1]:5.1f} px\n"
                    f"Time             render {self.timing[0]:.1f}s  flatten {self.timing[1]:.1f}s")
            msg = "\n".join(w for w in warn if w)
            if msg:
                self.ax_photo.text(0.015, 0.985, msg, transform=self.ax_photo.transAxes, va="top",
                                   fontsize=9, color="#a93226",
                                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#a93226", alpha=0.92))
            self.fig.canvas.draw_idle()

    app = App()
    app.recompute()
    if args.save:
        app.fig.savefig(args.save, dpi=100)
        print(f"saved {args.save}")
        if app.scores:
            print(f"cylinder RMS {app.scores[0]['rms_mm']:.2f} mm, naive RMS {app.scores[1]['rms_mm']:.2f} mm")
    else:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
