"""Sweep one viewing parameter and print accuracy against ground truth.

    python experiments/sweep.py elevation -30 0 20 40 60
    python experiments/sweep.py label_w 60 75 90 100
    python experiments/sweep.py f_err -30 -10 0 10 30
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from labelflat import Scene, make_label, flatten, score  # noqa: E402

VIEW = dict(elevation=17.0, rotation=0.0, lean=4.0, pan=3.0, distance=260.0, label_w=75.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("param", choices=list(VIEW) + ["f_err", "noise"])
    p.add_argument("values", type=float, nargs="+")
    p.add_argument("--size", default="1600x1200")
    args = p.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))

    print(f"{args.param:>10} | cyl RMS  cyl max | naive RMS | aspect err | wrap err | edge gaps px")
    for val in args.values:
        view, f_err, noise = dict(VIEW), 0.0, 1.5
        if args.param == "f_err":
            f_err = val
        elif args.param == "noise":
            noise = val
        else:
            view[args.param] = val
        sc = Scene.from_view(size=size, **view)
        gaps = "  ".join(f"{g:5.1f}" for g in sc.edge_gaps_px())
        if not sc.label_visible():
            print(f"{val:10g} | label edge hidden behind the outline          | {gaps}")
            continue
        img = sc.render(make_label(int(round(sc.lw)), 110), ss=2, noise=noise)
        try:
            r = flatten(img, sc.f * (1 + f_err / 100))
        except Exception as e:
            print(f"{val:10g} | failed: {type(e).__name__}: {e}")
            continue
        m, n = score(sc, r.mapxy), score(sc, r.naive_mapxy)
        a_err = 100 * (r.aspect / (sc.lw / sc.lh) - 1)
        w_err = np.degrees(r.model.width - (sc.th1 - sc.th0))
        print(f"{val:10g} | {m['rms_mm']:6.2f}  {m['max_mm']:7.2f} | {n['rms_mm']:9.2f} | "
              f"{a_err:+9.2f}% | {w_err:+7.2f}° | {gaps}")


if __name__ == "__main__":
    main()
