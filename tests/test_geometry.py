"""With perfect edge features the reconstruction must be exact; with rendered
photos it must be well under a millimetre for ordinary viewpoints."""
import numpy as np
import pytest
from labelflat import Scene, make_label, Model, flatten, score

VIEWS = [dict(), dict(elevation=0.0), dict(elevation=-25.0, rotation=-10.0),
         dict(elevation=50.0, lean=-20.0, pan=-8.0, label_w=70.0), dict(distance=160.0, label_w=60.0)]


@pytest.mark.parametrize("view", VIEWS)
def test_exact_features_give_exact_geometry(view):
    sc = Scene.from_view(**view)
    assert sc.label_visible()
    m = Model(sc.exact_features(), sc.f, (sc.size[0] / 2, sc.size[1] / 2)).fit_heights()
    assert np.degrees(m.width) == pytest.approx(np.degrees(sc.th1 - sc.th0), abs=1e-6)
    assert m.width / m.height == pytest.approx(sc.lw / sc.lh, rel=1e-6)


@pytest.mark.parametrize("view", VIEWS)
def test_exact_features_give_exact_flattening(view):
    sc = Scene.from_view(**view)
    img = np.zeros((sc.size[1], sc.size[0], 3), np.uint8)   # pixels unused for scoring
    r = flatten(img, sc.f, feats=sc.exact_features())
    e = score(sc, r.mapxy, step=2)                  # only straight-sided-cell error left
    assert e["rms_mm"] < 0.05 and e["max_mm"] < 0.35


@pytest.mark.parametrize("view", [dict(label_w=75.0), dict(elevation=40.0, rotation=-8.0, lean=10.0, label_w=70.0)])
def test_rendered_photo_sub_millimetre(view):
    sc = Scene.from_view(size=(1280, 960), **view)
    assert min(sc.edge_gaps_px()) > 5
    img = sc.render(make_label(int(sc.lw), 110), ss=2)
    r = flatten(img, sc.f)
    assert score(sc, r.mapxy)["rms_mm"] < 1.0
    assert score(sc, r.naive_mapxy)["rms_mm"] > 3.0       # the naive grid really is worse


def test_cell_error_shrinks_quadratically():
    sc = Scene.from_view(elevation=-25.0, rotation=-10.0)
    img = np.zeros((sc.size[1], sc.size[0], 3), np.uint8)
    e = [score(sc, flatten(img, sc.f, feats=sc.exact_features(), nc=n, nr=2 * n // 3).mapxy,
               step=2)["rms_mm"] for n in (24, 48)]
    assert 2.5 < e[0] / e[1] < 5.5          # halving the cell size cuts the error ~4x
