"""Root area when an arc is excluded (2026-11-10): which estimate for traits_DATE.xlsx?

  .venv/bin/python docs/development/area_estimate.py        (from the repo root; ~1 min)
  -> docs/development/area_estimate.png, area_estimate.csv

Three ways to estimate the whole root from the kept angles:
  scale    stele + kept cortex x 360 / kept degrees (the old root_area_est: assumes the excluded arc is average)
  fill     pipeline.root_area_filled: kept angles pixel by pixel, the root radius interpolated linearly across the arc
  fourier  a 2-harmonic fit of the kept radii, extended across the arc
Truth: intact real sections (both shipped example projects, stele found, nothing excluded); an arc of 60 or 120
degrees is cut at 8 positions and each estimate compared with the true pixel count. Plus synthetic ovals with the
edge torn in over the arc (what an excluded arc usually looks like).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import pipeline as P  # noqa: E402


def estimates(region, centre, arc):
    deg = np.arange(360)
    out = P.in_arcs(deg + 0.5, [arc])
    yy, xx = np.nonzero(region >= 1)
    kept_px = ~P.in_arcs(P.angle_of(yy, xx, centre), [arc])
    stele = (region == 2).sum()
    kept_ctx = (kept_px & (region[yy, xx] == 1)).sum()
    scale = stele + kept_ctx * 360 / (360 - out.sum())
    fill = P.root_area_filled(region, centre, [(arc[0], arc[1], "test")])
    r_rt = P.radial_profile(region, centre)[1]
    k = np.nonzero(~out)[0]
    th = np.radians(deg[k])
    X = lambda t: np.column_stack([np.ones_like(t)] + [f(h * t) for h in (1, 2) for f in (np.cos, np.sin)])
    coef = np.linalg.lstsq(X(th), r_rt[k], rcond=None)[0]
    rf = r_rt.copy()
    rf[out] = X(np.radians(deg[out])) @ coef
    fourier = 0.5 * np.sum((rf + 0.5) ** 2) * np.radians(1)
    return scale, fill, fourier


rows = []
N = 1000
yy, xx = np.indices((N, N))
C = (500.0, 500.0)
ang = P.angle_of(yy, xx, C)
for a_, b_ in ((400, 250), (330, 300)):
    reg = np.zeros((N, N), np.uint8)
    reg[((xx - C[1]) / a_) ** 2 + ((yy - C[0]) / b_) ** 2 <= 1] = 1
    reg[np.hypot(yy - C[0], xx - C[1]) <= 100] = 2
    true = (reg >= 1).sum()
    for side, mid in (("long side", 0), ("short side", 90), ("diagonal", 45)):
        for width in (60, 120):
            arc = ((mid - width / 2) % 360, (mid + width / 2) % 360)
            torn = reg.copy()
            torn[P.in_arcs(ang, [arc]) & (np.hypot(yy - C[0], xx - C[1]) > 0.75 * min(a_, b_)) & (torn == 1)] = 0
            s, f, fo = estimates(torn, C, arc)
            rows.append(dict(set="synthetic oval, edge torn in", img=f"oval {a_}x{b_} {side}", width=width,
                             scale=100 * (s / true - 1), fill=100 * (f / true - 1), fourier=100 * (fo / true - 1)))
for f in sorted(glob.glob(str(HERE.parent.parent / "examples/*/anatomy/masks/*.npz"))):
    cells, region, mx, _ = P.load_state(f)
    if not (region == 2).any() or P.load_sectors(f, 0):
        continue
    centre = P.centre_of(region, P.load_centre(f))
    true = (region >= 1).sum()
    for width in (60, 120):
        for mid in range(0, 360, 45):
            arc = ((mid - width / 2) % 360, (mid + width / 2) % 360)
            s, fi, fo = estimates(region, centre, arc)
            rows.append(dict(set="real intact section, arc cut", img=Path(f).stem, width=width,
                             scale=100 * (s / true - 1), fill=100 * (fi / true - 1), fourier=100 * (fo / true - 1)))
d = pd.DataFrame(rows)
d.to_csv(HERE / "area_estimate.csv", index=False)
real = d[d.set.str.startswith("real")]
print("real sections:", real.img.nunique())
summ = real.groupby("width")[["scale", "fill", "fourier"]].agg([lambda v: np.median(np.abs(v)), lambda v: np.max(np.abs(v))])
summ.columns = [f"{m} {'median' if i == 0 else 'max'} |err| %" for m, i in
                [(c, k) for c in ("scale", "fill", "fourier") for k in (0, 1)]]
print(summ.round(2).T.to_string())
print(d[d.set.str.startswith("synthetic")].round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
col = {"scale": "#999999", "fill": "#1f6fb5", "fourier": "#d0661c"}
for ax, w in zip(axes, (60, 120)):
    r = real[real.width == w]
    for j, m in enumerate(("scale", "fill", "fourier")):
        v = np.abs(r[m].to_numpy())
        ax.scatter(np.full(len(v), j) + np.random.default_rng(1).uniform(-.18, .18, len(v)), v, s=6, color=col[m],
                   alpha=.5, lw=0)
        ax.plot([j - .3, j + .3], [np.median(v)] * 2, color="k", lw=1.2)
        ax.text(j + .33, np.median(v), f"{np.median(v):.2f}", ha="left", va="center", fontsize=7)
    ax.set_xticks([0, 1, 2], ["scale by\nangle", "fill the\nedge", "Fourier\n(2 harmonics)"], fontsize=7)
    ax.set_title(f"{w}° arc cut out ({r.img.nunique()} sections x 8 positions)", fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axes[0].set_ylabel("|error| in root area (%)", fontsize=8)
fig.suptitle("Root area with an arc excluded: fill the edge is chosen (median shown)", fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "area_estimate.png", dpi=200)
