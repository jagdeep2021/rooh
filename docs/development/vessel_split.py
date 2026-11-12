"""Merged vessels (2026-11-12). Per image: vessels as saved vs after the vessel rule (P.clean_vessels: holes filled,
pen strokes dropped, two round bodies split), and what that does to mx_n, mean vessel area and conductance (per px
scale, so it holds with or without a scale). Draws every vessel the rule changed, with the cut.

  .venv/bin/python docs/development/vessel_split.py <project_dir> <out.csv> [figure.png]
Run on a copy of Hallie's reviewed web project "test" and on examples/: numbers in the 2026-11-12 vessel entry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pipeline as P  # noqa: E402

proj = Path(sys.argv[1])
anat = proj / "anatomy"
rows, show = [], []
for npz in sorted((anat / "masks").glob("*.npz")):
    z = np.load(npz, allow_pickle=False)
    raw, cells = z["mx"], z["cells"]
    new, n = P.clean_vessels(raw, cells)
    a0 = np.bincount(raw.ravel())[1:]
    a0 = a0[a0 > 0].astype(float)
    a1 = np.bincount(new.ravel())[1:]
    a1 = a1[a1 > 0].astype(float)
    k = lambda a: np.sum((a / np.pi) ** 2)          # sum r^4, px
    rows.append({"image": npz.stem, "mx_n_saved": len(a0), "mx_n_rule": len(a1), **n,
                 "mean_area_change": a1.mean() / a0.mean() - 1 if len(a0) and len(a1) else np.nan,
                 "conductance_change": k(a1) / k(a0) - 1 if len(a0) else np.nan})
    if len(sys.argv) > 3 and (n["split"] or n["dropped"]):
        for g in measure.regionprops(raw):
            ids = np.unique(new[g.slice][g.image])
            ids = ids[ids > 0]
            if len(ids) != 1 or g.area < 0.5 * len(ids):     # split or dropped
                show.append((npz, g.bbox, raw == g.label, new))
t = pd.DataFrame(rows)
t.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
ch = t[t.mx_n_saved != t.mx_n_rule]
print(f"\nimages changed {len(ch)}/{len(t)}; vessels {t.mx_n_saved.sum()} -> {t.mx_n_rule.sum()} "
      f"(split +{t.split.sum()}, dropped -{t.dropped.sum()}); hole px filled {t.holes_px.sum()}")
if len(ch):
    print("changed images, median: mean vessel area", f"{ch.mean_area_change.median():+.0%}",
          "conductance", f"{ch.conductance_change.median():+.0%}", "range", f"{ch.conductance_change.min():+.0%} to {ch.conductance_change.max():+.0%}")

if show:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imdir = P.image_dir(anat, json.loads((anat / "settings.json").read_text()))
    cols = 7
    fig, axs = plt.subplots(int(np.ceil(len(show) / cols)), cols, figsize=(cols * 3, np.ceil(len(show) / cols) * 3),
                            squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (npz, bb, was, new) in zip(axs.ravel(), show):
        gray = P.load_gray(imdir / str(np.load(npz)["image_file"]))
        y0, x0, y1, x1 = bb
        p = 20
        y0, x0 = max(y0 - p, 0), max(x0 - p, 0)
        sl = (slice(y0, y1 + p), slice(x0, x1 + p))
        ax.imshow(gray[sl], cmap="gray")
        ax.contour(was[sl], [0.5], colors="red", linewidths=1.2)
        nw = new[sl] * was[sl]
        for i in np.unique(nw[nw > 0]):
            ax.contour(nw == i, [0.5], colors="lime", linewidths=0.8)
        ax.set_title(npz.stem.replace("sibs__", ""), fontsize=8)
    fig.suptitle("vessels as saved (red), after the vessel rule (green): each green outline is one vessel", fontsize=11)
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=80)
