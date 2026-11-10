"""Vessel click, old vs new (2026-11-10). Every vessel in the shipped examples is erased and clicked at its centre:
red = the old rule (darkness only: pipeline.vessel_at, else lumen_at), yellow = the new one (pipeline.add_mx ->
vessel_space: space no other cell owns, dark, kept round). Prints how many land within 0.8-1.4x the original vessel.
Run from the repo root: .venv/bin/python docs/development/vessel_click.py docs/development/vessel_click.png"""
import json, sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import pipeline as P, run_batch as RB
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
panels, ratios = [], {"old": [], "new": []}
for ex in ["examples/hallie_SL-2026-004", "examples/amool_root_tip_to_mature"]:
    st = json.loads(Path(ex, "anatomy/settings.json").read_text()); imgdir = P.image_dir(Path(ex, "anatomy"), st)
    keys = {RB.image_key(r): r for r in RB.list_images(imgdir)}
    for f in sorted(glob.glob(f"{ex}/anatomy/masks/*.npz")):
        cells, region, mx, _ = P.load_state(f)
        if mx.max() == 0: continue
        gray = P.load_gray(imgdir / keys[Path(f).stem]); amax = 60 * P.typical_cell_area(cells)
        for v in np.unique(mx[mx > 0]):
            truth = mx == v
            y, x = [int(round(a)) for a in ndi.center_of_mass(truth)]
            if truth.sum() < 30 or not truth[y, x]: continue
            m0 = mx.copy(); m0[truth] = 0
            old = P.vessel_at(gray, y, x, amax)
            old = P.lumen_at(gray, y, x, amax) if old is None else old
            c2, m2 = cells.copy(), m0.copy(); P.add_mx(gray, m2, c2, y, x); new = m2 == m2.max()
            ratios["old"].append(old.sum() / truth.sum()); ratios["new"].append(new.sum() / truth.sum())
            ys, xs = np.nonzero(truth | old | new); pad = 40
            y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
            panels.append((f"{Path(f).stem[-12:]} v{v}", gray[y0:y1, x0:x1], old[y0:y1, x0:x1], new[y0:y1, x0:x1],
                           (y - y0, x - x0), cells[y0:y1, x0:x1]))
for k, r in ratios.items():
    r = np.array(r)
    print(k, "within 0.8-1.4x:", int(((r >= .8) & (r <= 1.4)).sum()), "| over 1.4x:", int((r > 1.4).sum()),
          "| under 0.3x:", int((r < .3).sum()), "of", len(r))
cols = 7; rows = (len(panels) + cols - 1) // cols
fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.7 * rows))
for a in ax.flat: a.axis("off")
for a, (t, g, o, n, (cy, cx), cl) in zip(ax.flat, panels):
    a.imshow(g, cmap="gray"); a.contour(find_boundaries(cl, mode="inner") & (cl > 0), [.5], colors="lime", linewidths=.2)
    a.contour(o, [.5], colors="red", linewidths=1.1); a.contour(n, [.5], colors="yellow", linewidths=1.1)
    a.plot(cx, cy, "m+", ms=8); a.set_title(t, fontsize=7)
fig.suptitle("Vessel click: red = old (darkness only) · yellow = new (space no cell owns, dark, round) · + = click", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=90)
