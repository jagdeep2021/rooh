"""How big is a cell clicked in by hand, next to Cellpose's outline of the same cell? (2026-11-12)

For 25 random cortical cells per image: delete the Cellpose cell, click it back in (pipeline.add_cell, the editors'
rule), compare the areas. Before CLICK_GROW_PX the click filled the lumen to the walls (+2 px): 0.79 x the Cellpose
area on Hallie's images. Here every growth 0-8 px is tried on the lumen fill, to choose the one that matches.

  .venv/bin/python docs/development/click_area.py [project_dir]     (default: examples/hallie_SL-2026-004)
Writes click_area.csv beside this script (one row per cell, ratio and IoU for each growth).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import pipeline as P  # noqa: E402

proj = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "examples/hallie_SL-2026-004"
KS = (0, 1, 2, 3, 4, 5, 6, 8)
P.CLICK_GROW_PX = 0                    # the lumen fill alone; the growths are tried below
rows = []
for npz in sorted((proj / "anatomy/masks").glob("*.npz")):
    gray = P.load_gray(proj / "raw" / str(np.load(npz)["image_file"]))
    cells, region, mx, _ = P.load_state(npz)
    props, _, _ = P.derive(cells, region, mx, 0, P.centre_of(region, P.load_centre(npz)))
    for _, c in props[props.tissue == "cortex"].sample(25, random_state=1).iterrows():
        l, y, x = int(c.label), int(round(c.y)), int(round(c.x))
        if cells[y, x] != l:
            continue
        c2 = cells.copy()
        c2[c2 == l] = 0
        new = P.add_cell(gray, c2, y, x)
        if not new:
            continue
        m, truth = c2 == new, cells == l
        dist = ndi.distance_transform_edt(~m)
        r = {"image": npz.stem, "cellpose_px": int(truth.sum())}
        for k in KS:
            g = m | ((c2 == 0) & (dist <= k))
            r[f"ratio_{k}"] = g.sum() / truth.sum()
            r[f"iou_{k}"] = (g & truth).sum() / (g | truth).sum()
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(HERE / "click_area.csv", index=False)
print(pd.DataFrame({k: [t[f"ratio_{k}"].median(), t[f"iou_{k}"].median()] for k in KS},
                   index=["area / Cellpose (median)", "IoU (median)"]).round(3).to_string())
print("per image, area ratio at 5 px:", t.groupby("image").ratio_5.median().round(2).to_dict())
