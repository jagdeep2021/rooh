"""Distance from the stele edge: along the line from the centre (Rooh) vs the shortest distance to the stele outline,
per cortical cell on real sections (2026-11-12). They differ where the line meets the stele at a slant (oval stele).

  .venv/bin/python docs/development/stele_distance.py [project_dir]   (default: examples/hallie_SL-2026-004)
Writes stele_distance.csv beside this script.
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
rows = []
for npz in sorted((proj / "anatomy/masks").glob("*.npz")):
    cells, region, mx, _ = P.load_state(npz)
    pr, _, _ = P.derive(cells, region, mx, 0, P.centre_of(region, P.load_centre(npz)))
    c = pr[pr.tissue == "cortex"]
    near = ndi.distance_transform_edt(~P.main_piece(region == 2))[c.y.round().astype(int), c.x.round().astype(int)]
    diff = c.dist_from_stele_px.values - near
    cell_d = 2 * np.sqrt(c.area_px.median() / np.pi)
    rows.append({"image": npz.stem, "cells": len(c), "median_diff_px": np.median(diff),
                 "p95_abs_diff_px": np.quantile(np.abs(diff), .95), "p95_in_cell_diameters": np.quantile(np.abs(diff), .95) / cell_d})
t = pd.DataFrame(rows)
t.to_csv(HERE / "stele_distance.csv", index=False)
print(t.round(2).to_string(index=False))
