"""Which cell files go all the way round? Evidence for 'Cells per file' (2026-11-11).

For every image of a project: the share of kept rays (every 0.5 degrees) that reach file k, the cells numbered k, the
share of them tagged epidermis, and the file count the rule keeps (P.complete_files: the FILE_COMPLETE_PCT percentile
of the files per ray). Outer files that only some rays reach mix layers: on thin sides they hold the epidermis.

  .venv/bin/python docs/development/file_rings.py [project_dir]     (default: examples/hallie_SL-2026-004)
Writes file_rings.csv beside this script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import pipeline as P  # noqa: E402

proj = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "examples/hallie_SL-2026-004"
rows = []
for npz in sorted((proj / "anatomy/masks").glob("*.npz")):
    cells, region, mx = P.load_state(npz)[:3]
    centre = P.centre_of(region, P.load_centre(npz))
    props, _, _ = P.derive(cells, region, mx, 0, centre)
    arcs = P.load_sectors(npz, 0)
    props = P.exclude_sectors(props, arcs)
    fl = P.cell_files(cells, props, centre, arcs)
    ray = np.array(fl.attrs["ray_files"])
    p = props.join(fl, on="label")
    p = p[p.tissue.isin(P.PATTERN_TISSUES)].dropna(subset=["file"])
    k = np.floor(p.file + 0.5).astype(int)
    keep = P.complete_files(ray)
    for f in range(1, int(k.max()) + 1):
        g = p[k == f]
        rows.append({"image": npz.stem, "file": f, "rays_reaching": round(float((ray >= f).mean()), 3),
                     "cells": len(g), "epidermis_share": round(float((g.tissue == "epidermis").mean()), 3) if len(g) else np.nan,
                     "counted": f <= keep})
t = pd.DataFrame(rows)
t.to_csv(HERE / "file_rings.csv", index=False)
print(t.pivot(index="image", columns="file", values="epidermis_share").iloc[:, :12].round(2).to_string())
print(t.groupby("counted")[["rays_reaching", "epidermis_share"]].median())
