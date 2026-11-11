"""Score a Rooh project of the synthetic sections (make_test_sections.py) against the truth: the auto pass, or the
same project after a person's review.

  .venv/bin/python tests/score_against_truth.py [sections_dir] [anatomy_dir]
      defaults: tests/output/sections and its anatomy/ (the auto pass from run_batch.py)

Cells: each true cortical cell is matched to the Rooh cell that covers most of it (IoU = overlap / union).
  found   IoU >= 0.5          merged   one Rooh cell covers >= 2 true cells (each >= 50% inside it)
  split   >= 2 Rooh cells share one true cell (each >= 25% of it)       missed   nothing covers >= 50% of it
Vessels: a true vessel is found if a Rooh vessel covers >= 50% of it; extra = Rooh vessels on no true vessel.
Numbers: summarise.main on the project (as Make results), next to truth.xlsx.
Writes <anatomy_dir>/../score_<anatomy_dir name>.xlsx and prints the per-image table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P      # noqa: E402
import summarise as SUM   # noqa: E402


def cell_score(tc, tl, rc):
    """tc truth cells, tl {label: layer} cortex labels, rc Rooh cells."""
    cort = [l for l in tl if (tc == l).any()]
    m = np.isin(tc, cort) | (rc > 0)
    key = tc[m].astype(np.int64) * 1_000_000 + rc[m]
    k, n = np.unique(key, return_counts=True)
    ov = pd.DataFrame({"t": k // 1_000_000, "r": k % 1_000_000, "n": n})
    ta = ov.groupby("t").n.sum() if False else pd.Series(np.bincount(tc.ravel()))
    ra = pd.Series(np.bincount(rc.ravel()))
    ov = ov[(ov.t > 0) & (ov.r > 0) & ov.t.isin(cort)]
    ov["frac_t"] = ov.n / ov.t.map(ta)
    ov["frac_r"] = ov.n / ov.r.map(ra)
    ov["iou"] = ov.n / (ov.t.map(ta) + ov.r.map(ra) - ov.n)
    best = ov.sort_values("iou").groupby("t").tail(1).set_index("t")
    found = best.index[best.iou >= 0.5]
    big = ov[ov.frac_t >= 0.5]
    merged_r = big.groupby("r").t.nunique()
    merged = big[big.r.isin(merged_r.index[merged_r >= 2])].t.unique()
    split_t = ov[ov.frac_t >= 0.25].groupby("t").r.nunique()
    split = split_t.index[split_t >= 2]
    covered = ov.groupby("t").frac_t.max()
    missed = [t for t in cort if covered.get(t, 0) < 0.5]
    return {"true_cortex_cells": len(cort), "found": len(found), "found_pct": 100 * len(found) / max(len(cort), 1),
            "merged": len(merged), "split": len(split), "missed": len(missed)}


def vessel_score(tm, rm):
    found = sum(1 for v in range(1, tm.max() + 1) if (rm[tm == v] > 0).mean() >= 0.5)
    extra = sum(1 for v in np.unique(rm) if v and not (tm[rm == v] > 0).any())
    return {"true_vessels": int(tm.max()), "vessels_found": found, "vessels_extra": extra}


def main(sec: Path, anat: Path):
    truth = pd.read_excel(sec / "truth/truth.xlsx").set_index("image")
    s = pd.read_excel(anat / "samples.xlsx", dtype={"image_file": str})
    s["um_per_px"] = s.image_file.map(truth.um_per_px)
    s.to_excel(anat / "samples.xlsx", index=False)
    SUM.main(anat)
    res = anat.parent / "results"
    got = pd.read_excel(sorted(res.rglob("anatomy_summary_*.xlsx"))[-1], sheet_name="per_image").set_index("image_file")
    rows = []
    for img in truth.index:
        t = np.load(sec / "truth/masks" / img.replace(".tif", ".npz"))
        f = anat / "masks" / img.replace(".tif", ".npz")
        rc, rr, rm, status = P.load_state(f)
        tl = dict(map(tuple, t["layer"]))
        g = got.loc[img] if img in got.index else pd.Series(dtype=float)
        row = {"image": img, "what": truth.at[img, "what"], "status": status,
               **cell_score(t["cells"], tl, rc), **vessel_score(t["mx"], rm)}
        for c in ("n_cortex_cells", "n_cortical_layers", "mx_n",
                  "stele_area_um2", "mean_cortex_cell_area_um2"):
            row[f"{c} truth"] = truth.at[img, c]
            row[f"{c} rooh"] = g.get(c, np.nan)
        rows.append(row)
    tab = pd.DataFrame(rows)
    out = anat.parent / f"score_{anat.name}.xlsx"
    tab.to_excel(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["image", "found_pct", "merged", "split", "missed", "vessels_found", "true_vessels", "vessels_extra",
            "n_cortex_cells truth", "n_cortex_cells rooh", "n_cortical_layers truth", "n_cortical_layers rooh",
            "stele_area_um2 truth", "stele_area_um2 rooh"]
    print(tab[show].round(1).to_string(index=False))
    print(out)


if __name__ == "__main__":
    here = Path(__file__).parent / "output/sections"
    sec = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    main(sec, Path(sys.argv[2]) if len(sys.argv) > 2 else sec / "anatomy")
