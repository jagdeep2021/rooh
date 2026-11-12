"""Stele and metaxylem on reviewed masks (2026-11-12). Per image, the reviewed masks taken as the reference:

  stele: the automatic rule (find_stele) run on the reviewed cells and root, vs the reviewed stele (area ratio, IoU);
         cells the reviewed outline cuts (20-80% of the cell inside); how much the stele area moves if the outline
         moves one boundary-cell layer (perimeter x median width of the cells just outside / area).
  mx:    the automatic rule (find_mx + propose's big-round-cell rule) vs the reviewed vessels (matched by overlap);
         reviewed vessels with solidity < 0.85 (two merged?), outside the stele, pixels both vessel and cell;
         stele cells that pass the vessel shape rule but were left as cells.
  conductance: share from the largest vessel; change if every vessel radius shifts by +-1 px.

  .venv/bin/python docs/development/stele_mx.py <project_dir> <out.csv> [figure.png]
Run on a copy of Hallie's reviewed web project "test" (38 images): numbers in the 2026-11-12 stele/metaxylem entry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pipeline as P  # noqa: E402

proj = Path(sys.argv[1])
anat = proj / "anatomy"
imdir = P.image_dir(anat, json.loads((anat / "settings.json").read_text()))


def auto_mx(gray, cells, stele):
    """propose()'s vessel rules on given cells and stele."""
    mx = P.find_mx(gray, cells, stele)
    st = pd.DataFrame(measure.regionprops_table(cells * stele, properties=("label", "area", "solidity", "eccentricity")))
    if len(st) > 5:
        big = st[(st.area > 5 * st.area.median()) & (st.solidity > 0.85) & (st.eccentricity < 0.85)]
        for lbl in big.label:
            m = (cells == lbl) & (mx == 0)
            if m.any():
                mx[m] = mx.max() + 1
    return measure.label(morphology.closing(mx > 0, morphology.disk(2))).astype(np.uint16)


def match(a, b):
    """Vessels of a that overlap a vessel of b by >= 50% of the smaller."""
    n = 0
    for g in measure.regionprops(a):
        hit = b[a == g.label]
        hit = hit[hit > 0]
        if len(hit):
            lb = np.bincount(hit).argmax()
            if np.bincount(hit).max() >= 0.5 * min(g.area, (b == lb).sum()):
                n += 1
    return n


rows, crops = [], []
for npz in sorted((anat / "masks").glob("*.npz")):
    cells, region, mx, status = P.load_state(npz)
    rel = str(np.load(npz)["image_file"])
    gray = P.load_gray(imdir / rel)
    root, stele = region >= 1, region == 2
    s_um = P.um_per_px(anat, rel)
    # stele: auto vs reviewed
    a_st = P.find_stele(cells, root)
    inter, union = (a_st & stele).sum(), (a_st | stele).sum()
    # cells cut by the outline, and one boundary layer
    lab = cells[cells > 0]
    tot = np.bincount(lab)
    ins = np.bincount(cells[stele & (cells > 0)], minlength=len(tot))
    frac = np.where(tot > 0, ins / np.maximum(tot, 1), np.nan)
    touch = set(np.unique(cells[morphology.dilation(stele, morphology.disk(2)) & ~morphology.erosion(stele, morphology.disk(2))])) - {0}
    cut = [c for c in touch if 0.2 <= frac[c] <= 0.8]
    outside = set(np.unique(cells[morphology.dilation(stele, morphology.disk(6)) & ~stele])) - {0}
    out_p = pd.DataFrame(measure.regionprops_table(cells, properties=("label", "axis_minor_length")))
    width = out_p[out_p.label.isin(outside) & (frac[out_p.label] < 0.2)].axis_minor_length.median()
    perim = measure.perimeter(stele)
    # metaxylem
    a_mx = auto_mx(gray, cells.copy(), stele)
    mp = pd.DataFrame(measure.regionprops_table(mx, properties=("label", "area", "solidity", "eccentricity", "centroid")))
    if len(mp):
        yi, xi = mp["centroid-0"].round().astype(int), mp["centroid-1"].round().astype(int)
        mp["in_stele"] = stele[yi, xi]
    sc = pd.DataFrame(measure.regionprops_table(cells * stele, properties=("label", "area", "solidity", "eccentricity")))
    med_st = sc.area.median() if len(sc) else np.nan
    left = sc[(sc.area > 5 * med_st) & (sc.solidity > 0.85) & (sc.eccentricity < 0.85)] if len(sc) > 5 else sc.iloc[:0]
    a = mp.area.to_numpy(float) if len(mp) else np.array([])
    r = np.sqrt(a / np.pi)
    k = lambda rr: np.sum(rr ** 4)
    rows.append({
        "image": npz.stem.replace("sibs__", ""), "status": status, "um_per_px": s_um,
        "stele_area_px": int(stele.sum()), "auto_stele_ratio": a_st.sum() / max(stele.sum(), 1),
        "auto_stele_iou": inter / max(union, 1),
        "n_cells_cut": len(cut), "n_cells_on_outline": len(touch), "boundary_cell_width_px": width,
        "one_layer_share": perim * width / max(stele.sum(), 1),
        "mx_n": len(mp), "auto_mx_n": int(a_mx.max()), "mx_found_by_auto": match(mx, a_mx) if len(mp) else 0,
        "auto_mx_extra": int(a_mx.max()) - match(a_mx, mx),
        "mx_median_r_px": float(np.median(r)) if len(r) else np.nan,
        "mx_low_solidity": int((mp.solidity < 0.85).sum()) if len(mp) else 0,
        "mx_outside_stele": int((~mp.in_stele).sum()) if len(mp) else 0,
        "mx_cell_overlap_px": int(((mx > 0) & (cells > 0)).sum()),
        "vessel_like_cells_left": len(left),
        "K_largest_share": r.max() ** 4 / k(r) if len(r) else np.nan,
        "K_change_1px": (k(r + 1) / k(r) - 1) if len(r) else np.nan,
    })
    if len(sys.argv) > 3 and stele.any():
        ys, xs = np.nonzero(stele)
        pad = 25
        y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
        crops.append((rows[-1]["image"], gray[y0:y1, x0:x1], stele[y0:y1, x0:x1], a_st[y0:y1, x0:x1],
                      mx[y0:y1, x0:x1], a_mx[y0:y1, x0:x1], cells[y0:y1, x0:x1], left.label.tolist()))

t = pd.DataFrame(rows)
t.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250)
print(t.drop(columns=["status"]).round(2).to_string(index=False))
print("\nmedians:\n", t.drop(columns=["image", "status"]).median().round(3).to_string())

if crops:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(crops)
    cols = 7
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 3, np.ceil(n / cols) * 3))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (name, g, st, ast, m, am, c, left) in zip(axs.ravel(), crops):
        ax.imshow(g, cmap="gray")
        ax.contour(st, [0.5], colors="cyan", linewidths=1)
        ax.contour(ast, [0.5], colors="yellow", linewidths=0.8, linestyles="dashed")
        ax.contour(m > 0, [0.5], colors="red", linewidths=0.8)
        ax.contour(am > 0, [0.5], colors="lime", linewidths=0.6, linestyles="dotted")
        if left:
            ax.contour(np.isin(c, left), [0.5], colors="magenta", linewidths=0.8)
        ax.set_title(name, fontsize=7)
    fig.suptitle("stele: reviewed (cyan), auto rule (yellow dashed)   vessels: reviewed (red), auto rule (green dotted), "
                 "vessel-shaped cells left as cells (magenta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=110)
