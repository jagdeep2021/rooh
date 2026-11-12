"""Are a project's results right? Recompute them from the masks without Rooh's code and compare (2026-11-12).

check_numbers.py proves the rules on synthetic sections with known answers; this proves that Make results reports
what a real project's masks hold: every pixel count, scale, angle, sector, vessel and cell row, by plain numpy on the
.npz files (pipeline.py is not imported). Rules that are algorithms (tissue tagging, ccfn, cell files, lacunae) are
proven in check_numbers.py; here only how their outputs are aggregated is checked.

  .venv/bin/python tests/check_results.py <project_dir>      (run Make results first, with --full for the lacunae;
                                                              reads the newest results)
Writes check_results.xlsx into the project's results folder and prints failures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ETA = 1.002e-9          # water at 20 C, MPa s
proj = Path(sys.argv[1])
anat, res = proj / "anatomy", proj / "results"
traits_f = sorted(res.glob("traits_*.xlsx"))[-1]
day = traits_f.stem.split("_", 1)[1]
img = pd.read_excel(traits_f, "Per image", dtype={"image_file": str})
qrt = pd.read_excel(traits_f, "Per quarter", dtype={"image_file": str})
cells_csv = pd.read_csv(res / f"all_cells_{day}.csv", dtype={"image_file": str})
cortex_csv = pd.read_csv(res / f"cortex_cells_{day}.csv", dtype={"image_file": str})
samples = pd.read_excel(anat / "samples.xlsx", dtype={"image_file": str}).set_index("image_file")
chk = pd.read_excel(traits_f, "Check", dtype={"image_file": str}).set_index("image_file")
prof = pd.read_excel(sorted(res.glob("cortex_pattern_*.xlsx"))[-1], "Profile", dtype={"image_file": str, "bin": int})
full = sorted((res / "full").glob("anatomy_summary_*.xlsx")) if (res / "full").exists() else []
lac = pd.read_excel(full[-1], "lacunae", dtype={"image_file": str}) if full else pd.DataFrame(columns=["image_file"])
rows = []


def check(image, name, reported, expected, tol=1e-6, rel=True):
    r, e = (np.nan if pd.isna(reported) else float(reported)), (np.nan if pd.isna(expected) else float(expected))
    if np.isnan(r) and np.isnan(e):
        ok = True
    elif np.isnan(r) or np.isnan(e):
        ok = False
    else:
        ok = abs(r - e) <= tol * (max(abs(e), 1e-30) if rel else 1)
    rows.append({"image": image, "check": name, "reported": r, "recomputed": e, "tolerance": tol, "result": "PASS" if ok else "FAIL"})


def in_arcs(deg, arcs):
    deg = np.mod(deg, 360)
    out = np.zeros(np.shape(deg), bool)
    for a, b in arcs:
        a, b = a % 360, b % 360
        out |= ((deg >= a) & (deg < b)) if a <= b else ((deg >= a) | (deg < b))
    return out


def edge_radius(mask, cy, cx, arcs_deg_bins):
    """Outer radius of mask in each 1-degree bin (farthest pixel), the bins as degree starts."""
    yy, xx = np.nonzero(mask)
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360
    r = np.hypot(yy - cy, xx - cx)
    b = np.floor(ang).astype(int)
    out = np.zeros(360)
    np.maximum.at(out, b, r)
    return out[arcs_deg_bins]


for _, row in img.iterrows():
    rel = row.image_file
    stem = rel.rsplit(".", 1)[0].replace("/", "__")
    npz = anat / "masks" / f"{stem}.npz"
    z = np.load(npz, allow_pickle=False)
    cells, region, mx = z["cells"], z["region"], z["mx"]
    arcs = [(float(a), float(b)) for a, b in zip(z["excl_start"], z["excl_end"])] if "excl_start" in z else []
    s = samples.at[rel, "um_per_px"] if rel in samples.index else np.nan
    s = float(s) if pd.notna(s) else np.nan
    um2 = s * s
    from scipy import ndimage as ndi

    def main(m):
        lab, n = ndi.label(m)
        return m if n <= 1 else lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    root, stele = region >= 1, main(region == 2)       # the stele is one piece (flecks go back to root)
    root_main = main(root)
    # centre: set by hand, else the stele centroid (largest stele piece), else the root centroid
    if "centre" in z:
        cy, cx = map(float, z["centre"])
    else:
        cy, cx = ndi.center_of_mass(stele if stele.any() else root)
    check(rel, "status", 1.0, float(row.status == str(z["status"])))
    check(rel, "um_per_px", row.um_per_px, s)
    grid = np.arange(0, 360, 0.01) + 0.005
    kept = 360 - in_arcs(grid, arcs).mean() * 360
    check(rel, "angle_kept_deg", row.angle_kept_deg, kept, tol=0.3, rel=False)
    check(rel, "root_area_um2", row.root_area_um2, root.sum() * um2)
    check(rel, "stele_area_um2", row.stele_area_um2, stele.sum() * um2 if stele.any() else np.nan)
    check(rel, "stele_diameter_um", row.stele_diameter_um, 2 * np.sqrt(stele.sum() / np.pi) * s if stele.any() else np.nan)
    yy, xx = np.nonzero(root & ~stele)
    keep_px = ~in_arcs(np.degrees(np.arctan2(-(yy - cy), xx - cx)), arcs)
    check(rel, "cortex_area_kept_um2", row.cortex_area_kept_um2, keep_px.sum() * um2, tol=0.002)
    # root area with excluded arcs filled: kept pixels + a circle sector per excluded degree at the radius
    # interpolated between the arc's edges (an independent version of the estimate; 2% tolerance)
    if not arcs:
        est = root.sum()
    elif kept < 180:
        est = np.nan
    else:
        deg = np.arange(360)
        r_rt = edge_radius(root_main, cy, cx, deg)
        out = in_arcs(deg + 0.5, arcs)
        k = np.nonzero(~out)[0]
        r = np.interp(deg, np.r_[k - 360, k, k + 360], np.r_[r_rt[k], r_rt[k], r_rt[k]])
        ryy, rxx = np.nonzero(root)
        est = (~in_arcs(np.degrees(np.arctan2(-(ryy - cy), rxx - cx)), arcs)).sum() + 0.5 * np.sum(r[out] ** 2) * np.radians(1)
    check(rel, "root_area_est_um2", row.root_area_est_um2, est * um2, tol=0.02)
    # derived columns from the reported estimate (arithmetic and definitions)
    E = row.root_area_est_um2
    check(rel, "root_diameter_um", row.root_diameter_um, 2 * np.sqrt(E / np.pi))
    check(rel, "cortex_area_est_um2", row.cortex_area_est_um2, E - row.stele_area_um2)
    check(rel, "stele_share", row.stele_share, row.stele_area_um2 / E)
    check(rel, "cortex_stele_ratio", row.cortex_stele_ratio, (E - row.stele_area_um2) / row.stele_area_um2)
    # root aspect from second moments of the main root piece
    ry, rx = np.nonzero(root_main)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([ry, rx]))))
    check(rel, "root_aspect", row.root_aspect, np.sqrt(ev[1] / ev[0]), tol=0.01)
    # metaxylem. The vessel rule (2026-11-12) may change the saved vessels: split merged pairs, drop pen strokes, fill
    # holes. So: count = saved + split - dropped (Check sheet), summed area close to the saved pixels (holes, strokes),
    # and every vessel statistic recomputed exactly from the vessel rows Rooh wrote.
    ids, area = np.unique(mx[mx > 0], return_counts=True)
    found = stele.any()
    ck = chk.loc[rel]
    v = cells_csv[(cells_csv.image_file == rel) & (cells_csv.tissue == "metaxylem")]
    va = v.area_px.to_numpy(float)
    check(rel, "mx_n = saved vessels + split - dropped", row.mx_n,
          len(ids) + ck.vessels_split - ck.vessels_dropped if found else np.nan)
    check(rel, "vessel rows = mx_n", len(v), row.mx_n if found else 0)
    check(rel, "vessel area summed vs saved pixels (10%: holes filled, strokes dropped)", va.sum(), area.sum(), tol=0.10)
    check(rel, "mx_area_mean_um2", row.mx_area_mean_um2, va.mean() * um2 if len(va) else np.nan)
    check(rel, "mx_area_median_um2", row.mx_area_median_um2, np.median(va) * um2 if len(va) else np.nan)
    check(rel, "mx_area_max_um2", row.mx_area_max_um2, va.max() * um2 if len(va) else np.nan)
    check(rel, "mx_area_total_um2", row.mx_area_total_um2, va.sum() * um2 if found else np.nan)
    rm = np.sqrt(va / np.pi) * s * 1e-6
    check(rel, "axial_conductance_m4_MPa_s", row.axial_conductance_m4_MPa_s,
          np.pi * np.sum(rm ** 4) / (8 * ETA) if found and np.isfinite(s) else np.nan)
    # cell rows: one per cell label, pixel count, centroid, angle, radius, kept, quarter
    c = cells_csv[(cells_csv.image_file == rel) & (cells_csv.tissue != "metaxylem")].set_index("cell")
    r_rt_all = edge_radius(root_main, cy, cx, np.arange(360))
    r_st_all = edge_radius(stele, cy, cx, np.arange(360)) if stele.any() else np.zeros(360)
    labels, counts = np.unique(cells[cells > 0], return_counts=True)
    check(rel, "cell rows = cell labels", len(c), len(labels))
    cnt = pd.Series(counts, labels)
    check(rel, "cell area_px = pixel count (max diff)", (c.area_px - cnt.reindex(c.index)).abs().max(), 0, tol=0, rel=False)
    check(rel, "cell area_um2 = area_px x scale² (max rel diff)",
          ((c.area_um2 - c.area_px * um2).abs() / (c.area_px * um2)).max() if np.isfinite(s) else 0, 0, tol=1e-9, rel=False)
    ys, xs = np.nonzero(cells)
    lv = cells[ys, xs]
    sy = np.bincount(lv, weights=ys)[labels] / counts
    sx = np.bincount(lv, weights=xs)[labels] / counts
    check(rel, "cell centroid (max px diff)", max((c.y_px - pd.Series(sy, labels)).abs().max(), (c.x_px - pd.Series(sx, labels)).abs().max()), 0, tol=1e-6, rel=False)
    check(rel, "centre (px diff)", np.hypot(c.centre_y_px.iloc[0] - cy, c.centre_x_px.iloc[0] - cx), 0, tol=1e-6, rel=False)
    ang = np.arctan2(-(c.y_px - cy), c.x_px - cx) % (2 * np.pi)
    check(rel, "cell angle (max rad diff)", (np.abs(np.angle(np.exp(1j * (c.angle_rad - ang))))).max(), 0, tol=1e-9, rel=False)
    rr = np.hypot(c.y_px - cy, c.x_px - cx) * s
    check(rel, "cell r_um (max µm diff)", (c.r_um - rr).abs().max() if np.isfinite(s) else 0, 0, tol=1e-6, rel=False)
    pat = c.tissue.str.replace("_excluded", "").isin(["cortex", "epidermis", "air_space", "unassigned"])
    want_kept = np.where(pat, np.where(in_arcs(np.degrees(ang), arcs), "no", "yes"), "yes")
    check(rel, "cell kept flag (wrong rows)", (c.kept.values != want_kept).sum(), 0, tol=0, rel=False)
    q = "Q" + (np.floor(np.degrees(ang) / 90).astype(int) % 4 + 1).astype(str)
    check(rel, "cell quarter (wrong rows)", (c.quarter.values != q.values).sum(), 0, tol=0, rel=False)
    # cortex_cells.csv = the kept cortex rows of all_cells
    cc = cortex_csv[cortex_csv.image_file == rel]
    check(rel, "cortex_cells rows = kept cortex cells", len(cc), ((c.tissue == "cortex") & (c.kept == "yes")).sum())
    # radial position: r_norm and dist_from_stele from the edges at the cell's own degree (farthest pixel; Rooh casts
    # a ray, so a dent can differ: compared at the 95th percentile)
    if stele.any():
        pc = c[c.tissue.isin(["cortex", "epidermis"]) & (c.kept == "yes")]
        d = np.floor(np.degrees(np.arctan2(-(pc.y_px - cy), pc.x_px - cx)) % 360).astype(int).values
        rr0 = np.hypot(pc.y_px - cy, pc.x_px - cx).values
        rs, rt_ = r_st_all[d], r_rt_all[d]
        rn = (rr0 - rs) / (rt_ - rs)
        check(rel, "r_norm: 95th pct |diff| <= 0.03", float(np.quantile(np.abs(pc.r_norm.values - rn), .95)), 0, tol=0.03, rel=False)
        check(rel, "dist_from_stele_um: 95th pct |diff| <= 2 px", float(np.quantile(np.abs(pc.dist_from_stele_um.values - (rr0 - rs) * s), .95)),
              0, tol=2 * s, rel=False)
        # the binned pattern = median area of the kept cortex + epidermis cells in each r_norm bin, per quarter
        for qn, g in pc.groupby("quarter"):
            pr = prof[(prof.image_file == rel) & (prof.quarter == qn)].set_index("bin")
            if not len(pr):
                continue
            b = (np.clip(g.r_norm, 0, 0.999) * 10).astype(int) + 1
            med = g.area_um2.groupby(b).median()
            diff = max((abs(pr.median_area_um2.get(k, np.nan) - v) for k, v in med.items()), default=0)
            check(rel, f"{qn} pattern: median area per r_norm bin (max diff µm²)", diff, 0, tol=1e-6, rel=False)
        # lacunae: area, angle, quarter, radial position of each
        for _, L in lac[lac.image_file == rel].iterrows():
            a = np.degrees(np.arctan2(-(L.y - cy), L.x - cx)) % 360
            dd = int(np.floor(a))
            check(rel, f"lacuna {L.lacuna}: angle", L.angle_deg, a, tol=1e-6, rel=False)
            check(rel, f"lacuna {L.lacuna}: quarter", L.quarter, int(a // 90) + 1)
            check(rel, f"lacuna {L.lacuna}: area_um2 = px x scale²", L.area_um2, L.area_px * um2)
            check(rel, f"lacuna {L.lacuna}: inner <= r_norm <= outer", float(L.r_norm_inner <= L.r_norm <= L.r_norm_outer), 1.0)
            rn_c = (np.hypot(L.y - cy, L.x - cx) - r_st_all[dd]) / (r_rt_all[dd] - r_st_all[dd])
            check(rel, f"lacuna {L.lacuna}: centre's r_norm within the lacuna's span", float(L.r_norm_inner - .03 <= rn_c <= L.r_norm_outer + .03), 1.0)
    # per quarter: kept degrees, width, median cell area; image means over used quarters
    used = []
    deg = np.arange(360)
    r_rt, r_st = edge_radius(root_main, cy, cx, deg), edge_radius(stele, cy, cx, deg) if stele.any() else np.zeros(360)
    for qi in range(4):
        qr = qrt[(qrt.image_file == rel) & (qrt.quarter == f"Q{qi + 1}")].iloc[0]
        qg = np.arange(qi * 90, qi * 90 + 90, 0.01) + 0.005
        qk = 90 - in_arcs(qg, arcs).mean() * 90
        check(rel, f"Q{qi + 1} kept_deg", qr.kept_deg, qk, tol=0.3, rel=False)
        qd = deg[(deg >= qi * 90) & (deg < qi * 90 + 90)]
        kd = qd[~in_arcs(qd + 0.5, arcs)]
        w = (r_rt[kd] - r_st[kd]).mean() * s if qk >= 45 and stele.any() else np.nan
        check(rel, f"Q{qi + 1} cortex_width_um (2%)", qr.cortex_width_um, w, tol=0.02)
        cq = c[(c.tissue == "cortex") & (c.kept == "yes") & (c.quarter == f"Q{qi + 1}")]
        check(rel, f"Q{qi + 1} median_cortex_cell_area_um2", qr.median_cortex_cell_area_um2,
              cq.area_um2.median() if qk >= 45 and len(cq) else np.nan)
        if qk >= 45:
            used.append(qr)
    u = pd.DataFrame(used)
    for k in ("ccfn", "cortex_width_um", "median_cortex_cell_area_um2", "aerenchyma_share"):
        check(rel, f"image {k} = mean of used quarters", row[k], u[k].mean() if len(u) and u[k].notna().any() else np.nan)

t = pd.DataFrame(rows)
t.to_excel(res / "check_results.xlsx", index=False)
bad = t[t.result == "FAIL"]
pd.set_option("display.width", 250)
print(bad.to_string(index=False) if len(bad) else "")
print(f"\n{(t.result == 'PASS').sum()} pass, {len(bad)} fail  ({t.image.nunique()} images)  ->  {res / 'check_results.xlsx'}")
print(bad.groupby("check").size().to_string() if len(bad) else "")
