"""Do the numbers come out right? Synthetic cross-sections whose answers are known, run through the same code as
Make results (summarise.main), then edited with the editors' own rules (pipeline.click_edit / outline_edit /
sector_edit / save_edit) and summarised again.

  .venv/bin/python tests/check_numbers.py [out_dir]      -> table of checks (PASS / FAIL), exit 1 on any FAIL

The section: centre (500, 500), stele r < 150 (small cells, 6 metaxylem vessels r = 15 at r = 105), cortex
8 layers x 36 cells (10 degrees each, widths below), epidermis 72 cells to r = 475. Walls drawn in a grey image so
click-to-add works as on a real picture. Expected values are geometry (pi r^2 ...), not what the code gave.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import segmentation

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import pipeline as P      # noqa: E402
import summarise as SUM   # noqa: E402

N = 1000
C = (500.0, 500.0)
R_STELE, R_ROOT = 150, 475
WIDTHS = [20, 30, 40, 50, 50, 45, 40, 30]                  # cortex layers, stele -> out (sum 305)
EDGES = np.cumsum([R_STELE] + WIDTHS)                       # 150 ... 455
R_EPI = R_ROOT
SECT = 36                                                    # cortex cells per layer
MX_R, MX_AT, MX_ANG = 15, 105, [15, 75, 135, 195, 255, 315]
UM = 0.5

yy, xx = np.mgrid[:N, :N]
RR = np.hypot(yy - C[0], xx - C[1])
TH = np.mod(np.degrees(np.arctan2(-(yy - C[0]), xx - C[1])), 360)


def at(r, deg):
    """pixel (y, x) at radius r, angle deg (counter-clockwise from the right)."""
    t = np.radians(deg)
    return int(round(C[0] - r * np.sin(t))), int(round(C[1] + r * np.cos(t)))


def cortex_label(layer, sector):            # layer 1..8, sector 0..35 (sector k spans 10k .. 10k+10 deg)
    return 1000 + 100 * layer + sector


def section():
    cells = np.zeros((N, N), np.int32)
    # stele: centre disc + rings of small cells
    cells[RR < 50] = 1
    for i, (r0, r1, n) in enumerate([(50, 100, 12), (100, 150, 24)]):
        m = (RR >= r0) & (RR < r1)
        cells[m] = 10 + 100 * i + (TH[m] // (360 / n)).astype(int)
    for k in range(8):
        m = (RR >= EDGES[k]) & (RR < EDGES[k + 1])
        cells[m] = cortex_label(k + 1, 0) + (TH[m] // (360 / SECT)).astype(int)
    m = (RR >= EDGES[-1]) & (RR < R_EPI)
    cells[m] = 5000 + (TH[m] // 5).astype(int)
    mx = np.zeros((N, N), np.int32)
    for i, a in enumerate(MX_ANG):
        y, x = at(MX_AT, a)
        mx[(yy - y) ** 2 + (xx - x) ** 2 <= MX_R ** 2] = i + 1
    cells[mx > 0] = 0
    region = np.where(RR < R_ROOT, 1, 0).astype(np.uint8)
    region[RR < R_STELE] = 2
    # grey picture: dark lumens, bright 2-px walls, dark vessels, black outside
    walls = segmentation.find_boundaries(cells + (mx > 0) * 9000, mode="thick")
    gray = np.where(region > 0, 0.25, 0.0)
    gray[walls & (region > 0)] = 1.0
    gray[mx > 0] = 0.05
    cells, _, _ = segmentation.relabel_sequential(cells)    # ids as Cellpose gives them (keep a map for tests)
    return cells, region, mx, gray


def ann_area(r0, r1, n=1):
    return np.pi * (r1 ** 2 - r0 ** 2) / n


# ------------------------------------------------------------------ expected, from geometry
CELL_AREA = [ann_area(EDGES[k], EDGES[k + 1], SECT) for k in range(8)]
EXPECT = {
    "root_area_px": np.pi * R_ROOT ** 2, "stele_area_px": np.pi * R_STELE ** 2,
    "cortex_area_est_px": np.pi * (R_ROOT ** 2 - R_STELE ** 2), "cortex_width_px": R_ROOT - R_STELE,
    "n_cortex_cells": 8 * SECT, "mean_cortex_cell_area_px": float(np.mean(CELL_AREA)),
    "n_air_spaces": 0, "mx_n": 6, "mx_mean_area_px": np.pi * MX_R ** 2,
    "n_cortical_layers": 8, "angle_kept_deg": 360, "low_coverage": 0,
    "cortex_area_kept_px": np.pi * (R_ROOT ** 2 - R_STELE ** 2), "root_area_est_px": np.pi * R_ROOT ** 2,
    "n_cortex_cells_est": 8 * SECT,
}
MED_CORTEX = float(np.median(np.repeat(CELL_AREA, SECT)))
EPI_AREA0 = np.pi * (475 ** 2 - 455 ** 2) / 72
MED_TISSUE = float(np.median(np.r_[np.repeat(CELL_AREA, SECT), np.repeat(EPI_AREA0, 72)]))   # the reference for rel
EPI_AREA = ann_area(EDGES[-1], R_EPI, 72)
TOL = {"root_area_px": .005, "root_diameter_eq_px": .005, "stele_area_px": .01, "cortex_area_px": .005, "cortex_width_px": .01,
       "mean_cortex_cell_area_px": .02, "mx_mean_area_px": .04, "cortex_area_kept_px": .005,
       "cortex_area_est_px": .005, "root_area_est_px": .005, "stele_share_est": .01, "epidermis_rel": .03,
       "lacuna_area_px": .03, "lacuna_r_norm_mean": .03}


def write_project(d: Path, images: dict):
    """images: name -> (cells, region, mx, gray, extra npz fields). A Rooh project as run_batch leaves it."""
    (d / "anatomy/masks").mkdir(parents=True)
    (d / "raw").mkdir()
    (d / "anatomy/settings.json").write_text(json.dumps({"image_dir": "../raw", "diameter": None,
                                                        "quadrant_angle_deg": 0}))
    rows = []
    for name, (cells, region, mx, gray, extra) in images.items():
        P.save_state(d / f"anatomy/masks/{name}.npz", cells, region, mx, image_file=name + ".tif", **extra)
        rows.append({"image_file": name + ".tif", "plant": "P1", "um_per_px": UM, "include": "yes"})
    pd.DataFrame(rows).to_excel(d / "anatomy/samples.xlsx", index=False)


def results(d: Path) -> dict:
    SUM.main(d / "anatomy")
    out = d / "results" / "old_format"         # the full record (old format, kept for one release)
    img = pd.read_excel(next(out.glob("anatomy_summary_*.xlsx")), sheet_name="per_image")
    prof_all = pd.read_excel(next(out.glob("cortex_pattern_full_*.xlsx")), sheet_name="per_image_profile", dtype={"bin": str})
    prof = prof_all[(prof_all.basis == "position") & (prof_all.scope == "section")]
    plant = pd.read_excel(next(out.glob("anatomy_summary_*.xlsx")), sheet_name="per_plant")
    cells = pd.read_excel(next(out.glob("all_cells_*.xlsx")), sheet_name="all_cells")
    img["quarters_used"] = img.quarters_used.astype(str)
    feats = pd.read_excel(next(out.glob("cortex_pattern_full_*.xlsx")), sheet_name="profile_features")
    xl = next(out.glob("anatomy_summary_*.xlsx"))
    lac = pd.read_excel(xl, sheet_name="lacunae")
    aer = pd.read_excel(next(out.glob("cortex_pattern_full_*.xlsx")), sheet_name="aerenchyma_profile")
    return {"img": img.set_index("image_file"), "prof": prof, "prof_all": prof_all, "plant": plant, "cells": cells,
            "feats": feats, "lac": lac, "aer": aer}


CHECKS: list[dict] = []


def check(case, what, got, want, tol=0.0, note=""):
    if isinstance(want, float) and np.isnan(want):
        ok = pd.isna(got)
    elif tol:
        ok = abs(got - want) <= tol * abs(want)
    else:
        ok = got == want
    CHECKS.append({"case": case, "check": what, "expected": want if not isinstance(want, float) else round(want, 2),
                   "got": got if not isinstance(got, float) else round(got, 2), "ok": "PASS" if ok else "FAIL",
                   "note": note})


def lab(cells, layer, deg):
    """the cell id at the middle of a cortex layer, at angle deg."""
    return int(cells[at((EDGES[layer - 1] + EDGES[layer]) / 2, deg)])


def run(out_dir: Path):
    cells, region, mx, gray = section()
    base = d = out_dir / "00_base"
    write_project(d, {"S": (cells, region, mx, gray, {}),
                      "S_rot": (np.rot90(cells).copy(), np.rot90(region).copy(), np.rot90(mx).copy(), gray, {})})
    r = results(d)
    i = r["img"].loc["S.tif"]
    for k, v in EXPECT.items():
        check("base", k, float(i[k]), float(v), TOL.get(k, 0))
    check("base", "µm²: root_area_um2 = px x 0.5²", float(i.root_area_um2), float(i.root_area_px) * UM ** 2, 1e-9)
    check("base", "µm: cortex_width_um = px x 0.5", float(i.cortex_width_um), float(i.cortex_width_px) * UM, 1e-9)
    check("base", "rotated copy: same cortex cells", int(r["img"].loc["S_rot.tif", "n_cortex_cells"]), 8 * SECT)
    check("base", "plant mean = mean of its 2 images", float(r["plant"].n_cortex_cells.iloc[0]),
          float(r["img"].n_cortex_cells.mean()), 1e-9)
    f = r["prof"][r["prof"].image_file == "S.tif"].set_index("bin")
    check("base", "profile: cortex + epidermis cells over the bins", int(f.n_cells.sum()), 8 * SECT + 72)
    check("base", "profile: epidermis is the outermost bin (10), alone", int(f.n_cells["10"]), 72)
    check("base", "profile: bin 10 median area = epidermis cell", float(f.median_area_px["10"]), EPI_AREA, .03)
    check("base", "epidermis_rel = epidermis / tissue median", float(i.epidermis_rel), EPI_AREA / MED_TISSUE, .03)
    for k in range(8):                                        # layer k's cells land in the bin of their r_norm
        r_mid = (EDGES[k] + EDGES[k + 1]) / 2
        b = str(int(min((r_mid - R_STELE) / (R_ROOT - R_STELE), 0.999) * SUM.N_BINS) + 1)
        check("base", f"layer {k + 1}: median area in bin {b}", float(f.median_area_px[b]), CELL_AREA[k], .03)

    # cell files (2026-11-03): every cortical cell gets the number of its layer
    ac = r["cells"][(r["cells"].image_file == "S.tif") & (r["cells"].tissue == "cortex")]
    layer_true = np.searchsorted(EDGES, ac.r_px.values, side="right")
    check("base", "file number = layer, every cortical cell", float((np.floor(ac.file + 0.5) == layer_true).mean()), 1.0)
    check("base", "every cortical cell has a file number", int(ac.file.notna().sum()), 8 * SECT)
    ae = r["cells"][(r["cells"].image_file == "S.tif") & (r["cells"].tissue == "epidermis")]
    check("base", "every epidermal cell is file 9", float((np.floor(ae.file + 0.5) == 9).mean()), 1.0)
    pf = r["prof_all"][(r["prof_all"].image_file == "S.tif") & (r["prof_all"].basis == "file")
                       & (r["prof_all"].scope == "section")].set_index("bin")
    check("base", "file pattern: 8 cortex files + the epidermis file", len(pf), 9)
    check("base", "file 9 = the epidermis", float(pf.median_area_px["9"]), EPI_AREA, .03)
    for k in range(8):
        check("base", f"file {k + 1}: median area", float(pf.median_area_px[str(k + 1)]), CELL_AREA[k], .03)
    # quarters: all four used, mean over 4
    fe = r["feats"][r["feats"].image_file == "S.tif"]
    check("base", "quarters used", r["img"].loc["S.tif", "quarters_used"], "1,2,3,4")
    qm = fe[(fe.basis == "position") & (fe.scope == "quarter_mean")]
    check("base", "quarter mean over 4 quarters", int(qm.n_quarters.iloc[0]), 4)
    sec = fe[(fe.basis == "position") & (fe.scope == "section")]
    check("base", "symmetric root: quarter-mean peak = section peak", float(qm.peak_pos_rel.iloc[0]),
          float(sec.peak_pos_rel.iloc[0]), .02)
    # diameter, lacunae
    check("base", "root_diameter_px (kept degrees)", float(i.root_diameter_px), 2 * R_ROOT, .01)
    check("base", "root_diameter_eq_px", float(i.root_diameter_eq_px), 2 * R_ROOT, .005)
    check("base", "no lacunae in an intact root", int(i.n_lacunae), 0)
    pd_ = r["prof_all"][(r["prof_all"].basis == "distance") & (r["prof_all"].scope == "section") &
                        (r["prof_all"].image_file == "S.tif")]
    check("base", f"distance basis: {P.DIST_BIN_UM:g} µm bins out to the epidermis (centre 157.5 µm -> bin 8)",
          len(pd_), 8)

    # missing tissue (2026-11-04, docs/development/loss_robustness.py): straight on the pipeline, no summarise
    def files_after(remove, region_cut=None):
        c, rg, m, _ = section()
        c[remove] = 0
        if region_cut is not None:
            c[region_cut], rg[region_cut] = 0, 0
        ctr = P.centre_of(rg, None)
        pr, _, _ = P.derive(c, rg, m, 0, ctr)
        fl = P.cell_files(c, pr, ctr, region=rg)
        pr = pr.join(fl, on="label")
        pr["true_layer"] = np.searchsorted(EDGES, np.hypot(pr.y - C[0], pr.x - C[1]), side="right")
        return pr
    pr = files_after((RR >= EDGES[0]) & (RR < EDGES[1]))
    check("loss", "inner file gone all round: cortex still found (was: all 'unassigned')",
          int((pr.tissue == "cortex").sum()), 7 * SECT)
    k = pr[pr.tissue.isin(["cortex", "epidermis"])]
    check("loss", "inner file gone: every cell keeps its file number (gap at the stele counted)",
          float((np.floor(k.file + 0.5) == k.true_layer).mean()), 1.0)
    pr = files_after((RR >= EDGES[3]) & (RR < EDGES[5]) & ((TH // 10) % 3 == 0))
    k = pr[pr.tissue.isin(["cortex", "epidermis"]) & (pr.true_layer >= 6)]
    check("loss", "aerenchyma (layers 4-5 gone, every 3rd sector): cells outside keep their file number",
          float((np.floor(k.file + 0.5) == k.true_layer).mean()), 1.0)
    pr = files_after(np.zeros_like(RR, bool), RR >= EDGES[7])
    k = pr[pr.tissue.isin(["cortex", "epidermis"])]
    check("loss", "outer layer + epidermis lost: file numbers unchanged (counted from the stele)",
          float((np.floor(k.file + 0.5) == k.true_layer).mean()), 1.0)
    docs = []                                    # output documentation: every column described, run info written
    for wb in sorted((base / "results").rglob("*.xlsx")):
        rd = pd.read_excel(wb, sheet_name="README")
        rd = rd[rd.column.notna()]
        docs += [f"{wb.name.split('_20')[0]}:{c}" for c in rd.column[rd["what it is"].fillna("").str.strip() == ""]]
        check("docs", f"{wb.name.split('_20')[0]}: run_info sheet present",
              "run_info" in pd.ExcelFile(wb).sheet_names, True)
    check("docs", "every output column has a description", ", ".join(docs) or "none missing", "none missing")
    check("docs", "root_diameter_eq_um written", "root_diameter_eq_um" in r["img"].columns, True)
    check("docs", "methods text written", any((base / "results").glob("methods_*.txt")), True)
    import re
    tr = (HERE / "docs/traits.md").read_text()
    miss = [c for c in r["img"].reset_index().columns
            if not any(f"`{b}`" in tr for b in {re.sub(r"_(um2|um|px)$", "", c),
                                                re.sub(r"_(est)$", "", re.sub(r"_(um2|um|px)$", "", c))})
            and c not in ("image_file", "plant", "tech_rep", "status", "um_per_px", "epidermis")]
    check("docs", "every per_image column is in docs/traits.md", ", ".join(miss) or "none missing", "none missing")

    def edit(case, fn, expect, note=""):
        """Copy the base project, edit image S with fn(npz, cells, region, mx), summarise, compare."""
        d = out_dir / case
        shutil.copytree(base, d)
        shutil.rmtree(d / "results", ignore_errors=True)
        npz = d / "anatomy/masks/S.npz"
        c, rg, m, _ = P.load_state(npz)
        fn(npz, c.copy(), rg.copy(), m.copy())
        i = results(d)["img"].loc["S.tif"]
        for k, v in expect.items():
            if k not in i.index:
                CHECKS.append({"case": case, "check": k, "expected": v, "got": "no such column", "ok": "FAIL",
                               "note": note})
                continue
            got = i[k]
            check(case, k, float(got) if pd.notna(got) else np.nan, float(v), TOL.get(k, 0), note)
        check(case, "status = reviewed", i.status, "reviewed")
        return i

    def sectors(*arcs):
        """Save excluded arcs on S (as the editors do)."""
        return lambda npz, c, rg, m: P.save_edit(npz, **P.sector_arrays(list(arcs)))

    # 1: delete a cortex cell (Cellpose saw a cell that is not there)
    def e1(npz, c, rg, m):
        y, x = at((EDGES[3] + EDGES[4]) / 2, 115)
        P.click_edit(gray, c, rg, m, "Cell", y, x, remove=True)
        P.save_edit(npz, cells=c)
    edit("01_delete_cell", e1, {"n_cortex_cells": 287, "n_cortical_layers": 8}, "one ray loses a layer; median stays 8")

    # 2: delete a cell, then click it back in (Cellpose missed a cell; the reviewer adds it)
    def e2(npz, c, rg, m):
        y, x = at((EDGES[3] + EDGES[4]) / 2, 135)
        P.click_edit(gray, c, rg, m, "Cell", y, x, remove=True)
        P.click_edit(gray, c, rg, m, "Cell", y, x)
        P.save_edit(npz, cells=c)
    edit("02_add_missed_cell", e2, {"n_cortex_cells": 288, "n_cortical_layers": 8})

    # 3: draw one cell over two (Cellpose split one cell; reviewer merges)
    def e3(npz, c, rg, m):
        P.outline_edit(c, rg, m, "Cell", (RR >= EDGES[3]) & (RR < EDGES[5]) & (TH >= 130) & (TH < 140))
        P.save_edit(npz, cells=c)
    edit("03_merge_two_cells", e3, {"n_cortex_cells": 287, "n_cortical_layers": 8})

    # 4: draw a cell over the outer half of one (Cellpose merged two; reviewer splits)
    def e4(npz, c, rg, m):
        mid = (EDGES[4] + EDGES[5]) / 2
        P.outline_edit(c, rg, m, "Cell", (RR >= mid) & (RR < EDGES[5]) & (TH >= 220) & (TH < 230))
        P.save_edit(npz, cells=c)
    edit("04_split_cell", e4, {"n_cortex_cells": 289, "n_cortical_layers": 8})

    # 5: exclude 180-270 (torn cortex): cortex from 270 degrees; stele, vessels, root outline whole; estimates scale
    ca = EXPECT["cortex_area_est_px"]
    edit("05_exclude_quarter_sector", sectors((180, 270, "torn")),
         {"n_cortex_cells": 216, "angle_kept_deg": 270, "n_cortical_layers": 8, "mx_n": 6,
          "stele_area_px": EXPECT["stele_area_px"], "root_area_px": EXPECT["root_area_px"],
          "cortex_width_px": R_ROOT - R_STELE, "cortex_area_kept_px": 0.75 * ca, "cortex_area_est_px": ca,
          "root_area_est_px": EXPECT["root_area_px"], "n_cortex_cells_est": 288,
          "stele_share_est": R_STELE ** 2 / R_ROOT ** 2, "low_coverage": 0,
          "mean_cortex_cell_area_px": EXPECT["mean_cortex_cell_area_px"], "epidermis_rel": EPI_AREA / MED_TISSUE},
         "stele + vessels whole; cortex from 3/4; _est back to the whole")

    # 6: an old project: dropped quarter 3 (before sectors) reads as the same 90 degree arc
    edit("06_old_dropped_quarter", lambda npz, c, rg, m: P.save_edit(npz, dropped_q=np.array([3], np.int16)),
         {"n_cortex_cells": 216, "angle_kept_deg": 270})

    # 6b: edge cells go by their centre: 3-23 degrees holds the centres at 5 and 15 (not 25): 2 per layer out
    edit("06b_edge_cells_by_centre", sectors((3, 23, "torn")), {"n_cortex_cells": 288 - 16, "angle_kept_deg": 340},
         "no half cells: a cell is in or out by its centre")

    # 6c: an arc across 0 degrees (350 -> 10)
    edit("06c_arc_across_zero", sectors((350, 10, "folded")), {"n_cortex_cells": 288 - 16, "angle_kept_deg": 340})

    # 6d: two arcs, overlapping: kept degrees count the overlap once
    edit("06d_overlapping_arcs", sectors((0, 60, "torn"), (40, 90, "debris")),
         {"n_cortex_cells": 216, "angle_kept_deg": 270})

    # 6e: a narrow kept sector: 40 degrees -> low_coverage; estimates still scale
    i = edit("06e_narrow_kept_sector", sectors((40, 360, "smeared / oblique")),
             {"n_cortex_cells": 32, "angle_kept_deg": 40, "low_coverage": 1, "n_cortex_cells_est": 288,
              "cortex_area_est_px": ca, "n_cortical_layers": 8}, f"below {P.MIN_KEPT_DEG} degrees: flagged")

    # 6f: exclude with a drag, then restore with a Ctrl/Cmd click (sector_edit, as the editors call it)
    def e6f(npz, c, rg, m):
        ctr = P.centre_of(rg)
        arcs = P.sector_edit([], ctr, at(300, 100), at(300, 170), "torn")
        arcs = P.sector_edit(arcs, ctr, at(300, 200), at(300, 250), "debris")
        arcs = P.sector_edit(arcs, ctr, at(300, 130), remove=True)        # the first comes back
        P.save_edit(npz, **P.sector_arrays(arcs))
    edit("06f_drag_then_restore", e6f, {"angle_kept_deg": 310, "n_cortex_cells": 288 - 5 * 8},
         "200-250: centres 205 ... 245 out")

    # 6g: excluded epidermis leaves the epidermis point; excluded cells stay in all_cells, tagged
    d = out_dir / "06_old_dropped_quarter"
    r6 = results(d)
    pe = r6["prof"][(r6["prof"].image_file == "S.tif") & (r6["prof"].bin == "10")]
    check("06g_excluded_cells", "outer bin (epidermis): kept cells only", int(pe.n_cells.iloc[0]), 54)
    ac = r6["cells"][r6["cells"].image_file == "S.tif"]
    check("06g_excluded_cells", "all_cells keeps excluded cortex, tagged", int((ac.tissue == "cortex_excluded").sum()), 72)
    check("06g_excluded_cells", "stele never excluded", int(ac.tissue.astype(str).str.startswith("stele_").sum()), 0)

    # 7-8: vessels
    def e7(npz, c, rg, m):
        y, x = at(MX_AT, 135)
        P.click_edit(gray, c, rg, m, "Metaxylem", y, x, remove=True)
        P.save_edit(npz, mx=m)
    edit("07_remove_vessel", e7, {"mx_n": 5})

    def e8(npz, c, rg, m):
        y, x = at(MX_AT, 45)                                 # draw a 7th vessel
        P.outline_edit(c, rg, m, "Metaxylem", (yy - y) ** 2 + (xx - x) ** 2 <= MX_R ** 2)
        P.save_edit(npz, cells=c, mx=m)
    edit("08_add_vessel", e8, {"mx_n": 7, "mx_mean_area_px": np.pi * MX_R ** 2})

    # 8b: a vessel removed, then clicked back in: the same vessel, no stele cell touched
    def e8b(npz, c, rg, m):
        y, x = at(MX_AT, 135)
        P.click_edit(gray, c, rg, m, "Metaxylem", y, x, remove=True)
        P.click_edit(gray, c, rg, m, "Metaxylem", y, x)
        P.save_edit(npz, cells=c, mx=m)
    edit("08b_click_vessel_back", e8b, {"mx_n": 6, "mx_mean_area_px": np.pi * MX_R ** 2})
    # 8c: the same, but a gap in the vessel wall opens onto a neighbouring stele cell whose lumen is as dark as the
    # vessel (as in Amool's B73 images): the click must not run into it (2026-11-10: it took 1.5-2x the vessel)
    def e8c(npz, c, rg, m):
        y, x = at(MX_AT, 135)
        g = gray.copy()
        nb = c[at(MX_AT + MX_R + 6, 135)]                   # the stele cell just outside the vessel
        g[c == nb] = 0.05
        gy, gx = at(MX_AT + MX_R, 135)
        g[gy - 3:gy + 4, gx - 3:gx + 4] = 0.05              # a gap in the wall between them
        P.click_edit(g, c, rg, m, "Metaxylem", y, x, remove=True)
        n0 = (c == nb).sum()
        P.click_edit(g, c, rg, m, "Metaxylem", y, x)
        check("08c_vessel_wall_gap", "neighbour cell kept whole", int((c == nb).sum()), int(n0))
        P.save_edit(npz, cells=c, mx=m)
    edit("08c_vessel_wall_gap", e8c, {"mx_n": 6, "mx_mean_area_px": np.pi * MX_R ** 2})

    # 9: stele redrawn wider, taking cortex layer 1
    def e9(npz, c, rg, m):
        P.outline_edit(c, rg, m, "Stele", RR < EDGES[1])
        P.save_edit(npz, region=rg)
    edit("09_stele_takes_layer1", e9, {"stele_area_px": np.pi * EDGES[1] ** 2, "n_cortex_cells": 7 * SECT,
                                       "n_cortical_layers": 7, "cortex_width_px": R_ROOT - EDGES[1]})

    # 10: root redrawn inside the epidermis: outer cortex layer becomes the edge
    def e10(npz, c, rg, m):
        P.outline_edit(c, rg, m, "Root", RR < EDGES[-1])
        P.save_edit(npz, region=rg)
    edit("10_root_without_epidermis", e10, {"root_area_px": np.pi * EDGES[-1] ** 2, "n_cortex_cells": 7 * SECT,
                                             "n_cortical_layers": 7})

    # 11: aerenchyma: 4 neighbouring cells of layer 5 gone (10-50 deg) = one lacuna; one layer-6 cell gone (205 deg)
    # = a one-cell gap (a missed cell, review hint), not a lacuna
    def e11(npz, c, rg, m):
        gone = [lab(c, 5, d) for d in (15, 25, 35, 45)] + [lab(c, 6, 205)]
        c[np.isin(c, gone)] = 0
        P.save_edit(npz, cells=c)
    r0, r1 = EDGES[4], EDGES[5]
    r_bar = 2 / 3 * (r1 ** 3 - r0 ** 3) / (r1 ** 2 - r0 ** 2)          # area-weighted mean radius of the annulus
    i11 = edit("11_aerenchyma_lacuna", e11, {"n_cortex_cells": 283, "n_lacunae": 1, "lacuna_area_px": 4 * CELL_AREA[4],
                                             "lacuna_r_norm_mean": (r_bar - R_STELE) / (R_ROOT - R_STELE),
                                             "n_gaps_missed_cell": 1},
               "the one-cell gap is a missed cell, not a lacuna")
    r11 = results(out_dir / "11_aerenchyma_lacuna")
    a = r11["aer"][(r11["aer"].image_file == "S.tif") & (r11["aer"].lacuna_share > 0)]
    check("11_aerenchyma_lacuna", "lacuna sits in bins 5-6 only (r_norm 0.43-0.58)", ",".join(map(str, a.bin)), "5,6")
    lt = r11["lac"][r11["lac"].image_file == "S.tif"]
    check("11_aerenchyma_lacuna", "lacuna inner edge (r_norm)", float(lt.r_norm_inner.iloc[0]),
          (r0 - R_STELE) / (R_ROOT - R_STELE), .03)
    check("11_aerenchyma_lacuna", "lacuna outer edge (r_norm)", float(lt.r_norm_outer.iloc[0]),
          (r1 - R_STELE) / (R_ROOT - R_STELE), .03)
    check("11_aerenchyma_lacuna", "lacuna quarter", int(lt.quarter.iloc[0]), 1)

    # 11f: the same lacuna marked "not a lacuna" by a reviewer (pipeline.lacuna_edit, Cmd/Ctrl-click inside it):
    # not counted, counted as overridden; the one-cell gap is untouched
    def e11f(npz, c, rg, m):
        gone = [lab(c, 5, d) for d in (15, 25, 35, 45)] + [lab(c, 6, 205)]
        c[np.isin(c, gone)] = 0
        props = P.derive(c, rg, m, 0, C)[0]
        y, x = at((EDGES[4] + EDGES[5]) / 2, 30)
        marks, _ = P.lacuna_edit(c, rg, m, props, C, [], [], [(y, x)], remove=True)
        P.save_edit(npz, cells=c, **P.not_lacuna_arrays(marks))
    edit("11f_lacuna_overridden", e11f, {"n_lacunae": 0, "lacuna_area_px": 0, "aerenchyma_share": 0,
                                         "n_lacunae_overridden": 1, "n_gaps_missed_cell": 1})

    # 11g: a reviewer draws around the same 4 cells ("this is a lacuna": Cellpose drew cells over a hole): the cells
    # go and the rule counts exactly the lacuna of case 11
    def e11g(npz, c, rg, m):
        props = P.derive(c, rg, m, 0, C)[0]
        r0, r1 = EDGES[4] + 1, EDGES[5] - 1
        pts = [at(r0, t) for t in np.linspace(10.5, 49.5, 40)] + [at(r1, t) for t in np.linspace(49.5, 10.5, 40)]
        marks, _ = P.lacuna_edit(c, rg, m, props, C, [], [], pts)
        P.save_edit(npz, cells=c, **P.not_lacuna_arrays(marks))
    edit("11g_lacuna_drawn", e11g, {"n_lacunae": 1, "n_lacunae_overridden": 0}, "walls between the 4 cells go too")

    # 11d: a tear against the stele (layer-1 cells gone over 60 deg) is not a lacuna
    def e11d(npz, c, rg, m):
        c[np.isin(c, [lab(c, 1, d) for d in range(185, 245, 10)] + [lab(c, 2, d) for d in range(185, 245, 10)])] = 0
        P.save_edit(npz, cells=c)
    edit("11d_tear_at_stele", e11d, {"n_lacunae": 0})

    # 11h: a reviewer draws a lacuna against the stele (layer 1-2 cells, 100-140 deg): the drawn lacuna counts even
    # though the automatic rule never counts a gap touching the stele (real aerenchyma can reach the endodermis;
    # decided 2026-11-10). The same gap undrawn is case 11d: not counted.
    def draw_lac(c, rg, m, r0, r1, t0, t1, drawn):
        props = P.derive(c, rg, m, 0, C)[0]
        pts = [at(r0, t) for t in np.linspace(t0, t1, 40)] + [at(r1, t) for t in np.linspace(t1, t0, 40)]
        return P.lacuna_edit(c, rg, m, props, C, [], [], pts, drawn=drawn)[0]
    def e11h(npz, c, rg, m):
        drawn = []
        draw_lac(c, rg, m, R_STELE - 2, EDGES[2] - 1, 100.5, 139.5, drawn)
        P.save_edit(npz, cells=c, **P.drawn_lacuna_arrays(drawn))
    i11h = edit("11h_drawn_at_stele", e11h, {"n_lacunae": 1, "n_lacunae_drawn": 1, "n_lacunae_overridden": 0},
                "drawn lacuna touching the stele counts")
    # 11i: a drawn lacuna of one cell (below the 2x-median size rule) counts too: the reviewer's call
    def e11i(npz, c, rg, m):
        drawn = []
        draw_lac(c, rg, m, EDGES[4] + 1, EDGES[5] - 1, 26, 34, drawn)
        P.save_edit(npz, cells=c, **P.drawn_lacuna_arrays(drawn))
    edit("11i_drawn_one_cell", e11i, {"n_lacunae": 1, "n_lacunae_drawn": 1, "n_gaps_missed_cell": 0},
         "a drawn one-cell lacuna counts, and is not a missed-cell hint")
    # 11j: Cmd/Ctrl-click on a drawn lacuna: not a lacuna after all (the newest call wins). It touches the stele, so
    # the automatic rule would not count it either: not "overridden" (that counts only gaps the rule would count)
    def e11j(npz, c, rg, m):
        drawn = []
        draw_lac(c, rg, m, R_STELE - 2, EDGES[2] - 1, 100.5, 139.5, drawn)
        props = P.derive(c, rg, m, 0, C)[0]
        marks, _ = P.lacuna_edit(c, rg, m, props, C, [], [], [at((R_STELE + EDGES[2]) / 2, 120)], remove=True,
                                 drawn=drawn)
        P.save_edit(npz, cells=c, **P.drawn_lacuna_arrays(drawn), **P.not_lacuna_arrays(marks))
    edit("11j_drawn_then_unmarked", e11j, {"n_lacunae": 0, "n_lacunae_drawn": 0, "n_lacunae_overridden": 0})

    # 11e: no stele (a root tip, or the stele cut off by the frame): stele / cortex traits blank, never 0
    def e11e(npz, c, rg, m):
        rg[rg == 2] = 1
        m[:] = 0
        P.save_edit(npz, region=rg, mx=m)
    edit("11e_no_stele", e11e, {"stele_found": 0, "n_cortical_layers": np.nan, "stele_area_um2": np.nan,
                                "n_cortex_cells": np.nan, "mx_n": np.nan, "n_lacunae": np.nan,
                                "root_diameter_eq_px": 2 * R_ROOT}, "root size still measured")

    # 11c: big epidermal cells (4 merged into 1): in the main curve (_rel, epidermis part of the tissue) the outer
    # end rises above the cortex peak, so the peak sits at the edge; without the tagged cells (_rel_cortex) it does not
    def e11c(npz, c, rg, m):
        e = (RR >= EDGES[-1]) & (RR < R_EPI) & (c > 0)
        c[e] = c.max() + 1 + (TH[e] // 20).astype(int)
        P.save_edit(npz, cells=c)
    edit("11c_big_epidermis", e11c, {"n_cortex_cells": 288})
    f11c = results(out_dir / "11c_big_epidermis")["feats"]
    f11c = f11c[(f11c.image_file == "S.tif") & (f11c.basis == "position") & (f11c.scope == "section")].iloc[0]
    check("11c_big_epidermis", "main curve: peak at the outer edge (peak_pos_rel > 0.9)", bool(f11c.peak_pos_rel > 0.9), True)
    check("11c_big_epidermis", "without the epidermis: peak inside (peak_pos_rel_cortex < 0.8)",
          bool(f11c.peak_pos_rel_cortex < 0.8), True)

    # 11b: a quarter with < 45 deg kept is not used for quarter features
    i11b = edit("11b_quarter_mostly_excluded", sectors((0, 50, "torn")), {"angle_kept_deg": 310})
    check("11b_quarter_mostly_excluded", "quarters used (Q1 has 40 deg)", i11b.quarters_used, "2,3,4")

    # 12: a lateral root breaking out at 70-80 deg: its cells replace the epidermis there and stick out past the
    # root outline (r 455-560). The root outline stays on the main root. None of its cells may count.
    def e12(npz, c, rg, m):
        lr = (RR >= EDGES[-1]) & (RR < 560) & (TH >= 70) & (TH < 80)
        rr = RR[lr]
        c[lr] = c.max() + 1 + ((rr - EDGES[-1]) // 25).astype(int)        # 5 lateral-root cells, stacked outward
        P.save_edit(npz, cells=c)
    edit("12_lateral_root_outside", e12, {"n_cortex_cells": 288, "n_cortical_layers": 8,
                                          "mean_cortex_cell_area_px": EXPECT["mean_cortex_cell_area_px"]},
         "cells outside the root outline must not count")

    # 13: the same lateral root, but the reviewer leaves the root outline round it (outline includes the LR)
    def e13(npz, c, rg, m):
        lr = (RR >= EDGES[-1]) & (RR < 560) & (TH >= 70) & (TH < 80)
        c[lr] = c.max() + 1 + ((RR[lr] - EDGES[-1]) // 25).astype(int)
        rg[lr] = 1
        P.save_edit(npz, cells=c, region=rg)
    edit("13_lateral_root_inside_outline", e13, {"n_cortex_cells": 288},
         "shows what happens if the outline takes in the lateral root")

    traits_checks(out_dir, base, gray)


def traits(d: Path) -> dict:
    """The sheets of traits_DATE.xlsx (after results(d) or SUM.main)."""
    x = next((d / "results").glob("traits_*.xlsx"))
    return {k: pd.read_excel(x, sheet_name=k) for k in ("README", "Per image", "Per quarter", "Cell area by file",
                                                        "Cells per file")}


def traits_checks(out_dir: Path, base: Path, gray):
    """traits_DATE.xlsx (2026-11-10): per image + per quarter, µm, nothing per plant."""
    A = UM ** 2
    t = traits(base)
    i = t["Per image"].set_index("image_file").loc["S.tif"]
    q = t["Per quarter"][t["Per quarter"].image_file == "S.tif"].set_index("quarter")
    ck = lambda what, got, want, tol=0.0, note="": check("T1_traits_base", what, got, want, tol, note)
    ck("root_area_um2 = pi R^2", float(i.root_area_um2), np.pi * R_ROOT ** 2 * A, .005)
    ck("nothing excluded: root_area_est = root_area", float(i.root_area_est_um2), float(i.root_area_um2), 1e-9)
    ck("root_diameter_um = 2 R", float(i.root_diameter_um), 2 * R_ROOT * UM, .005)
    ck("stele_area_um2", float(i.stele_area_um2), np.pi * R_STELE ** 2 * A, .01)
    ck("stele_share = (Rs/R)^2", float(i.stele_share), (R_STELE / R_ROOT) ** 2, .01)
    ck("cortex_stele_ratio", float(i.cortex_stele_ratio), (R_ROOT ** 2 - R_STELE ** 2) / R_STELE ** 2, .01)
    ck("mx_n", int(i.mx_n), 6)
    for k in ("mean", "median", "max"):
        ck(f"mx_area_{k}_um2 = pi r^2", float(i[f"mx_area_{k}_um2"]), np.pi * MX_R ** 2 * A, .04)
    ck("mx_area_total_um2", float(i.mx_area_total_um2), 6 * np.pi * MX_R ** 2 * A, .04)
    r_m = MX_R * UM * 1e-6
    ck("axial conductance = 6 pi r^4 / 8 eta", float(i.axial_conductance_m4_MPa_s),
       6 * np.pi * r_m ** 4 / (8 * P.WATER_VISCOSITY_MPA_S), .08, "r^4: 2% in area = 4% here")
    ck("n_quarters_used", int(i.n_quarters_used), 4)
    ck("ccfn (mean of quarters)", float(i.ccfn), 8.0)
    ck("cortex_width_um", float(i.cortex_width_um), (R_ROOT - R_STELE) * UM, .01)
    ck("aerenchyma_share", float(i.aerenchyma_share), 0.0)
    ck("4 quarter rows", len(q), 4)
    ck("quarter cortex areas sum to the cortex", float(q.cortex_area_um2.sum()),
       np.pi * (R_ROOT ** 2 - R_STELE ** 2) * A, .005)
    ck("each quarter ccfn 8", ",".join(str(int(v)) for v in q.ccfn), "8,8,8,8")
    ck("each quarter: 18 rays", ",".join(str(int(v)) for v in q.n_rays), "18,18,18,18")
    fa = t["Cell area by file"].set_index("image_file").loc["S.tif"]
    for k in range(8):
        ck(f"cell area file {k + 1} = layer {k + 1}", float(fa[f"file_{k + 1}"]), CELL_AREA[k] * A, .03)
    ck("file 9 = epidermis", float(fa["file_9"]), EPI_AREA * A, .03)
    fn = t["Cells per file"].set_index("image_file").loc["S.tif"]
    ck("cells per file 1-8 = 36, file 9 = 72", ",".join(str(int(fn[f"file_{k}"])) for k in range(1, 10)),
       "36,36,36,36,36,36,36,36,72")
    ck("no plant column (per image only)", "plant" in t["Per image"].columns, False)
    rd = t["README"]
    cols = {(s, c) for s in ("Per image", "Per quarter") for c in t[s].columns}
    blank = [c for s, c in cols if rd[(rd.sheet == s) & (rd.column == c)]["what it is"].fillna("").eq("").all()]
    ck("README: every column described", ", ".join(blank) or "none", "none")
    tr = (HERE / "docs/traits.md").read_text()
    miss = sorted({c for _, c in cols if f"`{c}`" not in tr} - {"image_file", "status", "um_per_px"})
    ck("every traits column is in docs/traits.md", ", ".join(miss) or "none missing", "none missing")

    # the layout (2026-11-10): three outputs on top, the old workbooks in old_format/
    top = sorted(p.name.split("_20")[0] for p in (base / "results").iterdir() if p.is_file())
    ck("results/: traits, cortex_pattern, 2 cell CSVs, methods", ",".join(top),
       "all_cells,cortex_cells,cortex_pattern,methods,traits")
    old = sorted(p.name.split("_20")[0] for p in (base / "results/old_format").iterdir())
    ck("old_format/: the 3 old workbooks", ",".join(old), "all_cells,anatomy_summary,cortex_pattern_full")
    ck("traits opens on START_HERE", pd.ExcelFile(next((base / "results").glob("traits_*.xlsx"))).sheet_names[0],
       "START_HERE")
    cc = pd.read_csv(next((base / "results").glob("cortex_cells_*.csv")))
    cc = cc[cc.image_file == "S.tif"]
    ck("cortex_cells.csv: 288 cells", len(cc), 288)
    ck("cortex_cells.csv: 72 per quarter", ",".join(str(v) for v in cc.quarter.value_counts().sort_index()),
       "72,72,72,72")
    ck("cortex_cells.csv: file = layer (1-8, 36 each)", ",".join(str(v) for v in
       np.floor(cc.file + 0.5).astype(int).value_counts().sort_index()), ",".join(["36"] * 8))
    mid = (EDGES[4] + EDGES[5]) / 2
    l5 = cc[np.floor(cc.file + 0.5) == 5]
    ck("cortex_cells.csv: layer 5 r_norm", float(l5.r_norm.median()), (mid - R_STELE) / (R_ROOT - R_STELE), .03)
    ck("cortex_cells.csv: layer 5 area_um2", float(l5.area_um2.median()), CELL_AREA[4] * A, .03)
    ck("cortex_cells.csv: area_um2 = area_px x 0.25", float((cc.area_um2 / cc.area_px).median()), A, 1e-9)
    ck("cortex_cells.csv: r_um / r_norm consistent (layer 5 r_um)", float(l5.r_um.median()), mid * UM, .03)
    ac = pd.read_csv(next((base / "results").glob("all_cells_*.csv")))
    ac = ac[ac.image_file == "S.tif"]
    ck("all_cells.csv: cell types", ",".join(f"{k}:{v}" for k, v in ac.tissue.value_counts().sort_index().items()),
       "cortex:288,epidermis:72,metaxylem:6,stele:37")
    ck("all_cells.csv: all kept (nothing excluded)", set(ac.kept), {"yes"})
    cf = pd.read_excel(next((base / "results").glob("cortex_pattern_*.xlsx")), sheet_name="Curve features")
    cf = cf[cf.image_file == "S.tif"].set_index("quarter")
    ck("curve features: Q1-Q4 + image", ",".join(cf.index), "Q1,Q2,Q3,Q4,image")
    ck("curve features: image = mean of quarters", float(cf.loc["image", "peak_height"]),
       float(cf.loc[["Q1", "Q2", "Q3", "Q4"], "peak_height"].mean()), 1e-9)
    ck("curve features: peak in the mid cortex (layer 6, r_norm ~0.65)", float(cf.loc["image", "peak_pos"]),
       ((EDGES[5] + EDGES[6]) / 2 - R_STELE) / (R_ROOT - R_STELE), .15)
    pf = pd.read_excel(next((base / "results").glob("cortex_pattern_*.xlsx")), sheet_name="Profile")
    pf = pf[pf.image_file == "S.tif"]
    ck("profile: 10 bins x (4 quarters + image)", len(pf), 50)
    pm = pf.groupby(["quarter"]).rel.sum()
    ck("profile: image rel = mean of quarters", float(pm["image"]), float(pm[["Q1", "Q2", "Q3", "Q4"]].mean()), 1e-9)
    chk = pd.read_excel(next((base / "results").glob("traits_*.xlsx")), sheet_name="Check").set_index("image_file")
    ck("Check: an unreviewed clean image says only 'not reviewed'", chk.at["S.tif", "flags"], "not reviewed")
    for sh in ("Per quarter", "Cell area by file", "Cells per file", "Check"):
        ck(f"no duplicate columns: {sh} has no status / um_per_px", bool({"status", "um_per_px"} & set(t[sh].columns)
                                                                         if sh in t else False), False)

    def tedit(case, fn):
        d = out_dir / case
        shutil.copytree(base, d)
        shutil.rmtree(d / "results", ignore_errors=True)
        npz = d / "anatomy/masks/S.npz"
        c, rg, m, _ = P.load_state(npz)
        fn(npz, c.copy(), rg.copy(), m.copy())
        SUM.main(d / "anatomy")
        t = traits(d)
        return (t["Per image"].set_index("image_file").loc["S.tif"],
                t["Per quarter"][t["Per quarter"].image_file == "S.tif"].set_index("quarter"))

    # T2: the edge torn in over 180-240 (outline follows the tear, r 350) and that arc excluded: fill-the-edge brings
    # back the circle; the measured outline area is smaller; cortex kept from the kept angles only
    def t2(npz, c, rg, m):
        rg[(TH >= 180) & (TH < 240) & (RR >= 350)] = 0
        P.save_edit(npz, region=rg, **P.sector_arrays([(180, 240, "torn")]))
    i, q = tedit("T2_torn_edge_filled", t2)
    torn = np.pi * R_ROOT ** 2 - (60 / 360) * np.pi * (R_ROOT ** 2 - 350 ** 2)
    check("T2_torn_edge_filled", "root_area_um2 = the torn outline", float(i.root_area_um2), torn * A, .005)
    check("T2_torn_edge_filled", "root_area_est_um2 = the whole circle", float(i.root_area_est_um2),
          np.pi * R_ROOT ** 2 * A, .005)
    check("T2_torn_edge_filled", "root_diameter_um = 2 R", float(i.root_diameter_um), 2 * R_ROOT * UM, .005)
    check("T2_torn_edge_filled", "cortex_area_est = circle - stele", float(i.cortex_area_est_um2),
          np.pi * (R_ROOT ** 2 - R_STELE ** 2) * A, .005)
    check("T2_torn_edge_filled", "cortex_area_kept = 300 deg of the annulus", float(i.cortex_area_kept_um2),
          300 / 360 * np.pi * (R_ROOT ** 2 - R_STELE ** 2) * A, .005)
    check("T2_torn_edge_filled", "Q3 (30 deg kept): blank", bool(pd.isna(q.loc["Q3", "ccfn"])), True)
    check("T2_torn_edge_filled", "Q3 kept_deg", float(q.loc["Q3", "kept_deg"]), 30.0)
    check("T2_torn_edge_filled", "n_quarters_used", int(i.n_quarters_used), 3)
    d2 = out_dir / "T2_torn_edge_filled" / "results"
    ac = pd.read_csv(next(d2.glob("all_cells_*.csv")))
    ac = ac[(ac.image_file == "S.tif") & (ac.tissue == "cortex")]
    full = pd.read_excel(next((d2 / "old_format").glob("anatomy_summary_*.xlsx")), sheet_name="per_image") \
        .set_index("image_file").loc["S.tif"]
    oc = pd.read_excel(next((d2 / "old_format").glob("all_cells_*.xlsx")), sheet_name="all_cells")
    check("T2_torn_edge_filled", "all_cells.csv: kept = no for every excluded cortex cell",
          int((ac.kept == "no").sum()), int(((oc.image_file == "S.tif") & (oc.tissue == "cortex_excluded")).sum()))
    check("T2_torn_edge_filled", "all_cells.csv: none of them in the kept angles",
          bool(P.in_arcs(np.degrees(ac[ac.kept == "no"].angle_rad), [(180, 240, "")]).all()), True)
    cc = pd.read_csv(next(d2.glob("cortex_cells_*.csv")))
    check("T2_torn_edge_filled", "cortex_cells.csv rows = n_cortex_cells", int((cc.image_file == "S.tif").sum()),
          int(full.n_cortex_cells))
    cf = pd.read_excel(next(d2.glob("cortex_pattern_*.xlsx")), sheet_name="Curve features")
    q3 = cf[(cf.image_file == "S.tif") & (cf.quarter == "Q3")]
    check("T2_torn_edge_filled", "curves: Q3 row there, blank", (len(q3), bool(q3.peak_height.isna().all())), (1, True))

    # T3: under 180 deg kept: no estimate (too little edge to interpolate), the measured area stays
    i, q = tedit("T3_little_kept", lambda npz, c, rg, m: P.save_edit(npz, **P.sector_arrays([(0, 200, "torn")])))
    check("T3_little_kept", "root_area_est blank under 180 deg", bool(pd.isna(i.root_area_est_um2)), True)
    check("T3_little_kept", "root_area_um2 still measured", float(i.root_area_um2), np.pi * R_ROOT ** 2 * A, .005)
    check("T3_little_kept", "stele still measured", float(i.stele_area_um2), np.pi * R_STELE ** 2 * A, .01)

    # T4: a quarter partly kept (Q1, 0-30 excluded -> 60 deg): shares, widths, ccfn yes; areas and counts blank
    i, q = tedit("T4_partial_quarter", lambda npz, c, rg, m: P.save_edit(npz, **P.sector_arrays([(0, 30, "debris")])))
    check("T4_partial_quarter", "Q1 kept 60", float(q.loc["Q1", "kept_deg"]), 60.0)
    check("T4_partial_quarter", "Q1 ccfn 8", float(q.loc["Q1", "ccfn"]), 8.0)
    check("T4_partial_quarter", "Q1 cortex width", float(q.loc["Q1", "cortex_width_um"]), (R_ROOT - R_STELE) * UM, .01)
    check("T4_partial_quarter", "Q1 aerenchyma_share 0", float(q.loc["Q1", "aerenchyma_share"]), 0.0)
    check("T4_partial_quarter", "Q1 cortex_area blank (not 90 deg)", bool(pd.isna(q.loc["Q1", "cortex_area_um2"])), True)
    check("T4_partial_quarter", "Q1 n_lacunae blank (not 90 deg)", bool(pd.isna(q.loc["Q1", "n_lacunae"])), True)
    check("T4_partial_quarter", "Q2 cortex_area = a quarter annulus", float(q.loc["Q2", "cortex_area_um2"]),
          np.pi * (R_ROOT ** 2 - R_STELE ** 2) / 4 * A, .005)

    # T5: aerenchyma in Q1: layer-5 cells over 0-60 deg gone (one lacuna). Cells present on those rays: 7; gap-aware
    # ccfn: 8 on every ray (the lacuna counts as the cell it replaced)
    def t5(npz, c, rg, m):
        c[np.isin(c, [lab(c, 5, d) for d in range(5, 60, 10)])] = 0
        P.save_edit(npz, cells=c)
    i, q = tedit("T5_lacuna_ccfn", t5)
    lac_a = 6 * CELL_AREA[4]
    check("T5_lacuna_ccfn", "Q1 ccfn (gap-aware) 8", float(q.loc["Q1", "ccfn"]), 8.0)
    check("T5_lacuna_ccfn", "Q1 cells present: 7 (12 of 18 rays cross the lacuna)",
          float(q.loc["Q1", "ccfn_cells_present"]), 7.0)
    check("T5_lacuna_ccfn", "Q1 n_lacunae 1", float(q.loc["Q1", "n_lacunae"]), 1.0)
    check("T5_lacuna_ccfn", "Q1 lacuna area = 6 cells", float(q.loc["Q1", "lacuna_area_um2"]), lac_a * A, .03)
    check("T5_lacuna_ccfn", "Q1 aerenchyma_share", float(q.loc["Q1", "aerenchyma_share"]),
          lac_a / (np.pi * (R_ROOT ** 2 - R_STELE ** 2) / 4), .03)
    check("T5_lacuna_ccfn", "Q2 untouched: share 0", float(q.loc["Q2", "aerenchyma_share"]), 0.0)
    check("T5_lacuna_ccfn", "image ccfn still 8", float(i.ccfn), 8.0)
    check("T5_lacuna_ccfn", "image aerenchyma_share = Q1 share / 4", float(i.aerenchyma_share),
          lac_a / (np.pi * (R_ROOT ** 2 - R_STELE ** 2) / 4) / 4, .03)

    # T5b: layer 1 gone all round (a gap between the stele edge and the cortex, as when the stele outline is drawn
    # too small or the cortex pulls away): ccfn does not count the gap (7 = cells present); a lacuna does (T5)
    i, q = tedit("T5b_gap_at_stele_not_counted",
                 lambda npz, c, rg, m: (c.__setitem__(np.isin(c, [lab(c, 1, d) for d in range(5, 360, 10)]), 0),
                                        P.save_edit(npz, cells=c)))
    check("T5b_gap_at_stele_not_counted", "image ccfn 7", float(i.ccfn), 7.0)
    check("T5b_gap_at_stele_not_counted", "= cells present", float(q.ccfn_cells_present.mean()), 7.0)

    # T6: no stele: root size stays, every stele / cortex / metaxylem number blank (never 0)
    def t6(npz, c, rg, m):
        rg[rg == 2] = 1
        m[:] = 0
        P.save_edit(npz, region=rg, mx=m)
    i, q = tedit("T6_no_stele", t6)
    check("T6_no_stele", "root area measured", float(i.root_area_um2), np.pi * R_ROOT ** 2 * A, .005)
    blank = [k for k in ("stele_area_um2", "cortex_area_est_um2", "stele_share", "mx_n", "axial_conductance_m4_MPa_s",
                         "ccfn", "cortex_width_um") if pd.notna(i[k])]
    check("T6_no_stele", "stele-dependent columns blank", ", ".join(blank) or "all blank", "all blank")
    check("T6_no_stele", "quarters blank", bool(q.ccfn.isna().all()), True)
    chk = pd.read_excel(next((out_dir / "T6_no_stele/results").glob("traits_*.xlsx")), sheet_name="Check")
    check("T6_no_stele", "Check flags: no stele", "no stele" in chk.set_index("image_file").at["S.tif", "flags"], True)

    # T7: samples.xlsx written when only S was measured (as when the scale is set after the first image), with a
    # renamed group column: Make results must still measure S_rot (it skipped it until 2026-11-10)
    d = out_dir / "T7_stale_samples"
    shutil.copytree(base, d)
    shutil.rmtree(d / "results", ignore_errors=True)
    pd.DataFrame([{"image_file": "S.tif", "plant": "P1", "um_per_px": UM, "include": "yes", "genotype": "B73"}]) \
        .to_excel(d / "anatomy/samples.xlsx", index=False)
    SUM.main(d / "anatomy")
    t7 = traits(d)
    s7 = pd.read_excel(d / "anatomy/samples.xlsx")
    check("T7_stale_samples", "both images in traits", sorted(t7["Per image"].image_file), ["S.tif", "S_rot.tif"])
    check("T7_stale_samples", "samples.xlsx: new row added, old row kept", len(s7), 2)
    check("T7_stale_samples", "old row's scale kept", float(s7.set_index("image_file").at["S.tif", "um_per_px"]), UM)
    check("T7_stale_samples", "renamed group column kept", "genotype" in t7["Per image"].columns, True)
    # T8: include = no: left out, and START_HERE says so
    s7.loc[s7.image_file == "S_rot.tif", "include"] = "no"
    s7.to_excel(d / "anatomy/samples.xlsx", index=False)
    SUM.main(d / "anatomy")
    t8 = traits(d)
    sh = pd.read_excel(next((d / "results").glob("traits_*.xlsx")), sheet_name="START_HERE")
    check("T8_include_no", "left out of traits", list(t8["Per image"].image_file), ["S.tif"])
    check("T8_include_no", "START_HERE names it", sh.astype(str).apply(lambda r: r.str.contains("S_rot.tif")).any().any(),
          True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rooh_check_"))
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    try:
        run(out)
    except Exception as e:                                    # report what ran
        CHECKS.append({"case": "ERROR", "check": repr(e)[:200], "ok": "FAIL"})
        import traceback
        traceback.print_exc()
    t = pd.DataFrame(CHECKS)
    pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 200)
    print(t.to_string(index=False))
    t.to_excel(out / "checks.xlsx", index=False)
    print(f"\n{(t.ok == 'PASS').sum()} pass, {(t.ok == 'FAIL').sum()} fail  ->  {out / 'checks.xlsx'}")
    sys.exit(int((t.ok == "FAIL").any()))
