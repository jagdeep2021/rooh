"""Masks -> tables. Run any time; it reads whatever is saved (auto or reviewed).

  python summarise.py <anatomy_dir> [--image-figures]     (also one cortex pattern figure per image)

Uses <anatomy_dir>/samples.xlsx (one row per image; made by run_batch.py, or here if missing). Fill it in:
  plant      the replicate; images of the same plant are averaged
  group_*    columns from the folder names; rename the headers (species, genotype ...)
             or add columns (treatment ...). Every column except image_file, plant,
             um_per_px and include is treated as a grouping column.
  um_per_px  microns per pixel, from the microscope/camera: always given by the user (scale bars are not read).
             Fill it any time after measuring: the scale is used only here, never for outlining.
  include    "no" drops the image
  notes      any column whose name starts with note / check / comment / remark: free text, copied to per_image
  epidermis  yes = intact, no = lost in sectioning (the outermost layer is then outer cortex, tagged epidermis by
             position); empty = not checked. The pattern counts every cell outside the endodermis either way.
Writes <anatomy_dir>/../results/anatomy_summary_YYYY-MM-DD.xlsx
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

import pattern_plots as PP
import pipeline as P

FIXED = ["image_file", "plant", "um_per_px", "include", "epidermis"]
NOTE_COLS = ("note", "check", "comment", "remark")   # samples.xlsx columns starting so = free text: copied to per_image,
                                                    # never a grouping column
N_BINS = 10   # radial bins across the cortex: 1 = next to the stele ... 10 = next to the epidermis; epidermis its own point
# what every column means: written into each workbook's README sheet. Columns ending _px are the same
# measure in pixels (kept at the end of each sheet, for images without a scale). Areas run to mid-wall.
COLUMNS = {
    "image_file": "image path inside the image folder (the tag that links every row to its picture)",
    "plant": "plant (biological replicate), from the folder name; images of one plant share it",
    "group_1": "first folder level (e.g. species); rename in anatomy/samples.xlsx",
    "group_2": "second folder level (e.g. genotype); rename in anatomy/samples.xlsx",
    "tech_rep": "image number within its plant (technical replicate)",
    "status": "auto = computer outlines only; reviewed = checked by hand (Save + next)",
    "um_per_px": "scale given in samples.xlsx (micrometres per pixel); empty = no µm columns",
    "n_images": "images averaged into this plant row",
    "n_reviewed": "of those, images checked by hand",
    "angle_kept_deg": "degrees of the section measured: 360 minus the excluded sectors (damaged tissue taken out by hand)",
    "excluded_sectors": "the excluded arcs, 'start-end reason' in degrees counter-clockwise from the image's right",
    "stele_found": "FALSE = no stele outlined (root tip, or stele cut off by the frame): stele, metaxylem, cortex "
                   "and aerenchyma columns are blank for this image (root size still measured)",
    "low_coverage": f"TRUE when fewer than {P.MIN_KEPT_DEG} degrees are kept: cortex numbers rest on few cells",
    "label": "cell number inside the image's mask",
    "tissue": "cortex | stele | epidermis | air_space | metaxylem | unassigned; '..._excluded' = centre in an excluded "
              "sector (kept in this table, out of every cortex number)",
    "sector": "kept or excluded (by the cell's centre)",
    "ring": "adjacency steps from the stele (a gap check; runs high)",
    "x": "cell centre, pixels from the left", "y": "cell centre, pixels from the top",
    "angle_rad": "direction of the cell centre from the centre, counter-clockwise from the right (radians)",
    "angle_deg": "the same in degrees",
    "r_um": "distance of the cell centre from the centre (µm)",
    "r_norm": "radial position of the cell centre: 0 = stele edge, 1 = root edge, along its own direction",
    "radial_bin": f"r_norm in {N_BINS} equal bins: 1 = next to the stele",
    "dist_from_stele_um": "distance from the stele edge to the cell centre (µm)",
    "centre_x_px": "the image's centre (stele centre, or set by hand), pixels from the left edge",
    "centre_y_px": "the image's centre, pixels from the top edge",
    "area_um2": "cell area (µm², to mid-wall)", "major_um": "cell length, long axis (µm)",
    "minor_um": "cell width, short axis (µm)",
    "root_area_um2": "whole cross-section area as outlined (µm²); with excluded sectors use root_area_est_um2",
    "stele_area_um2": "stele area, whole section (µm²; never sector dependent)",
    "cortex_area_kept_um2": "root outside the stele (cortex + epidermis) inside the kept angles (µm²)",
    "cortex_area_est_um2": "the same scaled to the whole circle (x 360 / angle_kept_deg): an ESTIMATE that assumes "
                           "the excluded arcs looked like the kept ones",
    "root_area_est_um2": "stele area + cortex_area_est_um2 (estimate)",
    "stele_share_est": "stele area / root_area_est (estimate; whole-section stele over the scaled root)",
    "cortex_width_um": "mean root radius minus stele radius over the kept degrees (µm)",
    "n_cortical_layers": "cortical cells crossed by a ray from the stele out, median over rays every 5 degrees in the "
                         "kept angles (_min / _max: the range: one side can carry an extra layer)",
    "n_cortical_layers_min": "fewest cortical layers on any 5-degree ray (kept angles)",
    "n_cortical_layers_max": "most cortical layers on any 5-degree ray; a wide min-max range = a wrong stele outline "
                             "or an extra layer on one side: check the image",
    "n_low_coverage": f"images of this plant with fewer than {P.MIN_KEPT_DEG} degrees kept",
    "n_cortex_cells": "cortical cells counted in the kept angles", "n_cortex_cells_est": "scaled to 360 degrees (estimate)",
    "mean_cortex_cell_area_um2": "mean cortical cell area, kept cells (µm²)",
    "median_cortex_cell_area_um2": "median cortical cell area, kept cells (µm²): the reference for rel",
    "n_air_spaces": "small spaces between kept cortical cells", "n_air_spaces_est": "scaled to 360 degrees (estimate)",
    "air_space_total_area_um2": "their total area (µm²)",
    "mx_n": "metaxylem vessels (whole section)", "mx_mean_area_um2": "mean metaxylem vessel area (µm²)",
    "mx_total_area_um2": "total metaxylem area (µm²)",
    "bin": f"position: 1..{N_BINS} = r_norm bins (1 next to the stele); file: the cell file number (the epidermis is "
           "the outermost file)",
    "x_position": "where the bin sits: position = bin centre (0-1); file = median file_rel of its cells (0-1); distance = bin centre, µm from the stele edge",
    "n_cells": "cells in this bin (kept angles, cortex + epidermis); few_cells = TRUE below " + str(P.MIN_CELLS_PER_BIN),
    "few_cells": f"TRUE when the bin has fewer than {P.MIN_CELLS_PER_BIN} cells: read it with care",
    "median_area_um2": "median cell area in this bin (µm²)",
    "rel": "median area in the bin / the image's median cell area (cortex + epidermis): shape without overall size",
    "epidermis_rel": "median area of the cells tagged epidermis (within 6 px of the outline) / the image's median",
    "min_cells_in_bin": "fewest cortical cells in any cortex bin of this image",
    "area_unit": "unit of the absolute sizes in this row: um2, or px when the image has no scale",
    "peak_height_rel": "largest value on the fitted curve; _rel = relative size (shape), all cells outside the "
                       "endodermis; _abs = the same in area_unit; _rel_cortex = relative, without the cells tagged "
                       "epidermis (a check) (same for the columns below)",
    "peak_pos_rel": "where the peak is on the 0-1 axis (stele side 0, epidermis side 1)",
    "rise_slope_rel": "(peak - first value) / distance: how fast cells grow from the stele to the peak",
    "decay_slope_rel": "slope just past the peak (over 20% of the range): how fast cells shrink outward",
    "outer_rise_rel": "last value - lowest value after the peak: an upturn at the epidermis",
    "basis": "position = cells binned by r_norm (0 stele edge - 1 root edge; the headline); file = cells grouped by "
             "cell file number (1 = next to the stele; gaps such as lacunae counted as the cells they would hold); "
             f"distance = {P.DIST_BIN_UM:g} µm bins of distance from the stele edge (images with a scale only): survives "
             "lost outer tissue but carries root size, a cross-check. For distance, x_position and peak_pos are µm "
             "from the stele edge and slopes are per µm",
    "scope": "section = all kept cells of the image; Q1..Q4 = one quarter (Q1 0-90 deg top right, counter-clockwise); "
             "quarter_mean / quarter_sd = mean and SD of the quarters' features",
    "n_quarters": "quarters averaged (a quarter needs >= 45 degrees kept)",
    "kept_deg": "degrees kept in this scope",
    "quarter": "quarter of the cell / lacuna centre: 1 = 0-90 deg (top right), counter-clockwise",
    "quarters_used": "quarters with >= 45 degrees kept (used for quarter features)",
    "file": "cell file number: median of the numbers given by rays every 0.5 deg (1 = next to the stele); a gap on a "
            "ray (lacuna, missed cell, gap at the stele) counts as the cells it would have held",
    "file_rel": "relative file position: median over rays of (k - 0.5) / cortical cells on that ray",
    "n_rays": "rays that crossed the cell (file number rests on them)",
    "epidermis": "from samples.xlsx: yes = intact, no = lost (outermost layer is outer cortex), empty = not checked",
    "root_diameter_um": "2 x mean root radius over the kept degrees, from the stele centre (µm)",
    "root_diameter_eq_um": "diameter of a circle with the root's area (µm; whole outline)",
    "n_giant_cortex_cells": "kept cortical cells > 5 x the median: possibly a lacuna Cellpose took for a cell, check",
    "n_lacunae": "aerenchyma lacunae: gaps in the cortex that are not cells, >= 2 x the median cortical cell, "
                 "touching neither the root edge nor the stele, centre in the kept angles",
    "epidermis_deg": "kept degrees (5-deg steps) holding a cell tagged epidermis: Cellpose often misses the epidermis, "
                     "so the outer bins then rest on hypodermis; below 90 compare _rel with _rel_cortex",
    "n_cells": "cells in the pattern (cortex + epidermis, kept angles)",
    "n_gaps_missed_cell": "gaps of about one cell (0.5-2 x the median) inside the cortex: usually a cell Cellpose "
                          "missed. Review hint: draw the cell",
    "lacuna_area_um2": "total lacuna area (µm²)", "aerenchyma_share": "lacuna area / cortex area kept",
    "lacuna_r_norm_mean": "area-weighted mean radial position of lacuna pixels (0 stele edge, 1 root edge)",
    "lacuna": "lacuna number in the image",
    "n_lacunae_overridden": "gaps the lacuna rule would count that a reviewer marked 'not a lacuna' (not in n_lacunae)",
    "n_lacunae_drawn": "lacunae a reviewer drew (Aerenchyma mode); counted in n_lacunae whatever the automatic size, "
                       "edge and stele rules say",
    "drawn": "TRUE = a reviewer drew this lacuna (counted whatever the automatic rules say)",
    "r_norm_inner": "innermost radial position of the lacuna", "r_norm_outer": "outermost radial position",
    "cortex_px": "cortex pixels in this bin (kept angles)", "lacuna_px": "lacuna pixels in this bin",
    "lacuna_share": "lacuna pixels / cortex pixels in this bin",
    "axis": "rel (the curve of relative size)", "fitted": "the curve's value at x",
}
FEATURE_SETS = ("_rel", "_abs", "_rel_cortex")


# Topics: the README's topic column and the column order of per_image / per_plant (first rule that matches wins).
TOPICS = (
    ("1 identity", lambda c, g: c in ("image_file", "plant", "tech_rep", "status", "um_per_px", "n_images",
                                      "n_reviewed", "epidermis", "label", "lacuna", "basis", "scope", "area_unit",
                                      "bin", "axis") or c in g or c.startswith("group_")),
    ("2 check first", lambda c, g: c in ("stele_found", "angle_kept_deg", "excluded_sectors", "low_coverage", "n_low_coverage",
                                         "quarters_used", "n_quarters", "kept_deg", "epidermis_deg",
                                         "min_cells_in_bin", "few_cells", "n_gaps_missed_cell",
                                         "n_giant_cortex_cells", "sector", "n_rays", "ring")
                                or c.startswith("n_cortical_layers_m")),
    ("3 whole root", lambda c, g: c.startswith("root_")),
    ("4 stele + metaxylem", lambda c, g: c.startswith(("stele_", "mx_"))),
    ("5 cortex", lambda c, g: "cortex" in c and "lacuna" not in c and not c.endswith("_rel_cortex")
                              or c.startswith(("n_cortical_layers", "n_air_spaces", "air_space"))),
    ("7 aerenchyma", lambda c, g: "lacuna" in c or "aerenchyma" in c),
    ("6 cortex pattern", lambda c, g: c.startswith(("peak_", "rise_", "decay_", "outer_rise", "epidermis_rel"))
                                      or c in ("rel", "x_position", "n_cells", "fitted", "radial_bin")
                                      or c.startswith("median_area")),
    ("8 cell position + size", lambda c, g: True),
)


def topic_of(col, groups=()):
    return next(t for t, rule in TOPICS if rule(col, groups))


def unit_of(col):
    c = col
    if c == "um_per_px":
        return "µm per pixel"
    if c.endswith(("_abs",)) or "_abs" in c:
        return "area_unit per unit position (per µm for basis distance)" if "slope" in c else \
            ("0-1 position (µm for basis distance)" if "pos" in c else "area_unit")
    if c.endswith("_um2"):
        return "µm²"
    if c.endswith("_um"):
        return "µm"
    if c.endswith("_px"):
        return "pixels²" if "area" in c else "pixels"
    if c.endswith("_deg") or c == "angle_deg":
        return "degrees"
    if c == "angle_rad":
        return "radians"
    if c in ("low_coverage", "few_cells", "stele_found"):
        return "TRUE / FALSE"
    if c in ("x", "y"):
        return "pixels"
    if c == "file_rel":
        return "0-1 (file position, stele side 0)"
    if c == "quarter":
        return "1-4"
    if c == "x_position" or c.startswith("peak_pos"):
        return "0-1 position (0 stele edge, 1 root edge); µm from the stele edge for basis distance"
    if "pos" in c or c.startswith(("r_norm", "lacuna_r_norm")):
        return "0-1 position (0 stele edge, 1 root edge)"
    if "slope" in c:
        return "relative size per unit position (per µm for basis distance)"
    if c in ("rel", "fitted", "epidermis_rel") or c.startswith(("peak_height", "outer_rise")):
        return "x the image's median cell (cortex + epidermis)"
    if "share" in c:
        return "fraction (0-1)"
    if c.startswith(("n_", "min_cells")) or c.endswith(("_n", "_est")) or c in ("tech_rep", "file", "bin",
                                                                                 "radial_bin", "lacuna", "label", "ring"):
        return "count / number"
    return "text"


def order_cols(df, groups=()):
    """per_image / per_plant: columns grouped by topic (identity, checks, root, stele, cortex, pattern, aerenchyma);
    within a topic µm before pixels."""
    cols = list(df.columns)
    return df[sorted(cols, key=lambda c: (topic_of(c, groups), c.endswith("_px") and c != "um_per_px",
                                          cols.index(c)))]


def rooh_version():
    """Commit of the Rooh code that made the numbers ('+ local changes' when the checkout is edited)."""
    import os
    import subprocess
    if os.environ.get("ROOH_VERSION"):
        return os.environ["ROOH_VERSION"]
    here = Path(__file__).resolve().parent
    try:
        h = subprocess.run(["git", "-C", str(here), "log", "-1", "--format=%h %cs"], capture_output=True,
                           text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (h + (" + local changes" if dirty else "")) if h else "unknown"
    except Exception:
        return "unknown"


def run_info(anatomy_dir, settings):
    """Provenance: written into every workbook (sheet run_info) so a table can always be traced to its rules."""
    return pd.DataFrame([
        ("made", dt.datetime.now().isoformat(timespec="minutes"), "date and time these tables were made"),
        ("rooh_version", rooh_version(), "Rooh code commit (github.com/jagdeep2021/cross-section-anatomy)"),
        ("anatomy_folder", str(anatomy_dir), "masks + samples.xlsx these tables were made from"),
        ("cell_model", "Cellpose cpsam" + (f" (custom: {settings['model_path']})" if settings.get("model_path")
                                           else ""), "proposes the cells; corrected by hand where status = reviewed"),
        ("cell_diameter", settings.get("diameter") or "auto", "Cellpose diameter setting (pixels)"),
        ("N_BINS", N_BINS, "radial bins across the tissue outside the endodermis"),
        ("MIN_KEPT_DEG", P.MIN_KEPT_DEG, "below this many degrees kept the image is flagged low_coverage"),
        ("MIN_QUARTER_DEG", P.MIN_QUARTER_DEG, "a quarter needs this many degrees kept to give quarter features"),
        ("MIN_CELLS_PER_BIN", P.MIN_CELLS_PER_BIN, "a bin with fewer cells is flagged few_cells"),
        ("MIN_EPI_DEG", P.MIN_EPI_DEG, "epidermis_deg below this: compare _rel with _rel_cortex"),
        ("FILE_STEP_DEG", P.FILE_STEP_DEG, "ray spacing (degrees) for cell file numbers"),
        ("LACUNA_MIN_CELLS", P.LACUNA_MIN_CELLS, "a lacuna is at least this many median cortical cells"),
        ("GAP_MIN_CELLS", P.GAP_MIN_CELLS, "gaps from this size up to LACUNA_MIN_CELLS = probably a missed cell"),
        ("MIN_PTS", PP.MIN_PTS, "fewest bins with cells for a fitted curve"),
    ], columns=["setting", "value", "what it is"])


def start_here(today, n_img, n_rev, no_scale, skipped=(), n_flagged=0):
    """The first sheet of traits_DATE.xlsx: the files, this run, the unit, what to check, where each answer is."""
    rows = [
        ("FILES", f"traits_{today}.xlsx", "this file: overall anatomy, per image and per quarter (µm)"),
        ("", f"cortex_pattern_{today}.xlsx", "cell size from the stele out: features of the curve per quarter, the "
                                             "binned profiles, the fitted curves, aerenchyma by position"),
        ("", f"cortex_cells_{today}.csv", "one row per kept cortical cell: quarter, position, radius, file, area"),
        ("", f"all_cells_{today}.csv", "the same for every object, with its cell type (tissue) and kept yes/no"),
        ("", f"methods_{today}.txt", "the Methods paragraph for these numbers, with the thresholds used"),
        ("", "figures/", "cortex pattern per group"),
        ("", "earlier_runs/", "results of earlier runs, moved out of the way (never deleted)"),
        ("", "README sheet", "every column of this file and of the two CSVs: unit and meaning"),
        ("THIS RUN", f"{n_img} images", f"{n_rev} of {n_img} checked by hand (status = reviewed); the rest are "
                                        "computer outlines only"),
        ("", f"{len(skipped)} measured image(s) left out", "; ".join(f"{r} ({why})" for r, why in skipped) or
         "none: every measured image is here"),
        ("", "scale", f"{no_scale} image(s) have no scale: their sizes are blank. Set um_per_px and run again"
         if no_scale else "every image has a scale"),
        ("", f"{n_flagged} image(s) flagged", "see the Check sheet before using them" if n_flagged else
         "the Check sheet found nothing"),
        ("UNIT", "the plant", "rows are images. Images of one plant are technical replicates: join plant and "
                              "treatment on image_file, then average per plant or use plant as a random effect. "
                              "Never use cells, quarters or images as n."),
        ("QUARTERS", "Q1-Q4", f"Q1 = 0-90 degrees (image right to top), then counter-clockwise. Blank under "
                              f"{P.MIN_QUARTER_DEG} degrees kept; areas and counts only with all 90 kept"),
        ("WHERE IS", "root size", "Per image: root_area_um2, root_area_est_um2, root_diameter_um"),
        ("", "stele + metaxylem", "Per image: stele_area_um2, stele_share, mx_n, mx_area_*, axial conductance"),
        ("", "CCFN, cortex width, cell size", "Per quarter; Per image = the mean of the quarters"),
        ("", "aerenchyma", "Per quarter: aerenchyma_share, lacuna_area_um2, n_lacunae; Per image: mean share"),
        ("", "cell size by file", "Cell area by file (+ Cells per file)"),
        ("", "the cortex pattern", f"cortex_pattern_{today}.xlsx"),
    ]
    return pd.DataFrame(rows, columns=["", "item", "what / where"])


CHECK_CCFN_GAP = 2   # Check sheet: gaps add >= 2 files in some quarter (ccfn - ccfn_cells_present) = cells missed?


def check_sheet(t_image, per_image, t_q) -> pd.DataFrame:
    """One row per image: plain-word reasons to look at it before using its numbers (flags), and the counts behind
    them that are not in Per image."""
    big = per_image.set_index("image_file")
    gap = (t_q.ccfn - t_q.ccfn_cells_present).groupby(t_q.image_file).max() if len(t_q) else pd.Series(dtype=float)
    rows = []
    for _, r in t_image.iterrows():
        b = big.loc[r.image_file]
        f = [] if r.status == "reviewed" else ["not reviewed"]
        if pd.isna(r.um_per_px):
            f.append("no scale")
        if not b.stele_found:
            f.append("no stele")
        elif r.n_quarters_used < 4:
            f.append(f"{4 - r.n_quarters_used} quarter(s) blank")
        if r.angle_kept_deg < P.MIN_EST_DEG:
            f.append("root area not estimated")
        if gap.get(r.image_file, 0) >= CHECK_CCFN_GAP:
            f.append("gaps add cell files: cells missed?")
        if b.n_giant_cortex_cells > 0:
            f.append(f"{b.n_giant_cortex_cells:.0f} giant cortex cell(s)")
        if b.n_gaps_missed_cell > 0:
            f.append(f"{b.n_gaps_missed_cell:.0f} missed-cell gap(s)")
        if b.stele_found and b.epidermis_deg < P.MIN_EPI_DEG:
            f.append("epidermis mostly missing")
        if b.n_lacunae_overridden > 0:
            f.append(f"{b.n_lacunae_overridden:.0f} lacuna(e) overruled by hand")
        rows.append({"image_file": r.image_file, "flags": "; ".join(f),
                     "n_giant_cortex_cells": b.n_giant_cortex_cells, "n_gaps_missed_cell": b.n_gaps_missed_cell,
                     "epidermis_deg": b.epidermis_deg, "n_lacunae_overridden": b.n_lacunae_overridden})
    return pd.DataFrame(rows, columns=["image_file", "flags", "n_giant_cortex_cells", "n_gaps_missed_cell",
                                       "epidermis_deg", "n_lacunae_overridden"])


def methods_text(per_image, n_plant, today):
    n, rev = len(per_image), int((per_image.status == "reviewed").sum())
    return f"""Methods: root cross-section anatomy (Rooh {rooh_version()}; tables made {today})

Text to adapt for a Methods section. The thresholds are this run's settings (sheet run_info); edit the species-specific wording.

Segmentation. Cells were segmented with Cellpose (cpsam model; Stringer et al. 2021, Pachitariu et al. 2025). The
root outline was derived from the cell walls and cells, the stele as the cluster of small cells near the centre, and
metaxylem vessels as large round lumens in the stele (area > 5 x the median stele cell, solidity > 0.85). All
outlines were proposed automatically and checked and corrected by hand in Rooh's editor ({rev} of {n} images in
this run; the rest are automatic only). Each pixel belongs to one object. The stele is everything inside the outer edge of the endodermis; cells
touching a 6-pixel band inside the root outline were tagged epidermis; the cells in between are cortex (including
the hypodermis). Objects smaller than 25% of their neighbours' median area with at most 5 neighbours were classed as
intercellular air spaces, not cells.

Damaged tissue. Torn, folded, oblique, out-of-focus or debris-covered parts of the cortex were excluded by hand as
arcs of angle around the stele centre, each with its reason; cells were kept or excluded by their centre. Biology
(aerenchyma, collapsed cells) was never a reason to exclude. Stele and metaxylem traits use the whole section.
Cortex counts and areas from the kept angles are also given scaled to 360 degrees (_est). Images with fewer than
{P.MIN_KEPT_DEG} degrees kept were flagged.

Scale. Pixel sizes (micrometres per pixel) were entered per image from the microscope calibration; areas are to the
mid-wall.

Cortical cell file number. Rays from the stele centre every 5 degrees (kept angles only) counted the cortical cells
crossed (a cell counts when the ray stays in it for at least 2 pixels); CCFN is the median over rays. Every cell
outside the endodermis was also given a file number (1 = next to the stele) as the median of its numbers along
rays every {P.FILE_STEP_DEG} degrees.

Radial pattern of cell size. Each cell's radial position was normalised along its own direction from the stele
centre: 0 at the stele edge, 1 at the root edge (r_norm). All cells outside the endodermis (cortex and epidermis,
treated as one tissue) in the kept angles were binned into {N_BINS} equal r_norm bins; cell area per bin was the
median, divided by the image's median cell area (relative size). A smoothing spline with the smoothing chosen by
generalised cross-validation (scipy make_smoothing_spline), weighted by cells per bin, was fitted through the bin
medians (at least {PP.MIN_PTS} bins with cells), and the peak height and position, the rise slope from the stele to
the peak, the decay slope over 20% of the range past the peak, and the outer rise were read off the curve. The same
was done by cell file instead of position, without the tagged epidermis (a check), and for each quarter around the
stele centre with at least {P.MIN_QUARTER_DEG} degrees kept (quarter mean and SD). Bins with fewer than
{P.MIN_CELLS_PER_BIN} cells were flagged.

Aerenchyma. Lacunae were non-cell gaps in the cortex of at least {P.LACUNA_MIN_CELLS:g} x the median cortical cell
area that touched neither the root edge nor the stele, with their centre in the kept angles. Aerenchyma was
expressed as lacuna area / cortex area kept, and by radial position as lacuna pixels / cortex pixels per r_norm bin.

Statistics. The plant was the experimental unit: per-cell values were summarised per image (medians), images
averaged per plant ({n} images from {n_plant} plants here); cells were never pooled across images.
"""


# ---------------------------------------------------------------- traits file (decided with Jagdeep 2026-11-10)
# traits_DATE.xlsx: the short overall file. Per image and per quarter only (never per plant: the user joins plant and
# treatment later by image_file). Sizes in µm (blank without a scale). The big workbooks stay the full record.
TRAITS = {
    "image_file": "image path inside the image folder: join plant, treatment and anything else on this",
    "status": "auto = computer outlines only; reviewed = checked by hand",
    "flags": "plain reasons to look at the image before using its numbers ('not reviewed' alone = a clean image "
             "nobody has checked yet)",
    "um_per_px": "scale from samples.xlsx (µm per pixel); empty = no sizes in this file",
    "angle_kept_deg": "degrees of the section kept (360 minus the excluded damaged arcs)",
    "excluded_sectors": "excluded arcs: start-end degrees and reason",
    "root_area_um2": "root area inside the outline as drawn (µm²); in excluded arcs the outline may follow damage",
    "root_area_est_um2": f"root area with excluded arcs filled: the root edge interpolated across each arc (µm²); "
                         f"= root_area_um2 when nothing is excluded; blank under {P.MIN_EST_DEG} degrees kept",
    "root_diameter_um": "diameter of the circle with root_area_est_um2 (µm)",
    "stele_area_um2": "stele area, whole section (µm²); the stele is never excluded",
    "cortex_area_kept_um2": "root minus stele in the kept angles only (µm²): measured, not estimated",
    "cortex_area_est_um2": "root_area_est_um2 minus stele_area_um2 (µm²)",
    "stele_share": "stele_area / root_area_est (0-1)",
    "cortex_stele_ratio": "cortex_area_est / stele_area",
    "mx_n": "metaxylem vessels, whole section",
    "mx_area_mean_um2": "mean area of one metaxylem vessel (µm²)",
    "mx_area_median_um2": "median area of one metaxylem vessel (µm²)",
    "mx_area_max_um2": "area of the largest metaxylem vessel (µm²)",
    "mx_area_total_um2": "summed metaxylem vessel area (µm²)",
    "axial_conductance_m4_MPa_s": "theoretical axial conductance of the metaxylem, Hagen-Poiseuille: pi x sum(r^4) / "
                                  "(8 x viscosity), r from each vessel's area, water at 20 C (Tyree & Zimmermann "
                                  "2002); m^4 MPa^-1 s^-1. A maximum: no end walls or pits",
    "n_quarters_used": f"quarters with at least {P.MIN_QUARTER_DEG} degrees kept (the quarter means use these)",
    "ccfn": "cortical cell file number, gap-aware: on each ray (every 5 degrees) the file number of the outermost "
            "cortical cell, a gap between cortical cells (lacuna, missed cell) counted as the cells it would hold (a "
            "gap at the stele edge is not); median over the rays. "
            "Per image: mean of the quarters",
    "cortex_width_um": "root radius minus stele radius, mean over the kept degrees (µm). Per image: mean of the quarters",
    "median_cortex_cell_area_um2": "median area of the kept cortical cells (µm²). Per image: mean of the quarters",
    "aerenchyma_share": "lacuna pixels / cortex pixels in the kept angles (0-1). Per image: mean of the quarters",
    "quarter": "Q1 = 0-90 degrees (image right to top), then counter-clockwise, around the stele centre",
    "kept_deg": f"degrees of this quarter kept; under {P.MIN_QUARTER_DEG} the quarter is blank",
    "n_rays": "rays (every 5 degrees) the quarter's ccfn is the median of",
    "ccfn_cells_present": "the same rays counting only the cells present (no gap counted; = the old n_cortical_layers "
                          "rule). Far below ccfn with no aerenchyma = gaps where cells were missed: check the cells",
    "cortex_area_um2": "root minus stele in this quarter (µm²); only when the whole quarter is kept (90 degrees)",
    "lacuna_area_um2": "aerenchyma lacuna area in this quarter (µm²); only when the whole quarter is kept",
    "n_lacunae": "aerenchyma lacunae with their centre in this quarter; only when the whole quarter is kept",
    "n_lacunae_drawn": "of n_lacunae, the ones drawn by hand (they count whatever the automatic rules say)",
}
TRAIT_SHEETS = {
    "Per image": "one row per image. Stele and metaxylem: whole section. Root area: as drawn, and estimated with "
                 "damaged arcs filled. Cortex traits: the mean of the quarters with enough kept",
    "Per quarter": f"one row per image x quarter. Blank under {P.MIN_QUARTER_DEG} degrees kept. Shares, medians, "
                   "widths and ccfn from the kept angles; areas and counts only when all 90 degrees are kept. "
                   "Nothing here is estimated",
    "Cell area by file": "one row per image: mean cell area (µm²) in each cell file, file 1 = next to the stele, the "
                         "outermost files hold the epidermis (the pattern tissue: cortex + epidermis). A cell's file = the "
                         "median of its numbers on rays every 0.5 degrees in the kept angles, gap-aware",
    "Cells per file": "cells in each file, all the way round: blank unless the whole section is kept (360 degrees), "
                      f"and blank for outer files that do not go all round (more than {P.FILE_COMPLETE_PCT}% of rays "
                      "end before them)",
}


def trait_rows(meta, status, s, region, props, mxp, tr, st, centre, arcs, lac, lmask, cells, ray_files=()):
    """(per-image row, per-quarter rows, file-area row, file-count row) for traits_DATE.xlsx. props: after
    exclude_sectors, with the file column; ray_files: cell_files(...).attrs["ray_files"]. Cells per file: only with
    the whole section kept (360 degrees), and only the files that go all round (P.complete_files)."""
    um2, um = s * s, s
    stele = float((region == 2).sum())
    found = bool(st["stele_found"])
    root_full = P.root_area_filled(region, centre, arcs)
    rays = P.ccfn_rays(cells, region, props, centre, arcs)
    if found:
        qs = P.quarter_traits(region, props, centre, arcs, lac, lmask, rays)
    else:                                   # no stele: no cortex numbers
        qs = [{"quarter": f"Q{q}", "kept_deg": P.quarter_kept_deg(arcs, q)} for q, *_ in P.QUARTERS]
    qrows = []
    for q in qs:
        r = {**meta, "quarter": q["quarter"], "kept_deg": q["kept_deg"]}
        for k in ("ccfn", "ccfn_cells_present", "n_rays"):
            r[k] = q.get(k, np.nan)
        r["cortex_width_um"] = q.get("cortex_width_px", np.nan) * um
        r["median_cortex_cell_area_um2"] = q.get("median_cortex_cell_area_px", np.nan) * um2
        r["cortex_area_um2"] = q.get("cortex_area_px", np.nan) * um2
        r["aerenchyma_share"] = q.get("aerenchyma_share", np.nan)
        r["lacuna_area_um2"] = q.get("lacuna_area_px", np.nan) * um2
        r["n_lacunae"] = q.get("n_lacunae", np.nan)
        r["n_lacunae_drawn"] = q.get("n_lacunae_drawn", np.nan)
        qrows.append(r)
    used = pd.DataFrame([r for r in qrows if r["kept_deg"] >= P.MIN_QUARTER_DEG])
    qmean = lambda k: float(used[k].mean()) if len(used) and used[k].notna().any() else np.nan
    va = mxp.area_px.to_numpy(float) if found and len(mxp) else np.array([])
    img = {**meta, "status": status, "um_per_px": s, "angle_kept_deg": st["angle_kept_deg"],
           "excluded_sectors": st["excluded_sectors"],
           "root_area_um2": tr["root_area_px"] * um2, "root_area_est_um2": root_full * um2,
           "root_diameter_um": 2 * np.sqrt(root_full / np.pi) * um,
           "stele_area_um2": stele * um2 if found else np.nan,
           "cortex_area_kept_um2": st["cortex_area_kept_px"] * um2 if found else np.nan,
           "cortex_area_est_um2": (root_full - stele) * um2 if found else np.nan,
           "stele_share": stele / root_full if found else np.nan,
           "cortex_stele_ratio": (root_full - stele) / stele if found else np.nan,
           "mx_n": len(va) if found else np.nan,
           "mx_area_mean_um2": va.mean() * um2 if len(va) else np.nan,
           "mx_area_median_um2": float(np.median(va)) * um2 if len(va) else np.nan,
           "mx_area_max_um2": va.max() * um2 if len(va) else np.nan,
           "mx_area_total_um2": va.sum() * um2 if found else np.nan,
           "axial_conductance_m4_MPa_s": P.axial_conductance(va, s) if found else np.nan,
           "n_quarters_used": len(used),
           **{k: qmean(k) for k in ("ccfn", "cortex_width_um", "median_cortex_cell_area_um2", "aerenchyma_share")}}
    c = props[props.tissue.isin(P.PATTERN_TISSUES)].dropna(subset=["file"]) if found else props.iloc[:0]
    k = np.floor(c.file + 0.5).astype(int)
    g = c.area_px.groupby(k).agg(["mean", "size"])
    g = g.reindex(range(1, int(g.index.max()) + 1)) if len(g) else g
    farea = {**meta, **{f"file_{i}": v * um2 for i, v in g["mean"].items()}}
    whole_ring = st["angle_kept_deg"] >= 360 - 1e-6
    n_full = P.complete_files(ray_files)
    fn = {**meta, **{f"file_{i}": (int(0 if pd.isna(v) else v) if whole_ring and i <= n_full else np.nan)
                     for i, v in g["size"].items()}}
    return img, qrows, farea, fn


CELLS = {   # the two cell CSVs (described in the traits README)
    "cell": "the cell's id in the image's mask (metaxylem: vessel number, its own numbering)",
    "tissue": "cell type: cortex | epidermis (the outermost cells, by position) | stele | metaxylem | air_space "
              "(small space at a cell junction) | unassigned (not connected to the cortex: debris, fragments)",
    "kept": "yes = centre in the kept angles; no = in an excluded (damaged) arc. Stele and metaxylem are always yes",
    "x_px": "centre, pixels from the image's left edge",
    "y_px": "centre, pixels from the image's top edge",
    "angle_rad": "direction of the centre from the stele centre, radians, counter-clockwise from the image's right",
    "r_um": "distance of the centre from the stele centre (µm)",
    "centre_x_px": "the image's centre for angles and distances (the stele centre, or the one set by hand), pixels "
                   "from the left edge: x_px - centre_x_px, centre_y_px - y_px give the cell's place in the root",
    "centre_y_px": "the same, pixels from the top edge",
    "dist_from_stele_um": "distance from the stele edge to the cell centre (µm), along the line from the centre "
                          "through the cell (on an oval stele up to ~0.2 cell diameters longer than the shortest "
                          "distance on the flanks)",
    "r_norm": "radial position: 0 = stele edge, 1 = root edge, both read along the cell's own direction",
    "file": "estimated cell file number (1 = next to the stele; cortex and epidermis only): the median of the "
            "cell's numbers on rays every 0.5 degrees, a gap between cells counted as the cells it would hold",
    "area_px": "cell area (pixels, to mid-wall)",
    "area_um2": "cell area (µm²)",
}
CURVES = {
    "quarter": "Q1-Q4 (Q1 = 0-90 degrees, image right to top, counter-clockwise); image = the mean of the quarters "
               "(its kept_deg and n_cells = the sums over those quarters)",
    "kept_deg": f"degrees of the quarter kept; under {P.MIN_QUARTER_DEG} the quarter is blank",
    "n_cells": "cells (cortex + epidermis, kept) behind the row",
    "min_cells_in_bin": f"fewest cells in any of the {N_BINS} bins; under {P.MIN_CELLS_PER_BIN} the curve is noisy there",
    "peak_height": "height of the fitted curve's peak: cell size relative to the image's median cell (1 = median)",
    "peak_pos": "where the peak sits: 0 = stele edge, 1 = root edge",
    "rise_slope": "(peak - first value) / distance from the first bin to the peak",
    "decay_slope": "slope from the peak over the next 20% of the cortex (local, so a late upturn cannot bend it)",
    "outer_rise": "last value - lowest value after the peak: ~0 for a plain hump, > 0 when the outer cells grow again",
    "bin": f"radial bin, 1 = next to the stele ... {N_BINS} = the root edge (equal steps of r_norm)",
    "x_position": "bin centre in r_norm (0 = stele edge, 1 = root edge)",
    "median_area_um2": "median cell area in the bin (µm²)",
    "rel": "median cell area in the bin / the image's median cell (the shape of the pattern, size removed)",
    "few_cells": f"TRUE = fewer than {P.MIN_CELLS_PER_BIN} cells in the bin",
    "fitted": "the smoothing spline (GCV, weights = cells per bin) through rel, at x_position 0, 0.05 ... 1 "
              "(blank outside the data): the input for fPCA",
    "lacuna_share": "lacuna pixels / cortex pixels in the bin, kept angles",
}
CURVE_SHEETS = {
    "Curve features": "one row per image x quarter (+ quarter = image, the mean of the quarters): features of the "
                      "fitted curve of relative cell size from the stele out (cortex + epidermis, kept cells)",
    "Profile": f"the binned curve itself: one row per image x quarter x bin ({N_BINS} equal r_norm bins)",
    "Fitted curves": "the fitted curve at 21 points, per image x quarter (long format)",
    "Aerenchyma by position": f"one row per image x bin: where the lacunae sit (whole image, kept angles)",
}


def _old_pattern(f: Path) -> bool:
    """A cortex_pattern workbook in the pre-2026-11-10 layout (it has per_image_profile)."""
    from openpyxl import load_workbook
    wb = load_workbook(f, read_only=True)
    try:
        return "per_image_profile" in wb.sheetnames
    finally:
        wb.close()


def write_book(path: Path, front: dict, tabs: dict, notes: dict, descr: dict, groups, info, extra_cols=()):
    """front: sheets before the README; tabs: data sheets {name: df}; notes: one line per sheet; descr: column ->
    meaning; extra_cols: (sheet, column, unit, meaning) rows for files outside the workbook (the CSVs)."""
    rows = [(name, "", "", note) for name, note in notes.items()]
    for name, df in tabs.items():
        for c in df.columns:
            what = (descr.get(c) or describe(c) or ("grouping column from samples.xlsx" if c in groups else "") or
                    (f"cell file {c[5:]} (1 = next to the stele)" if c.startswith("file_") else ""))
            unit = ("µm²" if name == "Cell area by file" else "cells") if c.startswith("file_") else \
                "m⁴ MPa⁻¹ s⁻¹" if c.startswith("axial_conductance") else unit_of(c)
            rows.append((name, c, unit, what))
    readme = pd.DataFrame([*rows, *extra_cols], columns=["sheet", "column", "unit", "what it is"])
    with pd.ExcelWriter(path) as w:
        for name, df in front.items():
            df.to_excel(w, sheet_name=name, index=False)
        readme.to_excel(w, sheet_name="README", index=False)
        for name, df in tabs.items():
            df.to_excel(w, sheet_name=name, index=False)
        info.to_excel(w, sheet_name="run_info", index=False)
    return readme


def write_traits(path: Path, start, check, per_image, per_quarter, file_area, file_n, groups, info, today):
    cell_cols = [(f"{kind}_{today}.csv", c, "" if c in ("cell", "tissue", "kept", "file") else unit_of(c) or
                  ("radians" if c == "angle_rad" else "pixels" if c.endswith("_px") else "0-1"), CELLS[c])
                 for kind, cols in (("cortex_cells", CORTEX_CSV), ("all_cells", ALL_CSV)) for c in cols
                 if c in CELLS]
    cell_cols = [(f"cortex_cells_{today}.csv", "", "", "one row per kept cortical cell; image_file and the grouping "
                  "columns first"), (f"all_cells_{today}.csv", "", "", "one row per segmented object, every tissue, "
                  "kept or not"), *cell_cols]
    return write_book(path, {"START_HERE": start},
                      {"Check": check, "Per image": per_image, "Per quarter": per_quarter,
                       "Cell area by file": file_area, "Cells per file": file_n},
                      {"Check": "one row per image: flags = plain reasons to look at the image before using it", 
                       **TRAIT_SHEETS}, TRAITS, groups, info, cell_cols)


def write_curves(path: Path, prof, feats, curves, aer, t_q, groups, info):
    """cortex_pattern_DATE.xlsx: the quarter rule of the traits file applied to the curves. Position basis, relative
    size only; the file and distance bases, _abs, _rel_cortex and whole-section fits are in full/cortex_pattern_full (summarise.py --full)."""
    ids = ["image_file", *groups]
    qs = [f"Q{q}" for q, *_ in P.QUARTERS]
    pos = feats[feats.basis == "position"] if len(feats) else feats
    fc = ["peak_height", "peak_pos", "rise_slope", "decay_slope", "outer_rise"]
    f = pos[pos.scope.isin([*qs, "quarter_mean"])][[*ids, "scope", "kept_deg", "n_cells", "min_cells_in_bin",
                                                     *[c + "_rel" for c in fc]]] if len(pos) else pd.DataFrame()
    f = f.rename(columns={"scope": "quarter", **{c + "_rel": c for c in fc}}).replace({"quarter": {"quarter_mean": "image"}})
    if len(f):                                  # the image row: kept degrees and cells of the quarters it averages
        qq = f[f.quarter.isin(qs)].groupby("image_file").agg(k=("kept_deg", "sum"), n=("n_cells", "sum"),
                                                              m=("min_cells_in_bin", "min"))
        im = f.quarter == "image"
        f.loc[im, "kept_deg"] = f.loc[im, "image_file"].map(qq.k).values
        f.loc[im, "n_cells"] = f.loc[im, "image_file"].map(qq.n).values
        f.loc[im, "min_cells_in_bin"] = f.loc[im, "image_file"].map(qq.m).values
    blank = t_q[t_q.kept_deg < P.MIN_QUARTER_DEG][[*ids, "quarter", "kept_deg"]] if len(t_q) else t_q
    f = pd.concat([f, blank], ignore_index=True)
    order = {q: i for i, q in enumerate([*qs, "image"])}
    f = f.sort_values(["image_file", "quarter"], key=lambda s: s.map(order) if s.name == "quarter" else s)
    pr = prof[(prof.basis == "position") & prof.scope.isin(qs)] if len(prof) else prof
    pr = pr[[*ids, "scope", "bin", "x_position", "n_cells", "median_area_um2", "rel", "few_cells"]] \
        .rename(columns={"scope": "quarter"}) if len(pr) else pd.DataFrame()
    if len(pr):
        im = pr.groupby([*ids, "bin", "x_position"], dropna=False, sort=False) \
            .agg(n_cells=("n_cells", "sum"), median_area_um2=("median_area_um2", "mean"), rel=("rel", "mean")) \
            .reset_index().assign(quarter="image")
        im["few_cells"] = im.n_cells < P.MIN_CELLS_PER_BIN
        pr = pd.concat([pr, im], ignore_index=True)
        pr["_b"] = pr.bin.astype(int)
        pr = pr.sort_values(["image_file", "quarter", "_b"], key=lambda s: s.map(order) if s.name == "quarter" else s) \
            .drop(columns="_b")
    cv = curves[(curves.basis == "position") & curves.scope.isin(qs) & (curves.axis == "rel")] if len(curves) else curves
    cv = cv[[*ids, "scope", "x_position", "fitted"]].rename(columns={"scope": "quarter"}) if len(cv) else pd.DataFrame()
    if len(cv):
        cv = pd.concat([cv, cv.groupby([*ids, "x_position"], dropna=False, sort=False).fitted.mean().reset_index()
                        .assign(quarter="image")], ignore_index=True)
        cv = cv.sort_values(["image_file", "quarter", "x_position"],
                            key=lambda s: s.map(order) if s.name == "quarter" else s)
    ae = aer[[*ids, "bin", "x", "lacuna_share"]].rename(columns={"x": "x_position"}) if len(aer) else aer
    return write_book(path, {}, {"Curve features": f, "Profile": pr, "Fitted curves": cv, "Aerenchyma by position": ae},
                      CURVE_SHEETS, CURVES, groups, info)


CORTEX_CSV = ["cell", "quarter", "x_px", "y_px", "centre_x_px", "centre_y_px", "angle_rad", "r_um",
              "dist_from_stele_um", "r_norm", "file", "area_px", "area_um2"]
ALL_CSV = ["cell", "tissue", "kept", "quarter", "x_px", "y_px", "centre_x_px", "centre_y_px", "angle_rad", "r_um",
           "dist_from_stele_um", "r_norm", "file", "area_px", "area_um2"]


def write_cell_csvs(out_dir: Path, today, all_cells, groups):
    """cortex_cells_DATE.csv (kept cortical cells) and all_cells_DATE.csv (every object, with tissue and kept)."""
    ids = ["image_file", *groups]
    a = all_cells.rename(columns={"label": "cell", "x": "x_px", "y": "y_px"})
    exc = a.tissue.astype(str).str.endswith("_excluded")
    a = a.assign(tissue=a.tissue.astype(str).str.removesuffix("_excluded"), kept=np.where(exc, "no", "yes"),
                 quarter="Q" + a.quarter.astype(int).astype(str))
    a[[*ids, *ALL_CSV]].to_csv(out_dir / f"all_cells_{today}.csv", index=False)
    a[(a.tissue == "cortex") & (a.kept == "yes")][[*ids, *CORTEX_CSV]] \
        .to_csv(out_dir / f"cortex_cells_{today}.csv", index=False)


def make_samples(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        rel = Path(str(np.load(f)["image_file"]))
        parts = list(rel.parent.parts)
        plant = (parts[-1] if len(parts) >= 3 else rel.stem).lower()   # "P1" and "p1" are one plant
        groups = parts[:-1] if len(parts) >= 3 else parts
        rows.append({"image_file": str(rel), "plant": plant, "um_per_px": None, "include": "yes", "epidermis": None,
                     **{f"group_{k + 1}": g for k, g in enumerate(groups)}})
    return pd.DataFrame(rows)


def write_samples(anatomy_dir: Path) -> Path:
    """samples.xlsx for the user to fill. Rows the user has are never changed; an image measured after the file was
    written gets a new row (before 2026-11-10 it was left out, and Make results then skipped it without a word).
    New rows take the folder names into the group columns by position (so a renamed group_1 still fills)."""
    sfile = anatomy_dir / "samples.xlsx"
    masks = sorted((anatomy_dir / "masks").glob("*.npz"))
    if not sfile.exists():
        make_samples(masks).to_excel(sfile, index=False)
        print(f"\nWrote {sfile}\nFill in um_per_px (microns per pixel) for every image, rename group_1, group_2 ... "
              "(e.g. species, genotype), check plant, then click Make results.")
        return sfile
    s = pd.read_excel(sfile, dtype={"image_file": str, "plant": str})
    if "image_file" not in s:
        return sfile
    have = set(s.image_file.dropna())
    new = make_samples([f for f in masks if str(np.load(f)["image_file"]) not in have])
    if len(new):
        gcols = [c for c in s.columns if c not in FIXED and not str(c).lower().startswith(NOTE_COLS)]
        new = new.rename(columns={f"group_{k + 1}": c for k, c in enumerate(gcols)})
        pd.concat([s, new], ignore_index=True).to_excel(sfile, index=False)
        print(f"Added {len(new)} newly measured image(s) to {sfile}: set their um_per_px.")
    return sfile


def px_last(df):
    """Pixel columns at the end: the µm columns are the ones to read."""
    px = [c for c in df.columns if c.endswith("_px") and c != "um_per_px"]
    return df[[c for c in df.columns if c not in px] + px]


def describe(col):
    if col in COLUMNS:
        return COLUMNS[col]
    if col.endswith("_px"):
        base = col[:-3]
        for unit in ("_um2", "_um"):
            if base + unit in COLUMNS:
                return COLUMNS[base + unit].split("(µm")[0].strip() + " (pixels)"
        return ""
    for sfx in FEATURE_SETS:                           # profile features: same meaning on another scale
        if col.endswith(sfx) and col[:-len(sfx)] + "_rel" in COLUMNS:
            return COLUMNS[col[:-len(sfx)] + "_rel"].split(";")[0] + f" ({sfx[1:]})"
    return ""


def to_um(d: dict, s: float) -> dict:
    """Pixel traits -> µm: *area*_px -> *_um2, *width_px -> *_um (NaN without a scale)."""
    out = {}
    for k, v in d.items():
        if k.endswith("_px") and "area" in k:
            out[k[:-3] + "_um2"] = v * s * s
        elif k.endswith("_px") and ("width" in k or "diameter" in k):
            out[k[:-3] + "_um"] = v * s
    return out


def main(anatomy_dir: Path, image_figures: bool = False, full: bool = False):
    """The project's tables (xlsx) and group figures. image_figures: also one cortex pattern figure per image
    (slow for big sets; the editors draw it live for the image on screen).
    Cortex numbers come from the kept angles only (excluded sectors out, cells by their centre); stele and metaxylem
    are whole-section. Unit of analysis: the plant. Per image first (medians per bin), then plant means; cells are
    never pooled across images."""
    settings = json.loads((anatomy_dir / "settings.json").read_text())
    files = sorted((anatomy_dir / "masks").glob("*.npz"))
    sfile = write_samples(anatomy_dir)
    samples = pd.read_excel(sfile, dtype={"image_file": str, "plant": str})
    samples["include"] = samples["include"].fillna("yes").astype(str).str.lower()
    notes = [c for c in samples.columns if str(c).lower().startswith(NOTE_COLS)]
    groups = [c for c in samples.columns if c not in FIXED and c not in notes]
    samples = samples.set_index("image_file")
    tech_rep = samples.groupby(["plant", *groups], dropna=False).cumcount() + 1   # image number within a plant

    angle = float(settings.get("quadrant_angle_deg", 0))       # only to read old dropped quarters as sectors
    fig_dir = anatomy_dir.parent / "results" / "figures" / "cortex_pattern"
    if image_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
    img_rows, cell_rows, prof_rows, all_rows, feat_rows, curve_rows, lac_rows, aer_rows = [], [], [], [], [], [], [], []
    grid = np.round(np.linspace(0, 1, 21), 2)
    skipped = []
    t_img, t_q, t_fa, t_fn = [], [], [], []
    for f in files:
        z = np.load(f)
        rel = str(z["image_file"])
        if rel not in samples.index or samples.at[rel, "include"] == "no":
            skipped.append((rel, "include = no" if rel in samples.index else "not in samples.xlsx"))
            continue
        cells, region, mx, status = P.load_state(f)
        marks = P.load_not_lacuna(f)              # reviewer's 'not a lacuna' marks
        drawn = P.load_drawn_lacuna(f)            # reviewer's drawn lacunae (count whatever the rules say)
        centre = P.centre_of(region, P.load_centre(f))
        props, mxp, tr = P.derive(cells, region, mx, angle, centre)
        arcs = P.load_sectors(f, angle)
        props = P.exclude_sectors(props, arcs)
        st = P.sector_traits(cells, region, props, centre, arcs)
        s = samples.at[rel, "um_per_px"]
        s = float(s) if pd.notna(s) else np.nan
        tech = int(tech_rep[rel])
        meta = {"image_file": rel, "plant": samples.at[rel, "plant"], **{g: samples.at[rel, g] for g in groups}}
        whole = {k: tr[k] for k in ("root_area_px", "root_diameter_eq_px", "stele_area_px", "mx_n", "mx_mean_area_px",
                                    "mx_total_area_px")}

        fl = P.cell_files(cells, props, centre, arcs)
        props = props.join(fl, on="label")
        props["quarter"] = P.quarter_of(np.degrees(props.angle_rad))
        c_all = props[props.tissue.isin(P.PATTERN_TISSUES)]          # rel reference: the tissue median
        ref = c_all.area_px.median() if len(c_all) else np.nan
        qs = [int(q) for q in st["quarters_used"].split(",") if q]
        scopes = {"section": (props, st["angle_kept_deg"])}
        scopes.update({f"Q{q}": (props[props.quarter == q], P.quarter_kept_deg(arcs, q)) for q in qs})
        um = not np.isnan(s)
        unit = "um2" if um else "px"
        base = {**meta, "tech_rep": tech, "status": status}
        prof = None
        for basis in ("position", "file", "distance"):
            if basis == "distance" and not um:                 # µm bins: needs the scale
                continue
            qfeat = []
            for scope, (sub, kdeg) in scopes.items():
                pr = P.sector_profile(sub, N_BINS, ref, basis, st["n_cortical_layers"], um_per_px=s)
                pr["median_area_um2"] = pr.median_area_px * s * s
                if basis == "position" and scope == "section":
                    prof = pr
                prof_rows.append(pr.rename(columns={"x": "x_position"}).assign(**base, basis=basis, scope=scope))
                allp = pr.dropna(subset=["rel"])
                w_all = allp.n_cells.where(allp.n_cells > 0)
                pc = P.sector_profile(sub, N_BINS, ref, basis, tissues=("cortex",), um_per_px=s).dropna(subset=["rel"])
                ycol = "median_area_um2" if um else "median_area_px"
                e_sub = sub[sub.tissue == "epidermis"]
                f = {**base, "basis": basis, "scope": scope, "area_unit": unit, "kept_deg": kdeg,
                     "low_coverage": st["low_coverage"] if scope == "section" else kdeg < P.MIN_QUARTER_DEG,
                     "n_cells": int(pr.n_cells.sum()), "min_cells_in_bin": int(pr.n_cells.min()) if len(pr) else 0,
                     "epidermis_deg": P.epidermis_deg(sub, arcs),
                     "epidermis_rel": e_sub.area_px.median() / ref if len(e_sub) else np.nan,
                     "epidermis": samples.at[rel, "epidermis"] if "epidermis" in samples.columns else None}
                for sfx, (x, y, ww) in {"_rel": (allp.x, allp.rel, w_all), "_abs": (allp.x, allp[ycol], w_all),
                                        "_rel_cortex": (pc.x, pc.rel, pc.n_cells.where(pc.n_cells > 0))}.items():
                    f.update({k + sfx: v for k, v in PP.profile_features(x, y, ww).items()})
                feat_rows.append(f)
                if scope != "section":
                    qfeat.append(f)
                sp = PP.spline(allp.x, allp.rel, w_all, at=grid)
                curve_rows += [{**base, "basis": basis, "scope": scope, "axis": "rel", "x_position": xv, "fitted": yv}
                               for xv, yv in zip(grid, sp[1] if sp is not None else [np.nan] * len(grid))]
            if qfeat:
                q = pd.DataFrame(qfeat)
                cols = [c for c in q.columns if c.endswith(FEATURE_SETS)]
                for scope, agg in (("quarter_mean", "mean"), ("quarter_sd", "std")):
                    feat_rows.append({**base, "basis": basis, "scope": scope, "area_unit": unit,
                                      "n_quarters": len(q), **q[cols].agg(agg).to_dict()})
        e_all = props[props.tissue == "epidermis"]
        lac, aprof, n_gaps, lmask = P.lacunae(cells, region, mx, props, centre, arcs, N_BINS, not_lacuna=marks,
                                              drawn=drawn, mask=True)
        lt = P.lacuna_traits(lac, n_gaps, st["cortex_area_kept_px"])
        ti, tq, tfa, tfn = trait_rows({"image_file": rel, **{g: samples.at[rel, g] for g in groups}}, status, s,
                                      region, props, mxp, tr, st, centre, arcs, lac, lmask, cells,
                                      fl.attrs["ray_files"])
        t_img.append(ti)
        t_q += tq
        t_fa.append(tfa)
        t_fn.append(tfn)
        lac_rows.append(lac.assign(**base, area_um2=lac.area_px * s * s))
        aer_rows.append(aprof.assign(**base))
        if not st["stele_found"]:                       # no stele: whole-section stele traits blank too, not 0
            whole.update({k: np.nan for k in P.NEEDS_STELE if k in whole})
            lt.update({k: np.nan for k in P.NEEDS_STELE if k in lt})
        img_rows.append({**meta, "tech_rep": tech, "status": status, "um_per_px": s,
                         "epidermis": samples.at[rel, "epidermis"] if "epidermis" in samples.columns else None,
                         **{n: samples.at[rel, n] for n in notes},
                         **whole, **st, **lt,
                         "epidermis_rel": e_all.area_px.median() / ref if len(e_all) else np.nan,
                         "min_cells_in_bin": int(prof.n_cells.min()),
                         **to_um({**whole, **st, **lt}, s)})
        ycol = "median_area_um2" if um else "median_area_px"
        if image_figures:
            PP.image_figure(prof.assign(y=prof[ycol]), "Cell area (µm²)" if um else "Cell area (px)",
                            f"{rel}  ·  {status}  ·  {st['angle_kept_deg']:.0f}° kept",
                            fig_dir / (Path(rel).with_suffix("").as_posix().replace("/", "__") + ".png"))

        mxt = pd.DataFrame(measure.regionprops_table(mx, properties=("label", "area", "centroid", "axis_major_length",
                                                                     "axis_minor_length"))) if mx.any() else pd.DataFrame()
        if len(mxt):
            mxt = mxt.rename(columns={"centroid-0": "y", "centroid-1": "x", "area": "area_px",
                                      "axis_major_length": "major_px", "axis_minor_length": "minor_px"})
            mxt = mxt.join(P.radial_position(mxt, region, centre))
            mxt["tissue"] = "metaxylem"
        allc = pd.concat([props, mxt], ignore_index=True)
        allc["sector"] = np.where(allc.tissue.astype(str).str.endswith("_excluded"), "excluded", "kept")
        allc["ring"] = allc.layer if "layer" in allc else np.nan
        allc["quarter"] = P.quarter_of(np.degrees(allc.angle_rad))
        for k in ("file", "file_rel", "n_rays"):
            if k not in allc:
                allc[k] = np.nan
        allc = allc.assign(area_um2=allc.area_px * s * s, major_um=allc.major_px * s, minor_um=allc.minor_px * s,
                           r_um=allc.r_px * s, dist_from_stele_um=allc.dist_from_stele_px * s,
                           angle_deg=np.degrees(allc.angle_rad), centre_x_px=centre[1], centre_y_px=centre[0])
        all_rows.append(allc.assign(**meta, tech_rep=tech, status=status)[
            [*meta, "tech_rep", "status", "label", "tissue", "sector", "ring", "x", "y",
             "angle_rad", "angle_deg", "quarter", "r_px", "r_um", "r_norm", "dist_from_stele_um", "file", "file_rel",
             "n_rays", "area_px", "area_um2", "major_um", "minor_um", "centre_x_px", "centre_y_px"]])

        c = props[props.tissue == "cortex"].copy()
        c["area_um2"] = c.area_px * s * s
        c["major_um"], c["minor_um"] = c.major_px * s, c.minor_px * s
        c["dist_from_stele_um"] = c.dist_from_stele_px * s
        c["radial_bin"] = (np.clip(c.r_norm, 0, 0.999) * N_BINS).astype(int) + 1
        cell_rows.append(c.assign(**meta, status=status)[[*meta, "status", "quarter", "r_norm", "radial_bin",
                                                          "file", "file_rel", "dist_from_stele_um",
                                                          "label", "area_px", "area_um2", "major_um", "minor_um"]])

    keys = ["plant", *groups]
    first = lambda df, cols: df[[*cols, *[c for c in df.columns if c not in cols]]]
    per_image = pd.DataFrame(img_rows)
    per_cell = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    per_image_profile = first(pd.concat(prof_rows, ignore_index=True),
                              ["image_file", *keys, "tech_rep", "status", "basis", "scope", "bin"]) \
        if prof_rows else pd.DataFrame()
    pos_section = per_image_profile[(per_image_profile.basis == "position") & (per_image_profile.scope == "section")] \
        if len(per_image_profile) else per_image_profile
    per_plant_profile = (pos_section.groupby([*keys, "bin", "x_position"], dropna=False)
                         .agg(n_images=("image_file", "size"), median_area_um2=("median_area_um2", "mean"),
                              rel=("rel", "mean"), n_cells=("n_cells", "sum")).reset_index()
                         .sort_values([*keys, "x_position"])) if len(per_image_profile) else pd.DataFrame()
    num = [c for c in per_image.columns if c.endswith(("_px", "_um2", "_um", "_deg", "_est", "_rel"))
           or c.startswith(("n_", "mx_", "min_"))]
    per_plant = (per_image.groupby(keys, dropna=False)
                 .agg(n_images=("image_file", "size"),
                      n_reviewed=("status", lambda s: int((s == "reviewed").sum())),
                      n_low_coverage=("low_coverage", "sum"),
                      **{c: (c, "mean") for c in num}).reset_index())
    all_cells = pd.concat(all_rows, ignore_index=True)
    profile_features = first(pd.DataFrame(feat_rows), ["image_file", *keys, "tech_rep", "status", "basis", "scope"])
    lacunae_t = first(pd.concat(lac_rows, ignore_index=True), ["image_file", *keys, "tech_rep", "status"]) \
        if lac_rows else pd.DataFrame()
    aerenchyma_profile = first(pd.concat(aer_rows, ignore_index=True), ["image_file", *keys, "tech_rep", "status"]) \
        if aer_rows else pd.DataFrame()
    spline_curves = pd.DataFrame(curve_rows)
    out_dir = anatomy_dir.parent / "results"
    out_dir.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()
    files_out = {
        f"full/cortex_pattern_full_{today}.xlsx": {
            "per_image_profile": (per_image_profile, f"one row per image x basis x scope x bin: every cell outside the "
                                  f"endodermis in the kept angles (centre rule), position: r_norm in {N_BINS} bins (1 = "
                                  "next to the stele); file: one row per cell file. rel = bin median / the image's median "
                                  f"cell area. few_cells = fewer than {P.MIN_CELLS_PER_BIN} cells in the bin"),
            "profile_features": (profile_features, "one row per image x basis (position | file) x scope (section, "
                                 "Q1..Q4, quarter_mean, quarter_sd): smoothing spline (GCV) through the bin medians "
                                 "(weights = cells per bin); features read off the curve. The pattern = every cell "
                                 "outside the endodermis (cortex + epidermis, one tissue). Basis distance = µm from the stele edge (cross-check for lost outer tissue). _rel = relative size (the "
                                 "main pattern), _abs = the same in area_unit, _rel_cortex = without the cells tagged "
                                 "epidermis (a check). "
                                 "peak_height, peak_pos; rise_slope = (peak - first) / distance; decay_slope = local, "
                                 "over 20% of the range past the peak; outer_rise = last value - minimum after the peak. "
                                 f"NaN = fewer than {PP.MIN_PTS} bins with cells or feature undefined (peak at an end). "
                                 "Check low_coverage and min_cells_in_bin before using a row"),
            "aerenchyma_profile": (aerenchyma_profile, f"one row per image x r_norm bin ({N_BINS} bins, 1 = next to the "
                                   "stele): lacuna pixels / cortex pixels in the kept angles. Tests where aerenchyma "
                                   "forms; plant = unit"),
            "spline_curves": (spline_curves, "the fitted relative-size curves at x_position 0, 0.05 ... 1 (long format; NaN outside "
                              "the data): the input for fPCA"),
            "per_plant_profile": (per_plant_profile, "plant x bin: mean of the image medians (display; analyse "
                                  "per_image_profile with plant as the unit)")},
        f"full/all_cells_{today}.xlsx": {
            "all_cells": (all_cells, "every segmented object: tissue = cortex | stele | epidermis | air_space | "
                          "unassigned | metaxylem, '..._excluded' when its centre is in an excluded sector. angle_rad "
                          "counter-clockwise from the image's right, around the stele centre. r_norm 0 = stele edge, "
                          "1 = root edge along the object's direction. ring = adjacency hops from the stele (gap check, runs high)")},
        f"full/anatomy_summary_{today}.xlsx": {
            "per_plant": (per_plant, "plant means of per_image; n_reviewed = images checked by hand; "
                          "n_low_coverage = images with too few degrees kept"),
            "per_image": (per_image, "per image. Stele, metaxylem, root_area: whole section. Cortex traits: kept angles "
                          "only; _est = scaled to 360 degrees (estimate). angle_kept_deg and excluded_sectors say what "
                          "was taken out and why"),
            "per_cell": (per_cell, "kept cortical cells (centre rule), one per row"),
            "lacunae": (lacunae_t, "one row per aerenchyma lacuna (centre in the kept angles): area, angle, quarter, "
                        "radial position (mean, inner, outer)")},
    }
    info = run_info(anatomy_dir, settings)
    n_plant = per_image.groupby(keys, dropna=False).ngroups
    import re
    earlier = out_dir / "earlier_runs"          # earlier outputs: kept, out of the way (top = this run only)
    for f in out_dir.iterdir():                 # earlier dates, and the old-layout workbooks of any date
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.(xlsx|csv|txt)$", f.name)
        if f.is_file() and m and (m.group(1) != today
                                  or (f.name.startswith(("anatomy_summary", "all_cells_")) and f.suffix == ".xlsx")
                                  or (f.name.startswith("cortex_pattern") and f.suffix == ".xlsx" and _old_pattern(f))):
            earlier.mkdir(exist_ok=True)
            f.replace(earlier / f.name)
    fcols = lambda df: df[[c for c in df.columns if not c.startswith("file_")] +
                          sorted([c for c in df.columns if c.startswith("file_")], key=lambda c: int(c[5:]))]
    t_image = pd.DataFrame(t_img)
    check = check_sheet(t_image, per_image, pd.DataFrame(t_q))
    sh = start_here(today, len(per_image), int((per_image.status == "reviewed").sum()),
                    int(per_image.um_per_px.isna().sum()), skipped,
                    int((check["flags"].str.replace("not reviewed", "").str.strip("; ") != "").sum()))
    write_traits(out_dir / f"traits_{today}.xlsx", sh, check, t_image, pd.DataFrame(t_q), fcols(pd.DataFrame(t_fa)),
                 fcols(pd.DataFrame(t_fn)), groups, info, today)
    write_curves(out_dir / f"cortex_pattern_{today}.xlsx", per_image_profile, profile_features, spline_curves,
                 aerenchyma_profile, pd.DataFrame(t_q), groups, info)
    write_cell_csvs(out_dir, today, all_cells, groups)
    if full:                                    # summarise.py --full: the long workbooks the checks and compare.py read
        (out_dir / "full").mkdir(exist_ok=True)
    for fname, tabs in (files_out.items() if full else ()):
        tabs = {name: (order_cols(df, [*groups, *notes]) if name in ("per_image", "per_plant") else px_last(df), note)
                for name, (df, note) in tabs.items()}
        readme = pd.DataFrame([(k, "", "", "", n) for k, (_, n) in tabs.items()],
                              columns=["sheet", "column", "topic", "unit", "what it is"])
        cols = pd.DataFrame([(name, c, topic_of(c, [*groups, *notes])[2:], unit_of(c), describe(c) or
                              ("grouping column from samples.xlsx" if c in groups else
                               "your note from samples.xlsx" if c in notes else ""))
                             for name, (df, _) in tabs.items() for c in df.columns],
                            columns=["sheet", "column", "topic", "unit", "what it is"])
        with pd.ExcelWriter(out_dir / fname) as w:
            pd.DataFrame({"FULL RECORD": [
                f"Written only with summarise.py --full: every intermediate table, for the checks and compare.py. "
                f"For analysis use traits_{today}.xlsx, cortex_pattern_{today}.xlsx and the cell CSVs in results/."
                ]}).to_excel(w, sheet_name="FULL_RECORD", index=False)
            pd.concat([readme, cols], ignore_index=True).to_excel(w, sheet_name="README", index=False)
            for name, (df, _) in tabs.items():
                df.to_excel(w, sheet_name=name, index=False)
            info.to_excel(w, sheet_name="run_info", index=False)
    paras = methods_text(per_image, n_plant, today).split("\n\n")         # one line per paragraph: pastes into Word
    (out_dir / f"methods_{today}.txt").write_text("\n\n".join(" ".join(x.split("\n")) for x in paras) + "\n")
    if len(per_image_profile) and groups:
        sp, gt = groups[0], groups[-1]            # e.g. species, genotype
        (out_dir / "figures").mkdir(exist_ok=True)
        for spv, d in pos_section.groupby(sp):
            um = d.median_area_um2.notna().any()   # µm² if any image has a scale; images without one are left out
            drop = d.loc[d.median_area_um2.isna(), "image_file"].nunique() if um else 0
            PP.group_figure(d[d.median_area_um2.notna()] if um else d, gt, spv,
                            "median_area_um2" if um else "median_area_px",
                            "Cortical cell area (µm²)" if um else "Cortical cell area (px)",
                            out_dir / "figures" / f"cortex_pattern_{spv}_{today}.png",
                            f"  {drop} image(s) without a scale left out." if drop else "")
    no_scale = per_image.loc[per_image.um_per_px.isna(), "image_file"].tolist()
    if no_scale:
        print(f"\nWARNING: {len(no_scale)} image(s) have no scale (um_per_px empty); their sizes "
              f"stay in pixels and they are left out of the um figures. Type um_per_px in {sfile.name} and run again:")
        print("  " + "\n  ".join(no_scale[:10]) + ("\n  ..." if len(no_scale) > 10 else ""))
    low = per_image.loc[per_image.low_coverage, "image_file"].tolist()
    if low:
        print(f"\nNOTE: {len(low)} image(s) keep fewer than {P.MIN_KEPT_DEG} degrees: low_coverage = TRUE.")
    if skipped:
        print(f"\nNOTE: {len(skipped)} measured image(s) left out: " + "; ".join(f"{r} ({why})" for r, why in skipped))
    print(f"{len(per_image)} of {len(files)} measured images ({(per_image.status == 'reviewed').sum()} reviewed), "
          f"{per_plant.shape[0]} plants -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), image_figures="--image-figures" in sys.argv, full="--full" in sys.argv)
