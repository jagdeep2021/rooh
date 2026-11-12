# Rooh traits: how each number is made now

The current definition of every trait, in the order the columns appear in `per_image`. For why each rule is what it
is, and the evidence, see the decision log (`docs/development/README.md`). This page states the rules; the log
explains how they came about.

**Changing a trait (anyone editing Rooh):** change the code, this page, the column description in `summarise.py`
(`COLUMNS`), and add a dated entry to the decision log, all in the same commit. Then run
`.venv/bin/python tests/check_numbers.py`; it must pass. It fails if a `per_image` column is missing from this page.

---

## 0. The objects everything is built from

| Object | How it is found (automatic proposal; a person corrects it in the editor) |
|---|---|
| **Image** | Brightfield images (dark walls on a bright field) are inverted to bright walls and their black frame blanked; fluorescence / DAPI unchanged. Then contrast-equalised (CLAHE). |
| **Cells** | Cellpose (cpsam model) on the grey image. One label per cell; masks run to mid-wall. |
| **Root outline** | Bright walls (Li threshold on the smoothed image) or any cell, closed (15 px), holes filled, largest piece kept, opened (5 px). |
| **Stele** | Everything inside the outer edge of the endodermis (the endodermis counts as stele). Auto: the cluster of small cells near the root centre (Otsu on log cell area, cells within 0.7 of the root radius), closed, filled, convex hull. **Misses on brightfield: draw it by hand.** |
| **Metaxylem** | Dark lumens inside the stele larger than 5× the median stele cell, solidity > 0.85, eccentricity < 0.85. |
| **Centre** | The stele centroid (the root centroid if there is no stele), unless the user set one. All angles and radial positions are measured from it. |
| **Excluded sectors** | Arcs of angle around the centre taken out by hand, each with a technical reason (torn, folded, smeared / oblique, out of focus, debris, other). **Biology is never a reason.** A cell is in or out by its centre. They affect the cortex, epidermis, air spaces and unassigned cells only; the stele and metaxylem are always whole-section. |

**Tissue of each cell** (one per cell, in this order):
1. **stele**: centre inside the stele outline.
2. **air_space**: smaller than 25% of its neighbours' median area, with at most 5 neighbours (a space at a cell junction, not a cell).
3. **epidermis**: touches the 6 px band inside the root outline. This is a position tag, not a cell type.
4. **cortex**: everything else inside the root that is connected, cell to cell, to the stele side. The trace starts from cells within 6 px of the stele and from the first cell on each 1° ray. Cells it cannot reach are **unassigned** (debris, fragments).
5. **…_excluded**: the same, with the centre in an excluded sector. Kept in `all_cells`, left out of every cortex number.

**Position of a cell:**
- `r_norm` = (r − r_stele(θ)) / (r_root(θ) − r_stele(θ)), where r is the distance from the centre and θ is the cell's own angle. 0 is the stele edge and 1 is the root edge, both read at that angle (1° steps).
- `dist_from_stele` = r − r_stele(θ).

**Scale:** µm per pixel is typed per image in `samples.xlsx`; it is never read from a scale bar. Areas convert by (µm/px)². Without a scale an image stays in pixels and is left out of µm figures.

---

## What Make results writes (since 2026-11-10)

```
results/
  traits_DATE.xlsx          START_HERE, README, Check, Per image, Per quarter, Cell area by file, Cells per file
  cortex_pattern_DATE.xlsx  Curve features, Profile, Fitted curves, Aerenchyma by position
  cortex_cells_DATE.csv     one row per kept cortical cell
  all_cells_DATE.csv        one row per object, every tissue, kept yes/no
  methods_DATE.txt
  figures/
  earlier_runs/             earlier results, and any old-layout workbook (moved here, never deleted)
  full/                     only with `summarise.py --full`: anatomy_summary, all_cells.xlsx, cortex_pattern_full,
                            every intermediate table (the checks and compare.py read these; sections 1-7 below)
```

Per image and per quarter only, never per plant: the user joins plant, treatment and anything else on `image_file`
later. Every row carries `image_file` and the grouping columns from `samples.xlsx`; no other column repeats across
sheets. Every image measured and not set to `include = no` is in every file; `START_HERE` lists any left out and why.

**Check sheet** (traits): one row per image, `flags` = plain reasons to look before using it (not reviewed, no
scale, no stele, blank quarters, root area not estimated, gaps add cell files ≥ 2 in a quarter, giant cortex cells,
missed-cell gaps, epidermis mostly missing, lacunae overruled by hand), plus the counts behind them
(`n_giant_cortex_cells`, `n_gaps_missed_cell`, `epidermis_deg`, `n_lacunae_overridden`, as in section 2).

**cortex_pattern** (the curves, quarter rules as below; position basis, relative size):
- `Curve features`: per image × quarter (+ `image` = the mean of the quarters; its `kept_deg` and `n_cells` are the
  sums): `peak_height`, `peak_pos`, `rise_slope`, `decay_slope`, `outer_rise` of the smoothing spline through the bin
  medians, relative to the image's median cell (section 6). Quarters under 45° kept: a blank row.
- `Profile`: per image × quarter × bin (10 equal r_norm bins): `n_cells`, `median_area_um2`, `rel`, `few_cells`.
- `Fitted curves`: the spline at r_norm 0, 0.05 … 1 (`fitted`), per image × quarter: the input for fPCA.
- `Aerenchyma by position`: per image × bin, `lacuna_share` (kept angles).
The file and distance bases, `_abs`, `_rel_cortex` and whole-section fits are in `full/cortex_pattern_full` (`--full` only).

**Cell CSVs**: `cell`, (`tissue`, `kept`: all_cells only), `quarter`, `x_px`, `y_px`, `angle_rad` (around the stele
centre, counter-clockwise from the image's right), `r_um` (from the stele centre), `r_norm` (0 stele edge, 1 root
edge, along the cell's direction), `file` (estimated cell file number, cortex and epidermis), `area_px`, `area_um2`.

## The traits file (`traits_DATE.xlsx`)

**Rules.** A quarter (Q1 = 0–90°, then counter-clockwise, around the centre) with < 45° kept is blank. Shares,
medians, widths and `ccfn` use the kept angles from 45° up; **absolute areas and counts per quarter only when all 90°
are kept**. Nothing per quarter is estimated. Per-image cortex traits = the mean of the quarters that pass.

**Sheet `Per image`**

| Column | How |
|---|---|
| `angle_kept_deg`, `excluded_sectors` | As in `per_image`. |
| `root_area_um2` | Pixels inside the root outline as drawn (where an arc is excluded the outline may follow the damage). |
| `root_area_est_um2` | **Fill the edge**: kept angles counted pixel by pixel; across each excluded arc the root radius (from the centre, 0.25° steps) is interpolated linearly between the kept radii on either side, and the wedge adds ½ (r + 0.5)² Δθ. Equals `root_area_um2` when nothing is excluded. Blank under 180° kept. (Tested against scaling by angle: `docs/development/area_estimate.py`.) |
| `root_diameter_um` | 2√(`root_area_est` / π). |
| `stele_area_um2` | Whole section. |
| `cortex_area_kept_um2` | Root outside the stele, kept angles only: measured. |
| `cortex_area_est_um2` | `root_area_est` − stele. |
| `stele_share` | stele / `root_area_est`. |
| `cortex_stele_ratio` | `cortex_area_est` / stele. |
| `mx_n`, `mx_area_mean_um2`, `mx_area_median_um2`, `mx_area_max_um2`, `mx_area_total_um2` | Metaxylem vessels, whole section: count and per-vessel area statistics. |
| `axial_conductance_m4_MPa_s` | Hagen–Poiseuille, π Σ r⁴ / (8 η): r from each vessel's area (equivalent circle), η = water at 20 °C (1.002 mPa s); m⁴ MPa⁻¹ s⁻¹ (Tyree & Zimmermann 2002). A theoretical maximum: no end walls, pits or lumen shape. |
| `n_quarters_used` | Quarters with ≥ 45° kept. |
| `ccfn`, `cortex_width_um`, `median_cortex_cell_area_um2`, `aerenchyma_share` | Mean of the quarter values (below). |

**Sheet `Per quarter`**

| Column | How |
|---|---|
| `quarter`, `kept_deg` | Q1–Q4 and its kept degrees. |
| `ccfn` | **Gap-aware CCFN**: on each 5° ray (as `n_cortical_layers`), the file number of the outermost cortical cell, where a gap **between cortical cells** on the ray (lacuna, missed cell) counts as the cells it would hold, round(gap / the neighbouring cells' length on the ray) (the `cell_files` rule). A gap between the stele edge and the first cell is not counted (nor in `cell_files` since 2026-11-11): on the shipped examples it came from stele outlines drawn too small or cortex pulled away, never lost cells. Median over the quarter's kept rays. |
| `ccfn_cells_present` | The same rays counting only the cells present (the `n_cortical_layers` rule). `ccfn` = this + the files in gaps; far above it with no aerenchyma = cells Cellpose missed. |
| `n_rays` | Rays behind `ccfn`. |
| `cortex_width_um` | Mean (root radius − stele radius) over the quarter's kept degrees. |
| `median_cortex_cell_area_um2` | Median area of the kept cortical cells (centre in the quarter). |
| `aerenchyma_share` | Lacuna pixels / root-outside-stele pixels, kept angles of the quarter. |
| `cortex_area_um2`, `lacuna_area_um2` | Root-outside-stele and lacuna pixels in the quarter; whole quarter kept only. |
| `n_lacunae`, `n_lacunae_drawn` | Lacunae with their centre in the quarter, and how many of them were drawn by hand; whole quarter kept only. |

**Sheets `Cell area by file` / `Cells per file`**: one row per image; `file_1` … `file_N` = mean cell area (µm²) and
cell count per cell file (file 1 next to the stele; cortex + epidermis, the pattern tissue, so the outermost files
hold the epidermis). File = the median of a cell's gap-aware numbers on rays every 0.5° (`cell_files`).
`Cells per file` (2026-11-11) is a count of a ring, so it is filled only when (a) the whole section is kept (360°) and
(b) the file goes all round: files up to the 10th percentile of the files per ray (`complete_files`,
`FILE_COMPLETE_PCT`). Outer files beyond it exist only where the cortex is thicker and mix layers (on thin sides they
hold the epidermis: on the Hallie examples, counted files median 0% epidermis cells, cut files 27%). `Cell area by file`
keeps every file (a mean per cell), but read its outer files with the same caution.

---

## 1. Identity

| Column | Meaning |
|---|---|
| `image_file`, `plant`, grouping columns, `tech_rep` | From `samples.xlsx`. `tech_rep` = image number within its plant. |
| `status` | `auto` = computer outlines only; `reviewed` = checked by hand (Save + next). |
| `um_per_px` | The scale typed in `samples.xlsx`. |
| `epidermis` | Typed in `samples.xlsx`: yes = intact, no = lost in sectioning, empty = not checked. |
| note columns | Any `samples.xlsx` column whose name starts with note / check / comment / remark: free text, never used to group. |

## 2. Check first

| Column | How | Read it as |
|---|---|---|
| `stele_found` | Any stele pixel. | FALSE: root tip or stele cut off. Every stele, cortex and aerenchyma column is then blank, never 0. |
| `angle_kept_deg` | 360 minus the excluded arcs. | |
| `excluded_sectors` | `start-end reason` in degrees, counter-clockwise from the image's right. | |
| `low_coverage` | `angle_kept_deg` < 45. | Cortex numbers rest on few cells. |
| `quarters_used` | Quarters (Q1 = 0–90°, top right, then counter-clockwise) with ≥ 45° kept. | Which quarters give quarter features. |
| `n_cortical_layers_min`, `n_cortical_layers_max` | Fewest and most cortical cells on any 5° ray. | A wide range means a wrong stele outline or an extra layer on one side. |
| `n_giant_cortex_cells` | Kept cortical cells > 5× the median. | Possibly a lacuna that Cellpose took for a cell. |
| `epidermis_deg` | Kept degrees (5° steps) holding an epidermis-tagged cell centre. | Below 90: Cellpose missed most of the epidermis; compare `_rel` with `_rel_cortex`. |
| `n_gaps_missed_cell` | Gaps in the cortex of 0.5–2× the median cortical cell (same rules as lacunae, below). | Usually a cell Cellpose missed: draw it. |
| `min_cells_in_bin` | Fewest cells in any position bin of the section pattern. | Bins under 5 cells are noisy. |

## 3. Whole root (whole outline; no sectors)

| Column | How |
|---|---|
| `root_area` | Pixels inside the root outline. |
| `root_diameter_eq` | 2√(root_area / π): the diameter of a circle with the root's area. |
| `root_diameter` | 2 × the mean root radius from the centre over the kept degrees (1° steps). |
| `root_area_est` | stele area + `cortex_area_est` (the cortex scaled to 360°). |

## 4. Stele and metaxylem (always whole-section)

| Column | How |
|---|---|
| `stele_area` | Pixels inside the stele outline. |
| `stele_share_est` | stele area / `root_area_est`. |
| `mx_n` | Number of metaxylem vessels. |
| `mx_mean_area`, `mx_total_area` | Mean and total vessel area. |

## 5. Cortex (kept angles only)

"Cortex area" means the root outside the stele (cortex + epidermis). "Cortical cells" means cells tagged cortex.

| Column | How |
|---|---|
| `n_cortical_layers` (CCFN) | Rays from the centre every 5° (offset 2.5° so none runs along a wall) in the kept angles. Each counts the cortical cells it stays in for ≥ 2 px; CCFN is the median over rays. It counts **cells present**, so aerenchyma lowers it. |
| `n_cortex_cells`, `_est` | Kept cortical cells; `_est` = × 360 / `angle_kept_deg`. |
| `mean_cortex_cell_area`, `median_cortex_cell_area` | Over the kept cortical cells (epidermis not included). |
| `cortex_width` | Mean of (root radius − stele radius) over the kept degrees. |
| `cortex_area_kept`, `cortex_area_est` | Root-outside-stele pixels in the kept angles; `_est` scaled to 360°. |
| `n_air_spaces`, `_est`, `air_space_total_area` | Objects tagged air_space in the kept angles. |

`_est` columns assume the excluded arcs looked like the kept ones. They are estimates, labelled as such.

## 6. Cortex pattern (sheet `cortex_pattern`)

**Cells used:** every kept cell outside the endodermis, cortex and epidermis treated as **one tissue**.

**Size measure:** `rel` = bin median area / the median area of those cells in the image (size without overall size). `_abs` is the same in µm² (px without a scale).

**Three bases:**
- **position** (headline): `r_norm` in 10 equal bins, x = bin centre (0–1). Survives aerenchyma and lost inner files; lost outer tissue stretches it (exclude a torn arc; `epidermis = no` flags loss all round).
- **file**: cells grouped by cell file number, x = the median `file_rel` of the file's cells (0–1). File number:
  - Rays every 0.5° in the kept angles number the cells 1, 2 … from the stele out.
  - **A gap between cells on a ray** (lacuna, missed cell) **counts as the cells it would have held**: round(gap length / the neighbouring cells' length on the ray). A gap between the stele edge and the first cell does not (2026-11-11, as `ccfn`): numbering starts at the first cell.
  - Each cell's `file` = the median of its numbers; `file_rel` = the median of (k − 0.5) / files on the ray.
  - The epidermis is the outermost file.
- **distance**: 20 µm bins of distance from the stele edge (images with a scale only); x is in µm. It survives every kind of tissue loss but carries root size, so it is a cross-check.

**Scopes:**
- `section`: all kept cells.
- `Q1`–`Q4`: one quarter, if ≥ 45° kept. A quarter's `rel` uses the section's median.
- `quarter_mean`, `quarter_sd`: mean and SD of the quarters' features.

**Curve and features:**
- A smoothing spline through the bin medians, smoothing chosen by generalised cross-validation (scipy `make_smoothing_spline`), weighted by cells per bin. It needs ≥ 5 bins with cells.
- Features read off the curve; the peak is searched over the whole curve, epidermis included:

| Feature | How |
|---|---|
| `peak_height`, `peak_pos` | Maximum of the curve and where it sits. |
| `rise_slope` | (peak − first value) / (peak_pos − first x). |
| `decay_slope` | Slope from the peak to 20% of the x-range further out. |
| `outer_rise` | Last value − the minimum after the peak (an upturn at the edge). |

Undefined features are blank: a peak at the first point has no rise, and a peak at the last point has no decay or outer rise.

**Feature sets:**
- `_rel`: the main one.
- `_abs`: µm².
- `_rel_cortex`: without the epidermis-tagged cells, a check for when `epidermis_deg` is low.

`epidermis_rel` (in `per_image`) = the median area of the epidermis-tagged cells / the same reference median.

## 7. Aerenchyma (kept angles)

**A lacuna:**
- Starts from non-cell pixels inside the root, outside the stele and the metaxylem. These are opened with a disc of radius max(2, 0.1 × √median cortical cell area), so thin walls and slits drop out, then grown back.
- Each piece is kept as a lacuna if it is ≥ 2× the median cortical cell area, touches neither the 6 px edge band nor the stele (dilated 3 px), and has its centre in the kept angles.
- **A lacuna a reviewer draws** (Aerenchyma mode) counts whatever the size, edge and stele rules say, as long as its centre is in the kept angles. Real aerenchyma can reach the endodermis; the automatic stele rule stays because the gaps it drops are almost always stele-outline errors (all 10 in the shipped examples, 2026-11-10). A later ⌘/Ctrl-click "not a lacuna" still wins.

| Column | How |
|---|---|
| `n_lacunae`, `lacuna_area` | Count and total area. |
| `aerenchyma_share` | lacuna area / `cortex_area_kept`. |
| `n_lacunae_overridden` | Gaps the rule would count that a reviewer marked **not a lacuna** (Aerenchyma mode, ⌘/Ctrl-click; marks kept in the image's npz). Not in `n_lacunae` or `aerenchyma_share`. Non-zero = a reviewer's judgement changed this image's aerenchyma. |
| `n_lacunae_drawn` | Lacunae a reviewer drew; included in `n_lacunae` and `aerenchyma_share`. Non-zero = a reviewer's judgement added aerenchyma the automatic rule would not count (or count differently). |
| `lacuna_r_norm_mean` | Mean `r_norm` of the lacuna pixels, which weights each lacuna by its area. Where in the cortex the aerenchyma sits. |
| sheet `lacunae` | One row per lacuna: drawn or not, area, angle, quarter, `r_norm` mean / inner / outer. |
| sheet `aerenchyma_profile` | Per `r_norm` bin: lacuna pixels / root-outside-stele pixels, kept angles. |

## 8. Analysis rules

- **The plant is the unit.** Per-cell values are summarised per image (medians per bin), and images are averaged per plant. Cells are never pooled across images, and neither cells nor images are n.
- `per_plant` = the plant mean of `per_image`, with `n_images`, `n_reviewed` and `n_low_coverage`.
- Every workbook carries `README` (every column: topic, unit, meaning) and `run_info` (Rooh commit, thresholds). `methods_DATE.txt` is a Methods draft written from the same constants.

## Thresholds (all in `pipeline.py` unless noted)

| Constant | Value | Used for |
|---|---|---|
| `MIN_KEPT_DEG` | 45 | `low_coverage` |
| `MIN_QUARTER_DEG` | 45 | quarter used |
| `MIN_CELLS_PER_BIN` | 5 | `few_cells` |
| `MIN_EPI_DEG` | 90 | `epidermis_deg` warning |
| `FILE_STEP_DEG` | 0.5 | file-number rays |
| `RAY_ANGLES` | every 5°, from 2.5° | CCFN rays |
| `DIST_BIN_UM` | 20 | distance basis |
| `LACUNA_MIN_CELLS` / `GAP_MIN_CELLS` | 2 / 0.5 | lacuna / missed-cell gap |
| `SPACE_RATIO` / `SPACE_MAX_NB` | 0.25 / 5 | air space |
| `N_BINS` (`summarise.py`) | 10 | position bins |
| `MIN_PTS` (`pattern_plots.py`) | 5 | fewest bins for a curve |
| epidermis band, edge band | 6 px | epidermis tag, lacuna edge |
