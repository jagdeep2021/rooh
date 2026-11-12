"""Cortex width on reviewed masks (2026-11-12): width per degree, its spread, degrees where the root runs off the
image, root aspect (long / short axis), alignment of cortical cells with the root's long axis (+1 = all along it;
an oblique cut would stretch them that way), and distance-from-stele along the line vs shortest (in cell diameters).

  .venv/bin/python docs/development/cortex_width.py <masks_dir> <out.csv>
Run on a copy of Hallie's reviewed web project "test" (38 images): numbers in the 2026-11-12 cortex width entry.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, ".")
import pipeline as P
rows = []
for npz in sorted(Path(sys.argv[1]).glob("*.npz")):
    c, rg, mx, st_ = P.load_state(npz)
    ctr = P.centre_of(rg, P.load_centre(npz)); arcs = P.load_sectors(npz, 0)
    r_st, r_rt = P.radial_profile(rg, ctr); w = (r_rt - r_st)
    keep = ~P.in_arcs(np.arange(360) + .5, arcs)
    t = np.radians(np.arange(360)); y = ctr[0] - r_rt * np.sin(t); x = ctr[1] + r_rt * np.cos(t)
    frame = ((y <= 1) | (y >= rg.shape[0] - 2) | (x <= 1) | (x >= rg.shape[1] - 2)) & keep
    rt = P.main_piece(rg >= 1); s = P.main_piece(rg == 2)
    pr = measure.regionprops(rt.astype(int))[0]; ps = measure.regionprops(s.astype(int))[0]
    props, _, _ = P.derive(c, rg, mx, 0, ctr)
    tb = pd.DataFrame(measure.regionprops_table(c, properties=("label", "orientation", "axis_major_length", "axis_minor_length", "centroid")))
    tb = tb[tb.label.isin(set(props.label[props.tissue == "cortex"])) & (tb.axis_minor_length > 0)]
    el = tb.axis_major_length / tb.axis_minor_length - 1
    align = np.average(np.cos(2 * (tb.orientation - pr.orientation)), weights=el)
    cx = props[props.tissue == "cortex"]
    near = ndi.distance_transform_edt(~s)[cx.y.round().astype(int), cx.x.round().astype(int)]
    cd = 2 * np.sqrt(cx.area_px.median() / np.pi)
    rows.append({"image": npz.stem.replace("sibs__", ""), "status": st_, "kept_deg": int(keep.sum()),
                 "width_px": w[keep].mean(), "width_cv": w[keep].std() / w[keep].mean(), "frame_deg": int(frame.sum()),
                 "root_aspect": pr.axis_major_length / pr.axis_minor_length, "stele_aspect": ps.axis_major_length / ps.axis_minor_length,
                 "cell_alignment": align, "dist_diff_p95_celldiam": np.quantile(np.abs(cx.dist_from_stele_px.values - near), .95) / cd})
t = pd.DataFrame(rows)
t.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
print("\ncorrelation root_aspect vs cell_alignment: r =", round(t.root_aspect.corr(t.cell_alignment), 2), "n =", len(t))
print("images with aspect >= 1.15:", int((t.root_aspect >= 1.15).sum()), "| and alignment >= 0.2:", int(((t.root_aspect >= 1.15) & (t.cell_alignment >= .2)).sum()),
      "| touching the frame:", int((t.frame_deg > 0).sum()), "| stele-distance diff p95 (cell diam) median", round(t.dist_diff_p95_celldiam.median(), 2))
