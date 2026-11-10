"""Gaps the automatic stele rule drops from aerenchyma (pipeline.lacunae), in the shipped examples: drawn to decide
whether to drop the rule (2026-11-10). Run from the repo root: .venv/bin/python docs/development/stele_rule_gaps.py docs/development/stele_rule_gaps.png
Result: all 10 are stele-outline errors, none aerenchyma, so the rule stays for automatic lacunae."""
import json, sys, numpy as np, glob
from pathlib import Path
sys.path.insert(0, ".")
import pipeline as P, run_batch as RB
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import segmentation
orig=P.morphology.dilation
crops=[]
for ex in ["examples/hallie_SL-2026-004","examples/amool_root_tip_to_mature"]:
    st=json.loads(Path(ex,"anatomy/settings.json").read_text()); imgdir=P.image_dir(Path(ex,"anatomy"),st)
    keys={RB.image_key(r):r for r in RB.list_images(imgdir)}
    for f in sorted(glob.glob(f"{ex}/anatomy/masks/*.npz")):
        cells,region,mx,_=P.load_state(f)
        if not (region==2).any(): continue
        centre=P.centre_of(region,P.load_centre(f)); arcs=P.load_sectors(f,float(st.get("quadrant_angle_deg",0)))
        props,_,_=P.derive(cells,region,mx,0,centre); props=P.exclude_sectors(props,arcs)
        la=P.lacunae(cells,region,mx,props,centre,arcs,mask=True)[3]
        P.morphology.dilation=lambda im,fp,_o=orig: np.zeros_like(im) if im.dtype==bool and im.sum()==(region==2).sum() else _o(im,fp)
        lb=P.lacunae(cells,region,mx,props,centre,arcs,mask=True)[3]; P.morphology.dilation=orig
        new=np.unique(lb[(lb>0)&(la<=0)])
        if not len(new): continue
        gray=P.load_gray(imgdir/keys[Path(f).stem])
        for l in new:
            m=lb==l; yy,xx=np.nonzero(m); pad=int(max(np.ptp(yy),np.ptp(xx))*0.8)+60
            y0,y1=max(0,yy.min()-pad),min(m.shape[0],yy.max()+pad); x0,x1=max(0,xx.min()-pad),min(m.shape[1],xx.max()+pad)
            crops.append((f"{ex.split('/')[-1][:6]} {Path(f).stem[-14:]}",gray[y0:y1,x0:x1],m[y0:y1,x0:x1],(region==2)[y0:y1,x0:x1],cells[y0:y1,x0:x1]))
n=len(crops); cols=5; rows=(n+cols-1)//cols
fig,ax=plt.subplots(rows,cols,figsize=(3.2*cols,3.3*rows))
for a in ax.flat: a.axis("off")
for i,(a,(t,g,m,s,c)) in enumerate(zip(ax.flat,crops)):
    a.imshow(g,cmap="gray"); a.contour(m,[.5],colors="orange",linewidths=1.6); a.contour(s,[.5],colors="cyan",linewidths=1.2)
    a.set_title(f"{i+1}. {t}",fontsize=8)
fig.suptitle("Gaps the stele rule throws out (orange); stele edge cyan",fontsize=10); fig.tight_layout()
fig.savefig(sys.argv[1],dpi=110); print(n)
