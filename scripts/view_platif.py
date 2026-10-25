"""Visualise the PlaTiF tibial plateau fracture dataset.

usage:
  python view_platif.py <platif_dir> <out_dir>               # overview + gallery
  python view_platif.py <platif_dir> <out_dir> <patient_id>  # one patient, e.g. 12

<platif_dir> holds Tibial_Plateau_Fracture_Metadata.xlsx and the unzipped
"Patient Data_Part N" folders (searched recursively for Patient_ID_###.mat).
Schatzker label 7 = normal (no fracture).
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

SCHATZKER = {
    1: "I: lateral split", 2: "II: lateral split-depression", 3: "III: lateral depression",
    4: "IV: medial plateau", 5: "V: bicondylar", 6: "VI: plateau + shaft dissociation", 7: "normal",
}


def find_mat(root, pid):
    hits = glob.glob(os.path.join(root, "**", f"Patient_ID_{int(pid):03d}.mat"), recursive=True)
    return hits[0] if hits else None


def load_patient(path):
    m = scipy.io.loadmat(path)
    s = m[[k for k in m if not k.startswith("__")][0]][0, 0]
    views = []
    for f in sorted(n for n in s.dtype.names if n.startswith("im")):
        v = s[f][0, 0]
        views.append({"name": f, "img": v["OriginalImage"], "mask": v["BW"].astype(bool),
                      "label": int(v["label"][0, 0])})
    ct = s["Coronal_CT"] if "Coronal_CT" in s.dtype.names else None
    return views, ct


def show_xray(ax, v, title=None):
    ax.imshow(v["img"], cmap="gray")
    ax.contour(v["mask"], [0.5], colors="lime", linewidths=1)
    ax.set_title(title or f"{v['name']}: Schatzker {SCHATZKER.get(v['label'], v['label'])}", fontsize=9)
    ax.axis("off")


def patient_figure(root, pid, out_dir):
    path = find_mat(root, pid)
    if not path:
        sys.exit(f"Patient_ID_{int(pid):03d}.mat not found under {root} (not unzipped yet?)")
    views, ct = load_patient(path)
    panels = len(views) + (ct is not None and ct.size > 0)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 6), squeeze=False)
    for ax, v in zip(axes[0], views):
        show_xray(ax, v)
    if ct is not None and ct.size > 0:
        axes[0, -1].imshow(ct, cmap="gray")
        axes[0, -1].set_title("coronal CT slice", fontsize=9)
        axes[0, -1].axis("off")
    fig.suptitle(f"PlaTiF patient {int(pid)} (green = tibia mask)")
    fig.tight_layout()
    out = os.path.join(out_dir, f"patient_{int(pid):03d}.png")
    fig.savefig(out, dpi=80)
    print("saved", out)


def overview(root, out_dir):
    meta = pd.read_excel(os.path.join(root, "Tibial_Plateau_Fracture_Metadata.xlsx"))
    tcol = [c for c in meta.columns if "Schatzker" in c][0]
    meta["type"] = meta[tcol].astype(str).str.strip()
    print(f"{len(meta)} patients; types:", meta["type"].value_counts().sort_index().to_dict())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    meta["type"].value_counts().sort_index().plot.bar(ax=ax[0], color="#8fb3d9")
    ax[0].set_title("Schatzker type (patients)")
    ax[1].hist(meta["Age"], bins=20, color="#e8a07a")
    ax[1].set_title("age")
    side = [c for c in meta.columns if "Right" in c][0]
    pd.crosstab(meta["type"], meta["Gender"]).plot.bar(ax=ax[2], stacked=True)
    ax[2].set_title(f"type by sex  (side R/L: {meta[side].value_counts().to_dict()})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metadata_summary.png"), dpi=80)
    print("saved metadata_summary.png")

    # Gallery: first available patient for each Schatzker label, X-ray + coronal CT
    examples = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "Patient_ID_*.mat"), recursive=True)):
        views, ct = load_patient(path)
        lab = views[0]["label"]
        if lab not in examples:
            examples[lab] = (os.path.basename(path)[:-4], views[0], ct)
        if len(examples) == 7:
            break
    labs = sorted(examples)
    fig, axes = plt.subplots(2, len(labs), figsize=(3.6 * len(labs), 8), squeeze=False)
    for i, lab in enumerate(labs):
        pid, v, ct = examples[lab]
        show_xray(axes[0, i], v, f"{SCHATZKER[lab]}\n{pid}")
        axes[1, i].axis("off")
        if ct is not None and ct.size > 0:
            axes[1, i].imshow(ct, cmap="gray")
            axes[1, i].set_title("coronal CT", fontsize=9)
    fig.suptitle("PlaTiF: one example per Schatzker type (top: X-ray + tibia mask, bottom: CT slice)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "schatzker_gallery.png"), dpi=80)
    print(f"saved schatzker_gallery.png ({len(labs)} types found in unzipped files)")


if __name__ == "__main__":
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if len(sys.argv) > 3:
        patient_figure(root, sys.argv[3], out_dir)
    else:
        overview(root, out_dir)
