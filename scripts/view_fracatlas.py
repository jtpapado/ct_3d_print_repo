"""Show FracAtlas X-rays with fracture outlines (COCO polygons + boxes).

usage: python view_fracatlas.py <FracAtlas_dir> <out_dir> [region] [n]
  region: leg (default), hand, hip, shoulder
Writes:
  fractured_<region>.png  - n fractured images with fracture outlines in red
  hardware_<region>.png   - n images with surgical hardware (plates/screws)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


def find_image(root, name):
    for sub in ("Fractured", "Non_fractured"):
        p = os.path.join(root, "images", sub, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def grid(root, rows, anns_by_name, title, out_path):
    n = len(rows)
    cols = min(4, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.5 * cols, 5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(Image.open(find_image(root, r.image_id)).convert("L"), cmap="gray")
        for a in anns_by_name.get(r.image_id, []):
            for poly in a.get("segmentation") or []:
                xy = list(zip(poly[::2], poly[1::2]))
                if len(xy) >= 3:
                    ax.add_patch(Polygon(xy, closed=True, fill=False, ec="red", lw=1.5))
            x, y, w, h = a["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="yellow", lw=1, ls="--"))
        views = "/".join(v for v in ("frontal", "lateral", "oblique") if r[v])
        tags = ["fractured" if r.fractured else "no fracture"]
        if r.hardware:
            tags.append("hardware")
        ax.set_title(f"{r.image_id}\n{', '.join(tags)} | {views}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print("saved", out_path)


def main(root, out_dir, region="leg", n=12):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(root, "dataset.csv"))
    coco = json.load(open(os.path.join(root, "Annotations", "COCO JSON", "COCO_fracture_masks.json")))
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_name = {}
    for a in coco["annotations"]:
        anns_by_name.setdefault(id2name[a["image_id"]], []).append(a)

    sel = df[df[region] == 1]
    print(f"{len(df)} images total; {region}: {len(sel)} "
          f"({int(sel.fractured.sum())} fractured, {int(sel.hardware.sum())} with hardware)")
    print("all regions:", {c: int(df[c].sum()) for c in ("hand", "leg", "hip", "shoulder")})

    frac = sel[sel.fractured == 1].sample(min(n, int(sel.fractured.sum())), random_state=0)
    grid(root, frac, anns_by_name, f"FracAtlas {region}: fractures (red = outline, yellow = box)",
         os.path.join(out_dir, f"fractured_{region}.png"))
    hw = sel[sel.hardware == 1]
    if len(hw):
        hw = hw.sample(min(n, len(hw)), random_state=0)
        grid(root, hw, anns_by_name, f"FracAtlas {region}: surgical hardware (plates / screws / nails)",
             os.path.join(out_dir, f"hardware_{region}.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "leg",
         int(sys.argv[4]) if len(sys.argv) > 4 else 12)
