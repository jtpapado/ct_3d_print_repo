"""Visualise a PENGWIN fractured-pelvis CT with its fragment labels.

PENGWIN label values: 1-10 sacrum, 11-20 left hip bone, 21-30 right hip bone
(the lowest value in each range is the main piece, higher values are fragments).

usage: python view_pengwin.py <ct.mha> <label.mha> <out_dir>
Writes slices.png (CT + label overlay) and fragments_3d_*.png (3D render),
plus one STL per fragment.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from skimage import measure

BONE = {0: "sacrum", 1: "left hip", 2: "right hip"}
MAIN_COLORS = {1: "#2f7fd0", 11: "#e05a5a", 21: "#3cb371"}  # main pieces
FRAG_COLORS = ["#ffb000", "#b04fd8", "#00c8c8", "#ff6fb5", "#9acd32", "#ff8c00", "#8a6dff", "#d2b48c"]


def piece_colors(vals):
    """Main pieces get fixed colours; fragments get distinct colours in order."""
    frags = [v for v in vals if v not in MAIN_COLORS]
    out = {v: FRAG_COLORS[i % len(FRAG_COLORS)] for i, v in enumerate(frags)}
    out.update({v: c for v, c in MAIN_COLORS.items() if v in vals})
    return out


def name(v):
    bone, k = BONE[(v - 1) // 10], (v - 1) % 10
    return f"{bone} main" if k == 0 else f"{bone} frag {k}"


def main(ct_path, lab_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ct_img, lab_img = sitk.ReadImage(ct_path), sitk.ReadImage(lab_path)
    ct = sitk.GetArrayFromImage(ct_img)
    lab = sitk.GetArrayFromImage(lab_img).astype(np.int16)
    sx, sy, sz = ct_img.GetSpacing()
    vals = [int(v) for v in np.unique(lab) if v]
    print(f"CT {ct.shape} (z,y,x), spacing {sz:.2f},{sy:.2f},{sx:.2f} mm, HU {ct.min()}..{ct.max()}")
    print(f"{len(vals)} labelled pieces:", ", ".join(f"{v}={name(v)}" for v in vals))

    color = piece_colors(vals)
    cmap = ListedColormap(["black"] + [color.get(v, "black") for v in range(1, 31)])
    overlay = np.ma.masked_equal(lab, 0)

    # Orthogonal slices through the centre of the labelled bone
    zc, yc, xc = (int(c) for c in np.argwhere(lab > 0).mean(axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (t, im, ov, asp) in zip(axes, [
        ("axial", ct[zc], overlay[zc], sy / sx),
        ("coronal", ct[:, yc, :], overlay[:, yc, :], sz / sx),
        ("sagittal", ct[:, :, xc], overlay[:, :, xc], sz / sy),
    ]):
        flip = t != "axial"
        ax.imshow(im[::-1] if flip else im, cmap="gray", vmin=-200, vmax=1500, aspect=asp)
        ax.imshow(ov[::-1] if flip else ov, cmap=cmap, vmin=0, vmax=30, alpha=0.5, aspect=asp,
                  interpolation="nearest")
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(os.path.basename(ct_path) + " - colours = fragments")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slices.png"), dpi=80)
    plt.close(fig)

    # 3D surface per fragment
    meshes = []
    for v in vals:
        m = np.pad(lab == v, 1)
        if m.sum() < 50:
            continue
        verts, faces, _, _ = measure.marching_cubes(m.astype(np.float32), 0.5, spacing=(sz, sy, sx), step_size=2)
        mesh = pv.PolyData(verts[:, ::-1], np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
        mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.1)
        mesh.save(os.path.join(out_dir, f"label_{v:02d}_{name(v).replace(' ', '_')}.stl"))
        meshes.append((v, mesh))

    allm = meshes[0][1].merge([m for _, m in meshes[1:]]) if len(meshes) > 1 else meshes[0][1]
    for view, d in {"front": (0, -1, 0), "back": (0, 1, 0), "oblique": (1, -1, 0.4)}.items():
        p = pv.Plotter(off_screen=True, window_size=(1100, 900))
        p.set_background("white")
        for v, m in meshes:
            p.add_mesh(m, color=color[v], smooth_shading=True, label=name(v))
        p.add_legend(bcolor="white", size=(0.22, 0.04 * len(meshes)))
        p.camera.focal_point = allm.center
        p.camera.position = np.array(allm.center) + 800 * np.array(d, dtype=float)
        p.camera.up = (0, 0, 1)
        p.reset_camera()
        p.screenshot(os.path.join(out_dir, f"fragments_3d_{view}.png"))
        p.close()
    print(f"saved slices.png, fragments_3d_*.png, {len(meshes)} STLs -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
