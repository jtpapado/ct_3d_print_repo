"""3D bone reconstruction from a CT volume (NIfTI from open_knee_ct.py).

Smooth -> threshold bone -> keep the largest connected pieces -> marching cubes
-> STL meshes + rendered views.

usage: python reconstruct_3d.py <volume.nii.gz> <out_dir> [threshold] [n_bones]
  threshold defaults to 750 (tuned for the Leeds XtremeCT knees; pass "otsu"
  for an automatic one); n_bones defaults to 4.
"""
import os
import sys

import numpy as np
import pyvista as pv
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import filters, measure

CLOSE_ITERS = 4  # voxels of morphological closing to seal the cortical shell
COLORS = ["#e9d8b4", "#8fb3d9", "#e8a07a", "#9cc99a", "#c9a8d0", "#d9d98f"]


def main(vol_path, out_dir, threshold=None, n_bones=4):
    os.makedirs(out_dir, exist_ok=True)
    img = sitk.ReadImage(vol_path)
    sx, sy, sz = img.GetSpacing()
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)
    print(f"volume {vol.shape}, spacing {sz:.3f},{sy:.3f},{sx:.3f} mm (z,y,x)")

    # Remove extreme outlier voxels, then smooth so trabecular bone reads as solid
    lo, hi = np.percentile(vol, [0.01, 99.99])
    vol = ndi.gaussian_filter(np.clip(vol, lo, hi), sigma=1.0)

    # Ignore the rim of the circular field of view (sample tube wall)
    nz, ny, nx = vol.shape
    yy, xx = np.ogrid[:ny, :nx]
    fov = (yy - ny / 2) ** 2 + (xx - nx / 2) ** 2 < (0.43 * min(ny, nx)) ** 2

    if threshold is None:
        threshold = filters.threshold_otsu(vol[:, fov][::4])
    print(f"bone threshold: {threshold:.0f}")
    print("percentiles 50/90/99:", np.percentile(vol[:, fov][::4], [50, 90, 99]).round())

    mask = (vol > threshold) & fov[None]
    del vol
    # Opening breaks thin bridges (tube wall, noise) between separate bones
    mask = ndi.binary_opening(mask, iterations=2)

    labels, n = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [i + 1 for i in order[:n_bones] if sizes[i] > 0.005 * sizes[order[0]]]
    voxel_ml = sx * sy * sz / 1000
    print(f"{n} connected pieces; keeping {len(keep)}:",
          ", ".join(f"{sizes[k - 1] * voxel_ml:.1f} ml" for k in keep))

    meshes = []
    for i, k in enumerate(keep):
        # Work in the bone's bounding box (padded so closing has room)
        bz, by, bx = ndi.find_objects((labels == k).astype(np.int8))[0]
        pad = CLOSE_ITERS + 2
        part = np.pad(labels[bz, by, bx] == k, pad)
        # Seal gaps in the cortical shell, then fill the marrow cavity:
        # closed 3D holes, then axial slice by slice (handles bones cut by the scan ends)
        part = ndi.binary_fill_holes(ndi.binary_closing(part, iterations=CLOSE_ITERS))
        part = np.stack([ndi.binary_fill_holes(s) for s in part])
        offset = np.array([bz.start, by.start, bx.start]) - pad
        verts, faces, _, _ = measure.marching_cubes(part.astype(np.float32), 0.5, spacing=(sz, sy, sx))
        verts = (verts + offset * np.array([sz, sy, sx]))[:, ::-1]  # (z,y,x) -> (x,y,z) in mm
        mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
        mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1).compute_normals()
        mesh.save(os.path.join(out_dir, f"bone_{i + 1}.stl"))
        print(f"bone_{i + 1}.stl: {mesh.n_points} vertices, {mesh.volume / 1000:.1f} ml, "
              f"bounds z {mesh.bounds[4]:.0f}..{mesh.bounds[5]:.0f} mm")
        meshes.append(mesh)

    combined = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
    combined.save(os.path.join(out_dir, "knee_all_bones.stl"))

    views = {"front": (0, -1, 0), "side": (1, 0, 0), "back": (0, 1, 0), "oblique": (1, -1, 0.5)}
    for name, direction in views.items():
        p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
        p.set_background("white")
        for i, m in enumerate(meshes):
            p.add_mesh(m, color=COLORS[i % len(COLORS)], smooth_shading=True, specular=0.3)
        p.camera_position = "xz"
        p.camera.position = np.array(combined.center) + 600 * np.array(direction, dtype=float)
        p.camera.focal_point = combined.center
        p.camera.up = (0, 0, -1)  # slice order puts the tibia at high z
        p.reset_camera()
        p.screenshot(os.path.join(out_dir, f"render_{name}.png"))
        p.close()
    print(f"saved STL meshes and render_*.png -> {out_dir}")


if __name__ == "__main__":
    thr = sys.argv[3] if len(sys.argv) > 3 else "750"
    main(sys.argv[1], sys.argv[2],
         None if thr == "otsu" else float(thr),
         int(sys.argv[4]) if len(sys.argv) > 4 else 4)
