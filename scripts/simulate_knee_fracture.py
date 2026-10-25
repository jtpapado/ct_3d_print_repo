"""SIMULATED tibial plateau fracture (Schatzker II: lateral split-depression)
on a real, intact knee from the VSDFullBodyBoneModels database.

This is a synthetic illustration, not patient data: the bones are real CT
reconstructions, the fracture is cut and displaced programmatically.

usage: python simulate_knee_fracture.py <Bones/NNN.mat> <out_dir> [side R|L] [platif_dir]
  platif_dir (optional): adds a real Schatzker II CT slice from PlaTiF for comparison.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.io
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage import measure

VOX = 0.4            # mm
SPLIT_AT = 0.22     # split reaches the joint surface at this fraction of plateau width lateral of centre
SPLIT_SLOPE = 0.30   # split line moves this many mm laterally per mm of depth (wedge shape)
SPLIT_CURVE = 0.004  # quadratic term: the split curves out to exit the lateral cortex
DEPRESS_AT = 0.13    # centre of the depressed joint-surface piece (fraction of width, lateral)
DEPRESS_R = 0.10     # radius of the depressed piece (fraction of width)
DEPRESS_DEPTH = 9.0  # mm thickness of the depressed piece
DEPRESS_DROP = 4.0   # mm it is pushed down
TILT_DEG = 4.0       # outward tilt of the split fragment about its lowest point
SHIFT_LAT = 2.0      # mm lateral shift of the split fragment
LABELS = {1: ("tibia (intact part)", "#e9d8b4"), 2: ("split fragment", "#e8a07a"),
          3: ("depressed fragment", "#d65a5a")}


def load_bones(path, side):
    B = scipy.io.loadmat(path)["B"]
    bones = {}
    for b in range(B.size):
        e = B[0, b]
        name = str(e[0][0])
        if name.endswith("_" + side):
            v, f = e[2][0][0][0], e[2][0][0][1] - 1
            bones[name[:-2]] = pv.PolyData(v.astype(float), np.hstack([np.full((len(f), 1), 3), f]).ravel())
    return bones


def _scatter(dims, idx):
    m = np.zeros(dims, bool)
    m[tuple(idx.T)] = True
    return m


def rodrigues(axis, deg):
    a = np.radians(deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def move(mask, R, hinge, t, origin):
    """Rigidly move a voxel mask: p -> R(p - hinge) + hinge + t (world mm)."""
    Rt = R.T
    offset = (Rt @ (origin - hinge - t) + hinge - origin) / VOX
    return ndi.affine_transform(mask.astype(np.uint8), Rt, offset=offset, order=0).astype(bool)


def to_mesh(mask, origin):
    m = ndi.gaussian_filter(np.pad(mask, 2).astype(np.float32), 1.0)  # removes voxel terraces
    verts, faces, _, _ = measure.marching_cubes(m, 0.5, spacing=(VOX,) * 3)
    verts = verts + origin - 2 * VOX
    mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    return mesh.smooth_taubin(n_iter=40, pass_band=0.08).compute_normals()


def main(mat_path, out_dir, side="R", platif_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    bones = load_bones(mat_path, side)
    tibia, fibula = bones["Tibia"], bones["Fibula"]
    z_top = tibia.bounds[5]

    # Voxelise the proximal tibia (closed mesh -> solid mask), padded so fragments can move
    prox = tibia.clip("z", origin=(0, 0, z_top - 120), invert=False)
    b = np.array(prox.bounds).reshape(3, 2)
    origin = b[:, 0] - 12
    dims = np.ceil((b[:, 1] - b[:, 0] + 24) / VOX).astype(int)
    grid = pv.ImageData(dimensions=dims, spacing=(VOX,) * 3, origin=origin)
    vox = tibia.voxelize_binary_mask(reference_volume=grid)
    solid = np.asarray(vox.point_data["mask"]).reshape(dims, order="F").astype(bool)
    print(f"tibia voxels: {solid.sum():,} ({solid.sum() * VOX ** 3 / 1000:.0f} ml in the proximal 120 mm)")

    # Anatomical frame: z up, L = towards the fibula (lateral), A = anteroposterior
    ii = np.argwhere(solid)
    P = origin + ii * VOX
    top = P[P[:, 2] > z_top - 15]
    c = top.mean(axis=0)
    # Lateral = the scan's left-right (x) axis, pointing to the fibula side
    # (the fibula sits posterolaterally, so pointing straight at it skews the frame)
    fib_top = fibula.points[fibula.points[:, 2] > fibula.bounds[5] - 60].mean(axis=0)
    L = np.array([np.sign(fib_top[0] - c[0]), 0.0, 0.0])
    Z = np.array([0, 0, 1.0])
    A = np.cross(Z, L)
    # Re-centre on the middle of the plateau's medial-lateral extent and scale cuts to its width
    lt = (top - c) @ L
    lmin, lmax = np.percentile(lt, [0.5, 99.5])
    W = lmax - lmin
    c = c + L * (lmin + lmax) / 2
    lat = (P - c) @ L
    h = P[:, 2] - z_top
    ant = (P - c) @ A
    tl = (top - c) @ L
    lat_plateau = top[(tl > 0.05 * W) & (tl < 0.4 * W)]
    a_dep = ((lat_plateau - c) @ A).mean()
    print(f"plateau width {W:.0f} mm; split at {SPLIT_AT * W:.0f} mm lateral, "
          f"depression centre {DEPRESS_AT * W:.0f} mm lateral, radius {DEPRESS_R * W:.0f} mm")

    # Smooth noise so fracture surfaces look irregular (+-1.5 mm)
    rng = np.random.default_rng(7)
    noise = ndi.gaussian_filter(rng.standard_normal(dims), 8)
    noise *= 1.5 / noise.std()
    nz = noise[tuple(ii.T)]

    # Depth below the joint surface, per (x, y) column
    surf = np.full(dims[:2], -np.inf)
    np.maximum.at(surf, (ii[:, 0], ii[:, 1]), P[:, 2])
    depth = surf[ii[:, 0], ii[:, 1]] - P[:, 2]

    d = -h
    split = (lat > SPLIT_AT * W + SPLIT_SLOPE * d + SPLIT_CURVE * d ** 2 + nz) & (d < 80)
    # keep only the piece attached to the joint surface (drop stray cortex bits)
    sl_lab, n = ndi.label(_scatter(dims, ii[split]))
    if n > 1:
        sizes = ndi.sum(sl_lab > 0, sl_lab, range(1, n + 1))
        keep = sl_lab[tuple(ii.T)] == (1 + int(np.argmax(sizes)))
        split &= keep
    rad = np.hypot(lat - DEPRESS_AT * W, ant - a_dep)
    depressed = (rad < DEPRESS_R * W + nz) & (depth < DEPRESS_DEPTH + 0.5 * nz) & ~split

    lab = np.zeros(dims, np.uint8)
    lab[tuple(ii.T)] = 1
    lab[tuple(ii[split].T)] = 2
    lab[tuple(ii[depressed].T)] = 3
    # Thin gap along the fracture lines
    frag = lab >= 2
    lab[(lab == 1) & ndi.binary_dilation(frag, iterations=1)] = 0
    print("fragment volumes (ml):", {LABELS[k][0]: round((lab == k).sum() * VOX ** 3 / 1000, 1) for k in (2, 3)})

    # Displace: split fragment tilts outward about its lower hinge and shifts; depressed piece sinks
    # Hinge = lowest point of the split fragment (where it stays attached to the cortex)
    sp = P[split]
    hinge = sp[np.argmin(sp[:, 2])]
    moved = np.zeros_like(lab)
    moved[lab == 1] = 1
    m2 = move(lab == 2, rodrigues(A, TILT_DEG), hinge, SHIFT_LAT * L + np.array([0, 0, -1.0]), origin)
    m3 = move(lab == 3, np.eye(3), np.zeros(3), np.array([0, 0, -DEPRESS_DROP]), origin)
    moved[m2 & (moved == 0)] = 2
    moved[m3 & (moved == 0)] = 3
    np.save(os.path.join(out_dir, "labels.npy"), moved)

    # Reduced (anatomic) fragments for fixation planning, and the void left under the
    # depressed piece once it is elevated (bone it was crushed into is gone -> graft)
    red = os.path.join(out_dir, "reduced")
    os.makedirs(red, exist_ok=True)
    back_up = move(moved == 3, np.eye(3), np.zeros(3), np.array([0, 0, DEPRESS_DROP]), origin)
    void = (lab == 3) & ~back_up
    for name, msk in [("split_reduced", lab == 2), ("depressed_reduced", back_up), ("graft_void", void)]:
        if msk.sum() > 20:
            to_mesh(msk, origin).save(os.path.join(red, f"{name}.stl"))
    json_path = os.path.join(red, "displacement.json")
    import json
    json.dump({"split": {"hinge": hinge.tolist(), "R": rodrigues(A, TILT_DEG).tolist(),
                         "t": (SHIFT_LAT * L + np.array([0, 0, -1.0])).tolist()},
               "depressed": {"t": [0, 0, -DEPRESS_DROP]},
               "frame": {"L": L.tolist(), "A": A.tolist(), "plateau_centre": c.tolist(), "z_top": z_top}},
              open(json_path, "w"), indent=1)
    print(f"reduced fragments + graft void ({void.sum() * VOX ** 3 / 1000:.2f} ml) -> {red}")

    meshes = {k: to_mesh(moved == k, origin) for k in (1, 2, 3)}
    intact = to_mesh(lab > 0, origin)
    for k, m in meshes.items():
        m.save(os.path.join(out_dir, f"tibia_{k}_{LABELS[k][0].split(' ')[0]}.stl"))
    femur = bones["Femur"].clip("z", origin=(0, 0, z_top + 110), invert=True)
    fib = fibula.clip("z", origin=(0, 0, z_top - 120), invert=False)
    others = {"femur": (femur, "#f3ead6"), "patella": (bones["Patella"], "#f3ead6"), "fibula": (fib, "#cfc6b3")}
    for n, (m, _) in others.items():
        m.save(os.path.join(out_dir, f"{n}.stl"))

    centre = np.array(meshes[1].center)

    def render(fname, direction, up, show_femur=True, femur_opacity=1.0, title=""):
        p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
        p.set_background("white")
        for n, (m, col) in others.items():
            if n == "femur" and not show_femur:
                continue
            if n == "patella" and not show_femur:
                continue
            p.add_mesh(m, color=col, smooth_shading=True, opacity=femur_opacity if n != "fibula" else 1)
        for k, m in meshes.items():
            p.add_mesh(m, color=LABELS[k][1], smooth_shading=True, specular=0.3, label=LABELS[k][0])
        p.add_legend(bcolor="white", size=(0.3, 0.12))
        p.add_text(title, font_size=11, color="black")
        p.camera.focal_point = centre
        p.camera.position = centre + 450 * np.asarray(direction, float)
        p.camera.up = up
        p.reset_camera()
        p.camera.zoom(1.25)
        p.screenshot(os.path.join(out_dir, fname))
        p.close()

    front = -A if A[1] > 0 else A           # anterior points to -y in this database
    render("fracture_front.png", front, (0, 0, 1), title="SIMULATED Schatzker II - front")
    render("fracture_lateral.png", L + 0.3 * front, (0, 0, 1), femur_opacity=0.25,
           title="SIMULATED - lateral side, femur transparent")
    render("fracture_top.png", Z, tuple(front), show_femur=False,
           title="SIMULATED - tibial plateau from above (femur removed)")
    render("fracture_oblique.png", front + L + 0.7 * Z, (0, 0, 1), show_femur=False,
           title="SIMULATED - oblique, femur removed")

    # Before/after
    p = pv.Plotter(off_screen=True, window_size=(1600, 800), shape=(1, 2))
    for i, (ttl, items) in enumerate([("intact", [(intact, LABELS[1][1])]),
                                      ("simulated fracture", [(meshes[k], LABELS[k][1]) for k in (1, 2, 3)])]):
        p.subplot(0, i)
        p.set_background("white")
        p.add_text(ttl, font_size=12, color="black")
        p.add_mesh(fib, color="#cfc6b3", smooth_shading=True)
        for m, col in items:
            p.add_mesh(m, color=col, smooth_shading=True)
        p.camera.focal_point = centre
        p.camera.position = centre + 450 * (front + 0.8 * L + 0.6 * Z)
        p.camera.up = (0, 0, 1)
        p.reset_camera()
    p.link_views()
    p.screenshot(os.path.join(out_dir, "before_after.png"))
    p.close()

    # Coronal cross-section through the depression, next to a real Type II CT slice
    # Coronal = x-z plane at the AP position of the depression; viewed from the front,
    # patient-right is on the image left (radiological convention)
    jy = int(round((c[1] + a_dep * A[1] - origin[1]) / VOX))
    sl = moved[:, jy, :].T[::-1]
    lo, hi = np.where(sl.any(axis=0))[0][[0, -1]]
    top_row = np.where(sl.any(axis=1))[0][0]
    sl = sl[max(0, top_row - 40): top_row + int(90 / VOX), max(0, lo - 20): hi + 20]
    real = None
    if platif_dir:
        for pth in sorted(glob.glob(os.path.join(platif_dir, "**", "Patient_ID_*.mat"), recursive=True)):
            s = scipy.io.loadmat(pth)
            s = s[[k for k in s if not k.startswith("__")][0]][0, 0]
            if int(s["im0"][0, 0]["label"][0, 0]) == 2 and "Coronal_CT" in s.dtype.names:
                real = (os.path.basename(pth)[:-4], s["Coronal_CT"])
                break
    fig, ax = plt.subplots(1, 2 if real else 1, figsize=(13 if real else 7, 6.5), squeeze=False)
    cmap = ListedColormap(["black"] + [LABELS[k][1] for k in (1, 2, 3)])
    ax[0, 0].imshow(sl, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax[0, 0].set_title("SIMULATED coronal section\n(orange = split, red = depressed)")
    ax[0, 0].axis("off")
    if real:
        ax[0, 1].imshow(real[1], cmap="gray")
        ax[0, 1].set_title(f"REAL Schatzker II CT slice (PlaTiF {real[0]})")
        ax[0, 1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "section_vs_real.png"), dpi=80)
    print("saved renders, STLs and section_vs_real.png ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "R",
         sys.argv[4] if len(sys.argv) > 4 else None)
