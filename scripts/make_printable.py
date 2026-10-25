"""Make STL meshes 3D-print ready: repair, check, simplify, place on the build plate.

usage:
  python make_printable.py <out_dir> part1.stl part2.stl ...
      [--scale 0.5]            print at half size (default 1.0 = life size, mm)
      [--faces 150000]         simplify each part to about this many triangles
      [--fuse NAME a.stl b.stl ...]   also make ONE solid NAME.stl from these parts,
                               closing gaps up to --gap mm so fragments join
      [--gap 2.0]

Every output is checked: watertight (no holes), manifold, size in mm, volume.
"""
import argparse
import os

import numpy as np
import pymeshfix
import pyvista as pv
from scipy import ndimage as ndi
from skimage import measure

BED = (220, 220, 250)  # typical printer build volume, mm


def repair(mesh):
    mesh = mesh.triangulate().clean()
    if mesh.n_open_edges or not mesh.is_manifold:
        fixer = pymeshfix.MeshFix(mesh)
        fixer.repair()
        mesh = fixer.mesh
    return mesh


def simplify(mesh, faces):
    if mesh.n_cells > faces:
        mesh = mesh.decimate_pro(1 - faces / mesh.n_cells, preserve_topology=True)
        mesh = repair(mesh)
    return mesh


def place_on_bed(mesh):
    """Put the largest flat face down (cut ends of bones make good bases), centre on the bed."""
    n = mesh.compute_normals(cell_normals=True, point_normals=False)
    area = n.compute_cell_sizes()["Area"]
    nz = n["Normals"][:, 2]
    up = area[nz > 0.98].sum()      # flat area facing +z
    down = area[nz < -0.98].sum()   # flat area facing -z
    if up > down * 1.5:
        mesh = mesh.rotate_x(180, point=mesh.center)
    b = mesh.bounds
    return mesh.translate((-(b[0] + b[1]) / 2, -(b[2] + b[3]) / 2, -b[4]))


def report(name, mesh):
    b = np.array(mesh.bounds)
    size = b[1::2] - b[::2]
    ok = mesh.n_open_edges == 0 and mesh.is_manifold
    fits = all(size <= np.array(BED))
    print(f"  {name:32s} {'OK ' if ok else 'BAD'} watertight={mesh.n_open_edges == 0} "
          f"manifold={mesh.is_manifold}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm  "
          f"{mesh.volume / 1000:.1f} ml  {mesh.n_cells:,} triangles"
          + ("" if fits else "  (TOO BIG for a 220 mm bed - use --scale)"))
    return ok


def fuse(paths, gap, voxel=0.3):
    """Union several closed meshes in voxel space, closing gaps <= gap mm."""
    meshes = [repair(pv.read(p)) for p in paths]
    lo = np.min([m.bounds[::2] for m in meshes], axis=0) - gap - 2
    hi = np.max([m.bounds[1::2] for m in meshes], axis=0) + gap + 2
    dims = np.ceil((hi - lo) / voxel).astype(int)
    grid = pv.ImageData(dimensions=dims, spacing=(voxel,) * 3, origin=lo)
    solid = np.zeros(dims, bool)
    for m in meshes:
        solid |= np.asarray(m.voxelize_binary_mask(reference_volume=grid).point_data["mask"]
                            ).reshape(dims, order="F").astype(bool)
    r = max(1, int(round(gap / 2 / voxel)))
    ball = ndi.generate_binary_structure(3, 1)
    solid = ndi.binary_closing(np.pad(solid, r + 2), ball, iterations=r)
    solid = ndi.binary_fill_holes(solid)
    lab, n = ndi.label(solid)
    if n > 1:  # keep the biggest piece if anything is still separate
        sizes = ndi.sum(solid, lab, range(1, n + 1))
        share = sizes.max() / sizes.sum()
        if share > 0.99:
            print(f"  note: dropped {n - 1} tiny loose speck(s) ({1 - share:.2%} of volume)")
        else:
            print(f"  WARNING: {n} pieces did not join; keeping the largest ({share:.0%} of volume)"
                  f" - try a bigger --gap")
        solid = lab == 1 + int(np.argmax(sizes))
    f = ndi.gaussian_filter(solid.astype(np.float32), 1.0)
    verts, faces, _, _ = measure.marching_cubes(f, 0.5, spacing=(voxel,) * 3)
    verts += lo - (r + 2) * voxel
    mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    return repair(mesh.smooth_taubin(n_iter=20, pass_band=0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--faces", type=int, default=150_000)
    ap.add_argument("--fuse", nargs="+", metavar=("NAME", "STL"))
    ap.add_argument("--gap", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    jobs = [(os.path.splitext(os.path.basename(p))[0], lambda p=p: repair(pv.read(p))) for p in a.parts]
    if a.fuse:
        jobs.append((a.fuse[0], lambda: fuse(a.fuse[1:], a.gap)))

    print(f"scale {a.scale:g} ({'life size' if a.scale == 1 else 'scaled'}), output -> {a.out_dir}")
    all_ok = True
    for name, load in jobs:
        mesh = simplify(load(), a.faces)
        if a.scale != 1:
            mesh = mesh.scale(a.scale)
        mesh = place_on_bed(mesh)
        all_ok &= report(name, mesh)
        mesh.save(os.path.join(a.out_dir, f"{name}.stl"), binary=True)
    print("all parts print-ready" if all_ok else "some parts need attention (see BAD)")


if __name__ == "__main__":
    main()
