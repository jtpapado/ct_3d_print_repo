"""Proximal tibial cut for total knee arthroplasty (TKA), planned on the simulated fractured knee.

Cut plane:
  - coronal: perpendicular to the tibial anatomic axis (0 deg varus/valgus)
  - sagittal: SLOPE deg posterior slope (plane drops towards the back)
  - depth: POLY (+ TRAY) mm below the centre of the intact MEDIAL plateau, measured
    perpendicular to the cut (stylus on the unaffected side - the lateral side is fractured),
    so bone removed = implant thickness and the joint line is restored.
Jig (schematic): intramedullary rod along the anatomic axis from the entry point, cutting
block with saw slot on the anterior tibia, stylus touching the medial plateau.

usage: python tka_tibial_cut.py <Bones/NNN.mat> <simulated_knee_dir> <axis_dir> <out_dir>
                                [--side R] [--slope 5] [--poly 9] [--tray 0] [--show]
Schematic planning illustration on SIMULATED data - not a surgical plan.
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tibia_axis import load_bones, write_html  # noqa: E402

BONE, SPLIT, DEPR = "#e9d8b4", "#e8a07a", "#d65a5a"
STEEL, IMPLANT, PLANE = "#8f9bab", "#c9d3df", "#3d8bfd"


def unit(v):
    return v / np.linalg.norm(v)


def box(center, ax_x, ax_y, ax_z, sx, sy, sz):
    """Oriented box: center, three unit axes, full sizes."""
    b = pv.Box(bounds=(-sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2)).triangulate()
    R = np.c_[ax_x, ax_y, ax_z]
    b.points = b.points @ R.T + center
    return b


def surface_hit(mesh, xy_point, n, above=200.0):
    """First point where a ray coming down along -n hits the mesh (stylus contact)."""
    pts, _ = mesh.ray_trace(xy_point + n * above, xy_point - n * above)
    if not len(pts):
        return None
    return pts[np.argmax(pts @ n)]


def split_by_plane(mesh, n, o):
    """Closed-surface clip -> (part above plane, part below plane), both capped.
    Normals must point consistently outward or the caps come out wrong."""
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    a = mesh.clip_closed_surface(normal=-n, origin=o)   # one side
    b = mesh.clip_closed_surface(normal=n, origin=o)    # other side
    def side(m):
        return (m.points.mean(axis=0) - o) @ n if m.n_points else 0
    # split normals at the sharp cut edge so the flat cap shades cleanly
    a, b = (m.compute_normals(split_vertices=True, feature_angle=40) if m.n_points else m for m in (a, b))
    return (a, b) if side(a) > side(b) else (b, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat"); ap.add_argument("sim_dir"); ap.add_argument("axis_dir"); ap.add_argument("out_dir")
    ap.add_argument("--side", default="R")
    ap.add_argument("--slope", type=float, default=5.0)
    ap.add_argument("--poly", type=float, default=9.0)
    ap.add_argument("--tray", type=float, default=0.0)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    depth = a.poly + a.tray

    bones = load_bones(a.mat, a.side)
    tibia_full = bones["Tibia"]
    m_ax, d_ax, entry = np.loadtxt(os.path.join(a.axis_dir, "axis.txt"))
    d_ax = unit(d_ax)
    pieces = {k: pv.read(os.path.join(a.sim_dir, f)) for k, f in
              [("tibia", "tibia_1_tibia.stl"), ("split", "tibia_2_split.stl"), ("depressed", "tibia_3_depressed.stl")]}
    prox_all = pieces["tibia"].merge([pieces["split"], pieces["depressed"]])

    # Frame perpendicular to the anatomic axis
    n0 = -d_ax                                               # up along the axis
    fib = bones["Fibula"].points
    fib_top = fib[fib[:, 2] > fib[:, 2].max() - 60].mean(axis=0)
    L = np.array([np.sign(fib_top[0] - m_ax[0]), 0, 0.0])   # lateral (fibula side)
    L = unit(L - (L @ n0) * n0)
    ANT = unit(np.cross(n0, L))
    if ANT[1] > 0:                                           # anterior is -y in this database;
        ANT = -ANT                                           # keep L on the fibula side

    # Sloped cut normal: tilt the normal posteriorly so the plane drops towards the back
    s = np.radians(a.slope)
    n = unit(np.cos(s) * n0 - np.sin(s) * ANT)
    t_ap = unit(ANT - (ANT @ n) * n)                         # in-plane anterior
    t_ml = unit(np.cross(n, t_ap))                           # in-plane, sign follows L
    if t_ml @ L < 0:
        t_ml = -t_ml

    # Plateau compartments (from the intact part of the proximal tibia)
    top = pieces["tibia"].points
    top = top[(top - entry) @ n0 > -15]
    lat = (top - entry) @ L
    lo, hi = np.percentile(lat, [1, 99])
    W = hi - lo
    mid = (lo + hi) / 2
    def compartment_centre(frac):
        sel = top[np.abs(lat - (mid + frac * W)) < 0.12 * W]
        p = entry + L * (mid + frac * W) + ANT * (((sel - entry) @ ANT).mean())
        return p - n0 * ((p - entry) @ n0)                  # on the axis-perpendicular plane through entry
    med_c, lat_c = compartment_centre(-0.25), compartment_centre(+0.25)

    ref = surface_hit(pieces["tibia"], med_c, n0)            # stylus contact, intact medial plateau
    o = ref - depth * n                                      # cut plane origin
    print(f"plateau width {W:.0f} mm")
    print(f"cut: perpendicular to the anatomic axis in the coronal plane, {a.slope:g} deg posterior slope")
    print(f"     {depth:g} mm below the medial plateau centre (poly {a.poly:g} + tray {a.tray:g})")

    # Check the plane's angles against the axis
    cor = np.degrees(np.arcsin(np.clip(n @ L, -1, 1)))
    sag = np.degrees(np.arctan2(-(n @ ANT), n @ n0))
    print(f"check: plane normal vs axis -> coronal {cor:+.2f} deg, sagittal slope {sag:+.2f} deg")

    # Resection thickness at each compartment (fractured lateral vs what it would be intact)
    lat_hit = surface_hit(prox_all, lat_c, n0)
    lat_intact = surface_hit(tibia_full, lat_c, n0)
    rl = (lat_hit - o) @ n if lat_hit is not None else float("nan")
    ri = (lat_intact - o) @ n if lat_intact is not None else float("nan")
    print(f"resection thickness: medial {(ref - o) @ n:.1f} mm | lateral {rl:.1f} mm "
          f"(would be {ri:.1f} mm without the fracture)")
    if rl < 0:
        print(f"  -> lateral bone surface is {-rl:.1f} mm BELOW the cut: a defect under the baseplate "
              f"(would need graft / cement / augment)")

    # Split every piece at the cut
    kept, removed = {}, {}
    for k, m in pieces.items():
        up_part, down_part = split_by_plane(m, n, o)
        kept[k], removed[k] = down_part, up_part
    vol = {k: (abs(kept[k].volume) / 1000, abs(removed[k].volume) / 1000) for k in pieces}
    print("bone removed by the cut (ml): " + ", ".join(f"{k} {v[1]:.1f}" for k, v in vol.items()))
    print("left below the cut (ml):      " + ", ".join(f"{k} {v[0]:.1f}" for k, v in vol.items()))
    print(f"  -> split fragment left under the baseplate: {vol['split'][0]:.1f} ml "
          f"(must be fixed, e.g. with screws, to support the implant)")

    # Cut cross-section -> plane patch and schematic implant (tray + 9 mm insert)
    sec = prox_all.slice(normal=n, origin=o)
    P2 = np.c_[(sec.points - o) @ t_ml, (sec.points - o) @ t_ap]
    hull = P2[ConvexHull(P2).vertices]
    c2 = hull.mean(axis=0)
    inset = c2 + (hull - c2) * (1 - 1.5 / np.linalg.norm(hull - c2, axis=1, keepdims=True))
    pts3 = o + inset[:, :1] * t_ml + inset[:, 1:] * t_ap
    base = pv.PolyData(pts3, faces=np.r_[len(pts3), np.arange(len(pts3))]).triangulate()
    implant = base.extrude(n * depth, capping=True).triangulate().clean()
    ext = np.abs(P2).max(axis=0) + 12
    plane = pv.Plane(center=o, direction=n, i_size=2 * ext[0], j_size=2 * ext[1])

    # Jig: IM rod, cutting block with saw slot, connecting bar, stylus
    rod = pv.Cylinder(center=entry + d_ax * 90, direction=d_ax, radius=4, height=220, resolution=40)
    ant_edge = (sec.points - o) @ t_ap
    front = o + t_ap * (ant_edge.max() + 2.5)                 # block's back face just in front of the cortex
    slot, bw, bh_low, bh_up, bd = 1.3, 60.0, 24.0, 6.0, 12.0
    lower = box(front + t_ap * bd / 2 - n * (slot / 2 + bh_low / 2), t_ml, t_ap, n, bw, bd, bh_low)
    upper = box(front + t_ap * bd / 2 + n * (slot / 2 + bh_up / 2), t_ml, t_ap, n, bw, bd, bh_up)
    rod_top = entry - d_ax * 25
    tower_top = front + t_ap * bd / 2 + n * (slot / 2 + bh_up)
    bar = pv.Tube(pointa=rod_top, pointb=rod_top + (tower_top + n * 25 - rod_top), radius=3)
    post = pv.Tube(pointa=tower_top, pointb=tower_top + n * 25, radius=3)
    stylus_base = front + t_ap * bd / 2 + n * (slot / 2 + bh_up) + t_ml * ((med_c - o) @ t_ml)
    s_top = stylus_base + n * ((ref - stylus_base) @ n + 12)
    s_over = s_top + t_ap * ((ref - s_top) @ t_ap)
    stylus = pv.Tube(pointa=stylus_base, pointb=s_top, radius=1.5).merge([pv.Tube(pointa=s_top, pointb=s_over, radius=1.5),
                                                            pv.Tube(pointa=s_over, pointb=ref, radius=1.5)])
    tip = pv.Sphere(radius=2.0, center=ref)

    fib_m = bones["Fibula"]
    femur = bones["Femur"].clip("z", origin=(0, 0, tibia_full.bounds[5] + 110), invert=True)
    shaft = tibia_full.clip("z", origin=(0, 0, tibia_full.bounds[5] - 125), invert=True)
    axis_line = pv.Tube(pointa=entry - d_ax * 40, pointb=entry + d_ax * 330, radius=0.8)

    jig = [(rod, STEEL, 1.0, "IM rod (on anatomic axis)"), (lower, STEEL, 1.0, "cutting block"),
           (upper, STEEL, 1.0, "cutting block (above slot)"), (bar, STEEL, 1.0, "alignment arm"),
           (post, STEEL, 1.0, "alignment post"), (stylus, "#e0b000", 1.0, f"stylus ({depth:g} mm)"),
           (tip, "#e0b000", 1.0, "stylus tip (medial plateau)")]
    before = [(shaft, BONE, 0.35, "tibia shaft"), (pieces["tibia"], BONE, 0.45, "proximal tibia"),
              (pieces["split"], SPLIT, 0.9, "split fragment"), (pieces["depressed"], DEPR, 0.9, "depressed fragment"),
              (fib_m, "#cfc6b3", 0.25, "fibula"), (axis_line, "#1f5fbf", 1.0, "anatomic axis"),
              (plane, PLANE, 0.35, f"cut plane ({a.slope:g} deg slope)")] + jig
    after = [(shaft, BONE, 1.0, "tibia shaft"), (kept["tibia"], BONE, 1.0, "tibia after cut"),
             (kept["split"], SPLIT, 1.0, "split fragment (left)"), (kept["depressed"], DEPR, 1.0, "depressed fragment (left)"),
             (removed["tibia"], BONE, 0.15, "resected bone"), (removed["split"], SPLIT, 0.15, "resected split"),
             (removed["depressed"], DEPR, 0.15, "resected depressed"),
             (implant, IMPLANT, 0.85, f"tibial component, {depth:g} mm (schematic)"),
             (fib_m, "#cfc6b3", 0.4, "fibula"), (femur, "#f3ead6", 0.3, "femur")]

    centre = np.array(prox_all.center)
    # Frame the knee + jig head (not the whole IM rod)
    roi = np.array(prox_all.merge([lower, upper, stylus, implant]).bounds)
    roi[::2] -= 8
    roi[1::2] += 8
    roi[4] -= 25    # a little shaft below the cut

    def render(items, fname, direction, title, zoom=1.0):
        p = pv.Plotter(off_screen=True, window_size=(1100, 1100))
        p.set_background("white")
        for mesh, col, op, name in items:
            p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True, label=name)
        p.add_legend(bcolor="white", size=(0.38, 0.3), loc="lower right")
        p.add_text(title, font_size=11, color="black")
        p.camera.focal_point = centre
        p.camera.position = centre + 600 * unit(np.asarray(direction, float))
        p.camera.up = tuple(n0)
        p.reset_camera(bounds=roi)
        p.camera.zoom(zoom)
        p.screenshot(os.path.join(a.out_dir, fname))
        p.close()

    render(before, "jig_front.png", ANT + 0.15 * n0, "Tibial cut jig - front (cut perpendicular to anatomic axis)")
    render(before, "jig_lateral.png", L, f"Tibial cut jig - lateral ({a.slope:g} deg posterior slope)")
    render(before, "jig_oblique.png", ANT + 0.9 * L + 0.5 * n0, "Tibial cut jig - oblique")
    render(after, "after_cut_implant.png", ANT + 0.7 * L + 0.45 * n0,
           f"After the cut: {depth:g} mm tibial component on the resection (schematic)")
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.set_background("white")
    for mesh, col, op, name in [(kept["tibia"], BONE, 1, ""), (kept["split"], SPLIT, 1, ""),
                                (kept["depressed"], DEPR, 1, ""), (tip, "#e0b000", 1, "")]:
        p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True)
    p.add_text("Resected surface from above (orange/red = fracture fragments left under the baseplate)",
               font_size=10, color="black")
    p.camera.focal_point = o
    p.camera.position = o + n * 400
    p.camera.up = tuple(t_ap)
    p.reset_camera()
    p.camera.zoom(1.5)
    p.screenshot(os.path.join(a.out_dir, "cut_surface_top.png"))
    p.close()

    for mesh, name in [(implant, "tibial_component_schematic"), (kept["tibia"], "tibia_after_cut"),
                       (removed["tibia"], "resected_bone"), (lower.merge(upper), "cutting_block")]:
        mesh.save(os.path.join(a.out_dir, f"{name}.stl"))
    write_html(before + [(implant, IMPLANT, 0.0, "tibial component (hidden: click legend)")],
               np.array([ref]), os.path.join(a.out_dir, "tka_cut_3d.html"))
    write_html(after, np.array([ref]), os.path.join(a.out_dir, "tka_after_cut_3d.html"))
    np.savetxt(os.path.join(a.out_dir, "cut_plane.txt"), np.vstack([o, n, ref]), fmt="%.4f",
               header="row1: point on cut plane (mm), row2: plane normal (up), row3: stylus contact point")
    print("saved jig_*.png, after_cut_implant.png, cut_surface_top.png, tka_cut_3d.html, "
          "tka_after_cut_3d.html, STLs ->", a.out_dir)

    if a.show:
        p = pv.Plotter(title="TKA tibial cut - drag to rotate, q to quit")
        p.set_background("white")
        for mesh, col, op, name in before:
            p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True, label=name)
        p.add_legend(bcolor="white", size=(0.25, 0.35))
        p.camera.focal_point = centre
        p.camera.position = centre + 600 * unit(ANT + 0.9 * L + 0.5 * n0)
        p.camera.up = tuple(n0)
        p.reset_camera()
        p.show()


if __name__ == "__main__":
    main()
