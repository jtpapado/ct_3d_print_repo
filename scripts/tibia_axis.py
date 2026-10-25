"""Anatomic axis of the tibia, drawn on the simulated fractured knee.

Anatomic axis = best-fit line through the centres of the tibial shaft (diaphysis).
Cross-sections are taken every SLICE_STEP mm between SHAFT_FROM and SHAFT_TO of the
tibia's length (measured from the top), perpendicular to the current axis estimate,
and the line is refitted until it settles.

usage: python tibia_axis.py <Bones/NNN.mat> <simulated_knee_dir> <out_dir> [side R|L] [--show]
  Always writes PNG renders and axis_3d.html (interactive, opens in a browser).
  --show also opens a rotatable 3D window.
"""
import os
import sys

import numpy as np
import pyvista as pv
import scipy.io

SHAFT_FROM, SHAFT_TO = 0.30, 0.80   # diaphysis portion used for the fit (fraction of length from top)
SLICE_STEP = 5.0                     # mm between cross-sections


def load_bones(path, side):
    B = scipy.io.loadmat(path)["B"]
    out = {}
    for b in range(B.size):
        e = B[0, b]
        name = str(e[0][0])
        if name.endswith("_" + side):
            v, f = e[2][0][0][0], e[2][0][0][1] - 1
            out[name[:-2]] = pv.PolyData(v.astype(float), np.hstack([np.full((len(f), 1), 3), f]).ravel())
    return out


def section_centroid(mesh, origin, normal):
    """Area centroid of the largest closed contour where a plane cuts the mesh."""
    sl = mesh.slice(normal=normal, origin=origin)
    if sl.n_points < 3:
        return None, 0.0
    best = (None, 0.0)
    for loop in sl.strip().cell:  # each strip is one contour, in order
        p = loop.points
        if len(p) < 3:
            continue
        # project onto the plane, shoelace formula
        n = normal / np.linalg.norm(normal)
        u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
        w = np.cross(n, u)
        q = np.c_[(p - origin) @ u, (p - origin) @ w]
        x, y = q[:, 0], q[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        area = cross.sum() / 2
        if abs(area) < 1e-6:
            continue
        cx, cy = ((x + x1) * cross).sum() / (6 * area), ((y + y1) * cross).sum() / (6 * area)
        if abs(area) > best[1]:
            best = (origin + cx * u + cy * w, abs(area))
    return best


def write_html(scene, centres, path, max_tris=40_000):
    """Standalone interactive 3D page (plotly) - open it in any web browser."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for mesh, col, op, name in scene:
        m = mesh.triangulate()
        if m.n_cells > max_tris:
            m = m.decimate_pro(1 - max_tris / m.n_cells, preserve_topology=True)
        f = m.faces.reshape(-1, 4)[:, 1:]
        v = m.points
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=col, opacity=op, name=name, showlegend=True, flatshading=False,
                                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.6)))
    fig.add_trace(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="markers",
                               marker=dict(size=3, color="#00a000"), name="shaft section centres"))
    fig.update_layout(title="Tibia anatomic axis on the SIMULATED Schatzker II fracture "
                            "(drag to rotate, scroll to zoom, click legend to hide parts)",
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False, camera=dict(eye=dict(x=-0.9, y=-1.6, z=0.3))),
                      margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
    fig.write_html(path, include_plotlyjs=True)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB) - open it in a web browser")


def fit_axis(tibia, iters=4):
    top, bot = tibia.bounds[5], tibia.bounds[4]
    length = top - bot
    d = np.array([0, 0, -1.0])                      # start: straight down
    p0 = np.array([*tibia.center[:2], top])
    for _ in range(iters):
        cents = []
        for s in np.arange(SHAFT_FROM * length, SHAFT_TO * length + 1e-6, SLICE_STEP):
            o = p0 + d * s / abs(d[2])              # point on current axis at depth s below the top
            c, a = section_centroid(tibia, o, d)
            if c is not None:
                cents.append(c)
        C = np.array(cents)
        m = C.mean(axis=0)
        _, _, vt = np.linalg.svd(C - m)
        d = vt[0] if vt[0][2] < 0 else -vt[0]        # unit direction, pointing distally
        p0 = m + d * (top - m[2]) / d[2]            # where the axis reaches the height of the top
    resid = np.linalg.norm(np.cross(C - m, d), axis=1)
    return m, d, C, resid, length


def main(mat, sim_dir, out_dir, side="R"):
    os.makedirs(out_dir, exist_ok=True)
    bones = load_bones(mat, side)
    tibia = bones["Tibia"]
    m, d, C, resid, length = fit_axis(tibia)

    # Anatomical frame (as in simulate_knee_fracture.py): z up, L lateral (fibula side), A posterior->anterior
    fib_top = bones["Fibula"].points[bones["Fibula"].points[:, 2] > bones["Fibula"].bounds[5] - 60].mean(axis=0)
    L = np.array([np.sign(fib_top[0] - m[0]), 0.0, 0.0])
    ANT = np.array([0, -1.0, 0])                   # anterior is -y in this database
    up = -d
    cor = np.degrees(np.arctan2(up @ L, up[2]))   # + = top of axis leans lateral
    sag = np.degrees(np.arctan2(up @ ANT, up[2])) # + = top of axis leans anterior

    # Where the axis exits the top of the (fractured) tibia: proximal entry point
    pieces = [pv.read(os.path.join(sim_dir, f)) for f in
              ("tibia_1_tibia.stl", "tibia_2_split.stl", "tibia_3_depressed.stl")]
    prox = pieces[0].merge(pieces[1:])
    start = m - d * 400
    pts, _ = prox.ray_trace(m, start)
    entry = pts[np.argmax(pts[:, 2])] if len(pts) else None
    pts_d, _ = tibia.ray_trace(m, m + d * 400)
    distal = pts_d[np.argmin(pts_d[:, 2])] if len(pts_d) else m + d * (m[2] - tibia.bounds[4]) / abs(d[2])

    top_pts = tibia.points[tibia.points[:, 2] > tibia.bounds[5] - 15]
    plateau_c = np.array([*((np.percentile(top_pts[:, :2], 1, axis=0) + np.percentile(top_pts[:, :2], 99, axis=0)) / 2),
                          tibia.bounds[5]])

    print(f"tibia length {length:.0f} mm; shaft fit on {len(C)} sections "
          f"({SHAFT_FROM:.0%}-{SHAFT_TO:.0%} of length), max deviation {resid.max():.2f} mm")
    print(f"axis direction (pointing down) {np.round(d, 4)}")
    print(f"tilt from vertical: {cor:+.1f} deg coronal (+ = leans lateral at top), "
          f"{sag:+.1f} deg sagittal (+ = leans anterior at top)")
    if entry is not None:
        off = entry - plateau_c
        print(f"proximal entry point: {off @ L:+.1f} mm lateral, {off @ ANT:+.1f} mm anterior "
              f"of the plateau's bounding-box centre")

    # Render: fractured proximal tibia + intact distal tibia, fibula, femur, axis
    distal_part = tibia.clip("z", origin=(0, 0, tibia.bounds[5] - 125), invert=True)
    axis_line = pv.Line(entry if entry is not None else m - d * length * 0.6, distal + d * 15)
    axis_tube = axis_line.tube(radius=1.2)
    ext = pv.Line(entry, entry - d * 40).tube(radius=0.8) if entry is not None else None
    cols = {"tibia_1": "#e9d8b4", "tibia_2": "#e8a07a", "tibia_3": "#d65a5a"}
    femur = bones["Femur"].clip("z", origin=(0, 0, tibia.bounds[5] + 110), invert=True)

    def render(fname, direction, title, femur_opacity=0.25):
        p = pv.Plotter(off_screen=True, window_size=(900, 1300))
        p.set_background("white")
        p.add_mesh(distal_part, color="#e9d8b4", smooth_shading=True, opacity=0.55)
        for (k, c), piece in zip(cols.items(), pieces):
            p.add_mesh(piece, color=c, smooth_shading=True, opacity=0.55 if k == "tibia_1" else 0.9)
        p.add_mesh(bones["Fibula"], color="#cfc6b3", smooth_shading=True, opacity=0.35)
        p.add_mesh(femur, color="#f3ead6", smooth_shading=True, opacity=femur_opacity)
        p.add_mesh(bones["Patella"], color="#f3ead6", smooth_shading=True, opacity=femur_opacity)
        p.add_mesh(axis_tube, color="#1f5fbf", label="anatomic axis")
        if ext is not None:
            p.add_mesh(ext, color="#1f5fbf", opacity=0.5)
            p.add_mesh(pv.Sphere(radius=2.5, center=entry), color="#ffd000", label="proximal entry point")
        p.add_points(C, color="#00a000", point_size=9, render_points_as_spheres=True, label="shaft section centres")
        p.add_legend(bcolor="white", size=(0.42, 0.09), loc="lower right")
        p.add_text(title, font_size=11, color="black")
        p.camera.focal_point = tibia.center
        p.camera.position = np.array(tibia.center) + 900 * np.asarray(direction, float)
        p.camera.up = (0, 0, 1)
        p.reset_camera()
        p.camera.zoom(1.15)
        p.screenshot(os.path.join(out_dir, fname))
        p.close()

    render("axis_front.png", ANT, f"Anatomic axis - front (coronal tilt {cor:+.1f} deg)")
    render("axis_side.png", L, f"Anatomic axis - lateral (sagittal tilt {sag:+.1f} deg)")
    render("axis_oblique.png", ANT + 0.8 * L + 0.25 * np.array([0, 0, 1]), "Anatomic axis - oblique")

    # Close-up of the plateau from above with the entry point
    p = pv.Plotter(off_screen=True, window_size=(900, 900))
    p.set_background("white")
    for (k, c), piece in zip(cols.items(), pieces):
        p.add_mesh(piece, color=c, smooth_shading=True)
    p.add_mesh(axis_tube, color="#1f5fbf")
    if entry is not None:
        p.add_mesh(pv.Sphere(radius=2.5, center=entry), color="#ffd000")
    p.add_text("Plateau from above: yellow = where the anatomic axis exits", font_size=11, color="black")
    p.camera.focal_point = plateau_c
    p.camera.position = plateau_c + np.array([0, 0, 400])
    p.camera.up = tuple(ANT)
    p.reset_camera()
    p.camera.zoom(1.6)
    p.screenshot(os.path.join(out_dir, "axis_entry_top.png"))
    p.close()

    # Scene for the interactive views: (mesh, colour, opacity, name)
    scene = [(distal_part, "#e9d8b4", 0.5, "tibia shaft"),
             (pieces[0], "#e9d8b4", 0.5, "proximal tibia"),
             (pieces[1], "#e8a07a", 1.0, "split fragment"),
             (pieces[2], "#d65a5a", 1.0, "depressed fragment"),
             (bones["Fibula"], "#cfc6b3", 0.3, "fibula"),
             (femur, "#f3ead6", 0.25, "femur"),
             (bones["Patella"], "#f3ead6", 0.25, "patella"),
             (axis_tube, "#1f5fbf", 1.0, "anatomic axis")]
    if ext is not None:
        scene += [(ext, "#1f5fbf", 0.5, "axis (above bone)"),
                  (pv.Sphere(radius=2.5, center=entry), "#ffd000", 1.0, "proximal entry point")]
    write_html(scene, C, os.path.join(out_dir, "axis_3d.html"))
    if SHOW:
        p = pv.Plotter(title="Tibia anatomic axis - drag to rotate, scroll to zoom, q to quit")
        p.set_background("white")
        for mesh, col, op, name in scene:
            p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True, label=name)
        p.add_points(C, color="#00a000", point_size=8, render_points_as_spheres=True, label="shaft section centres")
        p.add_legend(bcolor="white", size=(0.25, 0.3))
        p.camera.position = np.array(tibia.center) + 900 * (ANT + 0.5 * L)
        p.camera.focal_point = tibia.center
        p.camera.up = (0, 0, 1)
        p.reset_camera()
        p.show()

    np.savetxt(os.path.join(out_dir, "axis.txt"), np.vstack([m, d, entry if entry is not None else m]),
               header="row1: point on axis (mm), row2: unit direction (distal), row3: proximal entry point",
               fmt="%.4f")
    print("saved axis_front.png, axis_side.png, axis_oblique.png, axis_entry_top.png, axis.txt ->", out_dir)


SHOW = "--show" in sys.argv

if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if a != "--show"][:4])
