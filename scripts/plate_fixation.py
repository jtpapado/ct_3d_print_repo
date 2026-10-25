"""Anterolateral proximal tibial locking plate + screws for the SIMULATED Schatzker II fracture,
following the Medartis APTUS 3.5 TriLock screw rules.

Reference (verified from the FDA 510(k) K193633 summary for the APTUS 2.8/3.5 TriLock system):
  - titanium plates and screws, 3.5 mm screws
  - multidirectional locking: each screw may deviate up to +-15 deg from its hole axis
  - screw lengths 8-60 mm (2 mm steps assumed here)
Medartis lists no proximal tibia plate (FDA UDI database, Oct 2026: tibia plates are distal only),
so the plate OUTLINE and THICKNESS here are generic/schematic - only the screw rules are Medartis.

Steps
  1. Reduction: fragments back in anatomic position (from simulate_knee_fracture.py, <sim_dir>/reduced/);
     the void under the elevated piece = bone graft.
  2. Plate contoured to the bone by ray casting; several plate positions (angle from straight
     lateral towards anterior) are tried and the one whose TriLock-compliant screws best hold
     both fragments is kept.
  3. Screws: raft row (support the elevated joint surface), fragment screws (through the split
     fragment into the main tibia), bicortical shaft screws. Every screw must: stay within 15 deg of
     its hole axis, be 8-60 mm, stay >= MIN_CLEAR below the joint, keep clear of other screws,
     and actually hold its fragment.

usage: python plate_fixation.py <Bones/NNN.mat> <simulated_knee_dir> <axis_dir> <out_dir> [--side R] [--show]
Schematic illustration on SIMULATED data - not a surgical plan.
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tibia_axis import load_bones, write_html  # noqa: E402

# Medartis APTUS 3.5 TriLock rules
TRILOCK_MAX_DEV = 15.0  # deg off the hole axis
LEN_MIN, LEN_MAX, LEN_STEP = 8, 60, 2
SCREW_R = 1.75          # 3.5 mm
# Schematic plate / planning settings
ANGLES = [15, 10, 5, 20, 0, 25, 30]   # plate positions tried, deg from lateral towards anterior
THICK = 3.2
MIN_CLEAR = 3.0         # raft screws: minimum distance below the joint surface
MIN_GAP = 4.0           # minimum distance between screw axes
HOLE_GAP = 7.5          # minimum distance between hole centres
HOLE_EDGE = 3.5         # minimum distance from a hole centre to the plate edge
HEAD_END = -40.0        # plate head: holes above this level (mm below the joint line) are joint-parallel
BONE, SPLIT, DEPR, GRAFT = "#e9d8b4", "#e8a07a", "#d65a5a", "#7fbf7f"
PLATE_C = "#a7b4c2"     # titanium
SCREW_C = {"raft": "#d4a017", "fragment": "#2a9d8f", "shaft": "#8c96a3"}


def unit(v):
    return v / np.linalg.norm(v)


def rot(axis, deg, v):
    a = np.radians(deg)
    k = unit(axis)
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k @ v) * (1 - np.cos(a))


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1)))


def hits(mesh, a, b):
    p, _ = mesh.ray_trace(a, b)
    if not len(p):
        return np.empty((0, 3))
    return p[np.argsort(np.linalg.norm(p - a, axis=1))]


def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-9 else 0.0
    t = np.clip((b * s + f) / e, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1)
    return np.linalg.norm(p1 + d1 * s - (p2 + d2 * t))


def pt_seg(P, p, q):
    d = q - p
    t = np.clip(((P - p) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(P - (p + t[:, None] * d), axis=1)


def width(h):
    return np.interp(-h, [4, 7, 34, 52, 106, 112], [24, 34, 34, 13, 13, 8])


ROWS = np.arange(-4, -112.1, -1.5)
COLS = np.linspace(-1, 1, 15)


class Planner:
    def __init__(self, a):
        self.a = a
        bones = load_bones(a.mat, a.side)
        self.bones = bones
        tib_full = bones["Tibia"]
        m_ax, d_ax, entry = np.loadtxt(os.path.join(a.axis_dir, "axis.txt"))
        self.n0 = n0 = -unit(d_ax)
        S, R = a.sim_dir, os.path.join(a.sim_dir, "reduced")
        self.main_t = pv.read(f"{S}/tibia_1_tibia.stl")
        self.split_r = pv.read(f"{R}/split_reduced.stl")
        self.depr_r = pv.read(f"{R}/depressed_reduced.stl")
        self.graft = pv.read(f"{R}/graft_void.stl")
        self.displaced = [pv.read(f"{S}/tibia_2_split.stl"), pv.read(f"{S}/tibia_3_depressed.stl")]
        self.z_top = tib_full.bounds[5]
        self.shaft = tib_full.clip("z", origin=(0, 0, self.z_top - 128), invert=True)
        self.bone = self.main_t.merge([self.split_r, self.depr_r, self.graft, self.shaft])

        fib = bones["Fibula"].points
        fib_top = fib[fib[:, 2] > fib[:, 2].max() - 60].mean(axis=0)
        sx = np.sign(fib_top[0] - m_ax[0])
        self.L = unit(np.array([sx, 0, 0.0]) - (sx * n0[0]) * n0)
        ANT = unit(np.cross(n0, self.L))
        self.ANT = -ANT if ANT[1] > 0 else ANT    # anterior is -y in this database

        top = self.main_t.points[(self.main_t.points - entry) @ n0 > -15]
        lat = (top - entry) @ self.L
        lo, hi = np.percentile(lat, [1, 99])
        lat_c = entry + self.L * ((lo + hi) / 2 + 0.25 * (hi - lo))
        h_lat = hits(self.bone, lat_c + n0 * 80, lat_c - n0 * 80)[0]
        self.base = entry + n0 * ((h_lat - entry) @ n0)  # axis point at the lateral joint-line height

    # ---------------- plate ----------------
    def build_plate(self, angle):
        n0 = self.n0
        dir_c = unit(np.cos(np.radians(angle)) * self.L + np.sin(np.radians(angle)) * self.ANT)
        tang = unit(np.cross(n0, dir_c))

        def surface(h, w):
            ax = self.base + n0 * h
            p = hits(self.bone, ax + dir_c * 90 + tang * w, ax + tang * w - dir_c * 5)
            return p[0] if len(p) else None

        inner = np.full((len(COLS), len(ROWS), 3), np.nan)
        for j, h in enumerate(ROWS):
            for i, u in enumerate(COLS):
                s = surface(h, u * width(h) / 2)
                if s is not None:
                    inner[i, j] = s
        for j in range(len(ROWS)):                               # fill misses from the centre outwards
            for i in sorted(range(len(COLS)), key=lambda i: abs(i - len(COLS) // 2)):
                if np.isnan(inner[i, j, 0]):
                    k = i + (1 if i < len(COLS) // 2 else -1)
                    inner[i, j] = inner[k, j] + tang * (COLS[i] - COLS[k]) * width(ROWS[j]) / 2
        inner = np.stack([ndi.gaussian_filter(inner[..., k], 1.2, mode="nearest") for k in range(3)], -1)
        inner += dir_c * 0.4
        # outward plate normals from the grid (used as the TriLock hole axes)
        du = np.gradient(inner, axis=0)
        dv = np.gradient(inner, axis=1)
        nrm = np.cross(du, dv)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        if np.nanmean(nrm @ dir_c) < 0:
            nrm = -nrm
        outer = inner + nrm * THICK
        g = np.stack([inner, outer], axis=2)
        plate = pv.StructuredGrid(g[..., 0], g[..., 1], g[..., 2]).extract_surface(
            algorithm="dataset_surface").triangulate().clean()
        plate = plate.compute_normals(auto_orient_normals=True)

        def hole(h, w):
            """Hole centre on the bone side, and its nominal axis (into the bone).
            Plate design (as in anatomic proximal tibia plates): head holes are joint-parallel,
            i.e. the plate normal with its up/down component removed; shaft holes are
            perpendicular to the plate. TriLock then allows +-15 deg around these axes."""
            j = int(np.argmin(np.abs(ROWS - h)))
            i = int(np.argmin(np.abs(COLS - w / (width(h) / 2))))
            s = surface(h, w)
            axis = -nrm[i, j]
            if h > HEAD_END:
                axis = unit(axis - (axis @ n0) * n0)
            return s, axis
        return dict(angle=angle, dir_c=dir_c, tang=tang, plate=plate, surface=surface, hole=hole, normals=nrm)

    # ---------------- screws ----------------
    def plan(self, angle, verbose=False):
        n0, L, ANT = self.n0, self.L, self.ANT
        ctx = self.build_plate(angle)
        dir_c = ctx["dir_c"]
        placed, log = [], []

        def far_cortex(s, v):
            h = hits(self.bone, s + v * 0.5, s + v * 140)
            return np.linalg.norm(h[-1] - s) if len(h) else None

        def clearance(s, v, ln):
            cl = []
            for t in np.arange(6, ln - 2, 2.0):
                q = s + v * t
                h = hits(self.bone, q, q + n0 * 70)
                if len(h):
                    cl.append((h[-1] - q) @ n0)
            return min(cl) if cl else np.inf

        def trilock(v, axis):
            """Clamp a desired direction into the +-15 deg TriLock cone around the hole axis."""
            dev = angle_between(v, axis)
            if dev <= TRILOCK_MAX_DEV:
                return unit(v)
            return unit(rot(np.cross(axis, v), TRILOCK_MAX_DEV, axis))

        def try_place(kind, h, w, directions, bicortical=False, require=None, quiet=False):
            if abs(w) > width(h) / 2 - HOLE_EDGE or h > ROWS[0] - HOLE_EDGE:
                if not quiet:
                    log.append(f"  {kind} hole h={h:.1f} w={w:g}: too close to the plate edge")
                return False
            s, axis = ctx["hole"](h, w)
            if s is None:
                return False
            head = s + (-axis) * (0.4 + THICK)
            if any(np.linalg.norm(head - p["head"]) < HOLE_GAP for p in placed):
                return False
            tried = set()
            for v0 in list(directions) + [axis]:
                v = trilock(v0, axis)
                key = tuple(np.round(v, 3))
                if key in tried:
                    continue
                tried.add(key)
                far = far_cortex(s, v)
                if far is None:
                    continue
                extra = 0.4 + THICK
                if bicortical:
                    length = LEN_STEP * np.ceil((far + 2 + extra) / LEN_STEP)
                    if length > LEN_MAX:
                        continue
                else:
                    length = min(LEN_STEP * np.floor((far - 3 + extra) / LEN_STEP), LEN_MAX)
                if length < max(LEN_MIN, 15):
                    continue
                tip = head + v * length
                in_bone = length - extra
                clear = clearance(s, v, in_bone) if kind != "shaft" else np.inf
                gap = min([seg_dist(head, tip, p["head"], p["tip"]) for p in placed] + [np.inf])
                if clear < MIN_CLEAR or gap < MIN_GAP or (require and not require(head, tip)):
                    continue
                tilt = np.degrees(np.arcsin(np.clip(-(v @ n0), -1, 1)))
                placed.append(dict(kind=kind, head=head, tip=tip, v=v, length=length,
                                   dev=angle_between(v, axis), tilt=0.0 if abs(tilt) < 0.5 else tilt,
                                   clear=clear, gap=gap, capped=(not bicortical and far - 3 + extra > LEN_MAX)))
                return True
            if not quiet:
                log.append(f"  {kind} hole h={h:.1f} w={w:g}: no TriLock-compliant direction held its fragment")
            return False

        def horizontal(v):
            return unit(v - (v @ n0) * n0)

        def tilted(v, degs):
            return [rot(np.cross(n0, v), t, v) for t in degs]

        # raft screws: parallel, joint-parallel, aimed under the elevated fragment
        dp = self.depr_r.points
        d_bottom = (dp @ n0).min()
        a_lo, a_hi = np.percentile(dp @ ANT, [15, 85])
        raft_h = (d_bottom - SCREW_R - 1.5) - (self.base @ n0)

        def supports(hd, tp):
            return pt_seg(dp, hd, tp).min() - SCREW_R < 3.0
        edge = width(raft_h) / 2 - HOLE_EDGE
        raft_w = [-edge + k * HOLE_GAP for k in range(4)]
        s_mid, _ = ctx["hole"](raft_h, raft_w[1])
        if s_mid is not None:
            centre = L * np.median(dp @ L) + ANT * ((a_lo + a_hi) / 2) + n0 * (s_mid @ n0)
            v_raft = horizontal(centre - s_mid)
            for w in raft_w[:3]:
                try_place("raft", raft_h, round(w, 1), tilted(v_raft, np.arange(0, 11, 1.0)), require=supports)
        try_place("raft", raft_h, round(raft_w[3], 1), tilted(horizontal(-dir_c), np.arange(0, 11, 1.0)))

        # fragment screws: holes over the split fragment, into the main tibia
        sp = self.split_r.points

        def crosses(hd, tp):
            return len(hits(self.split_r, hd, tp)) > 0
        n_frag = 0
        for h in np.arange(raft_h - 6, -32, -2.0):
            for w in np.arange(-16, 10, 2.0):
                if n_frag >= 2:
                    break
                s, axis = ctx["hole"](h, w)
                if s is None or np.min(np.linalg.norm(sp - s, axis=1)) > 1.0:
                    continue
                toward = horizontal(self.base + n0 * h - ANT * 5 - s)
                cands = [c for da in (0, 10, -10, 20, -20) for c in tilted(rot(n0, da, toward), (0, 5, 10))]
                n_frag += try_place("fragment", h, w, cands, require=crosses, quiet=True)

        for h in (-72, -87, -102):
            s, axis = ctx["hole"](h, 0)
            if s is None:
                continue
            to_centre = horizontal(self.base + n0 * h - s)        # through the middle of the shaft
            try_place("shaft", h, 0, [to_centre, axis], bicortical=True)

        n_support = sum(1 for p in placed if p["kind"] == "raft" and supports(p["head"], p["tip"]))
        split_fixed = any(crosses(p["head"], p["tip"]) for p in placed)
        score = (split_fixed, n_support > 0, min(n_support, 2), min(n_frag, 2), len(placed))
        return dict(ctx=ctx, placed=placed, log=log, score=score, n_support=n_support,
                    n_frag=n_frag, split_fixed=split_fixed, supports=supports, crosses=crosses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat"); ap.add_argument("sim_dir"); ap.add_argument("axis_dir"); ap.add_argument("out_dir")
    ap.add_argument("--side", default="R")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    P = Planner(a)

    print("plate position search (Medartis TriLock rules: +-15 deg, 8-60 mm):")
    best = None
    for ang in ANGLES:
        r = P.plan(ang)
        print(f"  {ang:2d} deg: {len(r['placed'])} screws, split fragment held: {'yes' if r['split_fixed'] else 'NO '}, "
              f"raft screws supporting the elevated piece: {r['n_support']}, fragment screws: {r['n_frag']}")
        if best is None or r["score"] > best["score"]:
            best = r
    ctx, placed = best["ctx"], best["placed"]
    print(f"chosen plate position: {ctx['angle']} deg from straight lateral towards anterior")
    for line in best["log"]:
        print(line)

    print(f"{'screw':9s} {'length':>7s} {'off hole axis':>14s} {'tilt':>6s} {'below joint':>12s} {'nearest':>9s}")
    for p in placed:
        cl = f"{p['clear']:.1f} mm" if np.isfinite(p["clear"]) else "-"
        print(f"{p['kind']:9s} {p['length']:5.0f}mm {p['dev']:12.1f}° {p['tilt']:5.0f}° {cl:>12s} "
              f"{p['gap']:7.1f}mm" + ("  (60 mm max: stops short of the far cortex)" if p["capped"] else ""))
    assert all(p["dev"] <= TRILOCK_MAX_DEV + 1e-6 and LEN_MIN <= p["length"] <= LEN_MAX for p in placed)
    print("fixation check:")
    for k, p in enumerate(placed, 1):
        notes = []
        if best["crosses"](p["head"], p["tip"]):
            notes.append("crosses split fragment")
        if len(hits(P.graft, p["head"], p["tip"])):
            notes.append("through graft bed")
        dd = pt_seg(P.depr_r.points, p["head"], p["tip"]).min() - SCREW_R
        if dd < 3:
            notes.append(f"{max(dd, 0):.1f} mm under elevated fragment (supports it)")
        print(f"  #{k} {p['kind']:9s} " + ("; ".join(notes) if notes else "main tibia only"))
    print(f"  split fragment fixed: {'YES' if best['split_fixed'] else 'NO'} | "
          f"elevated fragment supported: {'YES' if best['n_support'] else 'NO'}")
    print(f"graft / bone substitute: {abs(P.graft.volume) / 1000:.2f} ml")

    # ---------------- output ----------------
    n0, L, ANT, dir_c = P.n0, P.L, P.ANT, ctx["dir_c"]
    plate = ctx["plate"]
    screws = []
    for p in placed:
        shaft_m = pv.Cylinder(center=(p["head"] + p["tip"]) / 2, direction=p["v"], radius=SCREW_R,
                              height=p["length"], resolution=24)
        head_m = pv.Cylinder(center=p["head"] - p["v"] * 0.6, direction=p["v"], radius=2.9, height=2.6, resolution=24)
        screws.append((shaft_m.merge(head_m), SCREW_C[p["kind"]], 1.0, f"{p['kind']} 3.5 TriLock {p['length']:.0f} mm"))
    bone_parts = [(P.shaft, BONE, 1.0, "tibia shaft"), (P.main_t, BONE, 1.0, "proximal tibia"),
                  (P.split_r, SPLIT, 1.0, "split fragment (reduced)"),
                  (P.depr_r, DEPR, 1.0, "depressed fragment (elevated)"),
                  (P.graft, GRAFT, 1.0, "bone graft (void)"), (P.bones["Fibula"], "#cfc6b3", 1.0, "fibula")]
    hardware = [(plate, PLATE_C, 1.0, "anterolateral plate (Ti, schematic)")] + screws
    hw_ids = {id(h[0]) for h in hardware}
    everything = bone_parts + hardware
    roi = np.array(P.main_t.merge([P.split_r, plate]).bounds)
    roi[::2] -= 6
    roi[1::2] += 6
    centre = np.array(P.main_t.center)

    def render(items, fname, direction, title, bone_op=None, up=None):
        p = pv.Plotter(off_screen=True, window_size=(1100, 1200))
        p.set_background("white")
        for mesh, col, op, name in items:
            o = bone_op if (bone_op is not None and id(mesh) not in hw_ids) else op
            p.add_mesh(mesh, color=col, opacity=o, smooth_shading=True, label=name)
        p.add_legend(bcolor="white", size=(0.38, 0.3), loc="lower right")
        p.add_text(title, font_size=11, color="black")
        p.camera.focal_point = centre
        p.camera.position = centre + 600 * unit(np.asarray(direction, float))
        p.camera.up = tuple(up if up is not None else n0)
        p.reset_camera(bounds=roi)
        p.screenshot(os.path.join(a.out_dir, fname))
        p.close()

    tag = f"TriLock rules, plate at {ctx['angle']} deg"
    render(everything, "plate_lateral.png", dir_c, f"Anterolateral plate - face on ({tag})")
    render(everything, "plate_oblique.png", dir_c + 0.8 * ANT + 0.4 * n0, f"Anterolateral plate - oblique ({tag})")
    render(everything, "xray_ap.png", ANT, "AP view, bone see-through (like an X-ray)", bone_op=0.25)
    render(everything, "xray_lateral.png", L, "Lateral view, bone see-through", bone_op=0.25)
    render(everything, "raft_from_above.png", n0, "From above, bone see-through", bone_op=0.2, up=ANT)

    p = pv.Plotter(off_screen=True, window_size=(1700, 900), shape=(1, 2))
    for i, (ttl, items) in enumerate([
        ("displaced fracture", [(P.shaft, BONE, 1, ""), (P.main_t, BONE, 1, ""), (P.displaced[0], SPLIT, 1, ""),
                                (P.displaced[1], DEPR, 1, ""), (P.bones["Fibula"], "#cfc6b3", 1, "")]),
        ("reduced + plate fixation", everything)]):
        p.subplot(0, i)
        p.set_background("white")
        p.add_text(ttl, font_size=12, color="black")
        for mesh, col, op, _ in items:
            p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True)
        p.camera.focal_point = centre
        p.camera.position = centre + 600 * unit(dir_c + 0.6 * ANT + 0.35 * n0)
        p.camera.up = tuple(n0)
        p.reset_camera(bounds=roi)
    p.link_views()
    p.screenshot(os.path.join(a.out_dir, "before_after_fixation.png"))
    p.close()

    for f in os.listdir(a.out_dir):                      # clear screws from earlier runs
        if f.startswith("screw_") and f.endswith(".stl"):
            os.remove(os.path.join(a.out_dir, f))
    plate.save(os.path.join(a.out_dir, "plate.stl"))
    for k, (m, _, _, _) in enumerate(screws):
        m.save(os.path.join(a.out_dir, f"screw_{k + 1:02d}_{placed[k]['kind']}_{placed[k]['length']:.0f}mm.stl"))
    femur = P.bones["Femur"].clip("z", origin=(0, 0, P.z_top + 110), invert=True)
    write_html(everything + [(femur, "#f3ead6", 0.2, "femur")], np.array([q["head"] for q in placed]),
               os.path.join(a.out_dir, "plate_fixation_3d.html"))
    print("saved renders, plate/screw STLs, plate_fixation_3d.html ->", a.out_dir)

    if a.show:
        p = pv.Plotter(title="Plate fixation - drag to rotate, q to quit")
        p.set_background("white")
        for mesh, col, op, name in everything:
            p.add_mesh(mesh, color=col, opacity=op, smooth_shading=True, label=name)
        p.add_legend(bcolor="white", size=(0.25, 0.4))
        p.camera.focal_point = centre
        p.camera.position = centre + 600 * unit(dir_c + 0.8 * ANT + 0.4 * n0)
        p.camera.up = tuple(n0)
        p.reset_camera(bounds=roi)
        p.show()


if __name__ == "__main__":
    main()
