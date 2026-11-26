bl_info = {
    "name": "Split & Pin",
    "author": "Claude",
    "version": (1, 5, 0),
    "blender": (3, 2, 0),
    "location": "View3D > Sidebar > Split & Pin",
    "description": "Split a mesh with a plane or a hand-drawn path and add pins and sockets for 3D printing",
    "category": "Object",
}

import math
import os
import re
import textwrap

import bmesh
import bpy
import gpu
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import interpolate_bezier

AXIS_INDEX = {'X': 0, 'Y': 1, 'Z': 2}
TAU = 2.0 * math.pi
ASSUMED_PRINT_SIZE_MM = 150.0  # used by Auto Size when scene units give an unrealistic size


class SplitError(Exception):
    """A problem worth showing to the user."""


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def _remove_object(ob):
    if ob is None:
        return
    data = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


def _mesh_object(name, bm):
    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return bpy.data.objects.new(name, mesh)


def _make_box(name, mn, mx):
    """Axis-aligned box object in world space (identity transform)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    c = (mn + mx) / 2.0
    h = (mx - mn) / 2.0
    for v in bm.verts:
        v.co = Vector((c.x + v.co.x * h.x, c.y + v.co.y * h.y, c.z + v.co.z * h.z))
    return _mesh_object(name, bm)


def _plane_basis(axis):
    """Right-handed (e1, e2, n) with n along the chosen world axis."""
    x, y, z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    return {0: (y, z, x), 1: (z, x, y), 2: (x, y, z)}[axis]


def _basis_from_normal(n):
    ref = Vector((1, 0, 0)) if abs(n.x) < 0.9 else Vector((0, 1, 0))
    e1 = (ref - n * ref.dot(n)).normalized()
    e2 = n.cross(e1)
    return e1, e2, n


def _cylinders_object(name, specs, basis, segments):
    """specs: list of (center, radius, length). Cylinders run along basis n."""
    e1, e2, n = basis
    bm = bmesh.new()
    for center, radius, length in specs:
        half = length / 2.0
        lo, hi = [], []
        for i in range(segments):
            a = TAU * i / segments
            off = e1 * (math.cos(a) * radius) + e2 * (math.sin(a) * radius)
            lo.append(bm.verts.new(center + off - n * half))
            hi.append(bm.verts.new(center + off + n * half))
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
        bm.faces.new(lo)
        bm.faces.new(hi)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _mesh_object(name, bm)


def _bake_modifiers(ob):
    """Replace the object's mesh with its evaluated result and drop its modifiers."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    new_mesh = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    name = old.name
    ob.modifiers.clear()
    ob.data = new_mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    new_mesh.name = name
    if len(new_mesh.polygons) == 0:
        raise SplitError(
            f"Boolean on '{ob.name}' produced an empty mesh. "
            "The model must be a closed (manifold) mesh and the cut must pass through it."
        )


def _boolean(ob, other, operation):
    mod = ob.modifiers.new("SplitPin", 'BOOLEAN')
    mod.operation = operation
    mod.operand_type = 'OBJECT'
    mod.object = other
    mod.solver = 'EXACT'
    _bake_modifiers(ob)


def _duplicate(src, name, coll):
    ob = src.copy()
    ob.data = src.data.copy()
    ob.name = name
    ob.data.name = name
    mw = src.matrix_world.copy()
    ob.parent = None
    ob.matrix_world = mw
    coll.objects.link(ob)
    return ob


# ----------------------------------------------------------------------------
# Plane cut
# ----------------------------------------------------------------------------

def _plane_cut(src, s, lo, hi, diag, cutters=True):
    axis = AXIS_INDEX[s.axis]
    pos = s.position
    if not (lo[axis] < pos < hi[axis]):
        raise SplitError("The cut plane doesn't pass through the object. "
                         "Adjust the position or use 'Center'.")
    margin = diag * 0.1 + 1e-3

    def box(name, a_lo, a_hi):
        mn = lo - Vector((margin,) * 3)
        mx = hi + Vector((margin,) * 3)
        mn[axis], mx[axis] = a_lo, a_hi
        return _make_box(name, mn, mx)

    cut_a = box("_cutter_A", lo[axis] - margin, pos) if cutters else None
    cut_b = box("_cutter_B", pos, hi[axis] + margin) if cutters else None
    basis = _plane_basis(axis)
    o0, o1 = (axis + 1) % 3, (axis + 2) % 3  # the plane basis' e1 and e2

    def field():
        pts, _area, centroid, seg_a, seg_b = _plane_section(src, axis, pos)
        if pts is None:
            raise SplitError("Could not find the cut face to place connectors on.")

        def surface(u, v):
            p = Vector((0.0, 0.0, 0.0))
            p[axis], p[o0], p[o1] = pos, u, v
            return p

        return {'surface': surface, 'center': centroid, 'segments': (seg_a, seg_b)}

    return cut_a, cut_b, basis, field


# ----------------------------------------------------------------------------
# Path cut
# ----------------------------------------------------------------------------

def _read_path_points(ob):
    """World-space points of the first spline. Returns (points, can_be_smoothed)."""
    cu = ob.data
    if not cu.splines:
        raise SplitError("The path curve has no spline.")
    sp = cu.splines[0]
    mw = ob.matrix_world
    pts = []
    if sp.type == 'BEZIER':
        bp = sp.bezier_points
        count = len(bp)
        res = max(3, cu.resolution_u)
        segs = count if sp.use_cyclic_u else count - 1
        for i in range(segs):
            a, b = bp[i], bp[(i + 1) % count]
            seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, res)
            pts.extend(seg[:-1])
        if not sp.use_cyclic_u:
            pts.append(bp[-1].co.copy())
        smooth_ok = False  # already a smooth curve
    else:
        pts = [p.co.xyz.copy() for p in sp.points]
        smooth_ok = True
    return [mw @ Vector(p) for p in pts], smooth_ok


def _catmull_rom_closed(pts, k):
    m = len(pts)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = pts[(i - 1) % m], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
        for j in range(k):
            t = j / k
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2.0 * p1) + (p2 - p0) * t
                              + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                              + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3))
    return out


def _linear_closed(pts, k):
    m = len(pts)
    out = []
    for i in range(m):
        a, b = pts[i], pts[(i + 1) % m]
        for j in range(k):
            out.append(a.lerp(b, j / k))
    return out


_bvh_cache = {'use_key': None, 'key': None, 'bvh': None}


def _get_bvh(src, depsgraph):
    """BVH of the evaluated mesh. Reused between calls only while the preview has set a
    cache key; the real split always builds a fresh one."""
    key = _bvh_cache['use_key']
    if key is not None and _bvh_cache['key'] == key:
        return _bvh_cache['bvh']
    bvh = BVHTree.FromObject(src, depsgraph)
    if key is not None:
        _bvh_cache['key'], _bvh_cache['bvh'] = key, bvh
    return bvh


def _snap_points(src, depsgraph, pts, eps):
    """Move points onto the mesh surface, nudged just outside along the normal."""
    bvh = _get_bvh(src, depsgraph)
    mw = src.matrix_world
    inv = mw.inverted()
    nm = inv.to_3x3().transposed()
    out = []
    for p in pts:
        loc, nrm, _idx, _dist = bvh.find_nearest(inv @ p)
        if loc is None:
            out.append(p)
            continue
        out.append(mw @ loc + (nm @ nrm).normalized() * eps)
    return out


def _path_normal(pts, c, mode):
    if mode in AXIS_INDEX:
        v = Vector((0.0, 0.0, 0.0))
        v[AXIS_INDEX[mode]] = 1.0
        return v
    arr = np.array([tuple(p - c) for p in pts], dtype=np.float64)
    _u, sv, vt = np.linalg.svd(arr, full_matrices=False)
    if len(sv) < 2 or sv[1] < sv[0] * 1e-3:
        raise SplitError("The path points are almost in a straight line. "
                         "Place points all the way around the model.")
    n = Vector(vt[-1].tolist()).normalized()
    i = max(range(3), key=lambda k: abs(n[k]))
    return -n if n[i] < 0 else n


def _wrap(a):
    return (a + math.pi) % TAU - math.pi


def _gaps(recs):
    m = len(recs)
    return [_wrap(recs[(i + 1) % m][0] - recs[i][0]) for i in range(m)]


def _build_path_cutter(name, c, basis, ring, R, h_cap, fan_h):
    """Closed volume bounded by the cut sheet on one side and a flat cap on the other."""
    e1, e2, n = basis

    def P(u, v, h):
        return c + e1 * u + e2 * v + n * h

    bm = bmesh.new()
    center = bm.verts.new(P(0.0, 0.0, fan_h))
    inner, outer, top = [], [], []
    for u, v, h in ring:
        r = math.hypot(u, v)
        ux, vy = u / r * R, v / r * R
        inner.append(bm.verts.new(P(u, v, h)))
        outer.append(bm.verts.new(P(ux, vy, h)))
        top.append(bm.verts.new(P(ux, vy, h_cap)))
    m = len(ring)
    for i in range(m):
        j = (i + 1) % m
        bm.faces.new((center, inner[i], inner[j]))
        bm.faces.new((inner[i], outer[i], outer[j], inner[j]))
        bm.faces.new((outer[i], top[i], top[j], outer[j]))
    bm.faces.new(top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    return _mesh_object(name, bm)


def _fan_height(ring, thetas, hc, u, v):
    """Height of the cut sheet at (u, v) inside the loop, or None outside it."""
    m = len(ring)
    th = math.atan2(v, u)
    for k in range(m):
        k2 = (k + 1) % m
        span = (thetas[k2] - thetas[k]) % TAU
        if (th - thetas[k]) % TAU <= span:
            uk, vk, hk = ring[k]
            ul, vl, hl = ring[k2]
            det = uk * vl - vk * ul
            if abs(det) < 1e-12:
                return None
            a = (u * vl - v * ul) / det
            b = (uk * v - vk * u) / det
            if a >= 0.0 and b >= 0.0 and a + b <= 1.0:
                return hc * (1.0 - a - b) + a * hk + b * hl
            return None
    return None


def _build_path_cut(src, s, lo, hi, diag, cutters=True):
    path_ob = s.path_object
    if path_ob is None or path_ob.type != 'CURVE':
        raise SplitError("Draw a cut path first, or pick a curve object as the path.")

    ctrl, smooth_ok = _read_path_points(path_ob)
    if len(ctrl) < 3:
        raise SplitError("The cut path needs at least 3 points.")

    c0 = sum(ctrl, Vector((0.0, 0.0, 0.0))) / len(ctrl)
    n = _path_normal(ctrl, c0, s.path_axis)
    basis = _basis_from_normal(n)
    e1, e2, n = basis

    if s.path_sort:
        ctrl.sort(key=lambda p: math.atan2((p - c0).dot(e2), (p - c0).dot(e1)))

    k = max(1, s.path_detail) if (s.path_curved or s.path_snap) else 1
    if s.path_curved and smooth_ok:
        dense = _catmull_rom_closed(ctrl, k)
    elif k > 1 and smooth_ok:
        dense = _linear_closed(ctrl, k)
    else:
        dense = list(ctrl)

    if s.path_snap:
        dense = _snap_points(src, bpy.context.evaluated_depsgraph_get(), dense, diag * 1e-4)

    c = sum(dense, Vector((0.0, 0.0, 0.0))) / len(dense)

    recs = []
    for p in dense:
        d = p - c
        u, v, h = d.dot(e1), d.dot(e2), d.dot(n)
        if math.hypot(u, v) < diag * 1e-6:
            continue
        recs.append((math.atan2(v, u), u, v, h))
    if s.path_sort:
        recs.sort(key=lambda r: r[0])

    clean = []
    for r in recs:
        if clean and abs(_wrap(r[0] - clean[-1][0])) < 1e-6:
            continue
        clean.append(r)
    if len(clean) > 1 and abs(_wrap(clean[0][0] - clean[-1][0])) < 1e-6:
        clean.pop()
    recs = clean
    if len(recs) < 3:
        raise SplitError("The cut path collapsed to fewer than 3 distinct points.")

    total = sum(_gaps(recs))
    if abs(abs(total) - TAU) > 1e-3:
        raise SplitError("The path doesn't go once around the model when viewed along the cut "
                         "direction. Enable 'Sort Points', or place points in a single loop.")
    if total < 0:
        recs.reverse()
    gaps = _gaps(recs)
    if min(gaps) < 1e-7 or max(gaps) > math.pi * 0.95:
        raise SplitError("The path has a gap that is too wide. Add points so the loop "
                         "surrounds the model evenly.")

    ring = [(u, v, h) for _t, u, v, h in recs]

    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    hs = [(q - c).dot(n) for q in corners]
    rs = [math.hypot((q - c).dot(e1), (q - c).dot(e2)) for q in corners]
    margin = diag * 0.1 + 1e-3
    R = max(max(rs), max(math.hypot(u, v) for u, v, _h in ring)) * 1.5 + margin
    h_top = max(max(hs), max(h for _u, _v, h in ring)) + margin
    h_bot = min(min(hs), min(h for _u, _v, h in ring)) - margin
    fan_h = sum(h for _u, _v, h in ring) / len(ring)

    cut_a = _build_path_cutter("_cutter_A", c, basis, ring, R, h_bot, fan_h) if cutters else None
    cut_b = _build_path_cutter("_cutter_B", c, basis, ring, R, h_top, fan_h) if cutters else None
    thetas = [math.atan2(v, u) for u, v, _h in ring]
    uv = np.array([(u, v) for u, v, _h in ring], dtype=np.float64)

    def field():
        def surface(u, v):
            h = _fan_height(ring, thetas, fan_h, u, v)
            return None if h is None else c + e1 * u + e2 * v + n * h

        return {'surface': surface,
                'center': tuple(float(x) for x in _polygon_centroid(uv)),
                'segments': (uv, np.roll(uv, -1, axis=0))}

    return cut_a, cut_b, basis, field


# ----------------------------------------------------------------------------
# Pins
# ----------------------------------------------------------------------------

def _inside_tester(src, depsgraph, eps):
    """Returns a function telling whether a world-space point is inside the solid."""
    bvh = _get_bvh(src, depsgraph)
    inv = src.matrix_world.inverted()
    dirs = [
        Vector((1.0, 0.0123, 0.0071)).normalized(),
        Vector((0.0087, 1.0, 0.0191)).normalized(),
        Vector((0.0151, 0.0049, 1.0)).normalized(),
    ]

    def inside(p):
        lp = inv @ p
        votes = 0
        for d in dirs:
            hits = 0
            o = lp.copy()
            for _ in range(256):
                loc, _n, _i, _dist = bvh.ray_cast(o, d)
                if loc is None:
                    break
                hits += 1
                o = loc + d * eps
            if hits % 2 == 1:
                votes += 1
        return votes >= 2

    return inside


def _layout_offsets(mode, count, spacing, angle):
    """(du, dv) offsets of the pins from the cut's centre, in the cut's (e1, e2) plane.

    CENTER: one pin in the middle, the others on a circle of radius `spacing`.
    RING:   every pin on a circle of radius `spacing` (a lone pin sits in the middle).
    LINE:   a straight row centred on the middle, `spacing` between neighbours.
    `angle` rotates the whole layout around the cut direction.
    """
    if count <= 0:
        return []
    if mode == 'LINE':
        ca, sa = math.cos(angle), math.sin(angle)
        return [(ca * spacing * (j - (count - 1) / 2.0), sa * spacing * (j - (count - 1) / 2.0))
                for j in range(count)]
    if mode == 'CENTER':
        out = [(0.0, 0.0)]
        m = count - 1
    else:
        if count == 1:
            return [(0.0, 0.0)]
        out = []
        m = count
    for k in range(m):
        a = angle + TAU * k / m
        out.append((spacing * math.cos(a), spacing * math.sin(a)))
    return out


def _min_distance_needed(mode, count, pin):
    """Smallest spacing that keeps neighbouring sockets from touching (pin = radius + clearance)."""
    gap = 3.0 * pin
    if count <= 1:
        return 0.0
    if mode == 'LINE':
        return gap
    m = count - 1 if mode == 'CENTER' else count
    need = gap / (2.0 * math.sin(math.pi / m)) if m >= 2 else gap
    return max(need, gap) if mode == 'CENTER' else need


def _seg_min_dist(c, seg_a, seg_b):
    """Smallest distance from point c to a set of 2D segments."""
    ab = seg_b - seg_a
    ac = c - seg_a
    t = np.einsum('ij,ij->i', ac, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-18)
    proj = seg_a + ab * np.clip(t, 0.0, 1.0)[:, None]
    return float(np.min(np.linalg.norm(proj - c, axis=1)))


def _ray_exit(c, d, seg_a, seg_b):
    """Distance along unit direction d from c to the first segment it crosses, or None."""
    e = seg_b - seg_a
    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(denom) > 1e-12
    safe = np.where(ok, denom, 1.0)
    w = seg_a - c
    t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / safe
    s = (w[:, 0] * d[1] - w[:, 1] * d[0]) / safe
    hit = ok & (t > 1e-9) & (s >= 0.0) & (s <= 1.0)
    return float(t[hit].min()) if hit.any() else None


def _line_half_length(c, angle, seg_a, seg_b):
    """How far a line through c at `angle` runs inside the outline, measured to the nearer end."""
    d = np.array([math.cos(angle), math.sin(angle)])
    fwd = _ray_exit(c, d, seg_a, seg_b)
    back = _ray_exit(c, -d, seg_a, seg_b)
    if fwd is None or back is None:
        return 0.0
    return min(fwd, back)


def _polygon_centroid(uv):
    pts = np.asarray(uv, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return pts.mean(axis=0)
    return np.array([((x + x2) * cr).sum(), ((y + y2) * cr).sum()]) / (6.0 * a)


def _auto_spacing(field, r, clear, count, mode, angle):
    """Spacing that spreads the pins comfortably inside the cut outline."""
    if count <= 1:
        return 0.0
    pin = r + clear
    margin = 1.4 * pin + 0.5 * r  # a pin, its socket wall and a little extra must stay inside
    need = _min_distance_needed(mode, count, pin)
    c = np.array(field['center'], dtype=np.float64)
    seg_a, seg_b = field['segments']
    if mode == 'LINE':
        half = _line_half_length(c, angle, seg_a, seg_b) - margin
        spacing = 0.85 * half / ((count - 1) / 2.0) if half > 0.0 else need
    else:
        usable = _seg_min_dist(c, seg_a, seg_b) - margin
        spacing = 0.7 * usable if usable > 0.0 else need
    return max(spacing, need)


def _max_pins(field, r, clear, mode, angle):
    """Most pins that can fit in the cut with this layout."""
    pin = r + clear
    margin = 1.4 * pin + 0.5 * r
    gap = 3.0 * pin
    c = np.array(field['center'], dtype=np.float64)
    seg_a, seg_b = field['segments']
    if mode == 'LINE':
        half = _line_half_length(c, angle, seg_a, seg_b) - margin
        return 1 + int(2.0 * half // gap) if half > 0.0 else 1
    usable = _seg_min_dist(c, seg_a, seg_b) - margin
    if mode == 'CENTER':
        if usable < gap:
            return 1
        return 1 + int(math.pi / math.asin(min(1.0, gap / (2.0 * usable))))
    if usable < gap / 2.0:
        return 1
    return max(1, int(math.pi / math.asin(min(1.0, gap / (2.0 * usable)))))


def _place_pins(field, offsets, fits, spacing, pin):
    """Turn layout offsets into points on the cut surface.

    First every pin is tried at its exact spot. Pins whose spot doesn't fit are then
    nudged a little, avoiding the pins already placed; if nothing nearby fits they are dropped.
    """
    surface = field['surface']
    cu, cv = field['center']
    min_gap = 3.0 * pin
    tol = max(0.25 * spacing, 1.5 * pin)
    nudges = []
    for rad in (tol / 3.0, 2.0 * tol / 3.0, tol):
        for k in range(8):
            a = k * math.pi / 4.0
            nudges.append((rad * math.cos(a), rad * math.sin(a)))

    def try_spot(du, dv, placed):
        p = surface(cu + du, cv + dv)
        if p is None:
            return None
        if any((p - q).length < min_gap * 0.999 for q in placed):
            return None
        return p if fits(p) else None

    placed, pending = [], []
    for du, dv in offsets:
        p = try_spot(du, dv, placed)
        if p is None:
            pending.append((du, dv))
        else:
            placed.append(p)

    for du, dv in pending:
        for su, sv in nudges:
            p = try_spot(du + su, dv + sv, placed)
            if p is not None:
                placed.append(p)
                break
    return placed


def _resolve_spacing(field, s):
    """The pin distance to use: automatic when 0, and never below what keeps sockets apart.
    Returns (spacing, raised) where raised is True if the typed value was too small."""
    r, clear = s.pin_radius, s.clearance
    spacing = s.pin_spacing
    if spacing <= 0.0:
        return _auto_spacing(field, r, clear, s.pin_count, s.pin_layout, s.pin_angle), False
    need = _min_distance_needed(s.pin_layout, s.pin_count, r + clear)
    if spacing < need:
        return need, True
    return spacing, False


def _select_pins(field, basis, src, s, diag, spacing):
    e1, e2, n = basis
    inside = _inside_tester(src, bpy.context.evaluated_depsgraph_get(), diag * 1e-5)
    r, clear, depth = s.pin_radius, s.clearance, s.pin_depth
    ring = (r + clear) * 1.4

    def fits(p):
        for off in (0.0, -depth, depth):
            c = p + n * off
            if not inside(c):
                return False
            for k in range(4):
                ang = k * math.pi / 2.0
                if not inside(c + e1 * (math.cos(ang) * ring) + e2 * (math.sin(ang) * ring)):
                    return False
        return True

    offsets = _layout_offsets(s.pin_layout, s.pin_count, spacing, s.pin_angle)
    return _place_pins(field, offsets, fits, spacing, r + clear)


# ----------------------------------------------------------------------------
# Auto sizing
# ----------------------------------------------------------------------------

def _mm_per_bu(scene):
    """Millimetres per Blender unit, matching what the STL export will write."""
    us = scene.unit_settings
    if scene.splitpin.export_use_unit_scale and us.system != 'NONE':
        return us.scale_length * 1000.0
    return 1.0


def _extents_2d(pts2):
    """(major, minor) extents of a 2D point cloud along its principal axes."""
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts2) < 3:
        return None
    q = pts2 - pts2.mean(axis=0)
    _u, _sv, vt = np.linalg.svd(q, full_matrices=False)
    proj = q @ vt.T
    ext = proj.max(axis=0) - proj.min(axis=0)
    return float(ext.max()), float(ext.min())


def _section_from_triangles(world, tris, axis, pos):
    """Cross-section of a closed triangle mesh at the plane coord[axis] == pos.

    world: (n, 3) vertex positions, tris: (m, 3) vertex indices.
    Returns (outline points (k, 3), area, centroid (u, v), segment starts, segment ends),
    where u, v are the coordinates along axes (axis+1)%3 and (axis+2)%3.
    Returns (None, 0.0, None, None, None) when the plane misses the mesh.
    The area comes from summing the oriented cut segments, so holes subtract
    and separate islands add.
    """
    tri = world[tris]
    d = tri[:, :, axis] - pos
    pos_side = d >= 0.0
    cnt = pos_side.sum(axis=1)
    sel = (cnt == 1) | (cnt == 2)
    if not sel.any():
        return None, 0.0, None, None, None
    tri, d, pos_side, cnt = tri[sel], d[sel], pos_side[sel], cnt[sel]

    # the "lone" vertex is the one on its own side of the plane
    lone = np.where((cnt == 1)[:, None], pos_side, ~pos_side).argmax(axis=1)
    rows = np.arange(len(tri))
    i1, i2 = (lone + 1) % 3, (lone + 2) % 3
    a, b, c = tri[rows, lone], tri[rows, i1], tri[rows, i2]
    da, db, dc = d[rows, lone], d[rows, i1], d[rows, i2]
    p1 = a + (b - a) * (da / (da - db))[:, None]
    p2 = a + (c - a) * (da / (da - dc))[:, None]

    # orient every segment counter-clockwise as seen from +axis
    e_a = np.zeros(3)
    e_a[axis] = 1.0
    normal = np.cross(b - a, c - a)
    flip = np.einsum('ij,ij->i', np.cross(p2 - p1, e_a), normal) < 0.0
    q1 = np.where(flip[:, None], p2, p1)
    q2 = np.where(flip[:, None], p1, p2)

    o0, o1 = (axis + 1) % 3, (axis + 2) % 3
    center = q1.mean(axis=0)
    u1, u2 = q1 - center, q2 - center
    cr = u1[:, o0] * u2[:, o1] - u1[:, o1] * u2[:, o0]
    signed = 0.5 * float(cr.sum())
    if abs(signed) > 1e-12:
        cx = float(((u1[:, o0] + u2[:, o0]) * cr).sum()) / (6.0 * signed) + center[o0]
        cy = float(((u1[:, o1] + u2[:, o1]) * cr).sum()) / (6.0 * signed) + center[o1]
    else:
        cx, cy = float(center[o0]), float(center[o1])
    return (np.vstack([q1, q2]), abs(signed), (cx, cy),
            q1[:, [o0, o1]], q2[:, [o0, o1]])


def _plane_section(src, axis, pos):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = src.evaluated_get(dg)
    mesh = ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3).astype(np.float64)
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', tris)
        tris = tris.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()
    if len(tris) == 0:
        return None, 0.0, None, None, None
    m = np.array(src.matrix_world)
    world = co @ m[:3, :3].T + m[:3, 3]
    return _section_from_triangles(world, tris, axis, pos)


def _polygon_area(uv):
    m = len(uv)
    return 0.5 * abs(sum(uv[i][0] * uv[(i + 1) % m][1] - uv[(i + 1) % m][0] * uv[i][1]
                         for i in range(m)))


def _measure_cut(src, s):
    """Measure the cut. Returns a dict (all lengths in Blender units):

    major, minor: extents of the cut outline; area: area of the cut face;
    room: distance from the cut to the nearest end of the model along the cut direction;
    center, segments: outline centroid and edges in the cut's 2D coordinates
    (the same format the pin layout uses).
    """
    corners = [src.matrix_world @ Vector(c) for c in src.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])

    if s.cut_mode == 'PLANE':
        axis = AXIS_INDEX[s.axis]
        pos = s.position
        if not (lo[axis] < pos < hi[axis]):
            raise SplitError("The cut plane doesn't pass through the object.")
        pts, area, center, seg_a, seg_b = _plane_section(src, axis, pos)
        if pts is None:
            raise SplitError("Couldn't find a cross-section at the cut plane.")
        o0, o1 = (axis + 1) % 3, (axis + 2) % 3
        ext = _extents_2d(pts[:, [o0, o1]])
        room = min(pos - lo[axis], hi[axis] - pos)
    else:
        path_ob = s.path_object
        if path_ob is None or path_ob.type != 'CURVE':
            raise SplitError("Draw a cut path first, or pick a curve object as the path.")
        ctrl, _smooth = _read_path_points(path_ob)
        if len(ctrl) < 3:
            raise SplitError("The cut path needs at least 3 points.")
        c0 = sum(ctrl, Vector((0.0, 0.0, 0.0))) / len(ctrl)
        e1, e2, n = _basis_from_normal(_path_normal(ctrl, c0, s.path_axis))
        uv = [((p - c0).dot(e1), (p - c0).dot(e2)) for p in ctrl]
        ext = _extents_2d(uv)
        if s.path_sort:
            uv.sort(key=lambda t: math.atan2(t[1], t[0]))
        area = _polygon_area(uv)
        arr = np.array(uv, dtype=np.float64)
        center = tuple(float(x) for x in _polygon_centroid(arr))
        seg_a, seg_b = arr, np.roll(arr, -1, axis=0)
        hs = [(q - c0).dot(n) for q in corners]
        room = min(-min(hs), max(hs))

    if ext is None:
        raise SplitError("The cut outline is too small to measure.")
    major, minor = ext
    if area <= 0.0 or area > major * minor * 1.02:
        area = math.pi / 4.0 * major * minor  # measurement failed; assume an ellipse
    return {'major': major, 'minor': minor, 'room': max(room, 1e-9), 'area': area,
            'center': center, 'segments': (seg_a, seg_b)}


# ----------------------------------------------------------------------------
# STL export
# ----------------------------------------------------------------------------

def _safe_name(name):
    return re.sub(r'[^\w\-. ]', '_', name).strip() or "part"


def _write_binary_stl(path, ob, scale):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    mesh = ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_tris = len(mesh.loop_triangles)

        verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', verts)
        verts = verts.reshape(-1, 3).astype(np.float64)

        idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', idx)
        idx = idx.reshape(-1, 3)

        m = np.array(ob.matrix_world)
        world = (verts @ m[:3, :3].T + m[:3, 3]) * scale
        tri = world[idx]
        if np.linalg.det(m[:3, :3]) < 0:  # mirrored object flips winding
            tri = tri[:, [0, 2, 1], :]

        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals /= lengths

        record = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
        data = np.zeros(n_tris, dtype=record)
        data['n'] = normals
        data['v'] = tri

        with open(path, 'wb') as f:
            f.write(b"Binary STL written by Split & Pin".ljust(80, b'\0'))
            f.write(np.uint32(n_tris).tobytes())
            f.write(data.tobytes())
    finally:
        ev.to_mesh_clear()
    return n_tris


# ----------------------------------------------------------------------------
# Viewport drawing for the path tool
# ----------------------------------------------------------------------------

_draw = {'handle': None, 'points': [], 'hover': None}


def _get_shader():
    for name in ('UNIFORM_COLOR', '3D_UNIFORM_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def _draw_callback():
    pts = _draw['points']
    hover = _draw['hover']
    if not pts and hover is None:
        return
    shader = _get_shader()
    if shader is None:
        return
    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')

        path = [tuple(p) for p in pts]
        if hover is not None:
            path.append(tuple(hover))

        shader.bind()
        if len(path) >= 2:
            gpu.state.line_width_set(2.0)
            shader.uniform_float("color", (1.0, 0.75, 0.1, 1.0))
            batch_for_shader(shader, 'LINE_STRIP', {"pos": path}).draw(shader)
            if len(pts) >= 2:  # faint closing segment
                shader.uniform_float("color", (1.0, 0.75, 0.1, 0.35))
                closing = [tuple(path[-1]), tuple(pts[0])]
                batch_for_shader(shader, 'LINES', {"pos": closing}).draw(shader)
        if pts:
            gpu.state.point_size_set(9.0)
            shader.uniform_float("color", (1.0, 0.3, 0.1, 1.0))
            batch_for_shader(shader, 'POINTS', {"pos": [tuple(p) for p in pts]}).draw(shader)
        gpu.state.blend_set('NONE')
    except Exception:  # noqa: BLE001
        pass


def _stop_drawing():
    if _draw['handle'] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw['handle'], 'WINDOW')
        except Exception:  # noqa: BLE001
            pass
        _draw['handle'] = None
    _draw['points'] = []
    _draw['hover'] = None


# ----------------------------------------------------------------------------
# Pin preview (wireframe overlay)
# ----------------------------------------------------------------------------

_preview = {'handle': None, 'sig': None, 'pending': False, 'lines': [], 'message': ""}

# Settings that change where or how big the pins are. Anything not listed (colour,
# line width, ...) only affects drawing and doesn't need the layout recalculated.
_PREVIEW_PROPS = (
    'cut_mode', 'axis', 'position', 'path_axis', 'path_curved', 'path_detail', 'path_snap',
    'path_sort', 'add_pins', 'pin_count', 'pin_radius', 'pin_depth', 'clearance',
    'pin_layout', 'pin_spacing', 'pin_angle', 'segments',
)


def _redraw_3d():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:  # noqa: BLE001
        pass


def _on_preview_toggle(self, context):
    _preview['sig'] = None
    _bvh_cache['key'] = None
    _bvh_cache['bvh'] = None
    _redraw_3d()


def _path_signature(ob):
    if ob is None or ob.type != 'CURVE' or not ob.data.splines:
        return None
    sp = ob.data.splines[0]
    if sp.type == 'BEZIER':
        pts = tuple((tuple(p.co), tuple(p.handle_left), tuple(p.handle_right))
                    for p in sp.bezier_points)
    else:
        pts = tuple(tuple(p.co) for p in sp.points)
    return (ob.name, sp.type, sp.use_cyclic_u, ob.data.resolution_u, pts,
            tuple(tuple(row) for row in ob.matrix_world))


def _preview_signature(context):
    s = context.scene.splitpin
    ob = context.active_object
    if ob is None or ob.type != 'MESH':
        return ('no-mesh',)
    sig = [tuple(getattr(s, n) for n in _PREVIEW_PROPS),
           (ob.name, ob.mode, len(ob.data.vertices), len(ob.modifiers),
            tuple(tuple(row) for row in ob.matrix_world))]
    if s.cut_mode == 'PATH':
        sig.append(_path_signature(s.path_object))
    return tuple(sig)


def _pin_wire_lines(positions, basis, radius, depth, segments):
    """Line-list vertices (pairs) outlining each pin: end rings, a ring on the cut, and struts."""
    e1, e2, n = basis
    lines = []
    for p in positions:
        rings = []
        for off in (-depth, 0.0, depth):
            c = p + n * off
            ring = []
            for i in range(segments):
                a = TAU * i / segments
                ring.append(tuple(c + e1 * (math.cos(a) * radius) + e2 * (math.sin(a) * radius)))
            rings.append(ring)
        for i in range(segments):
            j = (i + 1) % segments
            for ring in rings:
                lines += [ring[i], ring[j]]
            if i % 3 == 0:
                lines += [rings[0][i], rings[2][i]]
    return lines


def _compute_preview(context, s):
    """Returns (line vertices, status message) for the pins as they would be split now."""
    src = context.active_object
    if not s.add_pins:
        return [], "Pins are turned off."
    if src is None or src.type != 'MESH' or src.mode != 'OBJECT':
        return [], "Select a mesh in Object Mode."

    corners = [src.matrix_world @ Vector(c) for c in src.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    diag = (hi - lo).length

    _bvh_cache['use_key'] = (src.name, len(src.data.vertices), len(src.data.polygons),
                             len(src.modifiers))
    try:
        if s.cut_mode == 'PLANE':
            _a, _b, basis, field_fn = _plane_cut(src, s, lo, hi, diag, cutters=False)
        else:
            _a, _b, basis, field_fn = _build_path_cut(src, s, lo, hi, diag, cutters=False)
        field = field_fn()
        spacing, raised = _resolve_spacing(field, s)
        positions = _select_pins(field, basis, src, s, diag, spacing)
    except SplitError as e:
        return [], str(e)
    finally:
        _bvh_cache['use_key'] = None

    if not positions:
        return [], "No spot on the cut face is large enough for a pin. Try a smaller radius or depth."

    lines = _pin_wire_lines(positions, basis, s.pin_radius, s.pin_depth,
                            max(8, min(s.segments, 24)))
    msg = f"Previewing {len(positions)} pin(s)."
    if len(positions) < s.pin_count:
        msg += f" Only {len(positions)} of {s.pin_count} fit."
    if raised:
        msg += " Distance was too small, so it is raised to the minimum."
    return lines, msg


def _preview_refresh():
    """One-shot timer: recalculate the preview after settings stop changing."""
    _preview['pending'] = False
    sig = None
    try:
        context = bpy.context
        sig = _preview_signature(context)
        lines, message = _compute_preview(context, context.scene.splitpin)
    except Exception as e:  # noqa: BLE001
        lines, message = [], f"Preview failed: {e}"
    _preview['sig'] = sig if sig is not None else ('failed',)
    _preview['lines'] = lines
    _preview['message'] = message
    _redraw_3d()
    return None


def _preview_draw():
    try:
        context = bpy.context
        s = getattr(context.scene, "splitpin", None)
        if s is None or not s.show_preview:
            return
        ob = context.active_object
        if ob is None or ob.type != 'MESH' or ob.mode != 'OBJECT':
            _bvh_cache['key'] = None
            _bvh_cache['bvh'] = None
            return
        if not _preview['pending'] and _preview_signature(context) != _preview['sig']:
            _preview['pending'] = True
            bpy.app.timers.register(_preview_refresh, first_interval=0.05)

        lines = _preview['lines']
        if not lines:
            return
        shader = _get_shader()
        if shader is None:
            return
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')  # always drawn in front of the mesh
        gpu.state.line_width_set(s.preview_width)
        shader.bind()
        shader.uniform_float("color", tuple(s.preview_color))
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
    except Exception:  # noqa: BLE001
        pass


def _stop_preview():
    if _preview['handle'] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_preview['handle'], 'WINDOW')
        except Exception:  # noqa: BLE001
            pass
        _preview['handle'] = None
    try:
        if bpy.app.timers.is_registered(_preview_refresh):
            bpy.app.timers.unregister(_preview_refresh)
    except Exception:  # noqa: BLE001
        pass
    _preview.update(sig=None, pending=False, lines=[], message="")


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

def _is_curve(self, obj):
    return obj.type == 'CURVE'


class SplitPinSettings(bpy.types.PropertyGroup):
    cut_mode: EnumProperty(
        name="Cut Type",
        items=[
            ('PLANE', "Plane", "Flat cut perpendicular to a world axis"),
            ('PATH', "Path", "Cut along a loop you draw around the model"),
        ],
        default='PLANE',
    )

    # Plane
    axis: EnumProperty(
        name="Axis",
        description="The cut plane is perpendicular to this world axis",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='Z',
    )
    position: FloatProperty(
        name="Position",
        description="Where the cut plane sits along the chosen axis (world space)",
        subtype='DISTANCE', unit='LENGTH', default=0.0,
    )

    # Path
    path_object: PointerProperty(
        name="Path",
        description="Curve object whose first spline is the cut loop",
        type=bpy.types.Object, poll=_is_curve,
    )
    path_axis: EnumProperty(
        name="Direction",
        description="Direction the two parts are separated in",
        items=[
            ('AUTO', "Auto", "Best-fit direction through the path points"),
            ('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", ""),
        ],
        default='AUTO',
    )
    path_curved: BoolProperty(
        name="Smooth Curve",
        description="Pass a smooth spline through the points instead of straight lines",
        default=True,
    )
    path_detail: IntProperty(
        name="Detail",
        description="Extra points added between each pair of path points",
        min=1, max=32, default=6,
    )
    path_snap: BoolProperty(
        name="Follow Surface",
        description="Snap the interpolated path onto the mesh surface",
        default=True,
    )
    path_sort: BoolProperty(
        name="Sort Points",
        description="Order the points by angle around the loop so click order doesn't matter",
        default=True,
    )

    # Connectors
    add_pins: BoolProperty(
        name="Add Pins & Sockets",
        description="Add matching connectors on the cut faces",
        default=True,
    )
    printer_type: EnumProperty(
        name="Printer",
        description="Used by Auto Size to pick a clearance",
        items=[
            ('FDM', "FDM", "Filament printer (0.2 mm clearance)"),
            ('RESIN', "Resin", "Resin printer (0.1 mm clearance)"),
        ],
        default='FDM',
    )
    print_size: FloatProperty(
        name="Print Size (mm)",
        description="Length of the model's longest side once printed, in millimetres. "
                    "Auto Size uses it to pick realistic pin sizes. 0 = take it from the scene units",
        min=0.0, default=0.0,
    )
    pin_area_percent: FloatProperty(
        name="Pin Area",
        description="Auto Size makes each pin's cross-section this percentage of the cut's area",
        subtype='PERCENTAGE', min=0.1, max=30.0, default=4.0, precision=1,
    )
    pin_count: IntProperty(name="Count", min=1, max=12, default=2)
    show_preview: BoolProperty(
        name="Preview Pins",
        description="Show the pins as a wireframe in front of the model before you split",
        default=False, update=_on_preview_toggle,
    )
    preview_color: FloatVectorProperty(
        name="Preview Color",
        description="Color of the pin preview wireframe",
        subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.1, 0.85, 1.0, 1.0),
    )
    preview_width: FloatProperty(
        name="Line Width",
        description="Thickness of the preview lines",
        min=1.0, max=8.0, default=2.0,
    )
    pin_layout: EnumProperty(
        name="Layout",
        description="How the pins are arranged on the cut face",
        items=[
            ('CENTER', "Center", "One pin in the middle, the others on a circle around it"),
            ('RING', "Circle", "No pin in the middle; all pins sit on a circle around the center"),
            ('LINE', "Line", "Pins in a straight line"),
        ],
        default='CENTER',
    )
    pin_spacing: FloatProperty(
        name="Pin Distance",
        description="Center layout and Circle: distance from the middle to the surrounding pins "
                    "(the circle's radius). Line: distance between neighbouring pins. "
                    "0 = automatic. Auto Size sets this for you",
        subtype='DISTANCE', unit='LENGTH', min=0.0, default=0.0,
    )
    pin_angle: FloatProperty(
        name="Rotation",
        description="Rotates the pin layout around the cut direction "
                    "(around the Z axis for a Z plane cut)",
        subtype='ANGLE', unit='ROTATION', min=-TAU, max=TAU, default=0.0,
    )
    pin_radius: FloatProperty(
        name="Pin Radius", subtype='DISTANCE', unit='LENGTH',
        min=0.0001, default=2.0,
    )
    pin_depth: FloatProperty(
        name="Insertion Depth",
        description="How deep the pin goes into each part",
        subtype='DISTANCE', unit='LENGTH', min=0.0001, default=6.0,
    )
    clearance: FloatProperty(
        name="Clearance",
        description="Extra gap around and behind the pin so the parts fit after printing",
        subtype='DISTANCE', unit='LENGTH', min=0.0, default=0.2,
    )
    segments: IntProperty(name="Smoothness", min=8, max=96, default=32)
    swap_sides: BoolProperty(
        name="Pins on Upper Part",
        description="Put the pins on the upper part instead of the lower one",
        default=False,
    )
    separation: FloatProperty(
        name="Separate Parts",
        description="Move the upper part away from the cut for easier viewing",
        subtype='DISTANCE', unit='LENGTH', min=0.0, default=0.0,
    )

    # Export
    export_dir: StringProperty(name="Folder", subtype='DIR_PATH', default="//")
    export_use_unit_scale: BoolProperty(
        name="Scale to Millimetres",
        description="Convert using the scene's unit scale so slicers receive millimetres. "
                    "Turn off to export raw Blender units",
        default=True,
    )


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------

class SPLITPIN_OT_cut(bpy.types.Operator):
    """Split the active mesh and add matching pins and sockets"""
    bl_idname = "splitpin.cut"
    bl_label = "Split & Add Pins"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and context.mode == 'OBJECT'

    def execute(self, context):
        s = context.scene.splitpin
        src = context.active_object

        corners = [src.matrix_world @ Vector(c) for c in src.bound_box]
        lo = Vector([min(c[i] for c in corners) for i in range(3)])
        hi = Vector([max(c[i] for c in corners) for i in range(3)])
        diag = (hi - lo).length

        coll = bpy.data.collections.new(f"{src.name}_Parts")
        context.scene.collection.children.link(coll)
        created, helpers = [], []
        pin_count = 0
        extra = ""

        try:
            if s.cut_mode == 'PLANE':
                cut_a, cut_b, basis, field_fn = _plane_cut(src, s, lo, hi, diag)
            else:
                cut_a, cut_b, basis, field_fn = _build_path_cut(src, s, lo, hi, diag)
            for ob in (cut_a, cut_b):
                coll.objects.link(ob)
                helpers.append(ob)

            part_a = _duplicate(src, f"{src.name}_A", coll)
            created.append(part_a)
            part_b = _duplicate(src, f"{src.name}_B", coll)
            created.append(part_b)

            _boolean(part_a, cut_a, 'INTERSECT')
            _boolean(part_b, cut_b, 'INTERSECT')

            if s.add_pins:
                field = field_fn()
                r, clear, depth = s.pin_radius, s.clearance, s.pin_depth
                spacing, raised = _resolve_spacing(field, s)
                if raised:
                    extra += " Pin distance was too small for the sockets, so I increased it."
                positions = _select_pins(field, basis, src, s, diag, spacing)
                if not positions:
                    self.report({'WARNING'}, "No spot on the cut face is large enough for a pin. "
                                             "Try a smaller radius or depth. Parts were split without pins.")
                else:
                    pin_count = len(positions)
                    if pin_count < s.pin_count:
                        extra += f" Only {pin_count} of {s.pin_count} pins fit."
                    r, clear, depth = s.pin_radius, s.clearance, s.pin_depth
                    n = basis[2]
                    sign = -1.0 if s.swap_sides else 1.0

                    pin_specs = [(p, r, depth * 2.0) for p in positions]

                    # Socket runs from slightly behind the cut to depth + clearance past it.
                    a_rel = -sign * depth * 0.5
                    b_rel = sign * (depth + clear)
                    socket_specs = [
                        (p + n * ((a_rel + b_rel) / 2.0), r + clear, abs(b_rel - a_rel))
                        for p in positions
                    ]

                    pins = _cylinders_object("_pins", pin_specs, basis, s.segments)
                    sockets = _cylinders_object("_sockets", socket_specs, basis, s.segments)
                    for h in (pins, sockets):
                        coll.objects.link(h)
                        helpers.append(h)

                    pin_host, socket_host = (part_b, part_a) if s.swap_sides else (part_a, part_b)
                    _boolean(pin_host, pins, 'UNION')
                    _boolean(socket_host, sockets, 'DIFFERENCE')

            if s.separation > 0.0:
                part_b.location = part_b.location + basis[2] * s.separation

        except SplitError as e:
            self._cleanup(created, helpers, coll)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:  # noqa: BLE001
            self._cleanup(created, helpers, coll)
            self.report({'ERROR'}, f"Split failed: {e}")
            return {'CANCELLED'}

        for h in helpers:
            _remove_object(h)

        s.show_preview = False
        src.hide_set(True)  # original stays untouched, just hidden
        for ob in context.selected_objects:
            ob.select_set(False)
        for ob in created:
            ob.select_set(True)
        context.view_layer.objects.active = created[0]

        self.report({'INFO'}, f"Split into 2 parts with {pin_count} pin(s). "
                              f"Original '{src.name}' is hidden." + extra)
        return {'FINISHED'}

    @staticmethod
    def _cleanup(created, helpers, coll):
        for ob in created + helpers:
            try:
                _remove_object(ob)
            except Exception:  # noqa: BLE001
                pass
        try:
            bpy.data.collections.remove(coll)
        except Exception:  # noqa: BLE001
            pass


class SPLITPIN_OT_draw_path(bpy.types.Operator):
    """Click points on the mesh surface to define a cut loop around the model"""
    bl_idname = "splitpin.draw_path"
    bl_label = "Draw Cut Path"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None and ob.type == 'MESH' and context.mode == 'OBJECT'
                and context.area is not None and context.area.type == 'VIEW_3D')

    def invoke(self, context, event):
        self.area = context.area
        self.region = next((r for r in self.area.regions if r.type == 'WINDOW'), None)
        self.rv3d = self.area.spaces.active.region_3d
        if self.region is None or self.rv3d is None:
            self.report({'ERROR'}, "Start this from a 3D viewport.")
            return {'CANCELLED'}

        self.target = context.active_object
        self.mw = self.target.matrix_world.copy()
        self.inv = self.mw.inverted()
        self.bvh = BVHTree.FromObject(self.target, context.evaluated_depsgraph_get())

        _stop_drawing()
        self.points = _draw['points']
        _draw['handle'] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), 'WINDOW', 'POST_VIEW')

        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "Cut path:  LMB add point  |  Backspace remove last  |  Enter / RMB finish  |  Esc cancel")
        return {'RUNNING_MODAL'}

    # -- helpers ----------------------------------------------------------
    def _mouse_in_viewport(self, event):
        mx, my = event.mouse_x, event.mouse_y
        r = self.region
        if not (r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height):
            return None
        for other in self.area.regions:  # ignore clicks on sidebar / toolbar / header
            if other.type != 'WINDOW' and other.width > 1 and other.height > 1:
                if other.x <= mx < other.x + other.width and other.y <= my < other.y + other.height:
                    return None
        return (mx - r.x, my - r.y)

    def _pick(self, coord):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, coord)
        lo = self.inv @ origin
        ld = (self.inv.to_3x3() @ direction).normalized()
        loc, _n, _i, _d = self.bvh.ray_cast(lo, ld)
        return None if loc is None else self.mw @ loc

    def _end(self, context):
        _stop_drawing()
        try:
            context.workspace.status_text_set(None)
            self.area.tag_redraw()
        except Exception:  # noqa: BLE001
            pass

    # -- modal ------------------------------------------------------------
    def modal(self, context, event):
        try:
            self.area.tag_redraw()
        except Exception:  # noqa: BLE001
            self._end(context)
            return {'CANCELLED'}

        t = event.type
        if t in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE',
                 'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'NDOF_MOTION'}:
            return {'PASS_THROUGH'}
        if t.startswith('NUMPAD') and t != 'NUMPAD_ENTER':
            return {'PASS_THROUGH'}

        if t == 'MOUSEMOVE':
            coord = self._mouse_in_viewport(event)
            _draw['hover'] = self._pick(coord) if coord is not None and self.points else None
            return {'PASS_THROUGH'}

        if t == 'ESC' and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}

        if t in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            return self._finish(context)

        if (t == 'BACK_SPACE' or (t == 'Z' and event.ctrl)) and event.value == 'PRESS':
            if self.points:
                self.points.pop()
            return {'RUNNING_MODAL'}

        if t == 'LEFTMOUSE':
            if event.alt:
                return {'PASS_THROUGH'}
            coord = self._mouse_in_viewport(event)
            if coord is None:
                return {'PASS_THROUGH'}
            if event.value == 'PRESS':
                hit = self._pick(coord)
                if hit is not None:
                    self.points.append(hit)
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        pts = list(self.points)
        self._end(context)
        if len(pts) < 3:
            self.report({'WARNING'}, "A cut path needs at least 3 points. Nothing was created.")
            return {'CANCELLED'}

        name = f"{self.target.name}_CutPath"
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for pt, p in zip(sp.points, pts):
            pt.co = (p.x, p.y, p.z, 1.0)
        sp.use_cyclic_u = True

        ob = bpy.data.objects.new(name, cu)
        context.collection.objects.link(ob)
        ob.show_in_front = True

        s = context.scene.splitpin
        s.path_object = ob
        s.cut_mode = 'PATH'
        self.report({'INFO'}, f"Created '{name}' with {len(pts)} points. "
                              "Select it and press Tab to adjust points, then click Split.")
        return {'FINISHED'}


class SPLITPIN_OT_auto_size(bpy.types.Operator):
    """Pick pin radius, depth, clearance, count and distance from the area of the cut"""
    bl_idname = "splitpin.auto_size"
    bl_label = "Auto Size"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and context.mode == 'OBJECT'

    def execute(self, context):
        s = context.scene.splitpin
        ob = context.active_object
        try:
            meas = _measure_cut(ob, s)
        except SplitError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        major, minor, room, area = meas['major'], meas['minor'], meas['room'], meas['area']

        corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        longest_bu = max(
            max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3))
        longest_bu = max(longest_bu, 1e-9)

        # Millimetres per Blender unit. Only the clearance and the "smallest printable
        # pin" depend on it; everything else is proportional to the cut itself.
        k = _mm_per_bu(context.scene)
        note = ""
        if s.print_size > 0.0:
            k = s.print_size / longest_bu
        elif not (10.0 <= longest_bu * k <= 500.0):
            scene_mm = longest_bu * k
            k = ASSUMED_PRINT_SIZE_MM / longest_bu
            note = (f" Your scene units make the model {scene_mm:.0f} mm, which isn't a realistic "
                    f"print size, so I assumed {ASSUMED_PRINT_SIZE_MM:.0f} mm. "
                    "Set Print Size to override.")
        major_mm, minor_mm, room_mm = major * k, minor * k, room * k
        area_mm2 = area * k * k

        wanted = s.pin_area_percent
        radius = math.sqrt((wanted / 100.0) * area_mm2 / math.pi)
        radius = max(radius, 1.0)                        # smallest pin worth printing
        radius = max(min(radius, minor_mm / 4.0), 0.3)   # must still fit inside the cut
        clearance = 0.2 if s.printer_type == 'FDM' else 0.1
        depth = max(min(3.0 * radius, 0.4 * room_mm), 1.0)

        count = 1
        if major_mm >= 7.0 * (radius + clearance):
            count = 2
            if major_mm > 25.0 * radius:
                count += 1
            if major_mm > 40.0 * radius:
                count += 1

        r_bu, clear_bu = radius / k, clearance / k
        count = max(1, min(count, _max_pins(meas, r_bu, clear_bu, s.pin_layout, s.pin_angle)))
        spacing_bu = _auto_spacing(meas, r_bu, clear_bu, count, s.pin_layout, s.pin_angle)

        s.pin_radius = r_bu
        s.pin_depth = depth / k
        s.clearance = clear_bu
        s.pin_count = count
        s.pin_spacing = spacing_bu

        actual = 100.0 * math.pi * radius * radius / area_mm2
        limited = ""
        if abs(actual - wanted) > 0.2 * wanted:
            limited = " (limited by what fits or can be printed)"
        layout = {'CENTER': "center pin + circle", 'RING': "circle", 'LINE': "line"}[s.pin_layout]
        where = f", distance {spacing_bu * k:.1f} mm ({layout})" if count > 1 else ""
        msg = (f"{count} pin(s), radius {radius:.2f} mm = {actual:.1f}% of the cut area each"
               f"{limited}{where}, depth {depth:.1f} mm, clearance {clearance:.2f} mm. "
               f"Cut is about {major_mm:.0f} x {minor_mm:.0f} mm ({area_mm2:.0f} mm\u00b2)." + note)
        if minor_mm < 4.0:
            self.report({'WARNING'}, "This cut is very small, so pins may not fit. " + msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class SPLITPIN_OT_center(bpy.types.Operator):
    """Put the cut plane at the middle of the active object along the chosen axis"""
    bl_idname = "splitpin.center"
    bl_label = "Center"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        s = context.scene.splitpin
        axis = AXIS_INDEX[s.axis]
        ob = context.active_object
        vals = [(ob.matrix_world @ Vector(c))[axis] for c in ob.bound_box]
        s.position = (min(vals) + max(vals)) / 2.0
        return {'FINISHED'}


class SPLITPIN_OT_from_cursor(bpy.types.Operator):
    """Put the cut plane at the 3D cursor along the chosen axis"""
    bl_idname = "splitpin.from_cursor"
    bl_label = "From 3D Cursor"

    def execute(self, context):
        s = context.scene.splitpin
        s.position = context.scene.cursor.location[AXIS_INDEX[s.axis]]
        return {'FINISHED'}


class SPLITPIN_OT_export(bpy.types.Operator):
    """Export each selected mesh as its own binary STL file"""
    bl_idname = "splitpin.export"
    bl_label = "Export Selected as STL"

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        scene = context.scene
        s = scene.splitpin
        folder = bpy.path.abspath(s.export_dir)
        if not folder:
            self.report({'ERROR'}, "Choose an export folder first.")
            return {'CANCELLED'}
        os.makedirs(folder, exist_ok=True)

        scale = _mm_per_bu(scene)

        count = 0
        for ob in [o for o in context.selected_objects if o.type == 'MESH']:
            path = os.path.join(folder, _safe_name(ob.name) + ".stl")
            try:
                _write_binary_stl(path, ob, scale)
                count += 1
            except Exception as e:  # noqa: BLE001
                self.report({'ERROR'}, f"Could not export '{ob.name}': {e}")
                return {'CANCELLED'}

        self.report({'INFO'}, f"Exported {count} file(s) to {folder}")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

class VIEW3D_PT_splitpin(bpy.types.Panel):
    bl_label = "Split & Pin"
    bl_idname = "VIEW3D_PT_splitpin"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Split & Pin"

    def draw(self, context):
        s = context.scene.splitpin
        layout = self.layout

        box = layout.box()
        box.label(text="Cut", icon='MOD_BOOLEAN')
        box.row().prop(s, "cut_mode", expand=True)

        if s.cut_mode == 'PLANE':
            box.row().prop(s, "axis", expand=True)
            box.prop(s, "position")
            row = box.row(align=True)
            row.operator("splitpin.center")
            row.operator("splitpin.from_cursor")
        else:
            box.operator("splitpin.draw_path", icon='GREASEPENCIL')
            box.prop(s, "path_object")
            box.prop(s, "path_axis")
            box.prop(s, "path_snap")
            box.prop(s, "path_curved")
            sub = box.column()
            sub.enabled = s.path_curved or s.path_snap
            sub.prop(s, "path_detail")
            box.prop(s, "path_sort")
        box.prop(s, "separation")

        box = layout.box()
        box.prop(s, "add_pins", icon='PIVOT_ACTIVE')
        col = box.column()
        col.enabled = s.add_pins
        auto = col.box()
        auto.label(text="Auto Size", icon='AUTO')
        auto.prop(s, "pin_area_percent")
        auto.row().prop(s, "printer_type", expand=True)
        auto.prop(s, "print_size")
        auto.operator("splitpin.auto_size", icon='AUTO')
        col.row().prop(s, "pin_layout", expand=True)
        col.prop(s, "pin_count")
        col.prop(s, "pin_spacing")
        col.prop(s, "pin_angle")
        col.prop(s, "pin_radius")
        col.prop(s, "pin_depth")
        col.prop(s, "clearance")
        col.prop(s, "segments")
        col.prop(s, "swap_sides")

        box = layout.box()
        box.prop(s, "show_preview", icon='HIDE_OFF' if s.show_preview else 'HIDE_ON')
        if s.show_preview:
            row = box.row(align=True)
            row.prop(s, "preview_color", text="")
            row.prop(s, "preview_width")
            for line in textwrap.wrap(_preview['message'] or "Calculating...", 42):
                box.label(text=line)

        layout.operator("splitpin.cut", icon='SCULPTMODE_HLT')

        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(s, "export_dir")
        box.prop(s, "export_use_unit_scale")
        box.operator("splitpin.export")

        layout.label(text="Tip: use a mm scene (Unit Scale 0.001).")


classes = (
    SplitPinSettings,
    SPLITPIN_OT_cut,
    SPLITPIN_OT_draw_path,
    SPLITPIN_OT_auto_size,
    SPLITPIN_OT_center,
    SPLITPIN_OT_from_cursor,
    SPLITPIN_OT_export,
    VIEW3D_PT_splitpin,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splitpin = PointerProperty(type=SplitPinSettings)
    if _preview['handle'] is None:
        _preview['handle'] = bpy.types.SpaceView3D.draw_handler_add(
            _preview_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    _stop_drawing()
    _stop_preview()
    del bpy.types.Scene.splitpin
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
