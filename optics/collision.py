"""2-D polygon collision detection for optic layout feasibility.

All optical elements lie in the xy plane at the optical centre height.
Each element projects to an oriented rectangle matching its physical
body cross-section:

- mirror blank: D × edge_thickness, front face at optical vertex
- grating: size × thickness, front face at vertex
- HASMA slit: bore_radius × length, body behind vertex
- detector: package_width × package_height, offset behind die
  by glass_to_die_mm (glass window extends ahead)

Fixed-grating designs have no exit slit element; the beam path
ends at the detector.

Beam cones between consecutive beam-path elements are convex
quadrilaterals connecting aperture edges.  Vertices are angle-sorted
around their centroid to prevent self-intersection.  Overlap is
tested via the Separating Axis Theorem (SAT).

Two levels of check:

- **Optics-only** (``--optics_only``): bare optic body rectangles +
  beam-cone quadrilaterals.  Any overlap is infeasible.
- **Assembly** (default): optic+mount assembly polygons (via
  ``Mount.parent_label``).  Self-pairs (optic vs its own mount)
  are exempt.  Beam-cone checks use slab polygons at OCH.
"""

import math
from itertools import combinations

from optics.scene import (
    ElementPlacement, InfeasibleGeometry, Mount, Scene, perpendicular_xy,
)


# ── 2-D polygon primitives ──────────────────────────────────────────────

Vec2 = tuple[float, float]
Polygon = list[Vec2]

_CIRCLE_N = 16


def convex_hull(points: list[Vec2]) -> list[Vec2]:
    """Andrew's monotone-chain convex hull. Returns CCW-ordered vertices."""
    pts = sorted(points)
    if len(pts) <= 1:
        return list(pts)

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Vec2] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Vec2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _circle_polygon(cx: float, cy: float, r: float) -> Polygon:
    return [
        (cx + r * math.cos(2 * math.pi * i / _CIRCLE_N),
         cy + r * math.sin(2 * math.pi * i / _CIRCLE_N))
        for i in range(_CIRCLE_N)
    ]


def _rect_polygon(cx: float, cy: float,
                   half_w: float, half_h: float,
                   fwd: Vec2, right: Vec2) -> Polygon:
    fx, fy = fwd
    rx, ry = right
    corners = [
        (-half_w, -half_h),
        (+half_w, -half_h),
        (+half_w, +half_h),
        (-half_w, +half_h),
    ]
    return [
        (cx + dw * rx + dh * fx,
         cy + dw * ry + dh * fy)
        for dw, dh in corners
    ]


def _axis_2d(el: ElementPlacement) -> tuple[Vec2, Vec2]:
    """Normalised forward and right vectors in the xy plane."""
    fx, fy = el.axis[0], el.axis[1]
    ln = math.hypot(fx, fy)
    if ln > 1e-12:
        fx, fy = fx / ln, fy / ln
    rx, ry = perpendicular_xy(el.axis)
    return (fx, fy), (rx, ry)


def polygon_for_element(el: ElementPlacement) -> Polygon | None:
    """Physical-body footprint polygon in the xy plane.

    All elements project to rectangles at the optical center height:
    width = aperture diameter (perpendicular to axis), depth = substrate
    thickness (along axis).  The rectangle is placed with its front face
    at the element position (optical vertex) and extends backward by
    edge_thickness along −axis.
    """
    x, y = el.position[0], el.position[1]
    if el.kind == "mirror":
        half_w = el.params["diameter_mm"] / 2.0
        et = el.params.get("edge_thickness_mm", 0.0)
        fwd, right = _axis_2d(el)
        cx = x - fwd[0] * et / 2.0
        cy = y - fwd[1] * et / 2.0
        return _rect_polygon(cx, cy, half_w, et / 2.0, fwd, right)
    if el.kind == "grating":
        half = el.params["size_mm"] / 2.0
        et = el.params.get("edge_thickness_mm", 0.0)
        fwd, right = _axis_2d(el)
        cx = x - fwd[0] * et / 2.0
        cy = y - fwd[1] * et / 2.0
        return _rect_polygon(cx, cy, half, et / 2.0, fwd, right)
    if el.kind == "slit":
        bore_r = el.params.get("bore_radius_mm")
        length = el.params.get("slit_length_mm")
        if bore_r and length:
            fwd, right = _axis_2d(el)
            cx = x - fwd[0] * length / 2.0
            cy = y - fwd[1] * length / 2.0
            return _rect_polygon(cx, cy, bore_r, length / 2.0, fwd, right)
        return None
    if el.kind == "detector":
        pw = el.params.get("package_width_mm")
        pd = el.params.get("package_height_mm")
        if pw and pd:
            fwd, right = _axis_2d(el)
            glass = el.params["glass_to_die_mm"]
            offset = pd / 2.0 - glass
            cx = x - fwd[0] * offset
            cy = y - fwd[1] * offset
            return _rect_polygon(cx, cy, pw / 2.0, pd / 2.0, fwd, right)
        return None
    return None


def polygon_for_mount(mount: Mount, *, use_slab: bool = False) -> Polygon:
    """Mount footprint polygon.

    Default: foot convex hull (T-shape outline of the physical foot).
    Falls back to full AABB for mounts without a foot (detector, slit).

    When *use_slab* is True and the mount carries a ``slab_polygon_xy``
    (oriented rectangle of the slab cross-section at OCH), use that
    instead.  The slab polygon excludes the foot entirely.
    """
    if use_slab and mount.slab_polygon_xy is not None:
        return list(mount.slab_polygon_xy)
    if mount.foot_polygon_xy is not None:
        return list(mount.foot_polygon_xy)
    if mount.slab_polygon_xy is not None:
        return list(mount.slab_polygon_xy)
    x_lo, x_hi, y_lo, y_hi = mount.bbox_xy_mm
    return [
        (x_lo, y_lo),
        (x_hi, y_lo),
        (x_hi, y_hi),
        (x_lo, y_hi),
    ]


# ── Beam-cone quadrilateral ─────────────────────────────────────────────

def _edge_points_2d(el: ElementPlacement) -> tuple[Vec2, Vec2]:
    """Two aperture-edge points in the xy plane, perpendicular to axis."""
    x, y = el.position[0], el.position[1]
    if el.kind == "mirror":
        r = el.params["diameter_mm"] / 2.0
    elif el.kind == "grating":
        r = el.params["size_mm"] / 2.0
    elif el.kind == "slit":
        r = el.params["width_um"] * 1e-3 / 2.0
    elif el.kind == "detector":
        r = el.params.get("package_width_mm", 30.0) / 2.0
    else:
        r = 1.0
    px, py = perpendicular_xy(el.axis)
    return (
        (x + r * px, y + r * py),
        (x - r * px, y - r * py),
    )


def beam_cone_polygon(el_a: ElementPlacement,
                       el_b: ElementPlacement) -> Polygon:
    a0, a1 = _edge_points_2d(el_a)
    b0, b1 = _edge_points_2d(el_b)
    pts = [a0, a1, b0, b1]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    from math import atan2
    pts.sort(key=lambda p: atan2(p[1] - cy, p[0] - cx))
    return pts


# ── SAT overlap test ────────────────────────────────────────────────────

def _project(poly: Polygon, ax: float, ay: float) -> tuple[float, float]:
    dots = [px * ax + py * ay for px, py in poly]
    return min(dots), max(dots)


def polygons_overlap(a: Polygon, b: Polygon) -> bool:
    """True if convex polygons *a* and *b* overlap (SAT)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ax = -(y2 - y1)
            ay = x2 - x1
            ln = math.hypot(ax, ay)
            if ln < 1e-15:
                continue
            ax /= ln
            ay /= ln
            a_min, a_max = _project(a, ax, ay)
            b_min, b_max = _project(b, ax, ay)
            if a_max < b_min or b_max < a_min:
                return False
    return True


# ── High-level checks ───────────────────────────────────────────────────

def check_optic_collisions(scene: Scene, beam_path: list[ElementPlacement]
                            ) -> None:
    """Bare-optic collision check (``--optics_only`` mode).

    Raises ``InfeasibleGeometry`` if any optic polygon overlaps another
    optic or any beam-cone segment (excluding its own endpoints).
    """
    polys: dict[str, Polygon] = {}
    for el in scene.elements:
        p = polygon_for_element(el)
        if p is not None:
            polys[el.label] = p

    for (la, pa), (lb, pb) in combinations(polys.items(), 2):
        if polygons_overlap(pa, pb):
            raise InfeasibleGeometry(
                f"optic collision: {la} ↔ {lb}")

    for i in range(len(beam_path) - 1):
        ea = beam_path[i]
        eb = beam_path[i + 1]
        cone = beam_cone_polygon(ea, eb)
        seg = f"{ea.label}→{eb.label}"
        exempt = {ea.label, eb.label}
        for label, poly in polys.items():
            if label in exempt:
                continue
            if polygons_overlap(cone, poly):
                raise InfeasibleGeometry(
                    f"beam-cone clip: {seg} clips {label}")


def check_assembly_collisions(
    scene: Scene,
    mounts: list[Mount],
    beam_path: list[ElementPlacement],
) -> None:
    """Optic+mount assembly collision check (default mode).

    Each mount polygon is associated with its parent optic label.
    Self-pairs (optic vs its own mount) are exempt.  Beam-cone
    segments are checked against non-endpoint assemblies.
    """
    mount_polys: list[tuple[str, str, Polygon]] = []
    mount_slab_polys: list[tuple[str, str, Polygon]] = []
    for m in mounts:
        mount_polys.append((m.label, m.parent_label, polygon_for_mount(m)))
        mount_slab_polys.append(
            (m.label, m.parent_label, polygon_for_mount(m, use_slab=True)))

    optic_polys: list[tuple[str, str, Polygon]] = []
    for el in scene.elements:
        p = polygon_for_element(el)
        if p is not None:
            optic_polys.append((el.label, el.label, p))

    all_polys = optic_polys + mount_polys

    for i in range(len(all_polys)):
        for j in range(i + 1, len(all_polys)):
            name_a, parent_a, poly_a = all_polys[i]
            name_b, parent_b, poly_b = all_polys[j]
            if parent_a == parent_b:
                continue
            if polygons_overlap(poly_a, poly_b):
                raise InfeasibleGeometry(
                    f"assembly collision: {name_a} ↔ {name_b}")

    beam_polys = optic_polys + mount_slab_polys
    bp_labels = [el.label for el in beam_path]
    for i in range(len(beam_path) - 1):
        ea = beam_path[i]
        eb = beam_path[i + 1]
        cone = beam_cone_polygon(ea, eb)
        seg = f"{ea.label}→{eb.label}"
        endpoints = {ea.label, eb.label}
        optic_exempt = set(endpoints)
        if i > 0:
            optic_exempt.add(bp_labels[i - 1])
        if i + 2 < len(bp_labels):
            optic_exempt.add(bp_labels[i + 2])
        for name, parent, poly in beam_polys:
            if name == parent:
                if parent in optic_exempt:
                    continue
            else:
                if parent in endpoints:
                    continue
            if polygons_overlap(cone, poly):
                raise InfeasibleGeometry(
                    f"beam-cone clip: {seg} clips {name}")
