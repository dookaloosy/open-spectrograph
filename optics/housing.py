"""Pure-geometry functions for the unibody solid-subtract housing.

Computes mount pocket polygons, beam channel quads, HASMA jog
cut-outs, detector slot geometry, and insert hole positions.
All coordinates are in mm in the optical-plane XY frame.

No build123d imports — this module is consumed by the CAD builder
(housing_cad) and optionally by the raysect
world builder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from optics.assembly import _slit_mount_housing_dims
from optics.scene import Mount, Scene


# ---- Dimensional constants ------------------------------------------------

_HASMA_FLARE_HALF_ANGLE_DEG = 45.0


_DETECTOR_EXT_MARGIN_MM = 25.0
_DETECTOR_EXT_HALF_WIDTH_MM = 40.0

_WALL_THICKNESS_MM = 10.0

_SCREW_PILOT_R_MM = 1.0
_SCREW_PILOT_DEPTH_MM = 5.0

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass
class MountFastener:
    """A bolt or pusher hole position for a mount foot."""
    xy: Vec2
    floor_z: float
    kind: str  # "bolt" or "pusher"
    label: str = ""


@dataclass
class FloorStep:
    """A stepped floor region deeper than the base cavity floor."""
    polygons: list[list[Vec2]]
    floor_z: float
    label: str


def _mount_z_extent(
    mounts: list[Mount] | tuple[Mount, ...],
    scene: Scene,
    *,
    slit_flange_size_mm: float,
) -> tuple[float, float]:
    """Compute (z_bottom, z_top) from mount z_ranges and slit flanges."""
    z_top = 0.0
    z_bottom = 0.0
    for mount in mounts:
        if mount.z_range_mm is not None:
            z_bottom = min(z_bottom, mount.z_range_mm[0])
            z_top = max(z_top, mount.z_range_mm[1])
    for el in scene.elements:
        if el.kind == "slit":
            flange_half = 0.5 * slit_flange_size_mm
            z_top = max(z_top, flange_half)
            z_bottom = min(z_bottom, -flange_half)
        elif el.kind == "mirror":
            R = 0.5 * el.params["diameter_mm"]
            z_top = max(z_top, R)
            z_bottom = min(z_bottom, -R)
        elif el.kind == "grating":
            half = 0.5 * el.params["size_mm"]
            z_top = max(z_top, half)
            z_bottom = min(z_bottom, -half)
    return z_bottom - _WALL_THICKNESS_MM, z_top + _WALL_THICKNESS_MM


# ---- Floor step helpers ---------------------------------------------------

def _face_half_height_z(element) -> float:
    """Vertical (z) half-height of the optic aperture."""
    if element.kind == "slit":
        return float(element.params["bore_radius_mm"])
    if element.kind == "mirror":
        return 0.5 * float(element.params["diameter_mm"])
    if element.kind == "grating":
        return 0.5 * float(element.params["size_mm"])
    if element.kind == "detector":
        return 0.5 * float(element.params.get("package_height_mm", 0.0))
    return 0.0



def _compute_floor_steps(
    mount_floors: dict[str, tuple[float, list[Vec2], list[Vec2]]],
    beam_path,
) -> tuple[float, list[FloorStep]]:
    """Compute stepped floor levels and their xy regions.

    *mount_floors* maps element label → (floor_z, optic_poly, mount_hull_poly).
    Raw hull polygons are passed through — offset and keepout subtraction
    are handled in the CAD code with ``Kind.ARC`` for proper arc corners.

    Returns ``(shallowest_floor_z, [FloorStep for deeper levels])``.
    """
    if not mount_floors:
        return 0.0, []

    shallowest = max(fz for fz, _, _ in mount_floors.values())
    steps: list[FloorStep] = []

    for label, (floor_z, _optic_poly, hull_poly) in mount_floors.items():
        if floor_z >= shallowest - 0.1:
            continue
        steps.append(FloorStep(polygons=[list(hull_poly)], floor_z=floor_z,
                               label=label))

    return shallowest, steps


# ---- Output dataclass -----------------------------------------------------

@dataclass
class SolidHousingSpec:
    """All geometry data needed by the CAD and raysect builders."""

    housing_outline: list[Vec2]
    z_bottom: float
    z_top: float

    optic_polys: list[list[Vec2]]
    mount_hull_polys: list[list[Vec2]]
    beam_polys: list[list[Vec2]]
    wall_polys: list[list[Vec2]]

    wall_thickness_mm: float

    # HASMA
    hasma_bore_position: Vec3
    hasma_bore_axis: Vec2
    hasma_bore_radius_mm: float
    hasma_bore_flare_half_angle_rad: float
    hasma_boundary_mm: float
    hasma_hex_clearance_radius_mm: float
    hasma_thread_tpi: float | None
    hasma_thread_major_dia_mm: float | None

    # Detector
    detector_pcba_width: float
    detector_pcba_height: float
    detector_boundary_mm: float
    detector_glass_mm: float
    detector_slot_width: float
    detector_slot_height: float
    detector_slot_depth: float
    detector_package_width: float
    detector_package_height: float
    detector_pkg_slot_depth: float
    detector_plane_origin: Vec2
    detector_plane_x_dir: Vec2
    detector_plane_z_dir: Vec2

    # Insert holes
    insert_hole_positions: list[Vec3]
    insert_hole_radius_mm: float
    insert_hole_depth_mm: float
    corner_boss_radius_mm: float

    # Sensor-aligned bounding rect in uv coords (for downstream use)
    sensor_u_dir: Vec2 = field(repr=False, default=(1.0, 0.0))
    sensor_v_dir: Vec2 = field(repr=False, default=(0.0, 1.0))
    sensor_u_range: tuple[float, float] = field(repr=False, default=(0.0, 0.0))
    sensor_v_range: tuple[float, float] = field(repr=False, default=(0.0, 0.0))

    # Interior mounts (labels only — consumers look them up from the mount list)
    interior_mount_labels: list[str] = field(default_factory=list)

    # Sizing bboxes used for housing outline (HASMA + detector physical footprints)
    hasma_bbox_poly: list[Vec2] = field(default_factory=list)
    detector_bbox_poly: list[Vec2] = field(default_factory=list)

    # Stepped floor
    shallowest_floor_z: float = 0.0
    floor_steps: list[FloorStep] = field(default_factory=list)
    mount_floor_hulls: list[tuple[str, float, list[Vec2]]] = field(default_factory=list)

    # Mount fastener holes (bolts + pushers, drilled from housing bottom)
    mount_fasteners: list[MountFastener] = field(default_factory=list)
    bolt_clearance_dia_mm: float = 0.0
    bolt_counterbore_dia_mm: float = 0.0
    bolt_counterbore_depth_mm: float = 0.0
    pusher_access_dia_mm: float = 0.0

    # Controller pocket (bottom of housing)
    controller_pocket_origin: Vec2 | None = None
    controller_pocket_x_dir: Vec2 | None = None
    controller_board_width: float = 0.0
    controller_board_height: float = 0.0
    controller_pocket_depth: float = 0.0
    controller_solder_clearance_depth: float = 0.0
    controller_solder_protrusion: float = 0.0
    controller_insert_spacing_width: float = 0.0
    controller_insert_spacing_height: float = 0.0
    controller_corner_boss_radius: float = 0.0

    # Manufacturing
    print_tolerance_mm: float = 0.1


# ---- Detector dimension reads from BOM ------------------------------------

def _hasma_dims(slit_mount: dict) -> tuple[float, float, float, float]:
    """Read HASMA housing dimensions from BOM slit mount dict.

    Returns (bore_radius, hex_half, boundary, physical_depth).
    physical_depth is derived: length - boundary.
    """
    bore_r = float(slit_mount["bore_radius_mm"])
    hex_half = float(slit_mount["hex_half_mm"])
    boundary = float(slit_mount["boundary_mm"])
    length = float(slit_mount["length_mm"])
    return bore_r, hex_half, boundary, length - boundary


def _det_dims(det_el) -> tuple[float, float, float, float, float, float, float, float,
                               float, float, float, float]:
    """Read detector dimensions from element params (sourced from BOM).

    Returns (board_width, board_height, board_behind_die, glass_to_die,
    sensor_half_w, boundary, package_width, package_height,
    insert_spacing_width, insert_spacing_height, solder_clearance_depth,
    corner_boss_radius).
    """
    board_width = float(det_el.params["board_width_mm"])
    board_height = float(det_el.params["board_height_mm"])
    board_behind = float(det_el.params["board_behind_die_mm"])
    glass = float(det_el.params["glass_to_die_mm"])
    package_width = float(det_el.params["package_width_mm"])
    package_height = float(det_el.params["package_height_mm"])
    sensor_half = 0.5 * package_width
    boundary = float(det_el.params["boundary_behind_die_mm"])
    insert_w = float(det_el.params["insert_spacing_width_mm"])
    insert_h = float(det_el.params["insert_spacing_height_mm"])
    solder_depth = float(det_el.params["solder_clearance_depth_mm"])
    boss_r = float(det_el.params["corner_boss_radius_mm"])
    return (board_width, board_height, board_behind, glass, sensor_half,
            boundary, package_width, package_height,
            insert_w, insert_h, solder_depth, boss_r)


# ---- Pure geometry helpers ------------------------------------------------

def _face_half_width_mm(element) -> float:
    """XY-plane half-width of the optic face for beam channel sizing."""
    if element.kind == "slit":
        return 0.5 * element.params["width_um"] * 1e-3
    if element.kind == "mirror":
        return 0.5 * element.params["diameter_mm"]
    if element.kind == "grating":
        return 0.5 * element.params["size_mm"]
    return 0.0


def _mirror_aperture_xy(element) -> list[Vec2]:
    """Project the OAP aperture ellipse onto the z=0 plane.

    Uses the same aperture_overlay calculation as the beam spot
    diagrams: the elliptical aperture (r_major = r/cos(phi)) on the
    tilted recorder plane, projected onto XY via the plane's local-x
    axis.  Returns two (x, y) endpoints.
    """
    from raysect.core.math import Vector3D
    from optics.forward_trace import aperture_overlay, MM_TO_M
    from optics.elements.oap_mirror import _paraboloid_params

    otype, oparam = aperture_overlay(element, None)

    if otype == "ellipse":
        r_major = oparam[1]

        oaa = element.params["off_axis_angle_deg"]
        phi = math.radians(oaa / 2.0)
        fl_m = element.params["focal_length_mm"] * 1e-3
        ct_m = element.params["center_thickness_mm"] * 1e-3
        pa, pf, pv = _paraboloid_params(fl_m, oaa, focus_height_m=ct_m)
        z_oc = pv + pa * pa / (4.0 * pf)
        r_m = element.params["diameter_mm"] * 0.5 * 1e-3
        z_near = pv + (r_m - pa) ** 2 / (4.0 * pf)
        z_far = pv + (r_m + pa) ** 2 / (4.0 * pf)
        z_mid = 0.5 * (z_near + z_far)

        gd = element.axis
        fd = element.params["paraboloidal_focus_dir"]
        forward = Vector3D(gd[0], gd[1], 0.0).normalise()
        right = Vector3D(fd[0], fd[1], 0.0).normalise()
        cut_normal = Vector3D(
            math.sin(phi) * right.x + math.cos(phi) * forward.x,
            math.sin(phi) * right.y + math.cos(phi) * forward.y,
            0.0,
        ).normalise()
        local_x = Vector3D(-cut_normal.y, cut_normal.x, 0.0)

        ox = element.position[0] + (z_mid - z_oc) * forward.x * 1e3
        oy = element.position[1] + (z_mid - z_oc) * forward.y * 1e3

        return [
            (ox + r_major * local_x.x, oy + r_major * local_x.y),
            (ox - r_major * local_x.x, oy - r_major * local_x.y),
        ]

    # Spherical / flat mirror fallback: circle of diameter_mm
    r = element.params["diameter_mm"] / 2
    nx, ny = element.axis[0], element.axis[1]
    fx, fy = -ny, nx
    px, py = element.position[0], element.position[1]
    return [(px + fx * r, py + fy * r),
            (px - fx * r, py - fy * r)]


def _optic_edge_points(element, toward, beam_path=None,
                       endmill_r: float = 0.0) -> list[Vec2]:
    """Return two XY points defining the beam channel edge at this optic."""
    px, py = element.position[0], element.position[1]
    nx, ny = element.axis[0], element.axis[1]
    fx, fy = -ny, nx

    if element.kind == "detector":
        hw = 0.5 * float(element.params["array_length_mm"]) + endmill_r
        return [(px + fx * hw, py + fy * hw),
                (px - fx * hw, py - fy * hw)]

    if element.kind == "slit":
        hw = float(element.params["bore_radius_mm"])
        return [(px + fx * hw, py + fy * hw),
                (px - fx * hw, py - fy * hw)]

    if element.kind == "mirror":
        return _mirror_aperture_xy(element)

    if element.kind == "grating":
        hw = 0.5 * element.params["size_mm"]
        return [(px + fx * hw, py + fy * hw),
                (px - fx * hw, py - fy * hw)]

    return [(px, py), (px, py)]


def _beam_channel_polygons(beam_path, mounts, scene=None,
                           *, clearance: float = 1.0,
                           endmill_r: float = 0.0) -> list[list[Vec2]]:
    """Irregular quadrilateral channels between consecutive optics.

    Each quad's two side edges are pushed outward by *clearance*
    for lateral clearance.  The end edges at the optic faces are not
    expanded.
    """
    cl = clearance
    detector_el = None
    if scene:
        detector_el = next((e for e in scene.elements if e.kind == "detector"), None)

    channels: list[list[Vec2]] = []
    for i in range(len(beam_path) - 1):
        el_a, el_b = beam_path[i], beam_path[i + 1]
        if el_b.label == "exit_slit" and detector_el is not None:
            el_b = detector_el

        pts_a = _optic_edge_points(el_a, el_b.position, beam_path,
                                    endmill_r=endmill_r)
        pts_b = _optic_edge_points(el_b, el_a.position, beam_path,
                                    endmill_r=endmill_r)

        a0, a1 = pts_a
        b0, b1 = pts_b
        d00 = (a0[0] - b0[0]) ** 2 + (a0[1] - b0[1]) ** 2
        d01 = (a0[0] - b1[0]) ** 2 + (a0[1] - b1[1]) ** 2
        if d01 < d00:
            b0, b1 = b1, b0

        def _widen_face(p0, p1):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                return p0, p1
            ux, uy = dx / length, dy / length
            return ((p0[0] - ux * cl, p0[1] - uy * cl),
                    (p1[0] + ux * cl, p1[1] + uy * cl))

        a0, a1 = _widen_face(a0, a1)
        b0, b1 = _widen_face(b0, b1)

        # Angle-sort the four vertices around their centroid to
        # guarantee a convex, non-self-intersecting quad.  Widening
        # can re-introduce bowties on narrow beam segments.
        pts = [a0, a1, b0, b1]
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        channels.append(pts)
    return channels


def _pocket_rect_points(element, front_extent: float, back_extent: float,
                        half_width: float) -> list[Vec2]:
    """Return 4 XY corners of an oriented rectangle for a pocket."""
    nx, ny = element.axis[0], element.axis[1]
    fx, fy = -ny, nx
    px, py = element.position[0], element.position[1]

    return [
        (px + nx * front_extent + fx * half_width,
         py + ny * front_extent + fy * half_width),
        (px + nx * front_extent - fx * half_width,
         py + ny * front_extent - fy * half_width),
        (px - nx * back_extent - fx * half_width,
         py - ny * back_extent - fy * half_width),
        (px - nx * back_extent + fx * half_width,
         py - ny * back_extent + fy * half_width),
    ]


def _mirror_pocket_points(element, mount_dict) -> list[Vec2]:
    """XY polygon for a mirror + flexure mount pocket (raw, no clearance).

    Works for both spherical (rear_wall_mm) and OAP (slab_thickness_mm)
    mount types.
    """
    d = element.params["diameter_mm"]
    lip = element.params["edge_thickness_mm"]
    ot = element.params["center_thickness_mm"]
    if "slab_thickness_mm" in mount_dict:
        back_wall = mount_dict["slab_thickness_mm"]
    elif "rear_wall_mm" in mount_dict:
        back_wall = mount_dict["rear_wall_mm"]
    else:
        raise KeyError("mount requires slab_thickness_mm or rear_wall_mm")
    return _pocket_rect_points(
        element,
        front_extent=lip - ot,
        back_extent=ot + back_wall,
        half_width=0.5 * d,
    )


def _grating_pocket_points(element, mount_dict) -> list[Vec2]:
    """XY polygon for a ruled grating + jaw mount pocket (raw, no clearance)."""
    size = element.params["size_mm"]
    ot = element.params["center_thickness_mm"]
    lip = element.params["edge_thickness_mm"]
    rear_wall = mount_dict["rear_wall_mm"]
    return _pocket_rect_points(
        element,
        front_extent=lip - ot,
        back_extent=ot + rear_wall,
        half_width=0.5 * size,
    )



def _hasma_outside_pocket_points(element, hasma: tuple) -> list[Vec2]:
    """Outside pocket for HASMA: hex + body from boundary plane outward."""
    _, hex_half, boundary, _ = hasma
    return _pocket_rect_points(
        element,
        front_extent=-boundary,
        back_extent=hex_half,
        half_width=hex_half,
    )



def _detector_outside_pocket_points(element, det: tuple) -> list[Vec2]:
    """Outside pocket for detector: boundary plane to board back."""
    board_w, board_behind, glass, sensor_half, boundary = det
    return _pocket_rect_points(
        element,
        front_extent=boundary,
        back_extent=board_behind,
        half_width=0.5 * board_w,
    )


def _hasma_exterior_via_drop_lines(bp1_uv, bp2_uv, cavity_edges_uv,
                                   u0, u1, v0, v1,
                                   ent_el, ux, uy, vx, vy):
    """Split the housing rectangle at the HASMA boundary using drop lines.

    From each boundary endpoint, cast rays along sensor-aligned easy
    directions.  Keep rays that reach the housing edge without crossing
    any cavity boundary and make an obtuse angle with the boundary line.
    Among surviving candidates, pick the combination that maximises the
    cut polygon area.

    Returns (keep_poly, cut_poly) in (u,v) coords, or None on failure.
    """

    def _ray_seg_t(ox, oy, dx, dy, sx, sy, ex, ey):
        dsx, dsy = ex - sx, ey - sy
        denom = dx * dsy - dy * dsx
        if abs(denom) < 1e-10:
            return None
        t = ((sx - ox) * dsy - (sy - oy) * dsx) / denom
        s = ((sx - ox) * dy - (sy - oy) * dx) / denom
        if t > 1e-6 and -0.01 <= s <= 1.01:
            return t
        return None

    def _poly_area(poly):
        n = len(poly)
        a = 0.0
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            a += x0 * y1 - x1 * y0
        return abs(a) * 0.5

    boundary_vec = (bp2_uv[0] - bp1_uv[0], bp2_uv[1] - bp1_uv[1])

    easy_dirs = {
        (1, 0): 'u1', (-1, 0): 'u0',
        (0, 1): 'v1', (0, -1): 'v0',
    }

    def _find_candidates(pt_uv, bdir_uv):
        candidates = []
        for (du, dv), edge_label in easy_dirs.items():
            if du == 1:
                t_h = u1 - pt_uv[0]
            elif du == -1:
                t_h = pt_uv[0] - u0
            elif dv == 1:
                t_h = v1 - pt_uv[1]
            else:
                t_h = pt_uv[1] - v0
            if t_h <= 0:
                continue

            blocked = False
            for (s0, s1) in cavity_edges_uv:
                t_seg = _ray_seg_t(pt_uv[0], pt_uv[1], du, dv,
                                   s0[0], s0[1], s1[0], s1[1])
                if t_seg is not None and t_seg < t_h - 0.01:
                    blocked = True
                    break
            if blocked:
                continue

            dot = du * bdir_uv[0] + dv * bdir_uv[1]
            if dot >= 0:
                continue

            drop_pt = (pt_uv[0] + du * t_h, pt_uv[1] + dv * t_h)
            candidates.append((drop_pt, edge_label))
        return candidates

    cands1 = _find_candidates(bp1_uv, boundary_vec)
    cands2 = _find_candidates(bp2_uv, (-boundary_vec[0], -boundary_vec[1]))

    if not cands1 or not cands2:
        return None

    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    edge_to_idx = {'v0': 0, 'u1': 1, 'v1': 2, 'u0': 3}

    def _build_split(dp1, edge1, dp2, edge2):
        ei1 = edge_to_idx[edge1]
        ei2 = edge_to_idx[edge2]

        def _walk(start_ei, end_ei, start_pt, end_pt):
            path = [start_pt]
            i = start_ei
            for _ in range(5):
                i = (i + 1) % 4
                if i == (end_ei + 1) % 4:
                    path.append(end_pt)
                    return path
                path.append(corners[i])
            return None

        path_a = _walk(ei1, ei2, dp1, dp2)
        path_b = _walk(ei2, ei1, dp2, dp1)
        if path_a is None or path_b is None:
            return None, None

        inner_fwd = [dp1, bp1_uv, bp2_uv, dp2]
        poly_a = inner_fwd + list(reversed(path_a[1:-1]))
        inner_rev = [dp2, bp2_uv, bp1_uv, dp1]
        poly_b = inner_rev + list(reversed(path_b[1:-1]))

        ha = (ent_el.axis[0], ent_el.axis[1])
        test_pt = (ent_el.position[0] - ha[0] * 50,
                   ent_el.position[1] - ha[1] * 50)
        test_uv = (test_pt[0] * ux + test_pt[1] * uy,
                   test_pt[0] * vx + test_pt[1] * vy)

        def _pip(pt, poly):
            x, y = pt
            n = len(poly)
            inside = False
            j = n - 1
            for ii in range(n):
                xi, yi = poly[ii]
                xj, yj = poly[j]
                if ((yi > y) != (yj > y)) and \
                   (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = ii
            return inside

        if _pip(test_uv, poly_a):
            return poly_b, poly_a
        else:
            return poly_a, poly_b

    best = None
    best_area = -1
    for dp1, e1 in cands1:
        for dp2, e2 in cands2:
            keep, cut = _build_split(dp1, e1, dp2, e2)
            if cut is None:
                continue
            area = _poly_area(cut)
            if area > best_area:
                best_area = area
                best = (keep, cut)

    return best


def _expand_poly_bbox(poly: list[Vec2], margin: float) -> list[Vec2]:
    """Expand a polygon's axis-aligned bounding box by margin on all sides."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _poly_edges(poly: list[Vec2]) -> list[tuple[Vec2, Vec2]]:
    """Return consecutive edge pairs of a polygon."""
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


# ---- Builder ---------------------------------------------------------------

def build_solid_housing_spec(
    optics_scene,
    mounts,
    parts,
    beam_path,
    *,
    cavity_clearance_mm: float = 1.0,
    cavity_endmill_radius_mm: float = 1.0,
    pcba_pocket_clearance_mm: float = 2.5,
    controller_cavity_wall_mm: float = 5.0,
) -> SolidHousingSpec:
    """Compute all geometry for the unibody solid-subtract housing.

    Returns a SolidHousingSpec containing polygons, bore parameters,
    slot dimensions, and insert hole positions.  No build123d imports;
    consumers (CAD builder, raysect builder) apply their own
    extrude/subtract operations.

    The housing_outline returned here is the bounding rectangle of the
    expanded mount polys + HASMA/detector physical bboxes in sensor-
    aligned coords, expanded by wall thickness.  When build123d is
    available, the CAD builder may refine this using actual cavity
    sketch vertices (after offset + fillet).
    """
    # Z extent
    flange_mm, _ = _slit_mount_housing_dims(parts.slit_mount)
    z_bottom, z_top = _mount_z_extent(mounts, optics_scene,
                                      slit_flange_size_mm=flange_mm)

    # Map element labels to their mount dict and mount object.
    _label_mount = {
        "M1": parts.m1_mount, "M2": parts.m2_mount,
        "grating": parts.grating_mount,
        "F1": parts.f1_mount,
        "F2": parts.f2_mount,
    }
    mount_by_parent = {m.parent_label: m for m in mounts
                       if m.parent_label in _label_mount}
    interior_mount_labels = [m.label for m in mount_by_parent.values()]

    # Optic body oriented rectangles (OCH-plane projection).
    optic_polys: list[list[Vec2]] = []
    # Mount convex hulls (slab + foot combined).
    mount_hull_polys: list[list[Vec2]] = []
    # Per-mount floor data for step computation.
    mount_floors: dict[str, tuple[float, list[Vec2], list[Vec2]]] = {}
    for el in optics_scene.elements:
        md = _label_mount.get(el.label)
        if not md:
            continue
        if el.kind == "grating":
            op = _grating_pocket_points(el, md)
        else:
            op = _mirror_pocket_points(el, md)
        optic_polys.append(op)
        m = mount_by_parent.get(el.label)
        if m is not None:
            hull_pts = []
            if m.slab_polygon_xy is not None:
                hull_pts.extend(m.slab_polygon_xy)
            if m.foot_polygon_xy is not None:
                hull_pts.extend(m.foot_polygon_xy)
            if hull_pts:
                from optics.collision import convex_hull
                hull = convex_hull(hull_pts)
                mount_hull_polys.append(hull)
                if m.z_range_mm is not None:
                    mount_floors[el.label] = (m.z_range_mm[0], op, hull)

    # Wall mount keep-out polygons (slit wall, detector wall).
    wall_polys: list[list[Vec2]] = []
    for m in mounts:
        if m.slab_polygon_xy is not None and m.label == "slit_wall":
            wall_polys.append(list(m.slab_polygon_xy))

    # Beam channel polygons (already laterally expanded)
    beam_polys = _beam_channel_polygons(beam_path, mounts, scene=optics_scene,
                                        clearance=cavity_clearance_mm,
                                        endmill_r=cavity_endmill_radius_mm)

    # Stepped floor computation
    shallowest_floor_z, floor_steps = _compute_floor_steps(
        mount_floors, beam_path)

    # Detector dimensions from BOM
    det_el = next(e for e in optics_scene.elements if e.kind == "detector")
    det = _det_dims(det_el)
    (board_w, board_h, board_behind, glass, sensor_half, boundary, package_w, package_h,
     insert_sp_w, insert_sp_h, solder_depth, boss_r) = det

    # HASMA dimensions from BOM + physical bboxes for housing sizing
    ent_el = next(e for e in optics_scene.elements if e.label == "entrance_slit")
    hasma = _hasma_dims(parts.slit_mount)
    bore_r, hex_half, hasma_boundary, hasma_depth = hasma
    hasma_bbox_poly = _pocket_rect_points(
        ent_el,
        front_extent=bore_r,
        back_extent=hasma_boundary + hasma_depth,
        half_width=hex_half,
    )
    detector_bbox_poly = _pocket_rect_points(
        det_el,
        front_extent=glass,
        back_extent=board_behind,
        half_width=0.5 * board_w,
    )

    # Sensor-aligned bounding rectangle
    nx, ny = det_el.axis[0], det_el.axis[1]
    ux, uy = -ny, nx   # along sensor array (dispersion)
    vx, vy = nx, ny     # sensor normal (depth)
    wt = _WALL_THICKNESS_MM

    # Collect all cavity-influencing vertices: raw polys (clearance offset
    # is applied properly in the CAD builder sketch, not here).
    all_verts: list[Vec2] = []
    for poly in optic_polys + mount_hull_polys:
        all_verts.extend(poly)
    for poly in beam_polys:
        all_verts.extend(poly)
    all_verts.extend(hasma_bbox_poly)
    all_verts.extend(detector_bbox_poly)

    u_vals = [p[0] * ux + p[1] * uy for p in all_verts]
    v_vals = [p[0] * vx + p[1] * vy for p in all_verts]
    u0, u1 = min(u_vals) - wt, max(u_vals) + wt
    v0, v1 = min(v_vals) - wt, max(v_vals) + wt

    housing_outline: list[Vec2] = [
        (u0 * ux + v0 * vx, u0 * uy + v0 * vy),
        (u1 * ux + v0 * vx, u1 * uy + v0 * vy),
        (u1 * ux + v1 * vx, u1 * uy + v1 * vy),
        (u0 * ux + v1 * vx, u0 * uy + v1 * vy),
    ]

    # HASMA bore and tapered access pocket geometry.
    ha_axis = (ent_el.axis[0], ent_el.axis[1])
    hasma_bore_position: Vec3 = (ent_el.position[0], ent_el.position[1], 0.0)
    hasma_bore_axis: Vec2 = ha_axis

    # Tapered access trapezoid: narrow at the boundary wall (hex
    # clearance), flaring outward.  Flare half-angle is clamped to
    # min(2·θ_F1 − beam_widening, _HASMA_FLARE_HALF_ANGLE_DEG) so the
    # cut never reaches the F1→next beam cone.
    wf_depth = hasma_boundary
    wf_x = ent_el.position[0] - ha_axis[0] * wf_depth
    wf_y = ent_el.position[1] - ha_axis[1] * wf_depth
    ha_perp = (-ha_axis[1], ha_axis[0])

    beam_labels = [e.label for e in beam_path]
    max_flare_rad = math.radians(_HASMA_FLARE_HALF_ANGLE_DEG)
    slit_idx = beam_labels.index("entrance_slit")
    if slit_idx + 2 < len(beam_path):
        f1_el = beam_path[slit_idx + 1]
        next_el = beam_path[slit_idx + 2]
        sf = (f1_el.position[0] - ent_el.position[0],
              f1_el.position[1] - ent_el.position[1])
        sf_len = math.hypot(*sf)
        # 2·θ_F1 from the incidence angle to F1's normal
        inc_dot = -(sf[0] * f1_el.axis[0] + sf[1] * f1_el.axis[1]) / sf_len
        two_theta_f1 = 2.0 * math.acos(max(-1.0, min(1.0, inc_dot)))
        fm = (next_el.position[0] - f1_el.position[0],
              next_el.position[1] - f1_el.position[1])
        fm_len = math.hypot(*fm)
        r_next = _face_half_height_z(next_el)
        widening = math.atan2(r_next, fm_len) if fm_len > 1e-9 else 0.0
        safe = two_theta_f1 - widening
        if safe > 0:
            max_flare_rad = min(max_flare_rad, safe)
    hasma_bore_flare_half_angle_rad = math.radians(30.0)

    # Detector slot geometry
    da = det_el.axis
    det_bnd_pos: Vec2 = (
        det_el.position[0] - da[0] * boundary,
        det_el.position[1] - da[1] * boundary,
    )
    det_plane_x_dir: Vec2 = (-da[1], da[0])
    det_plane_z_dir: Vec2 = (da[0], da[1])
    pkg_slot_depth = boundary + glass

    # Insert hole positions (4 corners of the insert spacing rectangle)
    dx_dir = (-da[1], da[0])
    half_sp_w = 0.5 * insert_sp_w
    half_sp_h = 0.5 * insert_sp_h
    insert_positions: list[Vec3] = []
    for sx, sz in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        hx = det_bnd_pos[0] + sx * half_sp_w * dx_dir[0]
        hy = det_bnd_pos[1] + sx * half_sp_w * dx_dir[1]
        hz = sz * half_sp_h
        insert_positions.append((hx, hy, hz))

    # Mount fastener holes: bolts + pusher for each interior mount.
    # Local (v, u) bolt positions → world (x, y) via element frame.
    from optics.mounts import parse_mount_params, _perpendicular_in_dispersion_plane
    mount_fasteners: list[MountFastener] = []
    bolt_thread = None
    for el in optics_scene.elements:
        md = _label_mount.get(el.label)
        if md is None:
            continue
        m = mount_by_parent.get(el.label)
        if m is None or m.z_range_mm is None:
            continue
        try:
            mount_type = str(md["type"])
            mp = parse_mount_params(mount_type, md)
        except (KeyError, ValueError):
            continue
        if bolt_thread is None:
            bolt_thread = mp.foot_bolt_thread

        ax = el.axis
        ip = _perpendicular_in_dispersion_plane(ax)
        cx, cy = el.position[0], el.position[1]
        floor_z = m.z_range_mm[0]

        ct = float(el.params.get("center_thickness_mm", 0.0))
        if hasattr(mp, "rear_wall_mm"):
            u_wall_rear = -ct - mp.rear_wall_mm
        elif hasattr(mp, "slab_thickness_mm"):
            u_wall_rear = -ct - mp.slab_thickness_mm
        else:
            continue
        u_vertex = -ct if hasattr(mp, "slab_thickness_mm") else 0.0
        u_bolt = 0.5 * (u_wall_rear + u_vertex)
        u_front_bolt = (u_vertex + mp.front_bolt_offset_mm
                        if not hasattr(mp, "slab_thickness_mm")
                        else u_vertex + mp.front_bolt_offset_mm)
        v_bolt = mp.foot_bolt_spacing_mm
        u_pusher = u_vertex

        def _to_world(v_local, u_local):
            return (cx + v_local * ip[0] + u_local * ax[0],
                    cy + v_local * ip[1] + u_local * ax[1])

        for bv, bu in [(+v_bolt, u_bolt), (-v_bolt, u_bolt),
                        (0.0, u_front_bolt)]:
            mount_fasteners.append(MountFastener(
                xy=_to_world(bv, bu), floor_z=floor_z,
                kind="bolt", label=el.label))
        mount_fasteners.append(MountFastener(
            xy=_to_world(0.0, u_pusher), floor_z=floor_z,
            kind="pusher", label=el.label))

    # Bolt dimensions from BOM manufacturing section
    from optics.mounts_cad import _load_manufacturing
    mfg = _load_manufacturing()
    bt = bolt_thread or "M2.5"
    bolt_cl = float(mfg.bolt_dims[bt]["clearance_dia_mm"])
    bolt_cb = float(mfg.bolt_dims[bt]["head_dia_mm"]) + 0.5
    bolt_cb_depth = 5.0
    pusher_dia = float(mfg.bolt_dims["M2"]["clearance_dia_mm"])

    # Controller pocket: centered on detector along die array, offset
    # along detector normal to equalize 2-D clearance to M2 and grating
    # screw counterbore holes on the housing bottom face.
    ctrl_kwargs: dict = {}
    if parts.controller_board_width_mm is not None:
        det_x, det_y = det_el.position[0], det_el.position[1]
        r_cb = 0.5 * bolt_cb
        perp = det_plane_x_dir
        h_n = 0.5 * parts.controller_board_height_mm + pcba_pocket_clearance_mm
        h_p = 0.5 * parts.controller_board_width_mm + pcba_pocket_clearance_mm

        m2_bolts: list[tuple[float, float]] = []
        grat_bolts: list[tuple[float, float]] = []
        for mf in mount_fasteners:
            if mf.kind != "bolt":
                continue
            dx = mf.xy[0] - det_x
            dy = mf.xy[1] - det_y
            t_n = dx * da[0] + dy * da[1]
            t_p = dx * perp[0] + dy * perp[1]
            if mf.label == "M2":
                m2_bolts.append((t_n, t_p))
            elif mf.label == "grating":
                grat_bolts.append((t_n, t_p))

        def _min_gap(d, group):
            best = math.inf
            for t_n, t_p in group:
                gn = max(0.0, abs(t_n - d) - h_n)
                gp = max(0.0, abs(t_p) - h_p)
                g = math.hypot(gn, gp) - r_cb
                if g < best:
                    best = g
            return best

        m2_mean = sum(t for t, _ in m2_bolts) / len(m2_bolts)
        grat_mean = sum(t for t, _ in grat_bolts) / len(grat_bolts)
        lo = min(m2_mean, grat_mean)
        hi = max(m2_mean, grat_mean)
        f_lo = _min_gap(lo, m2_bolts) - _min_gap(lo, grat_bolts)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f_mid = _min_gap(mid, m2_bolts) - _min_gap(mid, grat_bolts)
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
            else:
                hi = mid
        ctrl_inward_offset = 0.5 * (lo + hi)

        ctrl_origin = (
            det_x + ctrl_inward_offset * da[0],
            det_y + ctrl_inward_offset * da[1],
        )
        ctrl_pocket_depth = (parts.controller_component_height_mm
                             + parts.controller_board_thickness_mm)
        ctrl_solder_depth = parts.controller_solder_clearance_depth_mm
        ctrl_total = ctrl_pocket_depth + ctrl_solder_depth

        # z_bottom must be deep enough that there is
        # controller_cavity_wall_mm between the solder clearance
        # ceiling and the interior cavity floor.
        z_bottom = min(z_bottom,
                       shallowest_floor_z - ctrl_total - controller_cavity_wall_mm)

        ctrl_kwargs = dict(
            controller_pocket_origin=ctrl_origin,
            controller_pocket_x_dir=det_plane_x_dir,
            controller_board_width=parts.controller_board_width_mm,
            controller_board_height=parts.controller_board_height_mm,
            controller_pocket_depth=ctrl_pocket_depth,
            controller_solder_clearance_depth=ctrl_solder_depth,
            controller_solder_protrusion=parts.controller_solder_protrusion_mm,
            controller_insert_spacing_width=parts.controller_insert_spacing_width_mm,
            controller_insert_spacing_height=parts.controller_insert_spacing_height_mm,
            controller_corner_boss_radius=parts.controller_corner_boss_radius_mm,
        )

    return SolidHousingSpec(
        housing_outline=housing_outline,
        z_bottom=z_bottom,
        z_top=z_top,
        optic_polys=optic_polys,
        mount_hull_polys=mount_hull_polys,
        beam_polys=beam_polys,
        wall_polys=wall_polys,
        wall_thickness_mm=_WALL_THICKNESS_MM,
        hasma_bore_position=hasma_bore_position,
        hasma_bore_axis=hasma_bore_axis,
        hasma_bore_radius_mm=bore_r,
        hasma_bore_flare_half_angle_rad=hasma_bore_flare_half_angle_rad,
        hasma_boundary_mm=hasma_boundary,
        hasma_hex_clearance_radius_mm=hex_half + cavity_clearance_mm,
        hasma_thread_tpi=parts.slit_mount.get("thread_tpi"),
        hasma_thread_major_dia_mm=parts.slit_mount.get("thread_major_dia_mm"),
        detector_pcba_width=board_w,
        detector_pcba_height=board_h,
        detector_boundary_mm=boundary,
        detector_glass_mm=glass,
        detector_slot_width=insert_sp_w,
        detector_slot_height=insert_sp_h,
        detector_slot_depth=solder_depth,
        detector_package_width=package_w,
        detector_package_height=package_h,
        detector_pkg_slot_depth=pkg_slot_depth,
        detector_plane_origin=det_bnd_pos,
        detector_plane_x_dir=det_plane_x_dir,
        detector_plane_z_dir=det_plane_z_dir,
        insert_hole_positions=insert_positions,
        insert_hole_radius_mm=_SCREW_PILOT_R_MM,
        insert_hole_depth_mm=_SCREW_PILOT_DEPTH_MM,
        corner_boss_radius_mm=boss_r,
        sensor_u_dir=(ux, uy),
        sensor_v_dir=(vx, vy),
        sensor_u_range=(u0, u1),
        sensor_v_range=(v0, v1),
        interior_mount_labels=interior_mount_labels,
        hasma_bbox_poly=hasma_bbox_poly,
        detector_bbox_poly=detector_bbox_poly,
        shallowest_floor_z=shallowest_floor_z,
        floor_steps=floor_steps,
        mount_floor_hulls=[(lbl, fz, list(hull))
                           for lbl, (fz, _op, hull) in mount_floors.items()],
        mount_fasteners=mount_fasteners,
        bolt_clearance_dia_mm=bolt_cl,
        bolt_counterbore_dia_mm=bolt_cb,
        bolt_counterbore_depth_mm=bolt_cb_depth,
        pusher_access_dia_mm=pusher_dia,
        print_tolerance_mm=parts.print_tolerance_mm,
        **ctrl_kwargs,
    )
