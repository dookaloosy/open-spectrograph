"""Procedural mount shape builders (raysect CSG).

Three flexure mount types, all 3D-printed with integrated pitch
adjustment via a thin living-hinge flexure at the rear wall:

  mirror_flexure — round bore for spherical/flat mirrors
  oap_flexure    — flat plate for off-axis parabolic mirrors
  grating_flexure — jaw pocket for rectangular gratings

Each mount has a shaped forward foot (tongue + boss), flexure gap,
and thin web connecting the slab to the foot.
"""


import math
from dataclasses import dataclass
from typing import Callable

from raysect.core.math import Point3D, rotate_basis, rotate_z, translate
from raysect.optical import ConstantSF
from raysect.optical.material.lambert import Lambert
from raysect.primitive import Box, Cylinder, Intersect, Subtract, Union

from optics.scene import ElementPlacement, Mount


_MM_TO_M = 1.0e-3
_HORIZONTAL_CLEARANCE_MM = 1.0


def _foot_hull_xy(element, v_half_mm, boss_half_mm, u_rear, u_vertex, u_front):
    """Convex hull of the T-shaped foot projected to the xy plane.

    Wide crossbar (±v_half_mm) from u_rear to u_vertex, narrow stem
    (±boss_half_mm) from u_vertex to u_front.  Returns vertices in
    convex-hull order (CCW).
    """
    n = element.axis
    ip = _perpendicular_in_dispersion_plane(n)
    cx, cy = element.position[0], element.position[1]

    def _xy(v, u):
        return (cx + v * ip[0] + u * n[0],
                cy + v * ip[1] + u * n[1])

    pts = [
        _xy(-v_half_mm, u_rear),
        _xy(+v_half_mm, u_rear),
        _xy(+v_half_mm, u_vertex),
        _xy(+boss_half_mm, u_vertex),
        _xy(+boss_half_mm, u_front),
        _xy(-boss_half_mm, u_front),
        _xy(-boss_half_mm, u_vertex),
        _xy(-v_half_mm, u_vertex),
    ]
    from optics.collision import convex_hull
    return tuple(convex_hull(pts))


def _foot_outline_xy(element, v_half_mm, boss_half_mm, u_rear, u_vertex, u_front):
    """Actual T-shaped foot outline projected to the xy plane.

    Same vertices as ``_foot_hull_xy`` but without the convex hull —
    preserves the concave notches at the crossbar-to-tongue junction.
    Returns vertices in CCW winding order.
    """
    n = element.axis
    ip = _perpendicular_in_dispersion_plane(n)
    cx, cy = element.position[0], element.position[1]

    def _xy(v, u):
        return (cx + v * ip[0] + u * n[0],
                cy + v * ip[1] + u * n[1])

    return (
        _xy(-v_half_mm, u_rear),
        _xy(+v_half_mm, u_rear),
        _xy(+v_half_mm, u_vertex),
        _xy(+boss_half_mm, u_vertex),
        _xy(+boss_half_mm, u_front),
        _xy(-boss_half_mm, u_front),
        _xy(-boss_half_mm, u_vertex),
        _xy(-v_half_mm, u_vertex),
    )


def _oriented_rect_xy(element, v_half_mm, u_lo, u_hi):
    """Compute an oriented rectangle polygon in the xy plane.

    Returns four (x, y) corners in CCW winding order for a rectangle
    spanning ±v_half_mm perpendicular to the element axis and
    u_lo..u_hi along it, centered on element.position.
    """
    n = element.axis
    ip = _perpendicular_in_dispersion_plane(n)
    cx, cy = element.position[0], element.position[1]

    def _pt(sv, u):
        return (cx + sv * v_half_mm * ip[0] + u * n[0],
                cy + sv * v_half_mm * ip[1] + u * n[1])

    return (_pt(-1, u_lo), _pt(+1, u_lo), _pt(+1, u_hi), _pt(-1, u_hi))


def _V3(v: tuple[float, float, float]):
    """Short alias for raysect `Vector3D` construction."""
    from raysect.core.math import Vector3D
    return Vector3D(*v)


def _perpendicular_in_dispersion_plane(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the in-dispersion-plane direction perpendicular to `normal`."""
    from optics.scene import perpendicular_xy
    px, py = perpendicular_xy(normal)
    return (px, py, 0.0)


@dataclass(frozen=True)
class ManufacturingParams:
    """Global manufacturing parameters from [manufacturing] in the BOM.

    Shared across all flexure mount types. Covers print tolerances,
    fillet radii, bolt dimensions, and the heat-set insert spec.
    """
    assembly_clearance_mm: float   # bore oversize for optic assembly fit
    print_tolerance_mm: float      # ± FDM/SLA dimensional tolerance
    fillet_radius_mm: float        # internal fillet radius (channel, tongue junction)
    # Bolt dimensions (keyed by thread string, e.g. "M3")
    bolt_dims: dict[str, dict[str, float]]
    # Heat-set insert: McMaster 94459A767, M2 brass
    insert_bore_dia_mm: float      # drill/print bore diameter
    insert_length_mm: float        # insert body length (flush when installed)
    insert_min_material_mm: float  # min wall thickness around insert
    insert_chamfer_mm: float       # snowplow relief chamfer at bore entry
    insert_flange_dia_mm: float    # insert flange outer diameter


@dataclass(frozen=True)
class RoundMirrorFlexureMountParams:
    """Dimensions for a flexure mirror mount with rear pivot and forward foot.

    Round-bore optic retention with flexure pitch adjustment via a thin
    living-hinge web at the rear wall.
    """
    # -- optic retention --
    optic_clearance_mm: float       # additional diametral bore clearance
    shoulder_width_mm: float       # annular shoulder width in bore
    channel_width_mm: float        # extraction channel width at bottom
    channel_extension_mm: float    # channel circle oversize beyond bore radius
    # -- slab geometry --
    wall_margin_mm: float          # material around bore (v and w sides)
    head_clearance_mm: float       # material above optic centre to top face
    foot_clearance_mm: float       # material below optic centre to slab bottom
    pusher_shelf_mm: float         # semicircular lip height above flexure gap
    rear_wall_mm: float            # material behind shoulder to back face
    # -- foot geometry --
    foot_thickness_mm: float       # foot slab thickness (w direction)
    foot_bolt_thread: str          # "M3" clearance holes in foot rails
    foot_bolt_spacing_mm: float    # bolt half-spacing in v-direction
    bolt_safety_mm: float          # clearance around bolt head to foot edge
    front_bolt_offset_mm: float    # u-offset of front tongue bolt from u_vertex
    # -- flexure --
    flexure_thickness_mm: float    # thin web dimension in u (layer-height controlled)
    flexure_gap_mm: float          # relief gap height (blade bending length)
    trim_angle_deg: float          # foot bottom trim angle for pitch preload
    # -- contact bumps (TPU, captive in bore pockets) --
    contact_radius_mm: float       # bump cylinder radius
    contact_offset_mm: float       # protrusion past bore wall into bore
    contact_separation_mm: float   # v-distance between the two bottom bumps


@dataclass(frozen=True)
class GratingFlexureMountParams:
    """Dimensions for a flexure grating mount with rear pivot and forward foot.

    Jaw-pocket retention (contact bumps + setscrew) with flexure pitch
    adjustment. Global manufacturing params (insert, fillet, tolerances)
    read from [manufacturing] in the BOM.
    """
    # -- optic retention --
    jaw_clearance_mm: float        # jaw arm material above/below grating edge
    optic_clearance_mm: float       # additional pocket clearance (diametral)
    contact_radius_mm: float       # bump cylinder radius
    contact_offset_mm: float       # protrusion past pocket wall into pocket
    contact_separation_mm: float   # v-distance between the two bottom bumps
    # -- slab geometry --
    head_clearance_mm: float       # material above grating edge to top face
    foot_clearance_mm: float       # material below grating edge to slab bottom
    pusher_shelf_mm: float         # semicircular lip height above flexure gap
    rear_wall_mm: float            # material behind grating to back face
    # -- foot geometry --
    foot_thickness_mm: float       # foot slab thickness (w direction)
    foot_bolt_thread: str          # "M3" clearance holes in foot rails
    foot_bolt_spacing_mm: float    # bolt half-spacing in v-direction
    bolt_safety_mm: float          # clearance around bolt head to foot edge
    front_bolt_offset_mm: float    # u-offset of front tongue bolt from u_vertex
    # -- flexure --
    flexure_thickness_mm: float    # thin web dimension in u
    flexure_gap_mm: float          # relief gap height
    trim_angle_deg: float          # foot bottom trim angle


@dataclass(frozen=True)
class OAPMirrorFlexureMountParams:
    """Dimensions for a flexure OAP mount with rear pivot and forward foot.

    Hex-bolt-pattern retention (6 holes on bolt circle + central
    locating pin) with flexure pitch adjustment. Global manufacturing
    params read from [manufacturing] in the BOM.
    """
    # -- optic retention --
    bolt_circle_radius_mm: float   # vendor bolt pattern radius
    n_holes: int                   # number of bolt holes (6 = hex)
    hole_phase_deg: float          # angular offset of first hole from 0°
    clearance_hole_dia_mm: float   # through-hole for optic bolts
    counterbore_dia_mm: float      # flat-head countersink major dia
    screw_part: str                # McMaster flat head screw part number
    # -- slab geometry --
    slab_thickness_mm: float       # total slab depth in u
    head_clearance_mm: float       # material above optic centre to top face
    foot_clearance_mm: float       # material below optic centre to slab bottom
    pusher_shelf_mm: float         # semicircular lip height above flexure gap
    # -- foot geometry --
    foot_thickness_mm: float       # foot slab thickness (w direction)
    foot_bolt_thread: str          # "M3" clearance holes in foot rails
    foot_bolt_spacing_mm: float    # bolt half-spacing in v-direction
    bolt_safety_mm: float          # clearance around bolt head to foot edge
    front_bolt_offset_mm: float    # u-offset of front tongue bolt from u_plate_front
    # -- flexure --
    flexure_thickness_mm: float    # thin web dimension in u
    flexure_gap_mm: float          # relief gap height
    trim_angle_deg: float          # foot bottom trim angle


# ── Flexure mounts (raysect CSG) ───────────────────────────────────────────


def build_mirror_flexure_mount(
    element: ElementPlacement, params: RoundMirrorFlexureMountParams, albedo: float,
    mfg: ManufacturingParams | None = None,
) -> Mount:
    """Flexure mirror mount for spherical/flat mirrors (raysect CSG).

    Simplified geometry matching the CAD: tombstone slab with bore,
    forward foot, flexure gap. No fillets, bolt holes, or extraction
    channel detail.
    """
    if mfg is None:
        from optics.mounts_cad import _load_manufacturing
        mfg = _load_manufacturing()

    optic_D = float(element.params["diameter_mm"])
    optic_thick = float(element.params["center_thickness_mm"])
    optic_R = 0.5 * optic_D

    u_vertex = 0.0
    u_shoulder = -optic_thick
    u_wall_rear = u_shoulder - params.rear_wall_mm

    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm

    w_top = optic_R + params.head_clearance_mm
    w_bot = -(optic_R + params.foot_clearance_mm)
    w_foot_bot = w_bot - params.flexure_gap_mm - params.foot_thickness_mm

    v_half_mm = optic_R

    front_bore_dia = optic_D + mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    rear_bore_dia = optic_D - 2.0 * params.shoulder_width_mm

    eps = 0.001  # CSG overcut to avoid coincident-face artifacts

    # Slab body: bottom rectangle + top arch.
    slab_bottom = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, 0.0, u_vertex * _MM_TO_M),
    )
    slab_depth = u_vertex - u_wall_rear
    slab_circle = Cylinder(
        radius=v_half_mm * _MM_TO_M,
        height=slab_depth * _MM_TO_M,
        transform=translate(0.0, 0.0, u_wall_rear * _MM_TO_M),
    )
    _d_shift = v_half_mm / math.sqrt(2.0)
    slab_diamond = Box(
        lower=Point3D(-0.5 * v_half_mm * _MM_TO_M, -0.5 * v_half_mm * _MM_TO_M, 0.0),
        upper=Point3D(+0.5 * v_half_mm * _MM_TO_M, +0.5 * v_half_mm * _MM_TO_M, slab_depth * _MM_TO_M),
    )
    slab_diamond.transform = (
        translate(0.0, _d_shift * _MM_TO_M, u_wall_rear * _MM_TO_M)
        * rotate_z(45.0)
    )
    slab_clip = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, 0.0, (u_wall_rear - eps) * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_top * _MM_TO_M, (u_vertex + eps) * _MM_TO_M),
    )
    slab_top = Intersect(Union(slab_circle, slab_diamond), slab_clip)
    slab = Union(slab_bottom, slab_top)

    # Subtract bores.
    front_bore = Cylinder(
        radius=0.5 * front_bore_dia * _MM_TO_M,
        height=(u_vertex - u_shoulder + 2 * eps) * _MM_TO_M,
        transform=translate(0.0, 0.0, (u_shoulder - eps) * _MM_TO_M),
    )
    rear_bore = Cylinder(
        radius=0.5 * rear_bore_dia * _MM_TO_M,
        height=(u_shoulder - u_wall_rear + 2 * eps) * _MM_TO_M,
        transform=translate(0.0, 0.0, (u_wall_rear - eps) * _MM_TO_M),
    )
    mount_csg = Subtract(Subtract(slab, front_bore), rear_bore)

    w_foot_top = w_bot - params.flexure_gap_mm
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    u_front_bolt = u_vertex + params.front_bolt_offset_mm

    # Foot: full width behind u_vertex, narrow tongue forward.
    foot_rear = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_vertex * _MM_TO_M),
    )
    foot_length = Box(
        lower=Point3D(-0.5 * boss_width_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_vertex * _MM_TO_M),
        upper=Point3D(+0.5 * boss_width_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_front_bolt * _MM_TO_M),
    )
    foot_boss = Cylinder(
        radius=0.5 * boss_width_mm * _MM_TO_M,
        height=(w_foot_top - w_foot_bot) * _MM_TO_M,
        transform=(
            translate(0.0, w_foot_bot * _MM_TO_M, u_front_bolt * _MM_TO_M)
            * rotate_basis(forward=_V3((0.0, 1.0, 0.0)), up=_V3((0.0, 0.0, 1.0)))
        ),
    )
    foot = Union(Union(foot_rear, foot_length), foot_boss)

    # Flexure web: thin connection at the back wall.
    flex_web = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, (u_wall_rear + params.flexure_thickness_mm) * _MM_TO_M),
    )

    mount_csg = Union(Union(mount_csg, foot), flex_web)
    mount_csg.material = Lambert(ConstantSF(albedo))

    mount_csg.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    in_plane = _perpendicular_in_dispersion_plane(element.axis)
    n = element.axis
    cx, cy = element.position[0], element.position[1]
    u_lo, u_hi = u_wall_rear, u_foot_front
    corners = [
        (cx + sv * v_half_mm * in_plane[0] + u * n[0],
         cy + sv * v_half_mm * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half_mm, u_wall_rear, u_vertex)
    foot_poly = _foot_hull_xy(element, v_half_mm, 0.5 * boss_width_mm,
                              u_wall_rear, u_vertex, u_foot_front)
    foot_outline = _foot_outline_xy(element, v_half_mm, 0.5 * boss_width_mm,
                                    u_wall_rear, u_vertex, u_foot_front)
    return Mount(label=f"{element.label}_mount", csg=mount_csg,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label=element.label,
                 z_range_mm=(w_foot_bot, w_top),
                 slab_polygon_xy=slab_poly,
                 foot_polygon_xy=foot_poly,
                 foot_outline_xy=foot_outline)


def build_grating_flexure_mount(
    element: ElementPlacement, params: GratingFlexureMountParams, albedo: float,
    mfg: ManufacturingParams | None = None,
) -> Mount:
    """Flexure grating mount (raysect CSG).

    Simplified geometry matching the CAD: rectangular slab with jaw pocket,
    forward foot, flexure gap.
    """
    if mfg is None:
        from optics.mounts_cad import _load_manufacturing
        mfg = _load_manufacturing()

    size_mm = float(element.params["size_mm"])
    optic_thick = float(element.params["center_thickness_mm"])
    grating_thick = optic_thick
    half_size = 0.5 * size_mm

    u_vertex = 0.0
    u_wall_rear = -(grating_thick + params.rear_wall_mm)

    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm

    w_top = half_size + params.head_clearance_mm
    w_bot = -(half_size + params.foot_clearance_mm)
    w_foot_bot = w_bot - params.flexure_gap_mm - params.foot_thickness_mm

    v_half_mm = half_size

    pocket_depth = grating_thick
    eps = 0.001  # CSG overcut to avoid coincident-face artifacts

    # Slab body: simple rectangle.
    slab = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_top * _MM_TO_M, u_vertex * _MM_TO_M),
    )

    # Subtract jaw pocket.
    pocket = Box(
        lower=Point3D(-half_size * _MM_TO_M, -half_size * _MM_TO_M, (u_vertex - pocket_depth - eps) * _MM_TO_M),
        upper=Point3D(+half_size * _MM_TO_M, +half_size * _MM_TO_M, (u_vertex + eps) * _MM_TO_M),
    )
    mount_csg = Subtract(slab, pocket)

    w_foot_top = w_bot - params.flexure_gap_mm
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    u_front_bolt = u_vertex + params.front_bolt_offset_mm

    # Foot: full width behind u_vertex, narrow tongue forward.
    foot_rear = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_vertex * _MM_TO_M),
    )
    foot_length = Box(
        lower=Point3D(-0.5 * boss_width_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_vertex * _MM_TO_M),
        upper=Point3D(+0.5 * boss_width_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_front_bolt * _MM_TO_M),
    )
    foot_boss = Cylinder(
        radius=0.5 * boss_width_mm * _MM_TO_M,
        height=(w_foot_top - w_foot_bot) * _MM_TO_M,
        transform=(
            translate(0.0, w_foot_bot * _MM_TO_M, u_front_bolt * _MM_TO_M)
            * rotate_basis(forward=_V3((0.0, 1.0, 0.0)), up=_V3((0.0, 0.0, 1.0)))
        ),
    )
    foot = Union(Union(foot_rear, foot_length), foot_boss)

    # Flexure web: thin connection at the back wall.
    flex_web = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, (u_wall_rear + params.flexure_thickness_mm) * _MM_TO_M),
    )

    mount_csg = Union(Union(mount_csg, foot), flex_web)
    mount_csg.material = Lambert(ConstantSF(albedo))

    mount_csg.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    in_plane = _perpendicular_in_dispersion_plane(element.axis)
    n = element.axis
    cx, cy = element.position[0], element.position[1]
    u_lo, u_hi = u_wall_rear, u_foot_front
    corners = [
        (cx + sv * v_half_mm * in_plane[0] + u * n[0],
         cy + sv * v_half_mm * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half_mm, u_wall_rear, u_vertex)
    foot_poly = _foot_hull_xy(element, v_half_mm, 0.5 * boss_width_mm,
                              u_wall_rear, u_vertex, u_foot_front)
    foot_outline = _foot_outline_xy(element, v_half_mm, 0.5 * boss_width_mm,
                                    u_wall_rear, u_vertex, u_foot_front)
    return Mount(label=f"{element.label}_mount", csg=mount_csg,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label=element.label,
                 z_range_mm=(w_foot_bot, w_top),
                 slab_polygon_xy=slab_poly,
                 foot_polygon_xy=foot_poly,
                 foot_outline_xy=foot_outline)


def build_oap_flexure_mount(
    element: ElementPlacement, params: OAPMirrorFlexureMountParams, albedo: float,
    mfg: ManufacturingParams | None = None,
) -> Mount:
    """Flexure OAP mount (raysect CSG).

    Simplified geometry matching the CAD: flat plate with through-holes
    (not modelled in CSG), forward foot, flexure gap.
    """
    if mfg is None:
        from optics.mounts_cad import _load_manufacturing
        mfg = _load_manufacturing()

    optic_D = float(element.params["diameter_mm"])
    optic_thick = float(element.params["center_thickness_mm"])
    optic_R = 0.5 * optic_D

    u_plate_front = -optic_thick
    u_wall_rear = u_plate_front - params.slab_thickness_mm

    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = max(bolt_head_mm + 2 * params.bolt_safety_mm,
                         params.front_bolt_offset_mm + 0.5 * boss_width_mm)
    u_foot_front = u_plate_front + foot_length_mm

    w_top = optic_R + params.head_clearance_mm
    w_bot = -(optic_R + params.foot_clearance_mm)
    w_foot_bot = w_bot - params.flexure_gap_mm - params.foot_thickness_mm

    v_half_mm = optic_R

    eps = 0.001  # CSG overcut to avoid coincident-face artifacts

    # Slab body: bottom rectangle + top semicircle.
    slab_bottom = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, 0.0, u_plate_front * _MM_TO_M),
    )
    slab_circle = Cylinder(
        radius=v_half_mm * _MM_TO_M,
        height=params.slab_thickness_mm * _MM_TO_M,
        transform=translate(0.0, 0.0, u_wall_rear * _MM_TO_M),
    )
    slab_clip = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, 0.0, (u_wall_rear - eps) * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_top * _MM_TO_M, (u_plate_front + eps) * _MM_TO_M),
    )
    slab_top = Intersect(slab_circle, slab_clip)
    mount_csg = Union(slab_bottom, slab_top)

    w_foot_top = w_bot - params.flexure_gap_mm
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    u_front_bolt = u_plate_front + params.front_bolt_offset_mm

    # Foot: full width behind u_plate_front, narrow tongue forward.
    foot_rear = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_plate_front * _MM_TO_M),
    )
    foot_length = Box(
        lower=Point3D(-0.5 * boss_width_mm * _MM_TO_M, w_foot_bot * _MM_TO_M, u_plate_front * _MM_TO_M),
        upper=Point3D(+0.5 * boss_width_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_front_bolt * _MM_TO_M),
    )
    foot_boss = Cylinder(
        radius=0.5 * boss_width_mm * _MM_TO_M,
        height=(w_foot_top - w_foot_bot) * _MM_TO_M,
        transform=(
            translate(0.0, w_foot_bot * _MM_TO_M, u_front_bolt * _MM_TO_M)
            * rotate_basis(forward=_V3((0.0, 1.0, 0.0)), up=_V3((0.0, 0.0, 1.0)))
        ),
    )
    foot = Union(Union(foot_rear, foot_length), foot_boss)

    # Flexure web: thin connection at the back wall.
    flex_web = Box(
        lower=Point3D(-v_half_mm * _MM_TO_M, w_foot_top * _MM_TO_M, u_wall_rear * _MM_TO_M),
        upper=Point3D(+v_half_mm * _MM_TO_M, w_bot * _MM_TO_M, (u_wall_rear + params.flexure_thickness_mm) * _MM_TO_M),
    )

    mount_csg = Union(Union(mount_csg, foot), flex_web)
    mount_csg.material = Lambert(ConstantSF(albedo))

    mount_csg.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    in_plane = _perpendicular_in_dispersion_plane(element.axis)
    n = element.axis
    cx, cy = element.position[0], element.position[1]
    u_lo, u_hi = u_wall_rear, u_foot_front
    corners = [
        (cx + sv * v_half_mm * in_plane[0] + u * n[0],
         cy + sv * v_half_mm * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half_mm, u_wall_rear, u_plate_front)
    foot_poly = _foot_hull_xy(element, v_half_mm, 0.5 * boss_width_mm,
                              u_wall_rear, u_plate_front, u_foot_front)
    foot_outline = _foot_outline_xy(element, v_half_mm, 0.5 * boss_width_mm,
                                    u_wall_rear, u_plate_front, u_foot_front)
    return Mount(label=f"{element.label}_mount", csg=mount_csg,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label=element.label,
                 z_range_mm=(w_foot_bot, w_top),
                 slab_polygon_xy=slab_poly,
                 foot_polygon_xy=foot_poly,
                 foot_outline_xy=foot_outline)


# ── Detector bbox (for feasibility checking) ──────────────────────────────


def build_detector_bbox(
    element: ElementPlacement, albedo: float,
) -> Mount:
    """Collision bounding box for the detector board.

    Not a physical mount — the board bolts directly to the housing.
    Dimensions from the element's params dict (sourced from BOM).
    The box spans from glass_to_die_mm in front of the sensor plane
    (toward M2) to board_behind_die_mm behind it (away from M2).
    """
    board_w = float(element.params["board_width_mm"])
    board_h = float(element.params["board_height_mm"])
    board_behind_die_mm = float(element.params["board_behind_die_mm"])
    glass_front_mm = float(element.params["glass_to_die_mm"])

    v_half = 0.5 * board_w
    w_half = 0.5 * board_h

    slab = Box(
        lower=Point3D(-v_half * _MM_TO_M, -w_half * _MM_TO_M,
                       -board_behind_die_mm * _MM_TO_M),
        upper=Point3D(+v_half * _MM_TO_M, +w_half * _MM_TO_M,
                       glass_front_mm * _MM_TO_M),
    )
    slab.material = Lambert(ConstantSF(albedo))

    slab.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    in_plane = _perpendicular_in_dispersion_plane(element.axis)
    n = element.axis
    cx, cy = element.position[0], element.position[1]
    u_lo, u_hi = -board_behind_die_mm, glass_front_mm
    corners = [
        (cx + sv * v_half * in_plane[0] + u * n[0],
         cy + sv * v_half * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    pkg_w = float(element.params["package_width_mm"])
    pkg_h = float(element.params["package_height_mm"])
    pkg_v_half = 0.5 * pkg_w
    pkg_u_lo = -0.5 * pkg_h
    pkg_u_hi = 0.5 * pkg_h
    slab_poly = _oriented_rect_xy(element, pkg_v_half, pkg_u_lo, pkg_u_hi)
    boundary_mm = float(element.params["boundary_behind_die_mm"])
    wall_poly = _oriented_rect_xy(element, pkg_v_half,
                                  -boundary_mm, glass_front_mm)
    return Mount(label="detector_bbox", csg=slab,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label="detector",
                 z_range_mm=(-w_half, w_half),
                 slab_polygon_xy=slab_poly,
                 foot_polygon_xy=wall_poly)


def build_slit_bbox(
    element: ElementPlacement, slit_mount: dict, albedo: float,
) -> Mount:
    """Collision bounding box for a HASMA slit adapter.

    Not a physical mount — the HASMA threads into the housing wall.
    Dimensions derived from the four fundamental HASMA dims in the BOM.
    """
    hex_half = float(slit_mount["hex_half_mm"])
    length = float(slit_mount["length_mm"])

    v_half = hex_half
    w_half = hex_half
    front = 0.0
    behind = length

    slab = Box(
        lower=Point3D(-v_half * _MM_TO_M, -w_half * _MM_TO_M,
                       -behind * _MM_TO_M),
        upper=Point3D(+v_half * _MM_TO_M, +w_half * _MM_TO_M,
                       front * _MM_TO_M),
    )
    slab.material = Lambert(ConstantSF(albedo))

    slab.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    n = element.axis
    cx, cy = element.position[0], element.position[1]
    in_plane = _perpendicular_in_dispersion_plane(n)
    u_lo, u_hi = -behind, front
    corners = [
        (cx + sv * v_half * in_plane[0] + u * n[0],
         cy + sv * v_half * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half, u_lo, u_hi)
    wall_v_half = float(slit_mount["hex_half_mm"]) + _HORIZONTAL_CLEARANCE_MM
    boundary = float(slit_mount["boundary_mm"])
    wall_poly = _oriented_rect_xy(element, wall_v_half, -boundary, front)
    return Mount(label="slit_bbox", csg=slab,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label="entrance_slit",
                 z_range_mm=(-w_half, w_half),
                 slab_polygon_xy=slab_poly,
                 foot_polygon_xy=wall_poly)


def build_slit_wall_mount(
    element: ElementPlacement, slit_mount: dict, albedo: float,
) -> Mount:
    """Virtual housing wall section around the HASMA bore (raysect CSG).

    Represents the minimum wall slab that the HASMA adapter threads
    through.  Shape is ``Box - Cylinder``: a rectangular wall with a
    bore hole drilled through.  When parented into the raysect world,
    rays that hit the wall material are absorbed (Lambert), and the
    forward trace naturally scores vignetting through reduced hit counts.

    The slab's front face sits at the slit plane (element.position) and
    extends ``boundary_mm`` behind it (opposite to element.axis, i.e.
    away from M1 / toward the exterior).
    """
    wall_half = float(slit_mount["hex_half_mm"]) + _HORIZONTAL_CLEARANCE_MM
    boundary = float(slit_mount["boundary_mm"])
    bore_r = float(slit_mount["bore_radius_mm"])

    v_half = wall_half
    w_half = wall_half
    eps = 0.001  # CSG overcut to avoid coincident-face artifacts

    # Local frame: +z = element.axis (toward M1).
    # Box spans from z = -boundary (rear / exterior face) to z = 0 (slit plane).
    box = Box(
        lower=Point3D(-v_half * _MM_TO_M, -w_half * _MM_TO_M,
                       -boundary * _MM_TO_M),
        upper=Point3D(+v_half * _MM_TO_M, +w_half * _MM_TO_M,
                       0.0),
    )

    # Cylinder drilled through the slab along the bore axis (local z).
    bore = Cylinder(
        radius=bore_r * _MM_TO_M,
        height=(boundary + 2 * eps) * _MM_TO_M,
        transform=translate(0.0, 0.0, -(boundary + eps) * _MM_TO_M),
    )

    wall_csg = Subtract(box, bore)
    wall_csg.material = Lambert(ConstantSF(albedo))

    wall_csg.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    # 2D bounding box in the XY plane (same pattern as build_slit_bbox).
    n = element.axis
    cx, cy = element.position[0], element.position[1]
    in_plane = _perpendicular_in_dispersion_plane(n)
    u_lo, u_hi = -boundary, 0.0
    corners = [
        (cx + sv * v_half * in_plane[0] + u * n[0],
         cy + sv * v_half * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half, u_lo, u_hi)
    return Mount(label="slit_wall", csg=wall_csg,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label="entrance_slit",
                 z_range_mm=(-w_half, w_half),
                 slab_polygon_xy=slab_poly)


def build_detector_wall_mount(
    element: ElementPlacement, albedo: float,
) -> Mount:
    """Virtual housing wall section around the detector package (raysect CSG).

    Shape is ``Box - Box``: a rectangular wall slab (board footprint ×
    boundary depth) with a rectangular cutout for the sensor package.
    The slab front face sits at the die plane (element.position) and
    extends ``boundary_behind_die_mm`` behind it.
    """
    board_w = float(element.params["board_width_mm"])
    board_h = float(element.params["board_height_mm"])
    boundary = float(element.params["boundary_behind_die_mm"])
    pkg_w = float(element.params["package_width_mm"])
    pkg_h = float(element.params["package_height_mm"])

    v_half = 0.5 * board_w
    w_half = 0.5 * board_h
    eps = 0.001

    # Local frame: +z = element.axis (toward beam / M2).
    # Box spans from z = -boundary (PCB bottom) to z = 0 (die plane).
    wall = Box(
        lower=Point3D(-v_half * _MM_TO_M, -w_half * _MM_TO_M,
                       -boundary * _MM_TO_M),
        upper=Point3D(+v_half * _MM_TO_M, +w_half * _MM_TO_M,
                       0.0),
    )

    cutout = Box(
        lower=Point3D(-0.5 * pkg_w * _MM_TO_M, -0.5 * pkg_h * _MM_TO_M,
                       -(boundary + eps) * _MM_TO_M),
        upper=Point3D(+0.5 * pkg_w * _MM_TO_M, +0.5 * pkg_h * _MM_TO_M,
                       eps * _MM_TO_M),
    )

    wall_csg = Subtract(wall, cutout)
    wall_csg.material = Lambert(ConstantSF(albedo))

    wall_csg.transform = (
        translate(*(p * _MM_TO_M for p in element.position))
        * rotate_basis(forward=_V3(element.axis), up=_V3((0.0, 0.0, 1.0)))
    )

    n = element.axis
    cx, cy = element.position[0], element.position[1]
    in_plane = _perpendicular_in_dispersion_plane(n)
    u_lo, u_hi = -boundary, 0.0
    corners = [
        (cx + sv * v_half * in_plane[0] + u * n[0],
         cy + sv * v_half * in_plane[1] + u * n[1])
        for sv in (-1, +1) for u in (u_lo, u_hi)
    ]
    bbox_xy_mm = (min(c[0] for c in corners), max(c[0] for c in corners),
                  min(c[1] for c in corners), max(c[1] for c in corners))

    slab_poly = _oriented_rect_xy(element, v_half, u_lo, u_hi)
    return Mount(label="detector_wall", csg=wall_csg,
                 bbox_xy_mm=bbox_xy_mm,
                 parent_label="detector",
                 z_range_mm=(-w_half, w_half),
                 slab_polygon_xy=slab_poly)


# ── Registry ────────────────────────────────────────────────────────────────

MOUNT_BUILDERS: dict[str, Callable[..., Mount]] = {
    "mirror_flexure":  build_mirror_flexure_mount,
    "oap_flexure":     build_oap_flexure_mount,
    "grating_flexure": build_grating_flexure_mount,
}


_PARAMS_CLASS: dict[str, type] = {
    "mirror_flexure":  RoundMirrorFlexureMountParams,
    "oap_flexure":     OAPMirrorFlexureMountParams,
    "grating_flexure": GratingFlexureMountParams,
}


def parse_mount_params(mount_type: str, mount_dict: dict):
    """Build the right params dataclass from a BOM mount sub-table dict."""
    cls = _PARAMS_CLASS.get(mount_type)
    if cls is None:
        raise ValueError(
            f"Unknown mount_type {mount_type!r}; "
            f"known types: {sorted(_PARAMS_CLASS)}"
        )
    # Drop the 'type' key — it's the dispatch tag, not a param field.
    fields = {k: v for k, v in mount_dict.items() if k != "type"}
    return cls(**fields)


def dispatch_mount(
    element: ElementPlacement,
    mount_type: str,
    params,
    albedo: float,
) -> Mount:
    """Build a Mount for an element using the BOM-selected mount_type.

    `albedo` is applied as the Lambert reflectance for the mount's
    Absorbing-style material — same value the housing walls use, since
    mounts and walls are the same printed-PLA material.
    """
    builder = MOUNT_BUILDERS.get(mount_type)
    if builder is None:
        raise ValueError(
            f"Unknown mount_type {mount_type!r} for element {element.label!r}; "
            f"known types: {sorted(MOUNT_BUILDERS)}"
        )
    return builder(element, params, albedo)
