"""build123d translator for the solid unibody housing.

Converts a ``SolidHousingSpec`` (from ``housing.py``) into
build123d ``Part``/``Compound`` objects for STEP export.

All positions are in mm (scene frame). z = out-of-plane.
"""


import math

from build123d import (
    Align,
    Axis,
    BuildPart,
    BuildSketch,
    Circle,
    Color,
    Compound,
    Cone,
    CounterBoreHole,
    FontStyle,
    GeomType,
    Location,
    Locations,
    Mode,
    Kind,
    Plane,
    Polygon,
    Rectangle,
    RectangleRounded,
    Text,
    Vector,
    add,
    export_step,
    extrude,
    fillet,
    offset,
)
from build123d.topology.three_d import Solid


# ─── Manufacturing constants ──────────────────────────────────────────────

_CAVITY_POCKET_CLEARANCE_MM = 1.0
_CAVITY_POCKET_FILLET_RADIUS_MM = 1.0     # 2mm dia, op1 (cavity + mounts)
_EXTERIOR_FILLET_RADIUS_MM = 10.0
_PCBA_POCKET_CLEARANCE_MM = 2.0
_PCBA_POCKET_FILLET_RADIUS_MM = 2.0 # 4mm dia, ops 2/3 (board pockets)
_TOP_COVER_DEPTH_MM = 3.0
_TOP_COVER_INSET_MM = 2.0
_TOP_COVER_LEDGE_MM = 6.0
_EMBOSS_DEPTH_MM = 0.5
_COVER_SCREW_DIA_MM = 2.4
_COVER_SCREW_HEAD_DIA_MM = 4.0
_COVER_SCREW_CBORE_DEPTH_MM = 2.0
_COVER_SCREW_PILOT_DIA_MM = 1.6
_COVER_SCREW_PILOT_DEPTH_MM = 5.0
_COVER_SCREW_MIN_SEGMENT_MM = 15.0
_PCBA_COVER_DEPTH_MM = 3.0
_PCBA_COVER_LEDGE_MM = 6.0
# Bottom-side cavities (exterior → interior):
#   1. Ledge pocket — perimeter shelf the bottom cover plate sits in.
#   2. Inner clear pocket — relief between cover plate and mount cavity
#      floors; mount screws pass through here.
#   3. Controller board pocket — rectangular pocket for PCB + components.
#   4. Solder clearance zone — insert-spacing rect minus corner boss
#      cylinders, directly above the board pocket.
#   5. Corner bosses — solid cylinders inside the solder zone for
#      heat-set inserts.
#   6. Cable channel — connects the controller pocket to the detector
#      PCBA pocket.
#   7. USB channel — routes from the controller USB edge through the
#      housing wall.
_BOTTOM_COVER_INSET_MM = 2.0
_BOTTOM_COVER_DEPTH_MM = 3.0           # cover plate thickness
_BOTTOM_COVER_CBORE_DEPTH_MM = 2.0
_BOTTOM_COVER_LEDGE_WIDTH_MM = 6.0     # ledge pocket shelf width
_BOTTOM_COVER_EXTENSION_MM = 13.0      # vertical depth below z_bottom
_BOTTOM_COVER_MIN_WALL_MM = 5.0        # floor between inner clear pocket and deepest cavity
_BOTTOM_COVER_WALL_MM = 3.0            # USB channel wall thickness
_CONTROLLER_Z_INSET_MM = 3.0           # board placement offset toward z_bottom_ext
_HOUSING_COLOR = Color(0.45, 0.45, 0.45)


# ─── Printability warnings ─────────────────────────────────────────────────

def printability_warnings(
    housing_spec,
    *,
    min_wall_mm: float = 1.2,
) -> list[str]:
    """Return a list of printability concerns (cheap checks only)."""
    warnings: list[str] = []

    if housing_spec is None:
        return warnings

    if housing_spec.wall_thickness_mm < min_wall_mm:
        warnings.append(
            f"housing wall_thickness_mm={housing_spec.wall_thickness_mm:.2f} "
            f"< min_wall_mm={min_wall_mm:.2f} (FDM min feature)"
        )

    return warnings


# ─── Housing CAD translator ──────────────────────────────────────────────


def _cut_through_next(face, target, direction):
    """Extrude *face* through the first contiguous solid of *target*.

    Builds a long extrusion along *direction*, intersects it with
    *target*, and returns only the nearest solid piece.  Subtracting
    this from *target* gives "through-to-next" cut semantics: the cut
    enters the body and stops at the first interior void.
    """
    direction = Vector(direction)
    max_dim = target.bounding_box().diagonal * 2
    extrusion = Solid.extrude(face, direction * max_dim)
    intersection = extrusion.intersect(target)
    if intersection is None:
        return None
    solids = intersection.solids().sort_by(Axis(face.center(), direction))
    return solids[0] if solids else None


def _fillet_convex(sk, r: float):
    """Fillet only the convex vertices of a sketch."""
    for v in sk.vertices():
        edges = [e for e in sk.edges() if v in e.vertices()]
        if len(edges) < 2:
            continue
        d_out = d_in = None
        for edge in edges:
            p0 = edge.position_at(0)
            at_start = (abs(p0.X - v.X) < 0.01
                        and abs(p0.Y - v.Y) < 0.01)
            t = edge.tangent_at(0 if at_start else 1)
            if at_start:
                d_out = (t.X, t.Y)
            else:
                d_in = (-t.X, -t.Y)
        if d_out is None or d_in is None:
            continue
        cross = d_out[0] * d_in[1] - d_out[1] * d_in[0]
        if cross > 0:
            try:
                fillet(v, radius=r)
            except Exception:
                pass


def _fillet_all(sk, r: float):
    """Fillet all vertices of a sketch by the endmill radius."""
    for v in sk.vertices():
        try:
            fillet(v, radius=r)
        except Exception:
            pass


def _build_clipped_cone(flare_plane, near_r, flare_rad,
                        bore_pos, bore_inward,
                        z_floor, z_height, extent):
    """Build the HASMA cone clipped to a z-bounded cuboid.

    Built outside any parent BuildPart so the intermediate solids
    don't union into the housing body.
    """
    with BuildPart() as _cone_bp:
        with BuildSketch(flare_plane):
            Circle(radius=near_r)
        extrude(amount=80.0, taper=-math.degrees(flare_rad))
    clip_plane = Plane(
        origin=Vector(bore_pos.X, bore_pos.Y, z_floor),
        x_dir=bore_inward,
        z_dir=Vector(0, 0, 1))
    with BuildPart() as _clip_bp:
        with BuildSketch(clip_plane):
            Rectangle(extent * 2, extent * 2)
        extrude(amount=z_height)
    return _cone_bp.part.intersect(_clip_bp.part)


def build_solid_housing_cad(
    spec,
    optics_scene,
    parts,
    *,
    design_wavelength_nm: float = 550.0,
    min_bw_nm: float = 300.0,
) -> Compound:
    """Build the unibody housing as a build123d Compound.

    Uses offset-from-surface and cut-through-all operations within a
    ``BuildPart`` context instead of standalone-solid boolean
    subtractions.  Sketches reference body surfaces directly;
    through-all cuts use generous ``both=True`` extrusions that the
    boolean engine clips to the body.

    Parameters
    ----------
    spec : SolidHousingSpec
        Pre-computed geometry from ``build_solid_housing_spec``.
    optics_scene
        The optical scene (for element positions/axes and
        ``place_all_in_scene_frame``).
    parts
        Loaded parts object (m1_part, m2_part, grating_part paths).

    Returns
    -------
    Compound
        Children = vendor optic placements + housing body solid.
    """
    from optics.mounts_cad import place_all_in_scene_frame

    z_bottom_ext = spec.z_bottom - _BOTTOM_COVER_EXTENSION_MM
    z_top = spec.z_top
    height = z_top - z_bottom_ext
    r_cavity = _CAVITY_POCKET_FILLET_RADIUS_MM
    r_pcba = _PCBA_POCKET_FILLET_RADIUS_MM

    # ── Interior cavity sketch (z = 0 plane) ─────────────────────────
    with BuildSketch(Plane.XY) as body_sk:
        for poly in spec.optic_polys:
            Polygon(*[(x, y) for (x, y) in poly], align=None)
        for poly in spec.mount_hull_polys:
            Polygon(*[(x, y) for (x, y) in poly], align=None)
        offset(body_sk.faces(), amount=_CAVITY_POCKET_CLEARANCE_MM,
               kind=Kind.INTERSECTION)

    # Union mounts + beam cones, then morphological close (dilate/erode
    # by endmill radius) to fill narrow notches between non-adjacent
    # mounts that would otherwise become pillars.
    with BuildSketch(Plane.XY) as _union_sk:
        add(body_sk.sketch)
        for poly in spec.beam_polys:
            Polygon(*[(x, y) for (x, y) in poly], align=None)
        offset(_union_sk.faces(), amount=r_cavity, kind=Kind.INTERSECTION)
        offset(_union_sk.faces(), amount=-r_cavity, kind=Kind.INTERSECTION)

    with BuildSketch(Plane.XY) as cavity_sk:
        add(_union_sk.sketch)
        for poly in spec.wall_polys:
            Polygon(*[(x, y) for (x, y) in poly], align=None,
                    mode=Mode.SUBTRACT)
        _fillet_convex(cavity_sk, r_cavity)

    # Top cover outline: housing rectangle inset, detector-axis aligned.
    _su, _sv = spec.sensor_u_dir, spec.sensor_v_dir
    _ho_u = [x * _su[0] + y * _su[1] for x, y in spec.housing_outline]
    _ho_v = [x * _sv[0] + y * _sv[1] for x, y in spec.housing_outline]
    _tc_u = (max(_ho_u) - min(_ho_u)) - 2 * _TOP_COVER_INSET_MM
    _tc_v = (max(_ho_v) - min(_ho_v)) - 2 * _TOP_COVER_INSET_MM
    _tc_cu = 0.5 * (min(_ho_u) + max(_ho_u))
    _tc_cv = 0.5 * (min(_ho_v) + max(_ho_v))
    _tc_cx = _tc_cu * _su[0] + _tc_cv * _sv[0]
    _tc_cy = _tc_cu * _su[1] + _tc_cv * _sv[1]
    _tc_plane = Plane(
        origin=Vector(_tc_cx, _tc_cy, 0),
        x_dir=Vector(_su[0], _su[1], 0),
        z_dir=Vector(0, 0, 1))
    with BuildSketch(_tc_plane) as _tc_outline_sk:
        RectangleRounded(_tc_u, _tc_v,
                         _EXTERIOR_FILLET_RADIUS_MM - _TOP_COVER_INSET_MM)

    # Detector cover outline: PCBA pocket footprint plus ledge.
    # All geometry uses the detector UV frame (u = sensor_u_dir,
    # v = sensor_v_dir = inward; outward = -v).
    _dco_da = spec.detector_plane_z_dir
    _dco_do = spec.detector_plane_origin
    _dco_dx = spec.detector_plane_x_dir
    _dco_outward = (-_dco_da[0], -_dco_da[1])
    _pcba_w = spec.detector_pcba_width + 2 * _PCBA_POCKET_CLEARANCE_MM
    _pcba_h = spec.detector_pcba_height + 2 * _PCBA_POCKET_CLEARANCE_MM
    _cover_w = _pcba_w + 2 * _PCBA_COVER_LEDGE_MM
    _cover_h = _pcba_h + 2 * _PCBA_COVER_LEDGE_MM
    _wall_depth_outward = max(
        (x - _dco_do[0]) * _dco_outward[0]
        + (y - _dco_do[1]) * _dco_outward[1]
        for x, y in spec.housing_outline
    )
    # Cover plane centered at det_origin, shifted to ledge floor depth.
    _ledge_offset = _wall_depth_outward - _PCBA_COVER_DEPTH_MM
    _det_cover_plane = Plane(
        origin=Vector(
            _dco_do[0] + _dco_outward[0] * _ledge_offset,
            _dco_do[1] + _dco_outward[1] * _ledge_offset,
            0),
        x_dir=Vector(_dco_dx[0], _dco_dx[1], 0),
        z_dir=Vector(_dco_outward[0], _dco_outward[1], 0))
    with BuildSketch(_det_cover_plane) as _cover_outline_sk:
        RectangleRounded(_cover_w, _cover_h, _PCBA_POCKET_FILLET_RADIUS_MM)

    # Bottom cover outline: housing rectangle inset, detector-axis aligned.
    _bc_u = (max(_ho_u) - min(_ho_u)) - 2 * _BOTTOM_COVER_INSET_MM
    _bc_v = (max(_ho_v) - min(_ho_v)) - 2 * _BOTTOM_COVER_INSET_MM
    _bc_plane = Plane(
        origin=Vector(_tc_cx, _tc_cy, 0),
        x_dir=Vector(_su[0], _su[1], 0),
        z_dir=Vector(0, 0, 1))
    with BuildSketch(_bc_plane) as _bc_outline_sk:
        RectangleRounded(_bc_u, _bc_v,
                         _EXTERIOR_FILLET_RADIUS_MM - _BOTTOM_COVER_INSET_MM)

    # Inner clear pocket: bottom cover outline inset by ledge.
    _bc_inner_u = _bc_u - 2 * _BOTTOM_COVER_LEDGE_WIDTH_MM
    _bc_inner_v = _bc_v - 2 * _BOTTOM_COVER_LEDGE_WIDTH_MM
    _bc_inner_r = _EXTERIOR_FILLET_RADIUS_MM - _BOTTOM_COVER_INSET_MM - _BOTTOM_COVER_LEDGE_WIDTH_MM
    with BuildSketch(_bc_plane) as _bc_inner_sk:
        RectangleRounded(_bc_inner_u, _bc_inner_v, max(_bc_inner_r, 1.0))

    # ── Housing body via BuildPart ────────────────────────────────────
    _usb_box_info = None
    _usb_slot_info = None

    with BuildPart() as bp:

        # Outer block: sketch on bottom surface, extrude to full height.
        with BuildSketch(Plane.XY.offset(z_bottom_ext)) as _outline_sk:
            Polygon(*[(x, y) for (x, y) in spec.housing_outline],
                    align=None)
            _fillet_all(_outline_sk, _EXTERIOR_FILLET_RADIUS_MM)
        extrude(amount=height)

        # Interior cavity — split into ceiling + stepped floor.
        # Precompute deeper step pocket sketches so we can union them
        # into the base floor (prevents undercuts where step pockets
        # extend beyond the cavity sketch).
        hull_by_label = {lbl: (fz, hull)
                         for lbl, fz, hull in spec.mount_floor_hulls}
        _MIN_CLEARANCE_MM = 1.0

        def _snap(poly, precision=0.01):
            inv = 1.0 / precision
            return [(round(x * inv) / inv, round(y * inv) / inv)
                    for x, y in poly]

        base_z = (spec.shallowest_floor_z if spec.floor_steps
                  else z_bottom_ext + _BOTTOM_COVER_EXTENSION_MM)

        # Compute compensated-offset pocket sketch for EVERY mount
        # (including shallowest) so all corners get fillet compensation.
        # Only deeper ones get extruded as step pockets.
        mount_pocket_sketches = []
        for lbl, (fz, hull) in hull_by_label.items():
            with BuildSketch(Plane.XY) as _mp_sk:
                Polygon(*_snap(hull), align=None)
                offset(_mp_sk.faces(), amount=_CAVITY_POCKET_CLEARANCE_MM,
                       kind=Kind.INTERSECTION)
                for other_lbl, (other_fz, other_hull) in hull_by_label.items():
                    if other_lbl == lbl:
                        continue
                    if other_fz <= fz:
                        continue
                    with BuildSketch(Plane.XY) as _ko_sk:
                        Polygon(*_snap(other_hull), align=None)
                    add(_ko_sk.sketch, mode=Mode.SUBTRACT)
                _fillet_convex(_mp_sk, r_cavity)
            depth = abs(fz - base_z)
            mount_pocket_sketches.append((_mp_sk.sketch, fz, depth, lbl))

        step_sketches = [(sk, fz, d) for sk, fz, d, _ in mount_pocket_sketches
                         if d > 0.01]

        # Base floor sketch = cavity ∪ all mount pocket sketches.
        with BuildSketch(Plane.XY) as _floor_sk:
            add(cavity_sk.sketch)
            for sk, _, _, _ in mount_pocket_sketches:
                add(sk)

        # Ceiling: z=0 upward (clipped by outer block at z_top).
        with BuildSketch(Plane.XY):
            add(_floor_sk.sketch)
        extrude(amount=height, mode=Mode.SUBTRACT)

        # Base floor: shallowest mount foot level up to z=0.
        with BuildSketch(Plane.XY.offset(base_z)):
            add(_floor_sk.sketch)
        extrude(amount=abs(base_z), mode=Mode.SUBTRACT)

        # Deeper step pockets.
        for sk, floor_z, depth in step_sketches:
            with BuildSketch(Plane.XY.offset(floor_z)):
                add(sk)
            extrude(amount=depth, mode=Mode.SUBTRACT)

        # Top cover ledge: widen cavity opening at top for recessed cover.
        with BuildSketch(Plane.XY.offset(z_top - _TOP_COVER_DEPTH_MM)):
            add(_tc_outline_sk.sketch)
        extrude(amount=_TOP_COVER_DEPTH_MM, mode=Mode.SUBTRACT)

        # HASMA bore: undersize pilot hole (5.5mm dia), then thread.
        bore_inward = Vector(spec.hasma_bore_axis[0],
                             spec.hasma_bore_axis[1], 0)
        bore_pos = Vector(spec.hasma_bore_position[0],
                          spec.hasma_bore_position[1], 0)
        bore_outside = bore_pos - bore_inward * 50
        bore_plane = Plane(origin=bore_outside, z_dir=bore_inward)

        # Step 1: Drill tap drill bore (diameter from BOM).
        _pilot_r = 0.5 * spec.hasma_tap_drill_dia_mm
        with BuildSketch(bore_plane) as bore_sk:
            Circle(radius=_pilot_r)
        bore_tool = _cut_through_next(
            bore_sk.sketch.faces()[0], bp.part, bore_inward)
        if bore_tool is not None:
            add(bore_tool, mode=Mode.SUBTRACT)



        # ── Detector features ────────────────────────────────────────
        # Reference planes: det_plane at the wall boundary (det_bnd_pos)
        # with z_dir pointing into the cavity; outward is -z_dir.
        da = spec.detector_plane_z_dir
        inward = Vector(da[0], da[1], 0)
        outward = Vector(-da[0], -da[1], 0)
        det_origin = Vector(spec.detector_plane_origin[0],
                            spec.detector_plane_origin[1], 0)
        det_x = Vector(spec.detector_plane_x_dir[0],
                       spec.detector_plane_x_dir[1], 0)
        det_plane = Plane(origin=det_origin, x_dir=det_x, z_dir=inward)

        # 1. PCBA outside pocket: from wall boundary outward through
        #    all exterior material (_cut_through_next).
        pcba_plane = Plane(origin=det_origin, x_dir=det_x, z_dir=outward)
        with BuildSketch(pcba_plane) as pcba_sk:
            Rectangle(spec.detector_pcba_width + 2 * _PCBA_POCKET_CLEARANCE_MM,
                    spec.detector_pcba_height + 2 * _PCBA_POCKET_CLEARANCE_MM)
            _fillet_all(pcba_sk, r_pcba)
        pcba_tool = _cut_through_next(
            pcba_sk.sketch.faces()[0], bp.part, outward)
        if pcba_tool is not None:
            bp._add_to_context(pcba_tool, mode=Mode.SUBTRACT)

        # 1b. Detector cover ledge: wider recess on exterior face.
        #     Draw fresh on det_cover_plane (same dims as pre-computed
        #     outline) to avoid add()-across-planes positioning issues.
        with BuildSketch(_det_cover_plane):
            RectangleRounded(_cover_w, _cover_h,
                             _PCBA_POCKET_FILLET_RADIUS_MM)
        extrude(amount=_PCBA_COVER_DEPTH_MM, mode=Mode.SUBTRACT)

        # 2. Heat-set insert holes: blind 5 mm from pocket floor into
        #    the wall (inward from det_bnd_pos toward cavity).
        for hx, hy, hz in spec.insert_hole_positions:
            insert_plane = Plane(
                origin=Vector(hx, hy, hz), x_dir=det_x,
                z_dir=inward,
            )
            with BuildSketch(insert_plane):
                Circle(radius=spec.insert_hole_radius_mm)
            extrude(amount=spec.insert_hole_depth_mm, mode=Mode.SUBTRACT)

        # 3. Solder-joint clearance: shallow blind pocket from wall
        #    boundary into cavity (same direction as inserts, wider).
        with BuildSketch(det_plane) as slot_sk:
            Rectangle(spec.detector_slot_width, spec.detector_slot_height)
            half_sw = 0.5 * spec.detector_slot_width
            half_sh = 0.5 * spec.detector_slot_height
            for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                with Locations([(sx * half_sw, sy * half_sh)]):
                    Circle(radius=spec.corner_boss_radius_mm, mode=Mode.SUBTRACT)
            _fillet_all(slot_sk, r_pcba)
        extrude(amount=spec.detector_slot_depth, mode=Mode.SUBTRACT)

        # 4. Detector package clearance: blind cut from solder pocket
        #    floor inward through remaining wall + glass.
        solder_floor = det_origin + inward * spec.detector_slot_depth
        pkg_plane = Plane(origin=solder_floor, x_dir=det_x, z_dir=inward)
        pkg_depth = spec.detector_pkg_slot_depth - spec.detector_slot_depth
        with BuildSketch(pkg_plane):
            RectangleRounded(spec.detector_package_width + 2 * _PCBA_POCKET_CLEARANCE_MM,
                             spec.detector_package_height + 2 * _PCBA_POCKET_CLEARANCE_MM,
                             r_pcba)
        extrude(amount=pkg_depth, mode=Mode.SUBTRACT)

        # ── Mount fastener holes (from housing bottom) ────────────────
        if spec.mount_fasteners:
            bottom_z = z_bottom_ext
            for mf in spec.mount_fasteners:
                hx, hy = mf.xy
                through_depth = mf.floor_z - bottom_z
                if through_depth < 0.5:
                    continue
                if mf.kind == "bolt":
                    cb_plane = Plane(
                        origin=Vector(hx, hy, bottom_z),
                        z_dir=Vector(0, 0, 1))
                    # Clearance hole through to mount foot
                    with BuildSketch(cb_plane):
                        Circle(radius=0.5 * spec.bolt_clearance_dia_mm)
                    extrude(amount=through_depth, mode=Mode.SUBTRACT)
                    # Tapered counterbore: spec'd at foot, widens
                    # 5° toward bottom for assembly access.
                    # Leave 2mm bearing surface under foot.
                    cb_depth = through_depth - 2.0
                    if cb_depth > 0.5:
                        foot_plane = Plane(
                            origin=Vector(hx, hy, bottom_z + cb_depth),
                            z_dir=Vector(0, 0, -1))
                        with BuildSketch(foot_plane):
                            Circle(radius=0.5 * spec.bolt_counterbore_dia_mm)
                        extrude(amount=cb_depth, taper=-5,
                                mode=Mode.SUBTRACT)
                else:
                    # Pusher hex key access: spec'd at foot,
                    # widens 10° toward bottom.
                    foot_plane = Plane(
                        origin=Vector(hx, hy, bottom_z + through_depth),
                        z_dir=Vector(0, 0, -1))
                    with BuildSketch(foot_plane):
                        Circle(radius=0.5 * spec.pusher_access_dia_mm)
                    extrude(amount=through_depth, taper=-5,
                            mode=Mode.SUBTRACT)

        # ── Controller pocket (from housing bottom) ────────────────────
        if spec.controller_pocket_origin is not None:
            cx, cy = spec.controller_pocket_origin
            ctrl_x = Vector(spec.controller_pocket_x_dir[0],
                            spec.controller_pocket_x_dir[1], 0)
            ctrl_up = Vector(0, 0, 1)

            # 1. Board pocket: full board envelope, components + PCB depth.
            ctrl_pocket_ext = spec.controller_pocket_depth + _BOTTOM_COVER_EXTENSION_MM
            pocket_plane = Plane(
                origin=Vector(cx, cy, z_bottom_ext),
                x_dir=ctrl_x, z_dir=ctrl_up)
            with BuildSketch(pocket_plane) as _ctrl_pcba_sk:
                Rectangle(spec.controller_board_width + 2 * _PCBA_POCKET_CLEARANCE_MM,
                          spec.controller_board_height + 2 * _PCBA_POCKET_CLEARANCE_MM)
                _fillet_all(_ctrl_pcba_sk, r_pcba)
            extrude(amount=ctrl_pocket_ext, mode=Mode.SUBTRACT)

            # 2. Solder clearance: insert spacing rect minus boss circles,
            #    above the board pocket.
            solder_z = z_bottom_ext + ctrl_pocket_ext
            solder_plane = Plane(
                origin=Vector(cx, cy, solder_z),
                x_dir=ctrl_x, z_dir=ctrl_up)
            half_isw = 0.5 * spec.controller_insert_spacing_width
            half_ish = 0.5 * spec.controller_insert_spacing_height
            with BuildSketch(solder_plane) as _ctrl_slot_sk:
                Rectangle(spec.controller_insert_spacing_width,
                          spec.controller_insert_spacing_height)
                for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    with Locations([(sx * half_isw, sy * half_ish)]):
                        Circle(radius=spec.controller_corner_boss_radius,
                               mode=Mode.SUBTRACT)
                _fillet_all(_ctrl_slot_sk, r_pcba)
            extrude(amount=spec.controller_solder_clearance_depth,
                    mode=Mode.SUBTRACT)

            # 3. Insert holes: from solder clearance ceiling downward
            #    into housing material.
            insert_z = solder_z + spec.controller_solder_clearance_depth
            ctrl_down = Vector(0, 0, -1)
            ctrl_y = Vector(-ctrl_x.Y, ctrl_x.X, 0)
            for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ix = cx + sx * half_isw * ctrl_x.X + sy * half_ish * ctrl_y.X
                iy = cy + sx * half_isw * ctrl_x.Y + sy * half_ish * ctrl_y.Y
                ins_plane = Plane(
                    origin=Vector(ix, iy, insert_z),
                    x_dir=ctrl_x, z_dir=ctrl_down)
                with BuildSketch(ins_plane):
                    Circle(radius=spec.insert_hole_radius_mm)
                extrude(amount=spec.insert_hole_depth_mm,
                        mode=Mode.SUBTRACT)

        # ── Controller cable channel ──────────────────────────────────
        # From controller pocket center toward detector wall, stops at
        # wall_thickness from the housing outline edge.
        if spec.controller_pocket_origin is not None:
            _ctrl_cable_w = 35.0
            da = spec.detector_plane_z_dir
            neg_da = (-da[0], -da[1])
            housing_edge_y = max(
                (vx - cx) * neg_da[0] + (vy - cy) * neg_da[1]
                for vx, vy in spec.housing_outline
            )
            _ch_setback = _BOTTOM_COVER_LEDGE_WIDTH_MM + _BOTTOM_COVER_INSET_MM
            ch_y_near = 0.0
            ch_y_far = housing_edge_y - _ch_setback
            ch_len = ch_y_far - ch_y_near
            if ch_len > 0:
                ch_cy = 0.5 * (ch_y_near + ch_y_far)
                with BuildSketch(pocket_plane) as _cable_sk:
                    with Locations([(0, ch_cy)]):
                        Rectangle(_ctrl_cable_w, ch_len)
                    _fillet_all(_cable_sk, r_pcba)
                extrude(amount=ctrl_pocket_ext, mode=Mode.SUBTRACT)

                # Plunge cut: from detector boundary to channel far
                # end (ledge+inset surface), full depth up to the
                # PCBA pocket floor.
                det_ox, det_oy = spec.detector_plane_origin
                det_bnd_y = ((det_ox - cx) * neg_da[0]
                             + (det_oy - cy) * neg_da[1])
                plunge_h = ch_y_far - det_bnd_y
                pcba_pocket_bottom_z = -(
                    0.5 * spec.detector_pcba_height
                    + _PCBA_POCKET_CLEARANCE_MM)
                plunge_depth = pcba_pocket_bottom_z - z_bottom_ext
                if plunge_h > 0.5 and plunge_depth > 0:
                    plunge_cy = ch_y_far - 0.5 * plunge_h
                    with BuildSketch(pocket_plane) as _plunge_sk:
                        with Locations([(0, plunge_cy)]):
                            Rectangle(_ctrl_cable_w, plunge_h)
                        _fillet_all(_plunge_sk, r_pcba)
                    extrude(amount=plunge_depth,
                            mode=Mode.SUBTRACT)

        # ── USB cable channel ─────────────────────────────────────────
        # From controller board -X edge (USB Micro-B on Teensy)
        # through the housing wall in the -X direction.
        if spec.controller_pocket_origin is not None:
            _usb_cable_w = 25.0
            _usb_z_offset = 11.165
            _usb_cable_h = _usb_z_offset + 5.0 + _BOTTOM_COVER_WALL_MM + 3.0
            _usb_x_offset = -29.6
            _usb_y_offset = -3.287
            ctrl_x_dir = spec.controller_pocket_x_dir
            housing_edge_neg_x = min(
                (vx - cx) * ctrl_x_dir[0] + (vy - cy) * ctrl_x_dir[1]
                for vx, vy in spec.housing_outline
            )
            usb_x_near = _usb_x_offset
            usb_x_far = housing_edge_neg_x
            usb_len = usb_x_near - usb_x_far
            if usb_len > 0:
                usb_cx = 0.5 * (usb_x_near + usb_x_far)
                with BuildSketch(pocket_plane):
                    with Locations([(usb_cx, _usb_y_offset)]):
                        Rectangle(usb_len, _usb_cable_w)
                extrude(amount=_usb_cable_h,
                        mode=Mode.SUBTRACT)
                _usb_box_info = (pocket_plane, usb_cx, _usb_y_offset,
                                 usb_len, _usb_cable_w, _usb_cable_h)

            # Save slot params for bottom cover cut.
            ctrl_y = Vector(-ctrl_x.Y, ctrl_x.X, 0)
            _usb_slot_info = (ctrl_x, ctrl_y,
                              Vector(cx, cy, 0), _usb_x_offset,
                              _usb_y_offset, _usb_z_offset)

        # ── HASMA conical flare (tapered extrude inside BuildPart) ──
        # Build the full cone, then intersect with a z-bounding box so
        # it stops MIN_WALL before the top cover ledge and the basement.
        flare_rad = spec.hasma_bore_flare_half_angle_rad
        if flare_rad > 0:
            bx, by = spec.hasma_bore_axis
            bore_inward = Vector(bx, by, 0)
            bore_pos = Vector(spec.hasma_bore_position[0],
                              spec.hasma_bore_position[1], 0)
            bnd_pos = bore_pos - bore_inward * spec.hasma_boundary_mm
            near_r = spec.hasma_hex_clearance_radius_mm

            _dfz = min(fz for _, fz, _ in spec.mount_floor_hulls)
            flare_z_floor = _dfz
            flare_z_ceil = z_top - _TOP_COVER_DEPTH_MM - _BOTTOM_COVER_MIN_WALL_MM
            flare_z_height = flare_z_ceil - flare_z_floor

            flare_plane = Plane(
                origin=bnd_pos,
                z_dir=-bore_inward)
            _clipped_cone = _build_clipped_cone(
                flare_plane, near_r, flare_rad,
                bore_pos, bore_inward,
                flare_z_floor, flare_z_height, height)
            add(_clipped_cone, mode=Mode.SUBTRACT)

        # ── Bottom cover pocket ──────────────────────────────────────
        # Ledge pocket (plate sits here).
        with BuildSketch(Plane.XY.offset(z_bottom_ext)):
            add(_bc_outline_sk.sketch)
        extrude(amount=_BOTTOM_COVER_DEPTH_MM, mode=Mode.SUBTRACT)

        # Inner clear pocket (z_basement): from ledge floor up to
        # _BOTTOM_COVER_MIN_WALL_MM below the deepest cavity floor.
        _deepest_floor_z = min(fz for _, fz, _ in spec.mount_floor_hulls)
        z_basement = _deepest_floor_z - _BOTTOM_COVER_MIN_WALL_MM
        _bc_pocket_floor = z_bottom_ext + _BOTTOM_COVER_DEPTH_MM
        _bc_pocket_depth = z_basement - _bc_pocket_floor
        if _bc_pocket_depth > 0.1:
            with BuildSketch(Plane.XY.offset(_bc_pocket_floor)):
                add(_bc_inner_sk.sketch)
            extrude(amount=_bc_pocket_depth, mode=Mode.SUBTRACT)

    body = bp.part
    body.label = "solid_housing"

    # ── Top cover (undersized by 2*print_tol for clearance fit) ──
    tol = spec.print_tolerance_mm
    with BuildSketch(Plane.XY) as _tc_fit_sk:
        add(_tc_outline_sk.sketch)
        try:
            offset(_tc_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.INTERSECTION)
        except ValueError:
            offset(_tc_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.ARC)

    # Find screw hole positions: midpoint of each straight edge > 30mm,
    # shifted inward by half the ledge so they land in the ledge center.
    tc_edges = _tc_outline_sk.sketch.edges()
    screw_positions: list[tuple[float, float]] = []
    _ledge_center_offset = _TOP_COVER_LEDGE_MM / 2
    for edge in tc_edges:
        if edge.length < _COVER_SCREW_MIN_SEGMENT_MM:
            continue
        mid = edge.position_at(0.5)
        tangent = edge.tangent_at(0.5)
        inward = Vector(-tangent.Y, tangent.X, 0)
        centroid = _tc_outline_sk.sketch.center()
        to_center = Vector(centroid.X - mid.X, centroid.Y - mid.Y, 0)
        if inward.dot(to_center) < 0:
            inward = -inward
        if edge.length >= 140.0:
            for t in (0.25, 0.75):
                pt = edge.position_at(t)
                screw_positions.append((
                    pt.X + inward.X * _ledge_center_offset,
                    pt.Y + inward.Y * _ledge_center_offset))
        else:
            screw_positions.append((
                mid.X + inward.X * _ledge_center_offset,
                mid.Y + inward.Y * _ledge_center_offset,
            ))

    with BuildPart() as tc_bp:
        with BuildSketch(Plane.XY.offset(z_top - _TOP_COVER_DEPTH_MM)):
            add(_tc_fit_sk.sketch)
        extrude(amount=_TOP_COVER_DEPTH_MM)

        # Counterbored screw holes through top cover.
        if screw_positions:
            screw_r = 0.5 * _COVER_SCREW_DIA_MM
            head_r = 0.5 * _COVER_SCREW_HEAD_DIA_MM
            for sx, sy in screw_positions:
                with Locations([(sx, sy, z_top)]):
                    CounterBoreHole(
                        radius=screw_r,
                        counter_bore_radius=head_r,
                        counter_bore_depth=_COVER_SCREW_CBORE_DEPTH_MM,
                        depth=_TOP_COVER_DEPTH_MM,
                    )

    top_cover = tc_bp.part
    top_cover.label = "top_cover"
    top_cover.color = _HOUSING_COLOR
    for s in top_cover.solids():
        s.label = "top_cover"

    # ── Chief-ray path emboss on top cover surface ───────────────
    from designs.czerny_assembly import CzernyAssembly as _CzAsm
    _beam_path = _CzAsm().resolve_beam_path(optics_scene)
    _RAY_LINE_WIDTH = 2.0
    _DET_TICK_LEN = 8.0
    if len(_beam_path) >= 2:
        with BuildPart() as _ray_bp:
            for i in range(len(_beam_path) - 1):
                x0, y0, _ = _beam_path[i].position
                x1, y1, _ = _beam_path[i + 1].position
                dx, dy = x1 - x0, y1 - y0
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                seg_plane = Plane(
                    origin=Vector(mx, my, z_top),
                    x_dir=Vector(dx / seg_len, dy / seg_len, 0),
                    z_dir=Vector(0, 0, 1))
                _ray_end_r = 0.5 * _RAY_LINE_WIDTH - 0.01
                with BuildSketch(seg_plane):
                    RectangleRounded(seg_len, _RAY_LINE_WIDTH,
                                    _ray_end_r)
                extrude(amount=_EMBOSS_DEPTH_MM)

            # Detector tick: perpendicular line at detector position.
            det_el = _beam_path[-1]
            if det_el.kind == "detector":
                _det_array_mm = 30.0
                dx, dy = det_el.axis[0], det_el.axis[1]
                perp_x, perp_y = -dy, dx
                det_plane = Plane(
                    origin=Vector(det_el.position[0],
                                  det_el.position[1], z_top),
                    x_dir=Vector(perp_x, perp_y, 0),
                    z_dir=Vector(0, 0, 1))
                with BuildSketch(det_plane):
                    Rectangle(_det_array_mm, _RAY_LINE_WIDTH)
                extrude(amount=_EMBOSS_DEPTH_MM)

            # Entrance slit dot.
            slit_el = _beam_path[0]
            if slit_el.kind == "slit":
                slit_plane = Plane(
                    origin=Vector(slit_el.position[0],
                                  slit_el.position[1], z_top),
                    z_dir=Vector(0, 0, 1))
                with BuildSketch(slit_plane):
                    Circle(radius=3.0)
                extrude(amount=_EMBOSS_DEPTH_MM)

        ray_solid = _ray_bp.part
        ray_solid.label = "top_cover_ray_path"
        ray_solid.color = Color(0.95, 0.95, 0.95)
        for s in ray_solid.solids():
            s.label = "top_cover_ray_path"
    else:
        ray_solid = None

    # ── Text emboss on top cover surface ─────────────────────────
    from importlib.metadata import version as pkg_version
    try:
        _ver = pkg_version("open-spectrograph")
    except Exception:
        _ver = "0.1.0"

    _fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    _na = 0.5 / _fnum
    _lam_min = design_wavelength_nm - min_bw_nm / 2.0
    _lam_max = design_wavelength_nm + min_bw_nm / 2.0

    _ux, _uy = _su
    _vx, _vy = _sv
    _line1 = "Open Spectrograph"
    _line2 = (f"v{_ver}"
              f"  {_lam_min:.0f}-{_lam_max:.0f}nm"
              f"  f/{_fnum:.1f}"
              f"  NA {_na:.2f}")

    def _fit_font_size(text, target_width, ref_size=10.0):
        with BuildSketch(Plane.XY) as _ms:
            Text(text, font_size=ref_size,
                 font="DejaVu Sans", font_style=FontStyle.BOLD)
        bb = _ms.sketch.bounding_box()
        fs = ref_size * target_width / bb.size.X
        hh = 0.5 * bb.size.Y * (fs / ref_size)
        return fs, hh

    _ray_v_vals = [
        el.position[0] * _vx + el.position[1] * _vy
        for el in _beam_path
    ] if _beam_path else [_tc_cv]
    _ray_v_min = min(_ray_v_vals) - 0.5 * _RAY_LINE_WIDTH
    _tc_v_min = _tc_cv - 0.5 * _tc_v
    _text_margin = 2.0
    _avail_h = (_ray_v_min - _text_margin) - (_tc_v_min + _TOP_COVER_LEDGE_MM + _text_margin)

    _target_w = 0.8 * _tc_u
    _fs1, _hh1 = _fit_font_size(_line1, _target_w)
    _fs2, _hh2 = _fit_font_size(_line2, _target_w)
    _line_gap = 0.4 * max(_hh1, _hh2)
    _total_h = _hh1 * 2 + _hh2 * 2 + _line_gap
    if _total_h > _avail_h:
        _scale = _avail_h / _total_h
        _fs1 *= _scale
        _hh1 *= _scale
        _fs2 *= _scale
        _hh2 *= _scale
        _line_gap *= _scale

    _v_center = 0.5 * ((_tc_v_min + _TOP_COVER_LEDGE_MM) + _ray_v_min)
    _v_base = _v_center - 0.5 * (_hh1 * 2 + _hh2 * 2 + _line_gap)
    _v2 = _v_base + _hh2
    _v1 = _v_base + _hh2 * 2 + _line_gap + _hh1

    with BuildPart() as _text_bp:
        for _line, _fs, _v in [(_line1, _fs1, _v1), (_line2, _fs2, _v2)]:
            _tx = _tc_cu * _ux + _v * _vx
            _ty = _tc_cu * _uy + _v * _vy
            _pl = Plane(
                origin=Vector(_tx, _ty, z_top),
                x_dir=Vector(-_ux, -_uy, 0),
                z_dir=Vector(0, 0, 1))
            with BuildSketch(_pl):
                Text(_line, font_size=_fs,
                     font="DejaVu Sans", font_style=FontStyle.BOLD)
            extrude(amount=_EMBOSS_DEPTH_MM)
    text_solid = _text_bp.part
    text_solid.label = "emboss_text"
    text_solid.color = Color(0.95, 0.95, 0.95)
    for s in text_solid.solids():
        s.label = "emboss_text"

    # ── Matching pilot holes in housing ledge ────────────────────
    if screw_positions:
        pilot_r = 0.5 * _COVER_SCREW_PILOT_DIA_MM
        with BuildPart() as _pilot_bp:
            add(body)
            for sx, sy in screw_positions:
                hole_plane = Plane(
                    origin=Vector(sx, sy, z_top - _TOP_COVER_DEPTH_MM),
                    z_dir=Vector(0, 0, -1))
                with BuildSketch(hole_plane):
                    Circle(radius=pilot_r)
                extrude(amount=_COVER_SCREW_PILOT_DEPTH_MM,
                        mode=Mode.SUBTRACT)
        body = _pilot_bp.part
        body.label = "solid_housing"

    # ── Detector cover plate (undersized by 2*print_tol) ───────────
    _det_outward = Vector(_dco_outward[0], _dco_outward[1], 0)
    with BuildSketch(_det_cover_plane) as _cover_fit_sk:
        RectangleRounded(_cover_w, _cover_h, _PCBA_POCKET_FILLET_RADIUS_MM)
        try:
            offset(_cover_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.INTERSECTION)
        except ValueError:
            offset(_cover_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.ARC)

    cover_edges = _cover_outline_sk.sketch.edges()
    cover_screw_positions: list[tuple[float, float, float]] = []
    _cover_ledge_center = _PCBA_COVER_LEDGE_MM / 2
    for edge in cover_edges:
        if edge.length >= _COVER_SCREW_MIN_SEGMENT_MM:
            mid = edge.position_at(0.5)
            tangent = edge.tangent_at(0.5)
            if abs(tangent.Z) > 0.5:
                continue
            perp = _det_outward.cross(tangent)
            pn = math.sqrt(perp.X ** 2 + perp.Y ** 2 + perp.Z ** 2)
            if pn < 1e-6:
                continue
            perp = perp * (1.0 / pn)
            centroid = _cover_outline_sk.sketch.center()
            to_center = Vector(centroid.X - mid.X,
                               centroid.Y - mid.Y,
                               centroid.Z - mid.Z)
            if perp.dot(to_center) < 0:
                perp = -perp
            cover_screw_positions.append((
                mid.X + perp.X * _cover_ledge_center,
                mid.Y + perp.Y * _cover_ledge_center,
                mid.Z + perp.Z * _cover_ledge_center,
            ))

    _det_u_vec = Vector(_dco_dx[0], _dco_dx[1], 0)
    _det_inward_vec = Vector(_dco_da[0], _dco_da[1], 0)

    with BuildPart() as cover_bp:
        with BuildSketch(_det_cover_plane) as _cover_plate_sk:
            RectangleRounded(_cover_w, _cover_h,
                             _PCBA_POCKET_FILLET_RADIUS_MM)
            try:
                offset(_cover_plate_sk.faces(), amount=-2 * tol,
                       kind=Kind.INTERSECTION)
            except ValueError:
                offset(_cover_plate_sk.faces(), amount=-2 * tol,
                       kind=Kind.ARC)
        extrude(amount=_PCBA_COVER_DEPTH_MM)

        with BuildSketch(Plane.XY.offset(z_bottom_ext)) as _clip_sk:
            Polygon(*[(x, y) for x, y in spec.housing_outline],
                    align=None)
            _fillet_all(_clip_sk, _EXTERIOR_FILLET_RADIUS_MM)
        extrude(amount=height, mode=Mode.INTERSECT)

        if cover_screw_positions:
            screw_r = 0.5 * _COVER_SCREW_DIA_MM
            head_r = 0.5 * _COVER_SCREW_HEAD_DIA_MM
            for sx, sy, sz in cover_screw_positions:
                ext = Vector(sx, sy, sz) + _det_outward * _PCBA_COVER_DEPTH_MM
                drill_plane = Plane(
                    origin=ext,
                    x_dir=_det_u_vec,
                    z_dir=_det_inward_vec)
                with BuildSketch(drill_plane):
                    Circle(radius=head_r)
                extrude(amount=_COVER_SCREW_CBORE_DEPTH_MM,
                        mode=Mode.SUBTRACT)
                with BuildSketch(drill_plane):
                    Circle(radius=screw_r)
                extrude(amount=_PCBA_COVER_DEPTH_MM,
                        mode=Mode.SUBTRACT)

    det_cover = cover_bp.part
    det_cover.label = "detector_cover"
    det_cover.color = _HOUSING_COLOR
    for s in det_cover.solids():
        s.label = "detector_cover"

    # Matching pilot holes in housing for detector cover screws.
    if cover_screw_positions:
        pilot_r = 0.5 * _COVER_SCREW_PILOT_DIA_MM
        with BuildPart() as _cover_pilot_bp:
            add(body)
            for sx, sy, sz in cover_screw_positions:
                hole_plane = Plane(
                    origin=Vector(sx, sy, sz),
                    x_dir=_det_u_vec,
                    z_dir=_det_inward_vec)
                with BuildSketch(hole_plane):
                    Circle(radius=pilot_r)
                extrude(amount=_COVER_SCREW_PILOT_DEPTH_MM,
                        mode=Mode.SUBTRACT)
        body = _cover_pilot_bp.part
        body.label = "solid_housing"

    # Bottom cover screw positions from outline edges.
    # Where the USB guide box splits an edge, place two screws
    # (one in each remaining segment) instead of one at the midpoint.
    bc_edges = _bc_outline_sk.sketch.edges()
    bc_screw_positions: list[tuple[float, float]] = []
    _bc_ledge_center = _BOTTOM_COVER_LEDGE_WIDTH_MM / 2

    # USB box footprint in world XY for edge splitting.
    _usb_box_half_w = 0.0
    _usb_box_center_xy = None
    _usb_box_tangent = None
    if _usb_box_info is not None:
        _ub_plane, _ub_cx, _ub_yoff, _ub_len, _ub_w, _ub_h_ext = _usb_box_info
        _usb_box_half_w = 0.5 * _ub_w
        _ub_origin = _ub_plane.origin
        _ub_xd = Vector(_ub_plane.x_dir.X, _ub_plane.x_dir.Y, 0)
        _ub_yd = Vector(-_ub_xd.Y, _ub_xd.X, 0)
        _usb_box_center_xy = Vector(
            _ub_origin.X + _ub_cx * _ub_xd.X + _ub_yoff * _ub_yd.X,
            _ub_origin.Y + _ub_cx * _ub_xd.Y + _ub_yoff * _ub_yd.Y, 0)
        _usb_box_tangent = _ub_xd

    for edge in bc_edges:
        if edge.length < _COVER_SCREW_MIN_SEGMENT_MM:
            continue
        mid = edge.position_at(0.5)
        tangent = edge.tangent_at(0.5)
        inward = Vector(-tangent.Y, tangent.X, 0)
        centroid = _bc_outline_sk.sketch.center()
        to_center = Vector(centroid.X - mid.X, centroid.Y - mid.Y, 0)
        if inward.dot(to_center) < 0:
            inward = -inward

        split = False
        if _usb_box_center_xy is not None:
            edge_start = edge.position_at(0)
            edge_vec = Vector(mid.X - edge_start.X,
                              mid.Y - edge_start.Y, 0).normalized()
            box_proj = (((_usb_box_center_xy.X - edge_start.X) * edge_vec.X
                         + (_usb_box_center_xy.Y - edge_start.Y) * edge_vec.Y))
            box_perp = abs((_usb_box_center_xy.X - mid.X) * inward.X
                           + (_usb_box_center_xy.Y - mid.Y) * inward.Y)
            if box_perp < _usb_box_half_w + 5.0:
                seg1_len = box_proj - _usb_box_half_w
                seg2_len = edge.length - box_proj - _usb_box_half_w
                if seg1_len >= _COVER_SCREW_MIN_SEGMENT_MM:
                    t1 = (0.5 * seg1_len) / edge.length
                    p1 = edge.position_at(t1)
                    bc_screw_positions.append((
                        p1.X + inward.X * _bc_ledge_center,
                        p1.Y + inward.Y * _bc_ledge_center))
                if seg2_len >= _COVER_SCREW_MIN_SEGMENT_MM:
                    t2 = (box_proj + _usb_box_half_w
                           + 0.5 * seg2_len) / edge.length
                    p2 = edge.position_at(t2)
                    bc_screw_positions.append((
                        p2.X + inward.X * _bc_ledge_center,
                        p2.Y + inward.Y * _bc_ledge_center))
                split = True

        if not split:
            if edge.length >= 140.0:
                for t in (0.25, 0.75):
                    pt = edge.position_at(t)
                    bc_screw_positions.append((
                        pt.X + inward.X * _bc_ledge_center,
                        pt.Y + inward.Y * _bc_ledge_center))
            else:
                bc_screw_positions.append((
                    mid.X + inward.X * _bc_ledge_center,
                    mid.Y + inward.Y * _bc_ledge_center,
                ))

    # ── Bottom cover (undersized by 2*print_tol) ────────────────────
    with BuildSketch(Plane.XY) as _bc_fit_sk:
        add(_bc_outline_sk.sketch)
        try:
            offset(_bc_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.INTERSECTION)
        except ValueError:
            offset(_bc_fit_sk.faces(), amount=-2 * tol,
                   kind=Kind.ARC)

    with BuildPart() as bc_bp:
        with BuildSketch(Plane.XY.offset(z_bottom_ext)):
            add(_bc_fit_sk.sketch)
        extrude(amount=_BOTTOM_COVER_DEPTH_MM)

        # Counterbored screw holes through bottom cover.
        # CounterBoreHole drills in -z_dir of the local plane.
        # Flip z_dir so it drills upward into the cover body.
        if bc_screw_positions:
            screw_r = 0.5 * _COVER_SCREW_DIA_MM
            head_r = 0.5 * _COVER_SCREW_HEAD_DIA_MM
            for sx, sy in bc_screw_positions:
                _bc_hp = Plane(
                    origin=Vector(sx, sy, z_bottom_ext),
                    x_dir=Vector(1, 0, 0),
                    z_dir=Vector(0, 0, -1))
                with Locations([_bc_hp.location]):
                    CounterBoreHole(
                        radius=screw_r,
                        counter_bore_radius=head_r,
                        counter_bore_depth=_BOTTOM_COVER_CBORE_DEPTH_MM,
                        depth=_BOTTOM_COVER_DEPTH_MM,
                    )

        # USB cable guide: box protruding from bottom cover into USB channel.
        if _usb_box_info is not None:
            _ub_plane, _ub_cx, _ub_yoff, _ub_len, _ub_w, _ub_h_ext = _usb_box_info
            _ub_box_w = _ub_w - 2 * tol
            _ub_box_h = _ub_h_ext - 2 * tol
            if _ub_box_h > 0.1:
                _ub_top_plane = Plane(
                    origin=_ub_plane.origin,
                    x_dir=_ub_plane.x_dir,
                    z_dir=_ub_plane.z_dir)
                with BuildSketch(_ub_top_plane):
                    with Locations([(_ub_cx, _ub_yoff)]):
                        Rectangle(_ub_len, _ub_box_w)
                extrude(amount=_ub_box_h)

                # Cable trough: inset 3mm on near side + both sides,
                # extended 3mm past housing edge to clear fillets.
                _ub_inset = _BOTTOM_COVER_WALL_MM
                _ub_cut_len = _ub_len
                _ub_cut_w = _ub_box_w - 2 * _ub_inset
                _ub_cut_cx = _ub_cx - 0.5 * _ub_inset
                _ub_cut_h = _ub_box_h - _ub_inset
                if _ub_cut_len > 0.1 and _ub_cut_w > 0.1 and _ub_cut_h > 0.1:
                    with BuildSketch(_ub_top_plane) as _ub_cut_sk:
                        with Locations([(_ub_cut_cx, _ub_yoff)]):
                            Rectangle(_ub_cut_len, _ub_cut_w)
                        _fillet_all(_ub_cut_sk, r_pcba)
                    extrude(amount=_ub_cut_h, mode=Mode.SUBTRACT)

        # USB connector slot through bottom cover guide box wall.
        if _usb_slot_info is not None:
            _sl_cx, _sl_cy, _sl_origin, _sl_xoff, _sl_yoff, _sl_zoff = _usb_slot_info
            _sl_center = (
                _sl_origin
                + _sl_cx * _sl_xoff
                + _sl_cy * _sl_yoff
                + Vector(0, 0, z_bottom_ext + _sl_zoff))
            # Stage 1: plug clearance hole, through all from inside face.
            _sl_plug_w = 8.0
            _sl_plug_h = 3.0
            _sl_plug_r = 1.0
            _sl_pin_plane = Plane(
                origin=_sl_center,
                x_dir=_sl_cy,
                z_dir=-_sl_cx)
            with BuildSketch(_sl_pin_plane) as _sl_pin_sk:
                RectangleRounded(_sl_plug_w, _sl_plug_h, _sl_plug_r)
            _sl_pin_tool = _cut_through_next(
                _sl_pin_sk.sketch.faces()[0], bc_bp.part, -_sl_cx)
            if _sl_pin_tool is not None:
                bc_bp._add_to_context(_sl_pin_tool, mode=Mode.SUBTRACT)

            # Stage 2: body clearance hole, 1mm from inside face.
            _usb_slot_w = 13.0
            _usb_slot_h = 9.0
            _usb_slot_r = 1.0
            _sl_skin = 1.0
            _sl_recess_plane = Plane(
                origin=_sl_center - _sl_cx * _sl_skin,
                x_dir=_sl_cy,
                z_dir=-_sl_cx)
            with BuildSketch(_sl_recess_plane) as _sl_sk:
                RectangleRounded(_usb_slot_w, _usb_slot_h, _usb_slot_r)
            _sl_tool = _cut_through_next(
                _sl_sk.sketch.faces()[0], bc_bp.part, -_sl_cx)
            if _sl_tool is not None:
                bc_bp._add_to_context(_sl_tool, mode=Mode.SUBTRACT)

    bottom_cover = bc_bp.part
    bottom_cover.label = "bottom_cover"
    bottom_cover.color = _HOUSING_COLOR
    for s in bottom_cover.solids():
        s.label = "bottom_cover"

    # ── Matching pilot holes in housing bottom ledge ───────────────
    if bc_screw_positions:
        pilot_r = 0.5 * _COVER_SCREW_PILOT_DIA_MM
        with BuildPart() as _bc_pilot_bp:
            add(body)
            for sx, sy in bc_screw_positions:
                hole_plane = Plane(
                    origin=Vector(sx, sy, z_bottom_ext + _BOTTOM_COVER_DEPTH_MM),
                    z_dir=Vector(0, 0, 1))
                with BuildSketch(hole_plane):
                    Circle(radius=pilot_r)
                extrude(amount=_COVER_SCREW_PILOT_DEPTH_MM,
                        mode=Mode.SUBTRACT)
        body = _bc_pilot_bp.part
        body.label = "solid_housing"

    body.color = _HOUSING_COLOR

    # ── Place vendor STEP optics in scene frame ──────────────────────
    placed, _, placed_by_label = place_all_in_scene_frame(
        optics_scene, parts.m1_part, parts.m2_part, parts.grating_part)

    # ── Place controller board in bottom pocket ──────────────────────
    ctrl_placed = None
    if spec.controller_pocket_origin is not None:
        from optics.mounts_cad import _load_vendor_controller
        from pathlib import Path
        ctrl_step = Path("data/step/Controller_T4_R3EB.step")
        if ctrl_step.exists():
            ctrl = _load_vendor_controller()
            cx, cy = spec.controller_pocket_origin
            x_dir = spec.controller_pocket_x_dir
            angle_deg = math.degrees(math.atan2(x_dir[1], x_dir[0])) + 180.0
            pcb_z = z_bottom_ext + ctrl_pocket_ext - _CONTROLLER_Z_INSET_MM
            ctrl = (ctrl
                    .rotate(Axis.Z, angle_deg)
                    .translate((cx, cy, pcb_z)))
            ctrl.label = "instrument_controller"
            ctrl.color = Color(0.15, 0.45, 0.15)
            ctrl_placed = ctrl

    # ── Place M2 pan head screws on top cover ──────────────────
    tc_screw_solids: list = []
    if screw_positions:
        from optics.mounts_cad import _load_or_procedural
        _raw_screw = _load_or_procedural("99461A915")
        for sx, sy in screw_positions:
            placed = _raw_screw.moved(Location(
                (sx, sy, z_top - _COVER_SCREW_CBORE_DEPTH_MM),
            ))
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            tc_screw_solids.append(placed)

    # ── Build named top-level subassemblies ──────────────────────────
    tc_children = [top_cover]
    if ray_solid is not None:
        tc_children.append(ray_solid)
    tc_children.append(text_solid)
    tc_children.extend(tc_screw_solids)
    top_cover_asm = Compound(children=tc_children)
    top_cover_asm.label = "top_cover_assembly"

    # ── Place M2 pan head screws on detector cover ─────────
    dc_screw_solids: list = []
    if cover_screw_positions:
        from optics.mounts_cad import _load_or_procedural as _lop_dc
        _raw_dc_screw = _lop_dc("99461A915")
        _outward_3d = Vector(_dco_outward[0], _dco_outward[1], 0)
        for sx, sy, sz in cover_screw_positions:
            screw_origin = (Vector(sx, sy, sz)
                            + _outward_3d * (_PCBA_COVER_DEPTH_MM
                                             - _COVER_SCREW_CBORE_DEPTH_MM))
            screw_plane = Plane(
                origin=screw_origin,
                x_dir=_det_u_vec,
                z_dir=_outward_3d)
            placed = _raw_dc_screw.moved(screw_plane.location)
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            dc_screw_solids.append(placed)

    dc_children = [det_cover]
    dc_children.extend(dc_screw_solids)
    det_cover_asm = Compound(children=dc_children)
    det_cover_asm.label = "detector_cover_assembly"

    # ── Place M2 pan head screws on bottom cover ─────────────
    bc_screw_solids: list = []
    if bc_screw_positions:
        from optics.mounts_cad import _load_or_procedural as _lop_bc
        from build123d import Rot
        _raw_bc_screw = _lop_bc("99461A915")
        for sx, sy in bc_screw_positions:
            placed = _raw_bc_screw.moved(Location(
                (sx, sy, z_bottom_ext + _BOTTOM_COVER_CBORE_DEPTH_MM),
                (180, 0, 0),
            ))
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            bc_screw_solids.append(placed)

    bc_children = [bottom_cover]
    bc_children.extend(bc_screw_solids)
    bottom_cover_asm = Compound(children=bc_children)
    bottom_cover_asm.label = "bottom_cover_assembly"

    # ── Place M2 pan head screws for detector board ────────────
    det_board_screw_solids: list = []
    if spec.insert_hole_positions:
        from optics.mounts_cad import _load_or_procedural as _lop_det
        _raw_det_screw = _lop_det("99461A915")
        _det_outward_3d = Vector(-spec.detector_plane_z_dir[0],
                                 -spec.detector_plane_z_dir[1], 0)
        _det_x_3d = Vector(spec.detector_plane_x_dir[0],
                           spec.detector_plane_x_dir[1], 0)
        _pcb_gap = 1.6
        for hx, hy, hz in spec.insert_hole_positions:
            screw_origin = Vector(hx, hy, hz) + _det_outward_3d * _pcb_gap
            screw_plane = Plane(
                origin=screw_origin,
                x_dir=_det_x_3d,
                z_dir=_det_outward_3d)
            placed = _raw_det_screw.moved(screw_plane.location)
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            det_board_screw_solids.append(placed)

    # ── Place M2 pan head screws for controller board ──────────
    ctrl_screw_solids: list = []
    if spec.controller_pocket_origin is not None:
        from optics.mounts_cad import _load_or_procedural as _lop_ctrl
        _raw_ctrl_screw = _lop_ctrl("99461A915")
        _ctrl_cx, _ctrl_cy = spec.controller_pocket_origin
        _ctrl_x_dir = Vector(spec.controller_pocket_x_dir[0],
                             spec.controller_pocket_x_dir[1], 0)
        _ctrl_y_dir = Vector(-_ctrl_x_dir.Y, _ctrl_x_dir.X, 0)
        _ctrl_half_isw = 0.5 * spec.controller_insert_spacing_width
        _ctrl_half_ish = 0.5 * spec.controller_insert_spacing_height
        _ctrl_pcb_gap = 1.6
        for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ix = (_ctrl_cx + sx * _ctrl_half_isw * _ctrl_x_dir.X
                  + sy * _ctrl_half_ish * _ctrl_y_dir.X)
            iy = (_ctrl_cy + sx * _ctrl_half_isw * _ctrl_x_dir.Y
                  + sy * _ctrl_half_ish * _ctrl_y_dir.Y)
            screw_origin = Vector(ix, iy,
                                  z_bottom_ext + _BOTTOM_COVER_EXTENSION_MM
                                  + spec.controller_pocket_depth
                                  - _ctrl_pcb_gap)
            screw_plane = Plane(
                origin=screw_origin,
                x_dir=_ctrl_x_dir,
                z_dir=Vector(0, 0, -1))
            placed = _raw_ctrl_screw.moved(screw_plane.location)
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            ctrl_screw_solids.append(placed)

    # ── Place M2 pan head screws for mount feet ───────────────
    mount_screw_solids: list = []
    if spec.mount_fasteners:
        from optics.mounts_cad import _load_or_procedural as _lop_mnt
        _raw_mnt_screw = _lop_mnt("99461A915")
        for mf in spec.mount_fasteners:
            if mf.kind != "bolt":
                continue
            hx, hy = mf.xy
            shelf_z = mf.floor_z - 2.0
            screw_plane = Plane(
                origin=Vector(hx, hy, shelf_z),
                x_dir=Vector(1, 0, 0),
                z_dir=Vector(0, 0, -1))
            placed = _raw_mnt_screw.moved(screw_plane.location)
            placed.label = "99461A915"
            placed.color = Color(0.75, 0.75, 0.78)
            mount_screw_solids.append(placed)

    housing_children = [body]
    housing_children.extend(det_board_screw_solids)
    housing_children.extend(ctrl_screw_solids)
    housing_children.extend(mount_screw_solids)
    housing_asm = Compound(children=housing_children)
    housing_asm.label = "housing_assembly"

    _ROLE_LABELS = {
        "F1": "F1_assembly",
        "M1": "M1_assembly",
        "grating": "grating_assembly",
        "M2": "M2_assembly",
        "F2": "F2_assembly",
    }

    top_children = [housing_asm, top_cover_asm, det_cover_asm, bottom_cover_asm]
    for role, asm_label in _ROLE_LABELS.items():
        parts_list = placed_by_label.get(role)
        if not parts_list:
            continue
        if len(parts_list) == 1:
            sub = parts_list[0]
            sub.label = asm_label
        else:
            sub = Compound(children=parts_list)
            sub.label = asm_label
        top_children.append(sub)

    for lbl in ("entrance_slit", "detector"):
        parts_list = placed_by_label.get(lbl)
        if parts_list:
            for p in parts_list:
                top_children.append(p)

    if ctrl_placed is not None:
        top_children.append(ctrl_placed)

    return (Compound(children=top_children), body, top_cover_asm,
            det_cover_asm, bottom_cover_asm, placed_by_label)
