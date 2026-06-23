"""build123d mount → STEP export.

CAD-grade implementations of the three mount types:
  - Round mount (spherical/flat mirrors)
  - OAP plate with hex bolt pattern (paraboloidal mirrors)
  - Jaw mount (gratings) with rotation shaft bore

All dimensions are imported from ``mounts.py`` so the CAD
and raysect CSG paths cannot drift. The CAD path adds detail the
ray tracer doesn't need: filleted channels, chamfers, etc.

Output: build123d `Part` ready for `export_step()` / `export_stl()`.
"""


import math
from math import atan2, degrees
from pathlib import Path

_BOM_PATH: Path | None = None

def set_bom_path(path: Path | str | None) -> None:
    global _BOM_PATH
    _BOM_PATH = Path(path) if path is not None else None

def _get_bom_path() -> Path:
    if _BOM_PATH is not None:
        return _BOM_PATH
    return Path(__file__).resolve().parent.parent / "data" / "czerny_bom_v0_design.toml"

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Color,
    Cone,
    Cylinder,
    FontStyle,
    GeomType,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    RectangleRounded,
    Sphere,
    Rot,
    Text,
    Until,
    Vector,
    chamfer,
    extrude,
    export_step,
    fillet as fillet2d,
    revolve,
)

_OPTIC_COLOR = Color(0.75, 0.78, 0.82)
_MOUNT_COLOR = Color(0.35, 0.37, 0.40)
_GRATING_COLOR = Color(0.20, 0.55, 0.25)
_DETECTOR_COLOR = Color(0.15, 0.45, 0.15)
_SLIT_COLOR = Color(0.55, 0.55, 0.60)
# Blind setscrew bore: leave a thin floor so a flat-tip setscrew
# pushes the optic through the wall instead of touching it directly.
# Eliminates the need for expensive nylon-tipped setscrews.
_SETSCREW_FLOOR_MM = 0.5

from optics.mounts import (
    GratingFlexureMountParams,
    ManufacturingParams,
    OAPMirrorFlexureMountParams,
    RoundMirrorFlexureMountParams,
    parse_mount_params,
)

def place_in_scene_frame(part, position, normal):
    """Rotate+translate a build123d part into the scene frame."""
    nx, ny, _ = normal
    theta_deg = degrees(atan2(nx, -ny))
    return (
        part
        .rotate(Axis.X, 90.0)
        .rotate(Axis.Z, theta_deg)
        .translate(position)
    )




def build_mirror_flexure_mount_cad(
    *,
    optic_diameter_mm: float,
    center_thickness_mm: float,
    params: RoundMirrorFlexureMountParams,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Flexure mirror mount (CAD-grade, sketch-driven).

    Frame: x=v, y=w (world +z), z=u (optic normal). Origin at vertex.
    Every feature is a sketch on a named plane, extruded with a mode.
    """
    if mfg is None:
        mfg = _load_manufacturing()

    optic_radius_mm = 0.5 * optic_diameter_mm

    # -- u positions (along optic normal, +u = toward beam) --
    u_vertex = 0.0
    u_shoulder = -center_thickness_mm
    u_wall_rear = u_shoulder - params.rear_wall_mm

    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    bolt_clearance_mm = mfg.bolt_dims[params.foot_bolt_thread]["clearance_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm

    # -- w positions (vertical, +w = up) --
    w_top = optic_radius_mm + params.head_clearance_mm
    w_bot = -(optic_radius_mm + params.foot_clearance_mm)
    slab_depth_mm = u_vertex - u_wall_rear
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    w_foot_mid = w_foot_bot + 0.5 * params.foot_thickness_mm

    # -- v positions (dispersion axis) --
    v_half_mm = optic_radius_mm + params.wall_margin_mm

    # -- bore diameters --
    front_bore_dia_mm = optic_diameter_mm + mfg.assembly_clearance_mm + mfg.print_tolerance_mm

    # -- roll flexure u-depth (slab depth only, no overhang) --
    roll_u_depth = slab_depth_mm
    u_roll_actuate = 0.5 * (u_wall_rear + u_vertex)

    # -- derived dimensions --
    slab_total_u_mm = u_foot_front - u_wall_rear
    full_height_mm = w_top - w_foot_bot
    w_center = 0.5 * (w_top + w_foot_bot)

    # -- foot geometry --
    v_bolt = params.foot_bolt_spacing_mm
    u_pusher = 0.0
    u_front_bolt = params.front_bolt_offset_mm
    u_setscrew = 0.5 * (u_wall_rear + u_vertex)

    # -- named planes --
    rear_plane = Plane(origin=(0, 0, u_wall_rear),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    bore_plane = Plane(origin=(0, 0, u_shoulder),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    front_cut_plane = Plane(origin=(0, 0, u_vertex),
                            x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    foot_bot_plane = Plane(origin=(0, w_foot_bot, 0),
                           x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    foot_profile_plane = Plane(origin=(0, w_foot_bot, 0),
                               x_dir=(-1, 0, 0), z_dir=(0, 1, 0))
    tilt_cos = math.cos(math.radians(params.trim_angle_deg))
    tilt_sin = math.sin(math.radians(params.trim_angle_deg))
    tilted_foot_plane = Plane(
        origin=(0, w_foot_bot, u_wall_rear),
        x_dir=(1, 0, 0),
        z_dir=(0, -tilt_cos, tilt_sin),
    )
    tilted_surface_plane = Plane(
        origin=(0, w_foot_bot, u_wall_rear),
        x_dir=(1, 0, 0),
        z_dir=(0, tilt_cos, -tilt_sin),
    )

    # -- bore geometry (needed by both hull and internal features) --
    front_bore_depth_mm = u_vertex - u_shoulder
    bore_r = 0.5 * (front_bore_dia_mm + params.optic_clearance_mm)
    contact_radius_mm = params.contact_radius_mm
    contact_offset_mm = params.contact_offset_mm
    v_contact_mm = 0.5 * params.contact_separation_mm
    u_mid = u_wall_rear + 0.5 * slab_total_u_mm

    with BuildPart() as mount:
        # ── Outer contour ─────────────────────────────────────────────

        # 1. Tombstone stock: sketch rear cross-section, extrude forward.
        with BuildSketch(rear_plane) as stock_sk:
            with Locations(Pos(0, w_center)):
                Rectangle(2 * v_half_mm, full_height_mm)
        extrude(stock_sk.sketch, amount=slab_total_u_mm)

        # 2. Side cuts: trim mount flush with mirror diameter.
        side_excess_mm = v_half_mm - optic_radius_mm
        if side_excess_mm > 0:
            for sign in (+1, -1):
                v_cut = sign * (optic_radius_mm + 0.5 * side_excess_mm)
                with Locations(Pos(v_cut, w_center,
                                   u_wall_rear + 0.5 * slab_total_u_mm)):
                    Box(side_excess_mm, full_height_mm, slab_total_u_mm,
                        mode=Mode.SUBTRACT)

        # 3. Face mill (sketched on body front face).
        #    Two bands: optic pocket (w_bot to w_top) and pitch shelf
        front_face = mount.faces().sort_by(Axis.Z)[-1]
        mill_depth_mm = u_foot_front - u_vertex
        w_slab_center = 0.5 * (w_top + w_shelf_bot)
        with BuildSketch(front_face) as mill_sk:
            with Locations(Pos(0, w_slab_center - w_center)):
                Rectangle(2 * v_half_mm, w_top - w_shelf_bot)
        extrude(mill_sk.sketch, amount=-mill_depth_mm, mode=Mode.SUBTRACT)

        # 4. Shape the foot (sketched on body bottom face).
        #    2 forward bolts at ±v_bolt, 1 aft bolt at center.
        #    Single wide tongue spanning both forward bolt positions.
        tongue_width_mm = 2 * v_bolt + boss_width_mm
        foot_face = (mount.faces()
                     .filter_by(Axis.Y).sort_by(Axis.Y)[0])
        foot_u_mid = foot_face.center().Z
        with BuildSketch(foot_face) as foot_sk:
            # Full-width rear section (slab footprint).
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_vertex)
                               - foot_u_mid)):
                Rectangle(2 * v_half_mm, slab_depth_mm)
            # Single wide tongue forward.
            tongue_u_len = u_front_bolt + 0.5 * boss_width_mm - u_wall_rear
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_front_bolt
                               + 0.5 * boss_width_mm) - foot_u_mid)):
                Rectangle(tongue_width_mm, tongue_u_len)
            # Fillet all four junction corners.
            fillet_verts = [
                v for v in foot_sk.vertices()
                if (abs(v.Y - (u_vertex - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
                or (abs(v.Y - (u_front_bolt + 0.5 * boss_width_mm
                               - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
            ]
            if fillet_verts:
                try:
                    fillet2d(fillet_verts, radius=mfg.fillet_radius_mm)
                except Exception:
                    pass
        extrude(foot_sk.sketch, amount=-(w_top - w_foot_bot),
                mode=Mode.INTERSECT)

    # 5. Pusher shelf (fused outside BuildPart to avoid coplanar issues).
    _shelf_plane = Plane(
        origin=(0, w_shelf_bot, u_pusher),
        x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    with BuildPart() as _shelf_bp:
        with BuildSketch(_shelf_plane):
            Circle(0.5 * boss_width_mm)
        extrude(amount=params.pusher_shelf_mm)
    result = mount.part.fuse(_shelf_bp.part)
    _shelf_fillet_r = min(mfg.fillet_radius_mm, 0.49 * params.pusher_shelf_mm)
    _shelf_r = 0.5 * boss_width_mm
    _fillet_edges = [
        e for e in result.edges()
        if abs(e.center().Z) < 0.1
        and w_shelf_bot < e.center().Y < w_shelf_bot + params.pusher_shelf_mm
        and abs(math.hypot(e.center().X, e.center().Z - u_pusher) - _shelf_r) < 0.5
    ]
    if _fillet_edges and _shelf_fillet_r > 0.1:
        try:
            result = result.fillet(_shelf_fillet_r, _fillet_edges)
        except Exception:
            pass

    # 6. Flexure relief: sketch on flexure front face, extrude forward.
    # 7. Trim foot bottom: wedge triangle in u-w plane, extrude in v.
    # These modify the outer contour but need to run after the shelf
    # fuse, so they operate on `result` via a fresh BuildPart.
    with BuildPart() as mount2:
        mount2._obj = result

        u_flexure_front = u_wall_rear + params.pitch_flexure_thickness_mm
        flexure_u_mm = u_foot_front - u_flexure_front
        flexure_plane = Plane(origin=(0, 0, u_flexure_front),
                              x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildSketch(flexure_plane) as flex_sk:
            w_flexure_mid = w_shelf_bot - 0.5 * params.pitch_flexure_gap_mm
            with Locations(Pos(0, w_flexure_mid)):
                Rectangle(2 * v_half_mm, params.pitch_flexure_gap_mm)
        extrude(flex_sk.sketch, until=Until.LAST, mode=Mode.SUBTRACT)

        trim_u_end = u_foot_front + boss_width_mm
        trim_u_span = trim_u_end - u_wall_rear
        wedge_rise = trim_u_span * math.tan(
            math.radians(params.trim_angle_deg))
        uw_plane = Plane(origin=(0, w_foot_bot, 0),
                         x_dir=(0, 0, 1), z_dir=(-1, 0, 0))
        with BuildSketch(uw_plane) as trim_sk:
            Polygon(
                (u_wall_rear, 0),
                (trim_u_end, 0),
                (trim_u_end, wedge_rise),
                align=None,
            )
        extrude(trim_sk.sketch, until=Until.LAST, both=True,
                mode=Mode.SUBTRACT)

        # 7b. Roll flexure blade cuts.
        if params.roll_flexure_height_mm > 0 and params.roll_blade_thickness_mm > 0:
            blade_t = params.roll_blade_thickness_mm
            v_insert = params.roll_insert_spacing_mm
            v_anchor = params.roll_blade_spacing_mm
            v_ped_inner = v_anchor + params.roll_pedestal_gap_mm
            ped_w = optic_radius_mm - v_ped_inner
            v_top_blade = v_anchor * (w_bot / w_roll_bot)

            dv = v_top_blade - v_anchor
            dw = w_bot - w_roll_bot
            blade_len = math.hypot(dv, dw)
            nv = dw / blade_len
            nw = -dv / blade_len
            ht = 0.5 * blade_t

            # Extend blade polygon past the relief rectangle on
            # both ends so blades merge into the rigid plates
            # rather than being cut off at the band boundary.
            # Bottom: continue centerline until v = mount wall.
            w_bot_ext = w_roll_bot * optic_radius_mm / v_anchor
            v_bot_ext = optic_radius_mm
            # Top: continue centerline to w = 0 (virtual pivot).
            v_top_ext = 0.0
            w_top_ext = 0.0
            blade_plus = [
                (v_bot_ext + ht * nv, w_bot_ext + ht * nw),
                (v_bot_ext - ht * nv, w_bot_ext - ht * nw),
                (v_top_ext - ht * nv, w_top_ext - ht * nw),
                (v_top_ext + ht * nv, w_top_ext + ht * nw),
            ]
            blade_minus = list(reversed([(-v, w) for v, w in blade_plus]))

            # Pedestals: trapezoidal, inner face parallel to blade
            # with constant perpendicular gap, outer face at mount wall.
            w_ped_top = w_bot
            w_ped_bot = w_roll_bot + 1.0
            gap = params.roll_pedestal_gap_mm
            total_off = ht + gap

            def _ped_inner_v(w_at):
                t = (w_at - w_roll_bot - total_off * nw) / dw
                return v_anchor + total_off * nv + t * dv

            v_inner_top = _ped_inner_v(w_ped_top)
            v_inner_bot = _ped_inner_v(w_ped_bot)

            # Center block: sidewalls parallel to blades, 1mm below w_bot.
            w_ctr_top = w_bot - 1.0
            w_ctr_bot = w_roll_bot

            def _ctr_wall_v(w_at):
                t = (w_at - w_roll_bot + total_off * nw) / dw
                return v_anchor - total_off * nv + t * dv

            v_ctr_top = _ctr_wall_v(w_ctr_top)
            v_ctr_bot = _ctr_wall_v(w_ctr_bot)

            roll_cut_plane = Plane(origin=(0, 0, u_wall_rear),
                                   x_dir=(1, 0, 0), z_dir=(0, 0, 1))
            w_roll_mid = 0.5 * (w_roll_bot + w_bot)
            with BuildSketch(roll_cut_plane) as roll_sk:
                with Locations(Pos(0, w_roll_mid)):
                    Rectangle(2 * optic_radius_mm,
                              params.roll_flexure_height_mm)
                Polygon(*blade_plus, align=None, mode=Mode.SUBTRACT)
                Polygon(*blade_minus, align=None, mode=Mode.SUBTRACT)
                for sign in (+1, -1):
                    Polygon(
                        (sign * optic_radius_mm, w_ped_top),
                        (sign * v_inner_top, w_ped_top),
                        (sign * v_inner_bot, w_ped_bot),
                        (sign * optic_radius_mm, w_ped_bot),
                        align=None, mode=Mode.SUBTRACT)
                Polygon(
                    (+v_ctr_top, w_ctr_top),
                    (-v_ctr_top, w_ctr_top),
                    (-v_ctr_bot, w_ctr_bot),
                    (+v_ctr_bot, w_ctr_bot),
                    align=None, mode=Mode.SUBTRACT)
            extrude(roll_sk.sketch, amount=roll_u_depth,
                    mode=Mode.SUBTRACT)

            # 7c. Roll setscrew insert bores — enter from foot
            # bottom, bore upward through foot + gap + into shelf.
            roll_bore_depth = (params.foot_thickness_mm
                               + params.pitch_flexure_gap_mm
                               + params.pusher_shelf_mm)
            roll_ss_plane = Plane(
                origin=(0, w_foot_bot, 0),
                x_dir=(1, 0, 0), z_dir=(0, 1, 0))
            for v_ss in [+v_insert, -v_insert]:
                with BuildSketch(roll_ss_plane) as roll_ss_sk:
                    with Locations(Pos(v_ss, -u_roll_actuate)):
                        Circle(radius=0.5 * mfg.insert_bore_dia_mm)
                extrude(roll_ss_sk.sketch,
                        amount=roll_bore_depth,
                        mode=Mode.SUBTRACT)
            # Chamfers on insert entries (foot bottom face).
            chamfer_top_r = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
            chamfer_bot_r = 0.5 * mfg.insert_bore_dia_mm
            for v_ss in [+v_insert, -v_insert]:
                with Locations(Pos(v_ss,
                                   w_foot_bot + 0.5 * mfg.insert_chamfer_mm,
                                   u_roll_actuate)):
                    Cone(bottom_radius=chamfer_bot_r,
                         top_radius=chamfer_top_r,
                         height=mfg.insert_chamfer_mm,
                     rotation=(90, 0, 0),
                     mode=Mode.SUBTRACT)

        # ── Internal features ─────────────────────────────────────────

        # 8. Cut bore: upper circle + wider lower circle with contact
        # bump pockets.  The bumps themselves are separate TPU bodies.
        bump_center_r = bore_r + contact_radius_mm - contact_offset_mm
        w_bump = -math.sqrt(bump_center_r**2 - v_contact_mm**2)
        with BuildSketch(bore_plane) as bore_sk:
            Circle(bore_r)
            Rectangle(2 * bore_r, bore_r,
                      align=(Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
            Circle(bore_r)
            with Locations(Pos(0, 0)):
                Rectangle(2 * v_half_mm + 0.2, 0.75 * optic_diameter_mm,
                          align=(Align.CENTER, Align.CENTER))
            with Locations(Pos(-v_contact_mm, w_bump),
                           Pos(+v_contact_mm, w_bump),
                           Pos(0, bump_center_r)):
                Circle(contact_radius_mm)
        extrude(bore_sk.sketch, amount=front_bore_depth_mm,
                mode=Mode.SUBTRACT)

        # 9. Setscrew insert bore from top (blind).
        ss_bore_depth_mm = (w_top - bore_r) - _SETSCREW_FLOOR_MM
        ss_plane = Plane(origin=(0, w_top, 0),
                         x_dir=(1, 0, 0), z_dir=(0, -1, 0))
        with BuildSketch(ss_plane) as ss_sk:
            with Locations(Pos(0, u_setscrew)):
                Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(ss_sk.sketch, amount=ss_bore_depth_mm, mode=Mode.SUBTRACT)

        # 10. Foot screw pilot holes (2 forward + 1 aft).
        foot_pilot_dia_mm = mfg.bolt_dims[params.foot_bolt_thread]["tap_drill_dia_mm"]
        foot_hole_plane = Plane(origin=(0, w_foot_bot, 0),
                                x_dir=(1, 0, 0), z_dir=(0, -1, 0))
        for bv, bu in [(+v_bolt, u_front_bolt), (-v_bolt, u_front_bolt),
                        (0, u_bolt)]:
            with BuildSketch(foot_hole_plane) as ins_sk:
                with Locations(Pos(bv, bu)):
                    Circle(radius=0.5 * foot_pilot_dia_mm)
            extrude(ins_sk.sketch, amount=-params.foot_thickness_mm,
                    mode=Mode.SUBTRACT)
        # Pusher insert bore (through foot).
        with BuildSketch(foot_hole_plane) as push_sk:
            with Locations(Pos(0, u_pusher)):
                Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(push_sk.sketch, amount=-params.foot_thickness_mm,
                mode=Mode.SUBTRACT)

        # 11. Chamfers on insert holes.
        chamfer_top_r = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
        chamfer_bot_r = 0.5 * mfg.insert_bore_dia_mm
        with Locations(Pos(0, w_top - 0.5 * mfg.insert_chamfer_mm, u_setscrew)):
            Cone(bottom_radius=chamfer_top_r, top_radius=chamfer_bot_r,
                 height=mfg.insert_chamfer_mm, rotation=(90, 0, 0),
                 mode=Mode.SUBTRACT)
        tilted_fwd = tilted_surface_plane.y_dir * -1
        tilted_n = tilted_surface_plane.z_dir
        chamfer_rot = (90 - params.trim_angle_deg, 0, 0)
        for cv, cu in [(0, u_pusher)]:
            surf = (tilted_surface_plane.origin
                    + tilted_surface_plane.x_dir * cv
                    + tilted_fwd * (cu - u_wall_rear))
            ctr = surf + tilted_n * (0.5 * mfg.insert_chamfer_mm)
            with Locations(Pos(ctr.X, ctr.Y, ctr.Z)):
                Cone(bottom_radius=chamfer_bot_r,
                     top_radius=chamfer_top_r,
                     height=mfg.insert_chamfer_mm,
                     rotation=chamfer_rot, mode=Mode.SUBTRACT)

    result = mount2.part
    result.color = _MOUNT_COLOR
    result.label = "flexure_mount"

    # Build captive TPU contact bumps as a separate body.
    _bump_color = Color(0.30, 0.60, 0.85)
    bump_positions = [
        (-v_contact_mm, w_bump),
        (+v_contact_mm, w_bump),
        (0, bump_center_r),
    ]
    bump_parts = []
    for bv, bw in bump_positions:
        bump = Cylinder(contact_radius_mm, front_bore_depth_mm,
                        align=(Align.CENTER, Align.CENTER, Align.MIN))
        bump = bump.translate((bv, bw, u_shoulder))
        bump.color = _bump_color
        bump_parts.append(bump)
    from build123d import Compound as _Cmp
    bumps = _Cmp(children=bump_parts)
    bumps.label = "contact_bumps_tpu"
    result._contact_bumps = bumps

    return result


def build_oap_flexure_mount_cad(
    *,
    optic_diameter_mm: float,
    center_thickness_mm: float,
    params: OAPMirrorFlexureMountParams,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Flexure OAP plate mount (CAD-grade, sketch-driven).

    Frame: x=v, y=w (world +z), z=u (optic normal). Origin at optical
    centre (chief-ray intersection on paraboloid surface).
    """
    if mfg is None:
        mfg = _load_manufacturing()

    optic_radius_mm = 0.5 * optic_diameter_mm
    hc = params.head_clearance_mm
    fc = params.foot_clearance_mm

    # -- u positions --
    u_plate_front = -center_thickness_mm
    u_wall_rear = u_plate_front - params.slab_thickness_mm
    slab_depth_mm = params.slab_thickness_mm
    pocket_depth_mm = center_thickness_mm

    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_plate_front + foot_length_mm
    slab_total_u_mm = u_foot_front - u_wall_rear

    # -- w positions --
    w_top = optic_radius_mm + hc
    w_bot = -(optic_radius_mm + fc)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm

    # -- v positions --
    v_half_mm = optic_radius_mm

    # -- roll flexure u-depth --
    roll_u_depth = slab_depth_mm
    u_roll_actuate = 0.5 * (u_wall_rear + u_plate_front)

    # -- derived --
    full_height_mm = w_top - w_foot_bot
    w_center = 0.5 * (w_top + w_foot_bot)

    # -- foot geometry --
    v_bolt = params.foot_bolt_spacing_mm
    u_pusher = u_plate_front
    u_front_bolt = u_plate_front + params.front_bolt_offset_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    u_mid = u_wall_rear + 0.5 * slab_total_u_mm

    # -- named planes --
    rear_plane = Plane(origin=(0, 0, u_wall_rear),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    foot_profile_plane = Plane(origin=(0, w_foot_bot, 0),
                               x_dir=(-1, 0, 0), z_dir=(0, 1, 0))
    tilted_surface_plane = Plane(
        origin=(0, w_foot_bot, u_wall_rear),
        x_dir=(1, 0, 0),
        z_dir=(0, math.cos(math.radians(params.trim_angle_deg)),
               -math.sin(math.radians(params.trim_angle_deg))),
    )

    with BuildPart() as mount:
        # 1. Tombstone stock: rectangle below w=0, semicircle above.
        rect_height_mm = abs(w_foot_bot)
        with BuildSketch(rear_plane) as stock_sk:
            with Locations(Pos(0, -0.5 * rect_height_mm)):
                Rectangle(2 * v_half_mm, rect_height_mm)
        extrude(stock_sk.sketch, amount=slab_total_u_mm)
        with BuildSketch(rear_plane) as arch_sk:
            Circle(v_half_mm)
            Rectangle(2 * v_half_mm, 2 * w_top,
                      align=(Align.CENTER, Align.MIN), mode=Mode.INTERSECT)
        extrude(arch_sk.sketch, amount=slab_total_u_mm)

        # 2. Bolt-circle through-holes + countersinks from back face.
        rear_face = mount.faces().sort_by(Axis.Z)[0]
        rear_center_y = rear_face.center().Y
        for i in range(params.n_holes):
            angle = math.radians(params.hole_phase_deg + i * 60.0)
            hx = params.bolt_circle_radius_mm * math.sin(angle)
            hy = params.bolt_circle_radius_mm * math.cos(angle)
            with BuildSketch(rear_face) as hole_sk:
                with Locations(Pos(hx, rear_center_y - hy)):
                    Circle(radius=0.5 * params.clearance_hole_dia_mm)
            extrude(hole_sk.sketch, amount=-params.slab_thickness_mm,
                    mode=Mode.SUBTRACT)
        # Countersinks (82° flat head cone).
        cs_top_r = 0.5 * params.counterbore_dia_mm
        cs_bot_r = 0.5 * params.clearance_hole_dia_mm
        cs_height = (cs_top_r - cs_bot_r) / math.tan(math.radians(41.0))
        for i in range(params.n_holes):
            angle = math.radians(params.hole_phase_deg + i * 60.0)
            hx = params.bolt_circle_radius_mm * math.sin(angle)
            hy = params.bolt_circle_radius_mm * math.cos(angle)
            with Locations(Pos(hx, hy, u_wall_rear + 0.5 * cs_height)):
                Cone(bottom_radius=cs_top_r, top_radius=cs_bot_r,
                     height=cs_height, mode=Mode.SUBTRACT)

        # 3. Face mill (sketched on body front face).
        front_face = mount.faces().sort_by(Axis.Z)[-1]
        front_center_y = front_face.center().Y
        mill_depth_mm = u_foot_front - u_plate_front
        w_slab_center = 0.5 * (w_top + w_shelf_bot)
        with BuildSketch(front_face) as mill_sk:
            with Locations(Pos(0, w_slab_center - front_center_y)):
                Rectangle(2 * v_half_mm, w_top - w_shelf_bot)
        extrude(mill_sk.sketch, amount=-mill_depth_mm, mode=Mode.SUBTRACT)

        # 4. Foot screw pilot holes (2 forward + 1 aft).
        foot_pilot_dia_mm = mfg.bolt_dims[params.foot_bolt_thread]["tap_drill_dia_mm"]
        foot_bottom_face = (mount.faces()
                            .filter_by(Axis.Y).sort_by(Axis.Y)[0])
        for bv, bu in [(+v_bolt, u_front_bolt), (-v_bolt, u_front_bolt),
                        (0, u_bolt)]:
            with BuildSketch(foot_bottom_face) as ins_sk:
                with Locations(Pos(bv, bu - u_mid)):
                    Circle(radius=0.5 * foot_pilot_dia_mm)
            extrude(ins_sk.sketch, amount=-params.foot_thickness_mm,
                    mode=Mode.SUBTRACT)
        # Pusher insert bore (through foot).
        with BuildSketch(foot_bottom_face) as push_sk:
            with Locations(Pos(0, u_pusher - u_mid)):
                Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(push_sk.sketch, amount=-params.foot_thickness_mm,
                mode=Mode.SUBTRACT)

        # 5. Shape the foot (sketched on body bottom face).
        #    Single wide tongue spanning both forward bolt positions.
        tongue_width_mm = 2 * v_bolt + boss_width_mm
        foot_face = (mount.faces()
                     .filter_by(Axis.Y).sort_by(Axis.Y)[0])
        foot_u_mid = foot_face.center().Z
        with BuildSketch(foot_face) as foot_sk:
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_plate_front)
                               - foot_u_mid)):
                Rectangle(2 * v_half_mm, slab_depth_mm)
            tongue_u_len = u_front_bolt + 0.5 * boss_width_mm - u_wall_rear
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_front_bolt
                               + 0.5 * boss_width_mm) - foot_u_mid)):
                Rectangle(tongue_width_mm, tongue_u_len)
            fillet_verts = [
                v for v in foot_sk.vertices()
                if (abs(v.Y - (u_plate_front - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
                or (abs(v.Y - (u_front_bolt + 0.5 * boss_width_mm
                               - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
            ]
            if fillet_verts:
                try:
                    fillet2d(fillet_verts, radius=mfg.fillet_radius_mm)
                except Exception:
                    pass
        extrude(foot_sk.sketch, amount=-(w_top - w_foot_bot),
                mode=Mode.INTERSECT)

        # 6. Flexure relief.
        u_flexure_front = u_wall_rear + params.pitch_flexure_thickness_mm
        flexure_plane = Plane(origin=(0, 0, u_flexure_front),
                              x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildSketch(flexure_plane) as flex_sk:
            w_flexure_mid = w_shelf_bot - 0.5 * params.pitch_flexure_gap_mm
            with Locations(Pos(0, w_flexure_mid)):
                Rectangle(2 * v_half_mm, params.pitch_flexure_gap_mm)
        extrude(flex_sk.sketch, until=Until.LAST, mode=Mode.SUBTRACT)

        # 7. Trim foot bottom: wedge triangle in u-w plane.
        trim_u_end = u_foot_front + boss_width_mm
        trim_u_span = trim_u_end - u_wall_rear
        wedge_rise = trim_u_span * math.tan(
            math.radians(params.trim_angle_deg))
        uw_plane = Plane(origin=(0, w_foot_bot, 0),
                         x_dir=(0, 0, 1), z_dir=(-1, 0, 0))
        with BuildSketch(uw_plane) as trim_sk:
            Polygon(
                (u_wall_rear, 0),
                (trim_u_end, 0),
                (trim_u_end, wedge_rise),
                align=None,
            )
        extrude(trim_sk.sketch, until=Until.LAST, both=True,
                mode=Mode.SUBTRACT)

        # 7b. Roll flexure blade cuts.
        if params.roll_flexure_height_mm > 0 and params.roll_blade_thickness_mm > 0:
            blade_t = params.roll_blade_thickness_mm
            v_insert = params.roll_insert_spacing_mm
            v_anchor = params.roll_blade_spacing_mm
            v_ped_inner = v_anchor + params.roll_pedestal_gap_mm
            ped_w = optic_radius_mm - v_ped_inner
            v_top_blade = v_anchor * (w_bot / w_roll_bot)

            dv = v_top_blade - v_anchor
            dw = w_bot - w_roll_bot
            blade_len = math.hypot(dv, dw)
            nv = dw / blade_len
            nw = -dv / blade_len
            ht = 0.5 * blade_t

            w_bot_ext = w_roll_bot * optic_radius_mm / v_anchor
            v_bot_ext = optic_radius_mm
            v_top_ext = 0.0
            w_top_ext = 0.0
            blade_plus = [
                (v_bot_ext + ht * nv, w_bot_ext + ht * nw),
                (v_bot_ext - ht * nv, w_bot_ext - ht * nw),
                (v_top_ext - ht * nv, w_top_ext - ht * nw),
                (v_top_ext + ht * nv, w_top_ext + ht * nw),
            ]
            blade_minus = list(reversed([(-v, w) for v, w in blade_plus]))

            w_ped_top = w_bot
            w_ped_bot = w_roll_bot + 1.0
            gap = params.roll_pedestal_gap_mm
            total_off = ht + gap

            def _ped_inner_v(w_at):
                t = (w_at - w_roll_bot - total_off * nw) / dw
                return v_anchor + total_off * nv + t * dv

            v_inner_top = _ped_inner_v(w_ped_top)
            v_inner_bot = _ped_inner_v(w_ped_bot)

            w_ctr_top = w_bot - 1.0
            w_ctr_bot = w_roll_bot

            def _ctr_wall_v(w_at):
                t = (w_at - w_roll_bot + total_off * nw) / dw
                return v_anchor - total_off * nv + t * dv

            v_ctr_top = _ctr_wall_v(w_ctr_top)
            v_ctr_bot = _ctr_wall_v(w_ctr_bot)

            roll_cut_plane = Plane(origin=(0, 0, u_wall_rear),
                                   x_dir=(1, 0, 0), z_dir=(0, 0, 1))
            w_roll_mid = 0.5 * (w_roll_bot + w_bot)
            with BuildSketch(roll_cut_plane) as roll_sk:
                with Locations(Pos(0, w_roll_mid)):
                    Rectangle(2 * optic_radius_mm,
                              params.roll_flexure_height_mm)
                Polygon(*blade_plus, align=None, mode=Mode.SUBTRACT)
                Polygon(*blade_minus, align=None, mode=Mode.SUBTRACT)
                for sign in (+1, -1):
                    Polygon(
                        (sign * optic_radius_mm, w_ped_top),
                        (sign * v_inner_top, w_ped_top),
                        (sign * v_inner_bot, w_ped_bot),
                        (sign * optic_radius_mm, w_ped_bot),
                        align=None, mode=Mode.SUBTRACT)
                Polygon(
                    (+v_ctr_top, w_ctr_top),
                    (-v_ctr_top, w_ctr_top),
                    (-v_ctr_bot, w_ctr_bot),
                    (+v_ctr_bot, w_ctr_bot),
                    align=None, mode=Mode.SUBTRACT)
            extrude(roll_sk.sketch, amount=roll_u_depth,
                    mode=Mode.SUBTRACT)

            # Roll insert bores — from foot bottom through foot + gap + shelf.
            roll_bore_depth = (params.foot_thickness_mm
                               + params.pitch_flexure_gap_mm
                               + params.pusher_shelf_mm)
            roll_ss_plane = Plane(
                origin=(0, w_foot_bot, 0),
                x_dir=(1, 0, 0), z_dir=(0, 1, 0))
            for v_ss in [+v_insert, -v_insert]:
                with BuildSketch(roll_ss_plane) as roll_ss_sk:
                    with Locations(Pos(v_ss, -u_roll_actuate)):
                        Circle(radius=0.5 * mfg.insert_bore_dia_mm)
                extrude(roll_ss_sk.sketch,
                        amount=roll_bore_depth,
                        mode=Mode.SUBTRACT)
            chamfer_top_r_roll = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
            chamfer_bot_r_roll = 0.5 * mfg.insert_bore_dia_mm
            for v_ss in [+v_insert, -v_insert]:
                with Locations(Pos(v_ss,
                                   w_foot_bot + 0.5 * mfg.insert_chamfer_mm,
                                   u_roll_actuate)):
                    Cone(bottom_radius=chamfer_bot_r_roll,
                         top_radius=chamfer_top_r_roll,
                         height=mfg.insert_chamfer_mm,
                     rotation=(90, 0, 0),
                     mode=Mode.SUBTRACT)

        # 8. Pusher insert chamfer on the tilted surface.
        chamfer_top_r = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
        chamfer_bot_r = 0.5 * mfg.insert_bore_dia_mm
        tilted_fwd = tilted_surface_plane.y_dir * -1
        tilted_n = tilted_surface_plane.z_dir
        chamfer_rot = (90 - params.trim_angle_deg, 0, 0)
        for cv, cu in [(0, u_pusher)]:
            surf = (tilted_surface_plane.origin
                    + tilted_surface_plane.x_dir * cv
                    + tilted_fwd * (cu - u_wall_rear))
            ctr = surf + tilted_n * (0.5 * mfg.insert_chamfer_mm)
            with Locations(Pos(ctr.X, ctr.Y, ctr.Z)):
                Cone(bottom_radius=chamfer_bot_r,
                     top_radius=chamfer_top_r,
                     height=mfg.insert_chamfer_mm,
                     rotation=chamfer_rot, mode=Mode.SUBTRACT)


    # Pusher shelf: semicircular lip at w_shelf_bot.
    _shelf_plane = Plane(
        origin=(0, w_shelf_bot, u_pusher),
        x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    with BuildPart() as _shelf_bp:
        with BuildSketch(_shelf_plane):
            Circle(0.5 * boss_width_mm)
        extrude(amount=params.pusher_shelf_mm)
    result = mount.part.fuse(_shelf_bp.part)
    _shelf_fillet_r = min(mfg.fillet_radius_mm, 0.49 * params.pusher_shelf_mm)
    _shelf_r = 0.5 * boss_width_mm
    _fillet_edges = [
        e for e in result.edges()
        if abs(e.center().Z) < 0.1
        and w_shelf_bot < e.center().Y < w_shelf_bot + params.pusher_shelf_mm
        and abs(math.hypot(e.center().X, e.center().Z - u_pusher) - _shelf_r) < 0.5
    ]
    if _fillet_edges and _shelf_fillet_r > 0.1:
        try:
            result = result.fillet(_shelf_fillet_r, _fillet_edges)
        except Exception:
            pass
    result = Part(result.wrapped)
    result.color = _MOUNT_COLOR
    result.label = "flexure_mount"
    return result


def build_grating_flexure_mount_cad(
    *,
    grating_size_mm: float,
    grating_thickness_mm: float,
    params: GratingFlexureMountParams,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Flexure grating mount (CAD-grade, sketch-driven).

    Frame: x=v, y=w (world +z), z=u (optic normal). Origin at grating
    front surface centre.
    """
    if mfg is None:
        mfg = _load_manufacturing()

    half_size_mm = 0.5 * grating_size_mm
    hc = params.head_clearance_mm
    fc = params.foot_clearance_mm

    # -- u positions (along optic normal, +u = toward beam) --
    u_vertex = 0.0
    u_wall_rear = -(grating_thickness_mm + params.rear_wall_mm)
    slab_depth_mm = u_vertex - u_wall_rear
    pocket_depth_mm = grating_thickness_mm
    u_pocket_back = u_vertex - pocket_depth_mm

    # -- w positions (vertical, +w = up) --
    w_top = half_size_mm + hc
    w_bot = -(half_size_mm + fc)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm

    # -- v positions (dispersion axis) --
    v_half_mm = half_size_mm

    # -- pocket dimensions --
    w_opening_upper = half_size_mm + 0.5 * (mfg.assembly_clearance_mm + mfg.print_tolerance_mm + params.optic_clearance_mm)
    w_opening_lower = -w_opening_upper

    # -- contact bumps (TPU, captive in pocket wall) --
    contact_radius_mm = params.contact_radius_mm
    contact_offset_mm = params.contact_offset_mm
    v_contact_mm = 0.5 * params.contact_separation_mm
    bump_center_w = w_opening_upper + contact_radius_mm - contact_offset_mm
    w_bump_lower = -bump_center_w
    w_bump_upper = bump_center_w

    # -- foot --
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    w_foot_mid = w_foot_bot + 0.5 * params.foot_thickness_mm

    # -- derived --
    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm
    slab_total_u_mm = u_foot_front - u_wall_rear
    full_height_mm = w_top - w_foot_bot
    w_center = 0.5 * (w_top + w_foot_bot)

    # -- roll flexure u-depth --
    roll_u_depth = slab_depth_mm
    u_roll_actuate = 0.5 * (u_wall_rear + u_vertex)

    # -- foot geometry --
    v_bolt = params.foot_bolt_spacing_mm
    u_pusher = 0.0
    u_front_bolt = params.front_bolt_offset_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    u_setscrew = -0.5 * grating_thickness_mm

    # -- named planes --
    rear_plane = Plane(origin=(0, 0, u_wall_rear),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    pocket_plane = Plane(origin=(0, 0, u_pocket_back),
                         x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    foot_profile_plane = Plane(origin=(0, w_foot_bot, 0),
                               x_dir=(-1, 0, 0), z_dir=(0, 1, 0))
    tilted_surface_plane = Plane(
        origin=(0, w_foot_bot, u_wall_rear),
        x_dir=(1, 0, 0),
        z_dir=(0, math.cos(math.radians(params.trim_angle_deg)),
               -math.sin(math.radians(params.trim_angle_deg))),
    )

    with BuildPart() as mount:
        # 1. Tombstone stock: sketch rear cross-section, extrude forward.
        with BuildSketch(rear_plane) as stock_sk:
            with Locations(Pos(0, w_center)):
                Rectangle(2 * v_half_mm, full_height_mm)
        extrude(stock_sk.sketch, amount=slab_total_u_mm)

        # 2. Pocket cut: rectangle plus contact bump pockets.
        pocket_w_mm = 2 * half_size_mm
        pocket_h_mm = w_opening_upper - w_opening_lower
        with BuildSketch(pocket_plane) as pocket_sk:
            with Locations(Pos(0, 0.5 * (w_opening_upper + w_opening_lower))):
                Rectangle(pocket_w_mm, pocket_h_mm)
            with Locations(Pos(-v_contact_mm, w_bump_lower),
                           Pos(+v_contact_mm, w_bump_lower),
                           Pos(0, w_bump_upper)):
                Circle(contact_radius_mm)
        extrude(pocket_sk.sketch, amount=pocket_depth_mm, mode=Mode.SUBTRACT)

        # Setscrew insert bore from top (blind — 0.5mm floor above pocket).
        ss_bore_depth_mm = (w_top - w_opening_upper) - _SETSCREW_FLOOR_MM
        top_face = mount.faces().sort_by(Axis.Y)[-1]
        u_mid = u_wall_rear + 0.5 * slab_total_u_mm
        with BuildSketch(top_face) as ss_sk:
            with Locations(Pos(0, -(u_setscrew - u_mid))):
                Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(ss_sk.sketch, amount=-ss_bore_depth_mm, mode=Mode.SUBTRACT)

        # 3. Face mill (sketched on body front face).
        front_face = mount.faces().sort_by(Axis.Z)[-1]
        mill_depth_mm = u_foot_front - u_vertex
        w_slab_center = 0.5 * (w_top + w_shelf_bot)
        with BuildSketch(front_face) as mill_sk:
            with Locations(Pos(0, w_slab_center - w_center)):
                Rectangle(2 * v_half_mm, w_top - w_shelf_bot)
        extrude(mill_sk.sketch, amount=-mill_depth_mm, mode=Mode.SUBTRACT)

        # 4. Foot screw pilot holes (2 forward + 1 aft).
        foot_pilot_dia_mm = mfg.bolt_dims[params.foot_bolt_thread]["tap_drill_dia_mm"]
        foot_bottom_face = (mount.faces()
                            .filter_by(Axis.Y).sort_by(Axis.Y)[0])
        for bv, bu in [(+v_bolt, u_front_bolt), (-v_bolt, u_front_bolt),
                        (0, u_bolt)]:
            with BuildSketch(foot_bottom_face) as ins_sk:
                with Locations(Pos(bv, bu - u_mid)):
                    Circle(radius=0.5 * foot_pilot_dia_mm)
            extrude(ins_sk.sketch, amount=-params.foot_thickness_mm,
                    mode=Mode.SUBTRACT)
        # Pusher insert bore (through foot).
        with BuildSketch(foot_bottom_face) as push_sk:
            with Locations(Pos(0, u_pusher - u_mid)):
                Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(push_sk.sketch, amount=-params.foot_thickness_mm,
                mode=Mode.SUBTRACT)

        # 5. Shape the foot (sketched on body bottom face).
        #    Single wide tongue spanning both forward bolt positions.
        tongue_width_mm = 2 * v_bolt + boss_width_mm
        foot_face = (mount.faces()
                     .filter_by(Axis.Y).sort_by(Axis.Y)[0])
        foot_u_mid = foot_face.center().Z
        with BuildSketch(foot_face) as foot_sk:
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_vertex)
                               - foot_u_mid)):
                Rectangle(2 * v_half_mm, slab_depth_mm)
            tongue_u_len = u_front_bolt + 0.5 * boss_width_mm - u_wall_rear
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_front_bolt
                               + 0.5 * boss_width_mm) - foot_u_mid)):
                Rectangle(tongue_width_mm, tongue_u_len)
            fillet_verts = [
                v for v in foot_sk.vertices()
                if (abs(v.Y - (u_vertex - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
                or (abs(v.Y - (u_front_bolt + 0.5 * boss_width_mm
                               - foot_u_mid)) < 0.1
                    and abs(abs(v.X) - 0.5 * tongue_width_mm) < 0.1)
            ]
            if fillet_verts:
                try:
                    fillet2d(fillet_verts, radius=mfg.fillet_radius_mm)
                except Exception:
                    pass
        extrude(foot_sk.sketch, amount=-(w_top - w_foot_bot),
                mode=Mode.INTERSECT)

        # 6. Flexure relief: sketch on flexure front face, extrude forward.
        u_flexure_front = u_wall_rear + params.pitch_flexure_thickness_mm
        flexure_plane = Plane(origin=(0, 0, u_flexure_front),
                              x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildSketch(flexure_plane) as flex_sk:
            w_flexure_mid = w_shelf_bot - 0.5 * params.pitch_flexure_gap_mm
            with Locations(Pos(0, w_flexure_mid)):
                Rectangle(2 * v_half_mm, params.pitch_flexure_gap_mm)
        extrude(flex_sk.sketch, until=Until.LAST, mode=Mode.SUBTRACT)

        # 7. Trim foot bottom: wedge triangle in u-w plane, extrude in v.
        trim_u_end = u_foot_front + boss_width_mm
        trim_u_span = trim_u_end - u_wall_rear
        wedge_rise = trim_u_span * math.tan(
            math.radians(params.trim_angle_deg))
        uw_plane = Plane(origin=(0, w_foot_bot, 0),
                         x_dir=(0, 0, 1), z_dir=(-1, 0, 0))
        with BuildSketch(uw_plane) as trim_sk:
            Polygon(
                (u_wall_rear, 0),
                (trim_u_end, 0),
                (trim_u_end, wedge_rise),
                align=None,
            )
        extrude(trim_sk.sketch, until=Until.LAST, both=True,
                mode=Mode.SUBTRACT)

        # 7b. Roll flexure blade cuts.
        if params.roll_flexure_height_mm > 0 and params.roll_blade_thickness_mm > 0:
            blade_t = params.roll_blade_thickness_mm
            v_insert = params.roll_insert_spacing_mm
            v_anchor = params.roll_blade_spacing_mm
            v_ped_inner = v_anchor + params.roll_pedestal_gap_mm
            ped_w = half_size_mm - v_ped_inner
            v_top_blade = v_anchor * (w_bot / w_roll_bot)

            dv = v_top_blade - v_anchor
            dw = w_bot - w_roll_bot
            blade_len = math.hypot(dv, dw)
            nv = dw / blade_len
            nw = -dv / blade_len
            ht = 0.5 * blade_t

            w_bot_ext = w_roll_bot * half_size_mm / v_anchor
            v_bot_ext = half_size_mm
            v_top_ext = 0.0
            w_top_ext = 0.0
            blade_plus = [
                (v_bot_ext + ht * nv, w_bot_ext + ht * nw),
                (v_bot_ext - ht * nv, w_bot_ext - ht * nw),
                (v_top_ext - ht * nv, w_top_ext - ht * nw),
                (v_top_ext + ht * nv, w_top_ext + ht * nw),
            ]
            blade_minus = list(reversed([(-v, w) for v, w in blade_plus]))

            w_ped_top = w_bot
            w_ped_bot = w_roll_bot + 1.0
            gap = params.roll_pedestal_gap_mm
            total_off = ht + gap

            def _ped_inner_v(w_at):
                t = (w_at - w_roll_bot - total_off * nw) / dw
                return v_anchor + total_off * nv + t * dv

            v_inner_top = _ped_inner_v(w_ped_top)
            v_inner_bot = _ped_inner_v(w_ped_bot)

            w_ctr_top = w_bot - 1.0
            w_ctr_bot = w_roll_bot

            def _ctr_wall_v(w_at):
                t = (w_at - w_roll_bot + total_off * nw) / dw
                return v_anchor - total_off * nv + t * dv

            v_ctr_top = _ctr_wall_v(w_ctr_top)
            v_ctr_bot = _ctr_wall_v(w_ctr_bot)

            roll_cut_plane = Plane(origin=(0, 0, u_wall_rear),
                                   x_dir=(1, 0, 0), z_dir=(0, 0, 1))
            w_roll_mid = 0.5 * (w_roll_bot + w_bot)
            with BuildSketch(roll_cut_plane) as roll_sk:
                with Locations(Pos(0, w_roll_mid)):
                    Rectangle(2 * half_size_mm,
                              params.roll_flexure_height_mm)
                Polygon(*blade_plus, align=None, mode=Mode.SUBTRACT)
                Polygon(*blade_minus, align=None, mode=Mode.SUBTRACT)
                for sign in (+1, -1):
                    Polygon(
                        (sign * half_size_mm, w_ped_top),
                        (sign * v_inner_top, w_ped_top),
                        (sign * v_inner_bot, w_ped_bot),
                        (sign * half_size_mm, w_ped_bot),
                        align=None, mode=Mode.SUBTRACT)
                Polygon(
                    (+v_ctr_top, w_ctr_top),
                    (-v_ctr_top, w_ctr_top),
                    (-v_ctr_bot, w_ctr_bot),
                    (+v_ctr_bot, w_ctr_bot),
                    align=None, mode=Mode.SUBTRACT)
            extrude(roll_sk.sketch, amount=roll_u_depth,
                    mode=Mode.SUBTRACT)

            # Roll insert bores — from foot bottom through foot + gap + shelf.
            roll_bore_depth = (params.foot_thickness_mm
                               + params.pitch_flexure_gap_mm
                               + params.pusher_shelf_mm)
            roll_ss_plane = Plane(
                origin=(0, w_foot_bot, 0),
                x_dir=(1, 0, 0), z_dir=(0, 1, 0))
            for v_ss in [+v_insert, -v_insert]:
                with BuildSketch(roll_ss_plane) as roll_ss_sk:
                    with Locations(Pos(v_ss, -u_roll_actuate)):
                        Circle(radius=0.5 * mfg.insert_bore_dia_mm)
                extrude(roll_ss_sk.sketch,
                        amount=roll_bore_depth,
                        mode=Mode.SUBTRACT)
            chamfer_top_r_roll = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
            chamfer_bot_r_roll = 0.5 * mfg.insert_bore_dia_mm
            for v_ss in [+v_insert, -v_insert]:
                with Locations(Pos(v_ss,
                                   w_foot_bot + 0.5 * mfg.insert_chamfer_mm,
                                   u_roll_actuate)):
                    Cone(bottom_radius=chamfer_bot_r_roll,
                         top_radius=chamfer_top_r_roll,
                         height=mfg.insert_chamfer_mm,
                     rotation=(90, 0, 0),
                     mode=Mode.SUBTRACT)

        # 8. Chamfers on insert holes.
        chamfer_top_r = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
        chamfer_bot_r = 0.5 * mfg.insert_bore_dia_mm
        # Setscrew chamfer from top.
        with Locations(Pos(0, w_top - 0.5 * mfg.insert_chamfer_mm, u_setscrew)):
            Cone(bottom_radius=chamfer_top_r, top_radius=chamfer_bot_r,
                 height=mfg.insert_chamfer_mm, rotation=(90, 0, 0),
                 mode=Mode.SUBTRACT)
        # Pusher insert chamfer on the tilted surface.
        tilted_fwd = tilted_surface_plane.y_dir * -1
        tilted_n = tilted_surface_plane.z_dir
        chamfer_rot = (90 - params.trim_angle_deg, 0, 0)
        for cv, cu in [(0, u_pusher)]:
            surf = (tilted_surface_plane.origin
                    + tilted_surface_plane.x_dir * cv
                    + tilted_fwd * (cu - u_wall_rear))
            ctr = surf + tilted_n * (0.5 * mfg.insert_chamfer_mm)
            with Locations(Pos(ctr.X, ctr.Y, ctr.Z)):
                Cone(bottom_radius=chamfer_bot_r,
                     top_radius=chamfer_top_r,
                     height=mfg.insert_chamfer_mm,
                     rotation=chamfer_rot, mode=Mode.SUBTRACT)


    # Pusher shelf: semicircular lip at w_shelf_bot.
    _shelf_plane = Plane(
        origin=(0, w_shelf_bot, u_pusher),
        x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    with BuildPart() as _shelf_bp:
        with BuildSketch(_shelf_plane):
            Circle(0.5 * boss_width_mm)
        extrude(amount=params.pusher_shelf_mm)
    result = mount.part.fuse(_shelf_bp.part)
    _shelf_fillet_r = min(mfg.fillet_radius_mm, 0.49 * params.pusher_shelf_mm)
    _shelf_r = 0.5 * boss_width_mm
    _fillet_edges = [
        e for e in result.edges()
        if abs(e.center().Z) < 0.1
        and w_shelf_bot < e.center().Y < w_shelf_bot + params.pusher_shelf_mm
        and abs(math.hypot(e.center().X, e.center().Z - u_pusher) - _shelf_r) < 0.5
    ]
    if _fillet_edges and _shelf_fillet_r > 0.1:
        try:
            result = result.fillet(_shelf_fillet_r, _fillet_edges)
        except Exception:
            pass
    result = Part(result.wrapped)
    result.color = _MOUNT_COLOR
    result.label = "flexure_mount"

    _bump_color = Color(0.30, 0.60, 0.85)
    bump_positions = [
        (-v_contact_mm, w_bump_lower),
        (+v_contact_mm, w_bump_lower),
        (0, w_bump_upper),
    ]
    bump_parts = []
    for bv, bw in bump_positions:
        bump = Cylinder(contact_radius_mm, pocket_depth_mm,
                        align=(Align.CENTER, Align.CENTER, Align.MIN))
        bump = bump.translate((bv, bw, u_pocket_back))
        bump.color = _bump_color
        bump_parts.append(bump)
    from build123d import Compound as _Cmp
    bumps = _Cmp(children=bump_parts)
    bumps.label = "contact_bumps_tpu"
    result._contact_bumps = bumps

    return result



_VENDOR_STEP_BY_PART = {
    "HASMA":         "data/step/HASMA.step",
    "94459A110":     "data/step/94459A110_Heat-Set Inserts for Plastic.STEP",
    "92605A043":     "data/step/92605A043_Stainless Steel Flat-Tip Set Screw.STEP",
    "92605A044":     "data/step/92605A044_Stainless Steel Flat-Tip Set Screw.STEP",
    "92605A912":     "data/step/92605A912_Stainless Steel Flat-Tip Set Screw.STEP",
    "91771A108":     "data/step/91771A108_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
    "91771A109":     "data/step/91771A109_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
    "91771A194":     "data/step/91771A194_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
    "99461A915":     "data/step/99461A915_Phillips Rounded Head Thread-Forming Screws.STEP",
    "TCD1304":       "data/step/TCD1304_SPI_Rev2EB.step",
    "Controller_T4_R3EB": "data/step/Controller_T4_R3EB.step",
}

# ── Flat head screw catalogue (imperial Phillips countersunk) ───────────────
# Dimensions from McMaster vendor STEP bounding boxes.
# head_dia_mm: countersink OD (= bbox XY), shaft_dia_mm: thread major dia,
# total_length_mm: tip-to-head top (= bbox Z), countersink_angle_deg: 82°.
_FLAT_HEAD_SCREW_CATALOG: dict[str, dict[str, float]] = {
    # #4-40 × 3/8" — head_height from vendor STEP (cone + flat rim)
    "91771A108": dict(
        head_dia_mm=5.385, shaft_dia_mm=2.845,
        total_length_mm=9.525, head_height_mm=1.702,
    ),
    # #4-40 × 7/16"
    "91771A109": dict(
        head_dia_mm=5.385, shaft_dia_mm=2.845,
        total_length_mm=11.113, head_height_mm=1.702,
    ),
    # #8-32 × 1/2"
    "91771A194": dict(
        head_dia_mm=7.925, shaft_dia_mm=4.166,
        total_length_mm=12.700, head_height_mm=2.540,
    ),
}


def _procedural_hasma() -> Part:
    """Procedural SMA-905 bulkhead adapter (hex nut + threaded cylinder).

    Oriented as vendor STEP: cylinder axis along +X, fiber endface at X=0,
    body extends to X = -length.

    Reads length_mm, boundary_mm, bore_radius_mm, and hex_half_mm from
    the BOM [slits.mount] section.  boundary_mm positions the hex inner
    face relative to the endface — this is the critical dimension that
    sets the fiber endface position relative to the housing wall.

    The thread OD (1/4-36 UNS = 6.35mm) and hex nut thickness (1.98mm)
    are SMA-905 standard dimensions not carried in the BOM.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    slit_mount = bom["slits"]["mount"]

    length_mm = float(slit_mount["length_mm"])
    boundary_mm = float(slit_mount["boundary_mm"])
    bore_r_mm = 1.8            # ferrule bore radius (fiber + ferrule clearance)
    thread_od_mm = 6.35        # 1/4-36 UNS major diameter (SMA-905 standard)
    hex_af_mm = 8.0            # hex across-flats (SMA-905 standard)
    front_thread_mm = boundary_mm
    hex_len_mm = 1.98          # SMA-905 hex nut thickness
    rear_thread_mm = length_mm - front_thread_mm - hex_len_mm

    with BuildPart() as bp:
        # All sketch planes normal to X axis; origin X shifts along axis.
        # Front thread: X = 0 to X = -front_thread_mm
        front_plane = Plane(origin=(0, 0, 0),
                            x_dir=(0, 1, 0), z_dir=(-1, 0, 0))
        with BuildSketch(front_plane):
            Circle(radius=0.5 * thread_od_mm)
        extrude(amount=front_thread_mm)

        # Hex nut: X = -front_thread_mm to X = -(front_thread+hex)
        hex_plane = Plane(origin=(-front_thread_mm, 0, 0),
                          x_dir=(0, 1, 0), z_dir=(-1, 0, 0))
        with BuildSketch(hex_plane):
            from build123d import RegularPolygon
            RegularPolygon(radius=0.5 * hex_af_mm / math.cos(math.radians(30)),
                           side_count=6, rotation=30)
        extrude(amount=hex_len_mm)

        # Rear thread: X = -(front_thread+hex) to X = -length
        rear_plane = Plane(origin=(-(front_thread_mm + hex_len_mm), 0, 0),
                           x_dir=(0, 1, 0), z_dir=(-1, 0, 0))
        with BuildSketch(rear_plane):
            Circle(radius=0.5 * thread_od_mm)
        extrude(amount=rear_thread_mm)

        # Central bore through entire length (along -X)
        bore_plane = Plane(origin=(0, 0, 0),
                           x_dir=(0, 1, 0), z_dir=(-1, 0, 0))
        with BuildSketch(bore_plane):
            Circle(radius=bore_r_mm)
        extrude(amount=length_mm, mode=Mode.SUBTRACT)

    bp.part.label = "HASMA_procedural"
    return bp.part


def _procedural_heat_set_insert() -> Part:
    """Procedural M2 brass heat-set insert (94459A110).

    Flanged cylinder with internal M2 tapped bore.  Reads dimensions
    from the BOM [manufacturing.insert] section.

    Axis along +Y, flange at max Y.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    ins = bom["manufacturing"]["insert"]

    body_dia_mm = float(ins["insert_bore_dia_mm"])  # knurl OD ≈ bore dia
    length_mm = float(ins["insert_length_mm"])
    flange_dia_mm = float(ins["insert_flange_dia_mm"])
    flange_h_mm = 0.3         # thin flange disc at top
    thread = ins.get("insert_thread", "M2")
    bore_dia_mm = float(bom["manufacturing"]["bolts"][thread]["tap_drill_dia_mm"])

    with BuildPart() as bp:
        # Main body cylinder: axis along Y, from Y=0 to Y=length
        Cylinder(
            radius=0.5 * body_dia_mm, height=length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(-90, 0, 0),
        )
        # Flange disc at top (max Y)
        with Locations(Pos(0, length_mm - 0.5 * flange_h_mm, 0)):
            Cylinder(
                radius=0.5 * flange_dia_mm, height=flange_h_mm,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                rotation=(-90, 0, 0),
            )
        # Internal tapped bore (through-hole)
        Cylinder(
            radius=0.5 * bore_dia_mm,
            height=length_mm + flange_h_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(-90, 0, 0),
            mode=Mode.SUBTRACT,
        )
    bp.part.label = "94459A110_procedural"
    return bp.part


def _procedural_pitch_set_screw() -> Part:
    """Procedural M2×5 flat-tip set screw (92605A044).

    Threaded cylinder with hex socket.  Reads thread diameter from the
    BOM [manufacturing.bolts.M2] section.

    Axis along +Y.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    major_dia_mm = float(bom["manufacturing"]["bolts"]["M2"]["thread_dia_mm"])
    length_mm = 5.0
    hex_socket_af_mm = 0.9   # M2 hex socket across-flats
    hex_depth_mm = 1.0       # socket depth from one end

    with BuildPart() as bp:
        Cylinder(
            radius=0.5 * major_dia_mm, height=length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(-90, 0, 0),
        )
        socket_plane = Plane(origin=(0, length_mm, 0),
                             x_dir=(1, 0, 0), z_dir=(0, -1, 0))
        with BuildSketch(socket_plane):
            from build123d import RegularPolygon
            RegularPolygon(
                radius=0.5 * hex_socket_af_mm / math.cos(math.radians(30)),
                side_count=6)
        extrude(amount=hex_depth_mm, mode=Mode.SUBTRACT)
    bp.part.label = "92605A044_procedural"
    return bp.part


def _procedural_retention_set_screw() -> Part:
    """Procedural M2×4 flat-tip set screw (92605A043).

    Threaded cylinder with hex socket.  Reads thread diameter from the
    BOM [manufacturing.bolts.M2] section.

    Axis along +X.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    major_dia_mm = float(bom["manufacturing"]["bolts"]["M2"]["thread_dia_mm"])
    length_mm = 4.0
    hex_socket_af_mm = 0.9
    hex_depth_mm = 1.0

    with BuildPart() as bp:
        Cylinder(
            radius=0.5 * major_dia_mm, height=length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(0, 90, 0),
        )
        socket_plane = Plane(origin=(length_mm, 0, 0),
                             x_dir=(0, 1, 0), z_dir=(-1, 0, 0))
        with BuildSketch(socket_plane):
            from build123d import RegularPolygon
            RegularPolygon(
                radius=0.5 * hex_socket_af_mm / math.cos(math.radians(30)),
                side_count=6)
        extrude(amount=hex_depth_mm, mode=Mode.SUBTRACT)
    bp.part.label = "92605A043_procedural"
    return bp.part


def _procedural_roll_set_screw() -> Part:
    """Procedural M2×8 flat-tip set screw (92605A912).

    Threaded cylinder with hex socket.  Reads thread diameter from the
    BOM [manufacturing.bolts.M2] section.

    Axis along +Y.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    major_dia_mm = float(bom["manufacturing"]["bolts"]["M2"]["thread_dia_mm"])
    length_mm = 8.0
    hex_socket_af_mm = 0.9
    hex_depth_mm = 1.0

    with BuildPart() as bp:
        Cylinder(
            radius=0.5 * major_dia_mm, height=length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(-90, 0, 0),
        )
        socket_plane = Plane(origin=(0, length_mm, 0),
                             x_dir=(1, 0, 0), z_dir=(0, -1, 0))
        with BuildSketch(socket_plane):
            from build123d import RegularPolygon
            RegularPolygon(
                radius=0.5 * hex_socket_af_mm / math.cos(math.radians(30)),
                side_count=6)
        extrude(amount=hex_depth_mm, mode=Mode.SUBTRACT)
    bp.part.label = "92605A912_procedural"
    return bp.part


def _procedural_pan_head_screw() -> Part:
    """Procedural M2×6 Phillips rounded head thread-forming screw (99461A915).

    Cylindrical head + cylindrical shaft.  Axis along +Z, head at top.
    Dimensions from vendor STEP bounding box.
    """
    head_dia_mm = 3.5
    head_height_mm = 1.4
    shaft_dia_mm = 2.0
    shaft_length_mm = 6.0

    slot_width_mm = 0.6
    slot_length_mm = 2.6
    slot_depth_mm = 0.8

    with BuildPart() as bp:
        Cylinder(
            radius=0.5 * shaft_dia_mm, height=shaft_length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        Cylinder(
            radius=0.5 * head_dia_mm, height=head_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        with BuildSketch(Plane.XY.offset(head_height_mm)):
            Rectangle(slot_length_mm, slot_width_mm)
            Rectangle(slot_width_mm, slot_length_mm)
        extrude(amount=-slot_depth_mm, mode=Mode.SUBTRACT)
    bp.part.label = "99461A915_procedural"
    return bp.part


def _procedural_flat_head_screw(
    head_dia_mm: float,
    shaft_dia_mm: float,
    total_length_mm: float,
    head_height_mm: float,
) -> Part:
    """Procedural Phillips flat head (countersunk) screw.

    Cone frustum head + cylindrical shaft.  A Phillips cross recess is
    approximated by a cylindrical bore in the head to match vendor
    STEP volume.  Head sits at max-Y end of the shaft (vendor STEP
    convention: axis along Y, head at max Y).

    head_height_mm is measured from the vendor STEP (cone + flat rim).
    """
    shaft_length_mm = total_length_mm - head_height_mm
    # Phillips recess: approximate as cylindrical bore in cone head
    recess_dia_mm = 0.5 * head_dia_mm   # recess width ~half head dia
    recess_depth_mm = 0.6 * head_height_mm

    with BuildPart() as bp:
        # Shaft: from Y=0 to Y=shaft_length
        Cylinder(
            radius=0.5 * shaft_dia_mm, height=shaft_length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            rotation=(-90, 0, 0),
        )
        # Countersunk head: cone from shaft_dia at bottom to head_dia at top
        with Locations(Pos(0, shaft_length_mm, 0)):
            Cone(
                bottom_radius=0.5 * shaft_dia_mm,
                top_radius=0.5 * head_dia_mm,
                height=head_height_mm,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                rotation=(-90, 0, 0),
            )
        # Phillips recess bore from top of head
        recess_plane = Plane(
            origin=(0, total_length_mm, 0),
            x_dir=(1, 0, 0), z_dir=(0, -1, 0))
        with BuildSketch(recess_plane):
            Circle(radius=0.5 * recess_dia_mm)
        extrude(amount=recess_depth_mm, mode=Mode.SUBTRACT)
    bp.part.label = "flat_head_screw_procedural"
    return bp.part


# Dispatch table: part number → procedural builder (no arguments).
_PROCEDURAL_BUILDERS: dict[str, callable] = {
    "HASMA":     _procedural_hasma,
    "94459A110": _procedural_heat_set_insert,
    "92605A044": _procedural_pitch_set_screw,
    "92605A043": _procedural_retention_set_screw,
    "92605A912": _procedural_roll_set_screw,
    "99461A915": _procedural_pan_head_screw,
}

# Flat head screws are parameterised — add lambda wrappers from the catalog.
for _pn, _dims in _FLAT_HEAD_SCREW_CATALOG.items():
    _PROCEDURAL_BUILDERS[_pn] = (
        lambda dims=_dims: _procedural_flat_head_screw(**dims)
    )


def _load_or_procedural(part_number: str) -> Part:
    """Try loading a vendor STEP; fall back to a procedural solid.

    Attempts ``import_step`` of the path registered in ``_VENDOR_STEP_BY_PART``.
    If the file is missing (``FileNotFoundError``) or the part is not in the
    dict (``KeyError``), builds a simplified procedural solid instead.

    Only hardware parts (fasteners, adapter) have procedural fallbacks.
    Optics and the detector board are handled by their own loader functions.
    """
    from build123d import import_step

    path = _VENDOR_STEP_BY_PART.get(part_number)
    if path is not None:
        try:
            return import_step(path)
        except FileNotFoundError:
            pass

    builder = _PROCEDURAL_BUILDERS.get(part_number)
    if builder is not None:
        return builder()

    raise FileNotFoundError(
        f"No vendor STEP and no procedural fallback for part {part_number!r}")


def _load_vendor_grating_at_front(part_number: str):
    """Import a vendor ruled grating STEP and translate so the
    ruled front face lands at local z=0 (= element.position).

    After translation: xy-centre at origin, front face at z=0, grating
    body at z in [-thickness, 0].

    Falls back to a procedural box when no vendor STEP is registered.
    """
    from build123d import import_step
    path = _VENDOR_STEP_BY_PART.get(part_number)
    if path is None:
        size_mm, thickness_mm, _ = _lookup_grating_in_bom(part_number)
        return _build_generic_grating(size_mm, thickness_mm)
    grating = import_step(path)
    faces = grating.faces()
    if not faces:
        raise RuntimeError(f"{part_number}: no faces in STEP")
    front = max(
        (f for f in faces if f.normal_at(f.center()).Z > 0.99),
        key=lambda f: f.area,
    )
    c = front.center()
    return grating.translate((-c.X, -c.Y, -c.Z))


def _load_vendor_fiber_adapter():
    """Import the HASMA SMA-905 bulkhead STEP and orient so the
    fiber endface (slit plane) is at local z=0 with the cylinder
    axis along +z. Body extends in -z (behind the housing wall).

    Vendor STEP convention:
      - Cylinder axis along STEP +X.
      - Fiber endface (flat) at STEP X=0 (closest to origin).
      - Body extends in STEP -X to X=-9.652.
      - Hex body centred at STEP (Y=35.98, Z=-18.63).

    Transform: recentre on cylinder axis, rotate STEP +X → local +z.

    Falls back to a procedural hex-prism + bore when no vendor STEP
    is available.
    """
    from build123d import Axis, import_step
    path = _VENDOR_STEP_BY_PART.get("HASMA")
    if path is not None:
        try:
            hasma = import_step(path)
            # Recentre on cylinder axis (Y/Z offset from STEP origin).
            hasma = hasma.translate((0.0, -35.977, 18.633))
            # Rotate STEP +X → local -z (body behind wall): rotate -90° about Y.
            hasma = hasma.rotate(Axis.Y, -90.0)
            return hasma
        except FileNotFoundError:
            pass

    # Procedural fallback: already oriented with axis along +X, endface at X=0.
    hasma = _procedural_hasma()
    # Rotate STEP +X → local -z (body behind wall): rotate -90° about Y.
    hasma = hasma.rotate(Axis.Y, -90.0)
    return hasma


def _load_vendor_detector():
    """Import the TCD1304 SPI detector board STEP and position
    so the sensor active surface lands at local z=0.

    Vendor STEP: drmcnelson TCD1304_SPI_Rev2EB, KiCad STEP export.
    Key Z planes (assembly coords, +Z toward PCB top):
      z=+4.685  SMD component tops (PCB top surface)
      z= 0.000  PCB bottom edge
      z=-3.085  DIP socket bottom
      z=-3.755  sensor ceramic back (toward PCB)
      z=-6.805  sensor glass window (away from PCB)

    Body thickness 3.05mm. Die is 1.7mm behind the front glass
    (TCD1304 datasheet), so die sits at -6.805 + 1.7 = -5.105.
    After translation the glass protrudes 1.7mm past the focal
    plane toward M2 (+Z after flip), which is physically correct.

    The exit_slit element normal points TOWARD M2, so local +Z maps
    toward the beam. We flip 180° about X so the sensor faces +Z.

    Array orientation: pin 1 (notch) of the TCD1304 is at STEP +X
    (the end with the signal-processing chain: op-amp, ADC).
    ``place_in_scene_frame`` maps STEP +X to raysect local +x,
    which is the short-wavelength (blue) end of the spectrum.
    """
    from build123d import Axis, import_step
    _GLASS_FACE_Z = -6.805     # sensor glass window in vendor STEP
    _DIE_OFFSET_MM = 1.7       # datasheet: glass to active surface
    _SENSOR_Z_IN_STEP = _GLASS_FACE_Z + _DIE_OFFSET_MM  # -5.105
    path = _VENDOR_STEP_BY_PART["TCD1304"]
    det = import_step(path)
    det = det.translate((0, 0, -_SENSOR_Z_IN_STEP))  # sensor to z=0
    det = det.rotate(Axis.X, 180.0)  # flip so sensor faces +z (toward M2)
    return det


def _load_vendor_controller():
    """Import the Controller T4 R3EB STEP and orient for bottom pocket.

    Vendor STEP: drmcnelson Controller_T4_R3EB, KiCad STEP export.
    Z planes in vendor STEP:
      z=0.00   PCB bottom
      z=4.60   PCB top (solder side modeled as thick board)
      z=17.39  tallest component top

    Physical board is 1.6mm; vendor z=4.60 includes solder protrusion.
    After flip and shift: components face -Z, PCB top at z=+1.6.
    """
    from build123d import Axis, import_step
    _VENDOR_PCB_TOP_Z = 4.60
    _PHYSICAL_PCB_THICKNESS = 1.6
    path = _VENDOR_STEP_BY_PART["Controller_T4_R3EB"]
    ctrl = import_step(path)
    ctrl = ctrl.translate((0, 0, -(_VENDOR_PCB_TOP_Z - _PHYSICAL_PCB_THICKNESS)))
    ctrl = ctrl.rotate(Axis.X, 180.0)
    return ctrl


def _load_vendor_f_mirror_at_face(part_number: str, *,
                                  cylindrical_orientation: str | None = None):
    """Import a vendor flat mirror STEP and translate so the
    reflective +z face lands at local z=0 (= element.position).

    Vendor STEP convention (verified for PF05-03-G01 / PF07-03-G01 /
    PF10-03-G01 — same family):
      - Cylindrical substrate with two big Ø disc faces perpendicular to z.
      - Reflective face is the +z face (largest area, +Z normal).
      - Substrate body extends in −z behind the reflective face.

    After translation: reflective face at local z=0, substrate body
    at z ∈ [−thickness, 0].

    ``cylindrical_orientation`` overrides the BOM lookup when the caller
    knows which section the part belongs to.
    """
    from build123d import import_step
    path = _VENDOR_STEP_BY_PART.get(part_number)
    if path is None:
        focal_mm, diameter_mm, thickness_mm, _ = _lookup_mirror_in_bom(part_number)
        mt, co = _lookup_mirror_shape(part_number)
        if cylindrical_orientation is not None:
            co = cylindrical_orientation
        if mt == "flat":
            return _build_generic_flat_mirror(diameter_mm, thickness_mm)
        return _build_generic_concave_mirror(
            focal_mm, diameter_mm, thickness_mm,
            mirror_type=mt, cylindrical_orientation=co)
    mirror = import_step(path)
    faces = mirror.faces()
    if not faces:
        raise RuntimeError(f"{part_number}: no faces in STEP")
    # Pick the +z-normal face with the largest area — the reflective disc.
    front = max(
        (f for f in faces if f.normal_at(f.center()).Z > 0.99),
        key=lambda f: f.area,
    )
    return mirror.translate((0, 0, -front.center().Z))


def build_f_mirror_assembly(part_number: str, *,
                            cylindrical_orientation: str | None = None):
    """Return a `build123d.Compound` with the fold mirror + flexure
    mount, reflective face at local z=0 (= element position), normal
    along +z.

    ``cylindrical_orientation`` overrides the BOM lookup when the same
    part number appears in both m1_options and f1_options with different
    orientations (e.g. CCM254-050-G01 as tangential M1 vs sagittal F1).
    """
    from build123d import Compound

    mirror = _load_vendor_f_mirror_at_face(
        part_number, cylindrical_orientation=cylindrical_orientation)
    bbox = mirror.bounding_box()
    cx = 0.5 * (bbox.min.X + bbox.max.X)
    cy = 0.5 * (bbox.min.Y + bbox.max.Y)
    mirror = mirror.translate((-cx, -cy, 0))

    _, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    params = _flexure_params_from_bom(mount_dict)
    mount = build_mirror_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
    )

    mirror.label = part_number
    mirror.color = _OPTIC_COLOR
    mount.color = _MOUNT_COLOR
    children = [mount, mirror]

    # --- heat-set inserts + setscrews (same as round mirror assembly) ---
    from build123d import Axis
    mfg = _load_manufacturing()
    insert_flange_r = 0.5 * mfg.insert_flange_dia_mm
    optic_R = 0.5 * diameter_mm
    u_vertex = 0.0
    u_wall_rear = -thickness_mm - params.rear_wall_mm
    boss_width_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"] + params.bolt_safety_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    u_setscrew = 0.5 * (u_wall_rear + u_vertex)
    w_top = optic_R + params.head_clearance_mm
    w_bot = -(optic_R + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm

    def _trim_offset(u):
        return (u + insert_flange_r - u_wall_rear) * math.tan(
            math.radians(params.trim_angle_deg))

    raw_insert = _load_or_procedural("94459A110")
    ins_bottom = raw_insert.rotate(Axis.X, 180.0)
    ins_bottom = ins_bottom.translate((0, -ins_bottom.bounding_box().min.Y, 0))
    ins_top = raw_insert.translate((0, -raw_insert.bounding_box().max.Y, 0))

    children.append(ins_top.translate((0, w_top, u_setscrew)))

    front_bore_dia = diameter_mm + mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    bore_r = 0.5 * (front_bore_dia + params.optic_clearance_mm)
    bump_center_r = bore_r + params.contact_radius_mm - params.contact_offset_mm
    w_bump_top = bump_center_r + params.contact_radius_mm

    raw_ss = _load_or_procedural("92605A043")
    ss_top = raw_ss.rotate(Axis.Z, 90.0)
    ss_top = ss_top.translate((0, -ss_top.bounding_box().min.Y, 0))
    children.append(ss_top.translate((0, w_bump_top, u_setscrew)))

    off_pusher = _trim_offset(0.0)
    children.append(ins_bottom.translate((0, w_foot_bot + off_pusher, 0.0)))

    raw_pusher = _load_or_procedural("92605A044")
    pusher = raw_pusher.rotate(Axis.X, 180.0)
    pusher = pusher.translate((0, -pusher.bounding_box().max.Y, 0))
    children.append(pusher.translate((0, w_shelf_bot, 0.0)))

    if params.roll_flexure_height_mm > 0:
        u_roll_actuate = 0.5 * (u_wall_rear + 0.0)
        v_insert = params.roll_insert_spacing_mm
        w_ped_bot = w_roll_bot + 1.0
        raw_roll_ss = _load_or_procedural("92605A912")
        roll_ss = raw_roll_ss.rotate(Axis.X, 180.0)
        roll_ss = roll_ss.translate((0, -roll_ss.bounding_box().max.Y, 0))
        for sign in (+1, -1):
            children.append(ins_bottom.translate((
                sign * v_insert, w_foot_bot + off_pusher, u_roll_actuate)))
            children.append(roll_ss.translate((
                sign * v_insert, w_ped_bot, u_roll_actuate)))

    asm = Compound(children=children)
    asm.label = f"assembly_{part_number}"
    return asm


def _load_vendor_mirror_at_vertex(part_number: str, thickness_mm: float):
    """Import a vendor concave mirror STEP and translate so the optical
    vertex lands at local z=0 (ready to drop into a mount at u=0).

    Vendor convention (verified for CM127-050-G01 and CM254-100-G01):
      thickness_mm is the distance from the back face to the optical
      vertex. The STEP's back face is at bbox.min.Z, so the vertex
      sits at bbox.min.Z + thickness_mm.
    """
    from build123d import import_step
    path = _VENDOR_STEP_BY_PART.get(part_number)
    if path is not None:
        mirror = import_step(path)
        bbox = mirror.bounding_box()
        vertex_z = bbox.min.Z + thickness_mm
        return mirror.translate((0, 0, -vertex_z))
    focal_mm, diameter_mm, ct_mm, _ = _lookup_mirror_in_bom(part_number)
    mt, co = _lookup_mirror_shape(part_number)
    return _build_generic_concave_mirror(focal_mm, diameter_mm, ct_mm,
                                         mirror_type=mt,
                                         cylindrical_orientation=co)


def _build_generic_concave_mirror(focal_mm, diameter_mm, center_thickness_mm,
                                  mirror_type="spherical",
                                  cylindrical_orientation=None):
    """Dispatch to the appropriate concave mirror builder."""
    if mirror_type == "cylindrical":
        return _build_generic_cylindrical_mirror(
            focal_mm, diameter_mm, center_thickness_mm,
            cylindrical_orientation or "tangential")
    return _build_generic_spherical_mirror(
        focal_mm, diameter_mm, center_thickness_mm)


def _build_generic_spherical_mirror(focal_mm, diameter_mm, center_thickness_mm):
    """Concave spherical mirror. Optical vertex at z=0, dish facing +z."""
    R = 2.0 * focal_mm
    half_d = 0.5 * diameter_mm
    sag = R - math.sqrt(R * R - half_d * half_d)
    blank = Cylinder(half_d, center_thickness_mm + sag + 1.0,
                     align=(Align.CENTER, Align.CENTER, Align.MIN))
    blank = blank.translate((0, 0, -center_thickness_mm))
    cut = Sphere(R).translate((0, 0, R))
    result = blank - cut
    return Part.cast(result.solids()[0].wrapped)


def _build_generic_cylindrical_mirror(focal_mm, diameter_mm,
                                      center_thickness_mm, orientation):
    """Concave cylindrical mirror. Optical vertex at z=0, dish facing +z.

    orientation="tangential": curvature in xz plane (cut cylinder axis along y).
    orientation="sagittal":   curvature in yz plane (cut cylinder axis along x).
    """
    R = 2.0 * focal_mm
    half_d = 0.5 * diameter_mm
    sag = R - math.sqrt(R * R - half_d * half_d)
    blank = Cylinder(half_d, center_thickness_mm + sag + 1.0,
                     align=(Align.CENTER, Align.CENTER, Align.MIN))
    blank = blank.translate((0, 0, -center_thickness_mm))
    cut_len = 2.0 * half_d + 2.0
    cut = Cylinder(R, cut_len)
    if orientation == "sagittal":
        cut = cut.rotate(Axis.Y, 90).translate((0, 0, R))
    else:
        cut = cut.rotate(Axis.X, 90).translate((0, 0, R))
    result = blank - cut
    return Part.cast(result.solids()[0].wrapped)


def _build_generic_flat_mirror(diameter_mm, thickness_mm):
    """Build a generic flat mirror disc. Front face at z=0, body at z<0."""
    half_d = 0.5 * diameter_mm
    with BuildPart() as p:
        Cylinder(half_d, thickness_mm,
                 align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part.translate((0, 0, -thickness_mm))


def _build_generic_grating(size_mm, thickness_mm):
    """Build a generic ruled grating as a rectangular box.

    Front (ruled) face at z=0, body at z in [-thickness, 0].
    XY centred on origin.
    """
    half_s = 0.5 * size_mm
    with BuildPart() as p:
        Box(size_mm, size_mm, thickness_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part.translate((0, 0, -thickness_mm))


def _build_generic_oap_mirror(focal_length_mm, diameter_mm,
                              center_thickness_mm,
                              off_axis_angle_deg=90.0,
                              n_profile_pts=100):
    """Build a procedural off-axis paraboloidal mirror.

    Optical centre at z=0, back face at z = -center_thickness_mm.
    The paraboloid surface faces +z (concave, toward incoming beam).

    Construction: revolve the parent paraboloid profile about its axis,
    intersect with a cylinder of the mirror aperture at the off-axis
    distance.
    """
    import numpy as np

    f = focal_length_mm
    r_mirror = 0.5 * diameter_mm
    off_axis = 2.0 * f * math.tan(math.radians(off_axis_angle_deg / 2.0))
    z_oc_parent = off_axis ** 2 / (4.0 * f)

    R_max = off_axis + r_mirror + 1.0
    r_vals = np.linspace(0, R_max, n_profile_pts)
    z_vals = r_vals ** 2 / (4.0 * f)

    pts = [(float(rv), float(zv)) for rv, zv in zip(r_vals, z_vals)]
    pts.append((float(R_max), -center_thickness_mm - 10.0))
    pts.append((0.0, -center_thickness_mm - 10.0))

    with BuildPart() as parab_solid:
        with BuildSketch(Plane.XZ):
            Polygon(*pts, align=None)
        revolve(axis=Axis.Z, revolution_arc=360)

    parab_body = parab_solid.part.translate((-off_axis, 0, -z_oc_parent))

    z_back = -center_thickness_mm
    z_far = ((off_axis + r_mirror) ** 2 / (4.0 * f)) - z_oc_parent
    z_top = max(z_far, 0) + 1.0
    cyl_height = z_top - z_back
    cyl = Cylinder(r_mirror, cyl_height,
                   align=(Align.CENTER, Align.CENTER, Align.MIN))
    cyl = cyl.translate((0, 0, z_back))

    return cyl & parab_body


def _load_vendor_oap_at_optical_centre(part_number: str, *, rotate_180: bool = False):
    """Import a vendor OAP mirror STEP and orient so the optical centre
    (chief-ray intersection on the paraboloid) sits at local z=0.

    This matches the convention used by ``_load_vendor_mirror_at_vertex``
    for spherical mirrors and by the raysect world builder's
    ``translate(0, 0, -z_optical_centre)`` shift.

    Vendor STEP convention (verified for MPD129-G01):
      - Cylinder axis along +Y; back face (with mounting holes) at Y=0.
      - Paraboloidal surface at the +Y end.
      - XZ plane is the circular cross-section.

    `rotate_180`: if True, rotate 180° about z after orienting. This
    flips the OAP's parent-axis direction, needed for M1 vs M2.

    After transform: optical centre at z=0, back face at z<0,
    paraboloid surface at z>0, XY centred on the cylinder axis.
    """
    from build123d import Axis, import_step
    path = _VENDOR_STEP_BY_PART.get(part_number)
    focal_mm, diameter_mm, center_thickness_mm, _ = _lookup_mirror_in_bom(part_number)
    if path is None:
        oap = _build_generic_oap_mirror(
            focal_mm, diameter_mm, center_thickness_mm,
            off_axis_angle_deg=90.0)
    else:
        oap = import_step(path)
        oap = oap.rotate(Axis.X, -90)
        oap = oap.rotate(Axis.X, 180)
        bb = oap.bounding_box()
        oap = oap.translate((0, 0, -bb.min.Z))
        oap = oap.translate((0, 0, -center_thickness_mm))

    if rotate_180:
        oap = oap.rotate(Axis.Z, 180)

    return oap


def _lookup_mirror_in_bom(part_number: str) -> tuple[float, float, float, dict]:
    """Find a mirror part in any mirror options table.

    Returns ``(focal_length_mm, diameter_mm, center_thickness_mm, mount_dict)``.
    Flat mirrors return ``focal_length_mm = inf``.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    for group in (
        bom["mirrors"]["m1_options"],
        bom["mirrors"]["m2_options"],
        bom["mirrors"].get("f1_options", {}),
        bom["mirrors"].get("f2_options", {}),
    ):
        if part_number in group:
            opt = group[part_number]
            focal = float(opt["focal_length_mm"]) if "focal_length_mm" in opt else float("inf")
            return (
                focal,
                float(opt["diameter_mm"]),
                float(opt["center_thickness_mm"]),
                dict(opt["mount"]),
            )
    raise KeyError(
        f"mirror part {part_number!r} not found in mirror options"
    )


def _lookup_mirror_shape(part_number: str) -> tuple[str, str | None]:
    """Return ``(mirror_type, cylindrical_orientation)`` for a mirror part."""
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    for group in (
        bom["mirrors"]["m1_options"],
        bom["mirrors"]["m2_options"],
        bom["mirrors"].get("f1_options", {}),
        bom["mirrors"].get("f2_options", {}),
    ):
        if part_number in group:
            opt = group[part_number]
            return (
                opt.get("mirror_type", "spherical"),
                opt.get("cylindrical_orientation"),
            )
    return ("spherical", None)


def _lookup_grating_in_bom(part_number: str) -> tuple[float, float, dict]:
    """Find a grating part in `[grating_options]`.

    Returns `(size_mm, center_thickness_mm, mount_dict)`.
    """
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    group = bom["grating_options"]
    if part_number in group:
        opt = group[part_number]
        return (
            float(opt["size_mm"]),
            float(opt["center_thickness_mm"]),
            dict(opt["mount"]),
        )
    raise KeyError(
        f"grating part {part_number!r} not found in [grating_options]"
    )




def _load_manufacturing() -> ManufacturingParams:
    """Load [manufacturing] from the active BOM TOML."""
    import tomllib
    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)
    m = bom["manufacturing"]
    ins = m["insert"]
    bolt_dims = {
        thread: {k: float(v) for k, v in dims.items()}
        for thread, dims in m["bolts"].items()
    }
    return ManufacturingParams(
        assembly_clearance_mm=float(m["assembly_clearance_mm"]),
        print_tolerance_mm=float(m["print_tolerance_mm"]),
        fillet_radius_mm=float(m["fillet_radius_mm"]),
        bolt_dims=bolt_dims,
        insert_bore_dia_mm=float(ins["insert_bore_dia_mm"]),
        insert_length_mm=float(ins["insert_length_mm"]),
        insert_min_material_mm=float(ins["insert_min_material_mm"]),
        insert_chamfer_mm=float(ins["insert_chamfer_mm"]),
        insert_flange_dia_mm=float(ins["insert_flange_dia_mm"]),
    )


def _flexure_params_from_bom(mount_dict: dict) -> RoundMirrorFlexureMountParams:
    """Build RoundMirrorFlexureMountParams from BOM mount dict. All fields explicit."""
    return RoundMirrorFlexureMountParams(
        optic_clearance_mm=float(mount_dict["optic_clearance_mm"]),
        shoulder_width_mm=float(mount_dict["shoulder_width_mm"]),
        channel_width_mm=float(mount_dict["channel_width_mm"]),
        channel_extension_mm=float(mount_dict["channel_extension_mm"]),
        wall_margin_mm=float(mount_dict["wall_margin_mm"]),
        head_clearance_mm=float(mount_dict["head_clearance_mm"]),
        foot_clearance_mm=float(mount_dict["foot_clearance_mm"]),
        pusher_shelf_mm=float(mount_dict["pusher_shelf_mm"]),
        rear_wall_mm=float(mount_dict["rear_wall_mm"]),
        foot_thickness_mm=float(mount_dict["foot_thickness_mm"]),
        foot_bolt_thread=str(mount_dict["foot_bolt_thread"]),
        foot_bolt_spacing_mm=float(mount_dict["foot_bolt_spacing_mm"]),
        bolt_safety_mm=float(mount_dict["bolt_safety_mm"]),
        front_bolt_offset_mm=float(mount_dict["front_bolt_offset_mm"]),
        pitch_flexure_thickness_mm=float(mount_dict["pitch_flexure_thickness_mm"]),
        pitch_flexure_gap_mm=float(mount_dict["pitch_flexure_gap_mm"]),
        trim_angle_deg=float(mount_dict["trim_angle_deg"]),
        roll_flexure_height_mm=float(mount_dict["roll_flexure_height_mm"]),
        roll_blade_thickness_mm=float(mount_dict["roll_blade_thickness_mm"]),
        roll_insert_spacing_mm=float(mount_dict["roll_insert_spacing_mm"]),
        roll_blade_spacing_mm=float(mount_dict["roll_blade_spacing_mm"]),
        roll_pedestal_gap_mm=float(mount_dict["roll_pedestal_gap_mm"]),
        contact_radius_mm=float(mount_dict["contact_radius_mm"]),
        contact_offset_mm=float(mount_dict["contact_offset_mm"]),
        contact_separation_mm=float(mount_dict["contact_separation_mm"]),
    )


def build_mirror_flexure_assembly(
    part_number: str,
    params: RoundMirrorFlexureMountParams | None = None,
):
    """Return a Compound of flexure mount + vendor mirror STEP."""
    from build123d import Compound

    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _flexure_params_from_bom(mount_dict)
    mount = build_mirror_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
    )
    mirror = _load_vendor_mirror_at_vertex(part_number, thickness_mm)
    mirror.label = part_number
    mirror.color = _OPTIC_COLOR
    mount.color = _MOUNT_COLOR
    asm = Compound(children=[mount, mirror])
    asm.label = f"assembly_{part_number}"
    return asm


def export_flexure_mount_step(
    part_number: str,
    output_path: Path | None = None,
    params: RoundMirrorFlexureMountParams | None = None,
) -> Path:
    """Export the flexure mount alone as STEP."""
    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _flexure_params_from_bom(mount_dict)
    mount = build_mirror_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
    )
    if output_path is None:
        output_path = Path(f"output/flexure_mount_{part_number}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(mount, str(output_path))
    return output_path


def export_flexure_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: RoundMirrorFlexureMountParams | None = None,
) -> Path:
    """Export flexure mount + mirror assembly as STEP."""
    assembly = build_mirror_flexure_assembly(part_number, params)
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(assembly, str(output_path))
    return output_path


def _oap_flexure_params_from_bom(mount_dict: dict) -> OAPMirrorFlexureMountParams:
    """Build OAPMirrorFlexureMountParams from BOM mount dict. All fields explicit."""
    return OAPMirrorFlexureMountParams(
        bolt_circle_radius_mm=float(mount_dict["bolt_circle_radius_mm"]),
        n_holes=int(mount_dict["n_holes"]),
        hole_phase_deg=float(mount_dict["hole_phase_deg"]),
        clearance_hole_dia_mm=float(mount_dict["clearance_hole_dia_mm"]),
        counterbore_dia_mm=float(mount_dict["counterbore_dia_mm"]),
        screw_part=str(mount_dict["screw_part"]),
        slab_thickness_mm=float(mount_dict["slab_thickness_mm"]),
        head_clearance_mm=float(mount_dict["head_clearance_mm"]),
        foot_clearance_mm=float(mount_dict["foot_clearance_mm"]),
        pusher_shelf_mm=float(mount_dict["pusher_shelf_mm"]),
        foot_thickness_mm=float(mount_dict["foot_thickness_mm"]),
        foot_bolt_thread=str(mount_dict["foot_bolt_thread"]),
        foot_bolt_spacing_mm=float(mount_dict["foot_bolt_spacing_mm"]),
        bolt_safety_mm=float(mount_dict["bolt_safety_mm"]),
        front_bolt_offset_mm=float(mount_dict["front_bolt_offset_mm"]),
        pitch_flexure_thickness_mm=float(mount_dict["pitch_flexure_thickness_mm"]),
        pitch_flexure_gap_mm=float(mount_dict["pitch_flexure_gap_mm"]),
        trim_angle_deg=float(mount_dict["trim_angle_deg"]),
        roll_flexure_height_mm=float(mount_dict["roll_flexure_height_mm"]),
        roll_blade_thickness_mm=float(mount_dict["roll_blade_thickness_mm"]),
        roll_insert_spacing_mm=float(mount_dict["roll_insert_spacing_mm"]),
        roll_blade_spacing_mm=float(mount_dict["roll_blade_spacing_mm"]),
        roll_pedestal_gap_mm=float(mount_dict["roll_pedestal_gap_mm"]),
    )


def export_oap_flexure_mount_step(
    part_number: str,
    output_path: Path | None = None,
    params: OAPMirrorFlexureMountParams | None = None,
) -> Path:
    """Export the OAP flexure mount alone as STEP."""
    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _oap_flexure_params_from_bom(mount_dict)
    mount = build_oap_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
    )
    if output_path is None:
        output_path = Path(f"output/flexure_oap_mount_{part_number}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(mount, str(output_path))
    return output_path


def export_oap_flexure_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: OAPMirrorFlexureMountParams | None = None,
    *,
    role: str = "M1",
) -> Path:
    """Export OAP flexure mount + vendor mirror assembly as STEP."""
    from build123d import Compound

    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _oap_flexure_params_from_bom(mount_dict)
    mount = build_oap_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
    )
    oap = _load_vendor_oap_at_optical_centre(part_number, rotate_180=(role == "M1"))
    oap.label = part_number
    oap.color = _OPTIC_COLOR
    mount.color = _MOUNT_COLOR
    asm = Compound(children=[mount, oap])
    asm.label = f"assembly_{part_number}"
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(asm, str(output_path))
    return output_path


def build_oap_flexure_full_assembly(
    part_number: str,
    params: OAPMirrorFlexureMountParams | None = None,
    *,
    role: str = "M1",
    mfg: ManufacturingParams | None = None,
):
    """OAP flexure mount + vendor mirror + heat-set inserts + pusher setscrew + flat head screws.

    Hardware placed:
      - 1 pusher insert (centre, through foot)
      - 1 flat-tip pusher setscrew (92605A044) at flexure gap
      - 3 flat head screws on alternating bolt-circle holes (odd for M1, even for M2)

    All positions derived from BOM params. No hardcoded dimensions.
    """
    from build123d import Axis, Compound

    if mfg is None:
        mfg = _load_manufacturing()

    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _oap_flexure_params_from_bom(mount_dict)

    mount = build_oap_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
        mfg=mfg,
    )

    try:
        oap = _load_vendor_oap_at_optical_centre(part_number, rotate_180=(role == "M1"))
        oap.label = part_number
        oap.color = _OPTIC_COLOR
        children = [mount, oap]
    except KeyError:
        children = [mount]

    insert_flange_r = 0.5 * mfg.insert_flange_dia_mm
    optic_R = 0.5 * diameter_mm
    u_plate_front = -thickness_mm
    u_wall_rear = u_plate_front - params.slab_thickness_mm
    boss_width_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"] + params.bolt_safety_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    w_bot = -(optic_R + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    v_bolt = params.foot_bolt_spacing_mm

    def _trim_offset(u):
        return (u + insert_flange_r - u_wall_rear) * math.tan(
            math.radians(params.trim_angle_deg))

    # --- heat-set inserts (pusher only; foot bolts are self-tapping
    #     M2.5, no inserts) ---
    raw_insert = _load_or_procedural("94459A110")
    ins = raw_insert.rotate(Axis.X, 180.0)
    ins = ins.translate((0, -ins.bounding_box().min.Y, 0))

    u_pusher = u_plate_front
    off_pusher = _trim_offset(u_pusher)
    children.append(ins.translate((0, w_foot_bot + off_pusher, u_pusher)))

    # --- flat-tip pusher setscrew (closed tip at w_shelf_bot) ---
    raw_pusher = _load_or_procedural("92605A044")
    pusher = raw_pusher.rotate(Axis.X, 180.0)
    pusher = pusher.translate((0, -pusher.bounding_box().max.Y, 0))
    children.append(pusher.translate((0, w_shelf_bot, u_pusher)))

    # --- flat head screws on bolt circle (3 of 6 holes) ---
    raw_screw = _load_or_procedural(params.screw_part)
    fhs = raw_screw.rotate(Axis.X, 180.0)
    fhs = fhs.translate((0, 0, -fhs.bounding_box().min.Z))

    holes = [1, 3, 5] if role == "M1" else [0, 2, 4]
    for i in holes:
        angle = math.radians(params.hole_phase_deg + i * 60.0)
        v = params.bolt_circle_radius_mm * math.sin(angle)
        w = params.bolt_circle_radius_mm * math.cos(angle)
        children.append(fhs.translate((v, w, u_wall_rear)))

    # Roll pusher hardware (inserts in foot bottom + 6mm setscrews)
    if params.roll_flexure_height_mm > 0:
        u_roll_actuate = 0.5 * (u_wall_rear + u_plate_front)
        v_insert = params.roll_insert_spacing_mm
        w_ped_bot = w_roll_bot + 1.0
        ins_bottom = raw_insert.rotate(Axis.X, 180.0)
        ins_bottom = ins_bottom.translate((0, -ins_bottom.bounding_box().min.Y, 0))
        raw_roll_ss = _load_or_procedural("92605A912")
        roll_ss = raw_roll_ss.rotate(Axis.X, 180.0)
        roll_ss = roll_ss.translate((0, -roll_ss.bounding_box().max.Y, 0))
        for sign in (+1, -1):
            children.append(ins_bottom.translate((
                sign * v_insert, w_foot_bot + off_pusher, u_roll_actuate)))
            children.append(roll_ss.translate((
                sign * v_insert, w_ped_bot, u_roll_actuate)))

    asm = Compound(children=children)
    asm.label = f"assembly_{part_number}_{role}"
    return asm


def export_oap_flexure_full_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: OAPMirrorFlexureMountParams | None = None,
    *,
    role: str = "M1",
) -> Path:
    """Export OAP flexure mount + vendor mirror + all hardware as STEP."""
    asm = build_oap_flexure_full_assembly(part_number, params, role=role)
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure_full_{role}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(asm, str(output_path))
    return output_path


def build_mirror_flexure_full_assembly(
    part_number: str,
    params: RoundMirrorFlexureMountParams | None = None,
    *,
    mfg: ManufacturingParams | None = None,
):
    """Round mirror flexure mount + vendor mirror + heat-set inserts + setscrews.

    Hardware placed:
      - 1 top insert + flat-tip setscrew (92605A043) for optic retention
      - 1 pusher insert (centre, through foot)
      - 1 flat-tip pusher setscrew (92605A044) at flexure gap
    """
    from build123d import Axis, Compound

    if mfg is None:
        mfg = _load_manufacturing()

    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    if params is None:
        params = _flexure_params_from_bom(mount_dict)

    mount = build_mirror_flexure_mount_cad(
        optic_diameter_mm=diameter_mm,
        center_thickness_mm=thickness_mm,
        params=params,
        mfg=mfg,
    )

    try:
        mirror = _load_vendor_mirror_at_vertex(part_number, thickness_mm)
        mirror.label = part_number
        mirror.color = _OPTIC_COLOR
        children = [mount, mirror]
    except KeyError:
        children = [mount]

    insert_flange_r = 0.5 * mfg.insert_flange_dia_mm
    optic_R = 0.5 * diameter_mm
    u_vertex = 0.0
    u_wall_rear = -thickness_mm - params.rear_wall_mm
    boss_width_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"] + params.bolt_safety_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    u_setscrew = 0.5 * (u_wall_rear + u_vertex)
    w_top = optic_R + params.head_clearance_mm
    w_bot = -(optic_R + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    v_bolt = params.foot_bolt_spacing_mm

    def _trim_offset(u):
        return (u + insert_flange_r - u_wall_rear) * math.tan(
            math.radians(params.trim_angle_deg))

    # --- heat-set inserts (setscrew + pusher only; foot bolts are
    #     self-tapping M2.5, no inserts) ---
    raw_insert = _load_or_procedural("94459A110")
    ins_bottom = raw_insert.rotate(Axis.X, 180.0)
    ins_bottom = ins_bottom.translate((0, -ins_bottom.bounding_box().min.Y, 0))
    ins_top = raw_insert.translate((0, -raw_insert.bounding_box().max.Y, 0))

    # Top insert (setscrew bore, at u_bolt)
    children.append(ins_top.translate((0, w_top, u_setscrew)))

    front_bore_dia = diameter_mm + mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    bore_r = 0.5 * (front_bore_dia + params.optic_clearance_mm)
    bump_center_r = bore_r + params.contact_radius_mm - params.contact_offset_mm
    w_bump_top = bump_center_r + params.contact_radius_mm

    raw_ss = _load_or_procedural("92605A043")
    ss_top = raw_ss.rotate(Axis.Z, 90.0)
    ss_top = ss_top.translate((0, -ss_top.bounding_box().min.Y, 0))
    children.append(ss_top.translate((0, w_bump_top, u_setscrew)))

    off_pusher = _trim_offset(0.0)
    children.append(ins_bottom.translate((0, w_foot_bot + off_pusher, 0.0)))

    raw_pusher = _load_or_procedural("92605A044")
    pusher = raw_pusher.rotate(Axis.X, 180.0)
    pusher = pusher.translate((0, -pusher.bounding_box().max.Y, 0))
    children.append(pusher.translate((0, w_shelf_bot, 0.0)))

    # Roll pusher hardware (inserts in foot bottom + 6mm setscrews)
    if params.roll_flexure_height_mm > 0:
        u_roll_actuate = 0.5 * (u_wall_rear + 0.0)
        v_insert = params.roll_insert_spacing_mm
        w_ped_bot = w_roll_bot + 1.0
        raw_roll_ss = _load_or_procedural("92605A912")
        roll_ss = raw_roll_ss.rotate(Axis.X, 180.0)
        roll_ss = roll_ss.translate((0, -roll_ss.bounding_box().max.Y, 0))
        for sign in (+1, -1):
            children.append(ins_bottom.translate((
                sign * v_insert, w_foot_bot + off_pusher, u_roll_actuate)))
            children.append(roll_ss.translate((
                sign * v_insert, w_ped_bot, u_roll_actuate)))

    asm = Compound(children=children)
    asm.label = f"assembly_{part_number}"
    return asm


def export_mirror_flexure_full_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: RoundMirrorFlexureMountParams | None = None,
) -> Path:
    """Export round mirror flexure mount + vendor mirror + all hardware as STEP."""
    asm = build_mirror_flexure_full_assembly(part_number, params)
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure_full.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(asm, str(output_path))
    return output_path


def build_grating_flexure_full_assembly(
    part_number: str,
    params: GratingFlexureMountParams | None = None,
    *,
    mfg: ManufacturingParams | None = None,
):
    """Grating flexure mount + vendor grating + heat-set inserts + setscrews.

    Hardware placed:
      - 1 top insert + flat-tip setscrew (92605A043) for optic retention
      - 1 pusher insert (centre, through foot)
      - 1 flat-tip pusher setscrew (92605A044) at flexure gap
    """
    from build123d import Axis, Compound

    if mfg is None:
        mfg = _load_manufacturing()

    size_mm, thickness_mm, mount_dict = _lookup_grating_in_bom(part_number)
    if params is None:
        params = _grating_flexure_params_from_bom(mount_dict)

    mount = build_grating_flexure_mount_cad(
        grating_size_mm=size_mm,
        grating_thickness_mm=thickness_mm,
        params=params,
        mfg=mfg,
    )

    try:
        grating = _load_vendor_grating_at_front(part_number)
        grating.label = part_number
        grating.color = _GRATING_COLOR
        children = [mount, grating]
    except KeyError:
        children = [mount]

    insert_flange_r = 0.5 * mfg.insert_flange_dia_mm
    half_size = 0.5 * size_mm
    u_wall_rear = -(thickness_mm + params.rear_wall_mm)
    boss_width_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"] + params.bolt_safety_mm
    u_bolt = u_wall_rear + 0.5 * boss_width_mm
    u_setscrew = -0.5 * thickness_mm
    w_top = half_size + params.head_clearance_mm
    w_bot = -(half_size + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    v_bolt = params.foot_bolt_spacing_mm

    def _trim_offset(u):
        return (u + insert_flange_r - u_wall_rear) * math.tan(
            math.radians(params.trim_angle_deg))

    # --- heat-set inserts (setscrew + pusher only; foot bolts are
    #     self-tapping M2.5, no inserts) ---
    raw_insert = _load_or_procedural("94459A110")
    ins_bottom = raw_insert.rotate(Axis.X, 180.0)
    ins_bottom = ins_bottom.translate((0, -ins_bottom.bounding_box().min.Y, 0))
    ins_top = raw_insert.translate((0, -raw_insert.bounding_box().max.Y, 0))

    # Top insert (at u_setscrew, centred on grating)
    children.append(ins_top.translate((0, w_top, u_setscrew)))

    w_opening_upper = half_size + 0.5 * (mfg.assembly_clearance_mm + mfg.print_tolerance_mm + params.optic_clearance_mm)
    bump_center_w = w_opening_upper + params.contact_radius_mm - params.contact_offset_mm
    w_bump_top = bump_center_w + params.contact_radius_mm

    raw_ss = _load_or_procedural("92605A043")
    ss_top = raw_ss.rotate(Axis.Z, 90.0)
    ss_top = ss_top.translate((0, -ss_top.bounding_box().min.Y, 0))
    children.append(ss_top.translate((0, w_bump_top, u_setscrew)))

    off_pusher = _trim_offset(0.0)
    children.append(ins_bottom.translate((0, w_foot_bot + off_pusher, 0.0)))
    raw_pusher = _load_or_procedural("92605A044")
    pusher = raw_pusher.rotate(Axis.X, 180.0)
    pusher = pusher.translate((0, -pusher.bounding_box().max.Y, 0))
    children.append(pusher.translate((0, w_shelf_bot, 0.0)))

    # Roll pusher hardware (inserts in foot bottom + 6mm setscrews)
    if params.roll_flexure_height_mm > 0:
        u_roll_actuate = 0.5 * (u_wall_rear + 0.0)
        v_insert = params.roll_insert_spacing_mm
        w_ped_bot = w_roll_bot + 1.0
        raw_roll_ss = _load_or_procedural("92605A912")
        roll_ss = raw_roll_ss.rotate(Axis.X, 180.0)
        roll_ss = roll_ss.translate((0, -roll_ss.bounding_box().max.Y, 0))
        for sign in (+1, -1):
            children.append(ins_bottom.translate((
                sign * v_insert, w_foot_bot + off_pusher, u_roll_actuate)))
            children.append(roll_ss.translate((
                sign * v_insert, w_ped_bot, u_roll_actuate)))

    asm = Compound(children=children)
    asm.label = f"assembly_{part_number}"
    return asm


def export_grating_flexure_full_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: GratingFlexureMountParams | None = None,
) -> Path:
    """Export grating flexure mount + vendor grating + all hardware as STEP."""
    asm = build_grating_flexure_full_assembly(part_number, params)
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure_full.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(asm, str(output_path))
    return output_path


def _grating_flexure_params_from_bom(mount_dict: dict) -> GratingFlexureMountParams:
    """Build GratingFlexureMountParams from BOM mount dict. All fields explicit."""
    return GratingFlexureMountParams(
        jaw_clearance_mm=float(mount_dict["jaw_clearance_mm"]),
        optic_clearance_mm=float(mount_dict["optic_clearance_mm"]),
        contact_radius_mm=float(mount_dict["contact_radius_mm"]),
        contact_offset_mm=float(mount_dict["contact_offset_mm"]),
        contact_separation_mm=float(mount_dict["contact_separation_mm"]),
        head_clearance_mm=float(mount_dict["head_clearance_mm"]),
        foot_clearance_mm=float(mount_dict["foot_clearance_mm"]),
        pusher_shelf_mm=float(mount_dict["pusher_shelf_mm"]),
        rear_wall_mm=float(mount_dict["rear_wall_mm"]),
        foot_thickness_mm=float(mount_dict["foot_thickness_mm"]),
        foot_bolt_thread=str(mount_dict["foot_bolt_thread"]),
        foot_bolt_spacing_mm=float(mount_dict["foot_bolt_spacing_mm"]),
        bolt_safety_mm=float(mount_dict["bolt_safety_mm"]),
        front_bolt_offset_mm=float(mount_dict["front_bolt_offset_mm"]),
        pitch_flexure_thickness_mm=float(mount_dict["pitch_flexure_thickness_mm"]),
        pitch_flexure_gap_mm=float(mount_dict["pitch_flexure_gap_mm"]),
        trim_angle_deg=float(mount_dict["trim_angle_deg"]),
        roll_flexure_height_mm=float(mount_dict["roll_flexure_height_mm"]),
        roll_blade_thickness_mm=float(mount_dict["roll_blade_thickness_mm"]),
        roll_insert_spacing_mm=float(mount_dict["roll_insert_spacing_mm"]),
        roll_blade_spacing_mm=float(mount_dict["roll_blade_spacing_mm"]),
        roll_pedestal_gap_mm=float(mount_dict["roll_pedestal_gap_mm"]),
    )


def export_grating_flexure_mount_step(
    part_number: str,
    output_path: Path | None = None,
    params: GratingFlexureMountParams | None = None,
) -> Path:
    """Export the grating flexure mount alone as STEP."""
    size_mm, thickness_mm, mount_dict = _lookup_grating_in_bom(part_number)
    if params is None:
        params = _grating_flexure_params_from_bom(mount_dict)
    mount = build_grating_flexure_mount_cad(
        grating_size_mm=size_mm,
        grating_thickness_mm=thickness_mm,
        params=params,
    )
    if output_path is None:
        output_path = Path(f"output/flexure_grating_mount_{part_number}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(mount, str(output_path))
    return output_path


def build_grating_flexure_assembly(
    part_number: str,
    params: GratingFlexureMountParams | None = None,
):
    """Grating flexure mount + vendor grating + heat-set inserts + setscrews.

    Hardware placed:
      - 1 top insert + flat-tip setscrew (92605A043) for optic retention
      - 1 pusher insert (centre, through foot)
      - 1 flat-tip pusher setscrew (92605A044) at flexure gap
      - 2 roll inserts + 2 roll setscrews (92605A912) when roll flexure active
    """
    from build123d import Axis, Compound

    mfg = _load_manufacturing()
    size_mm, thickness_mm, mount_dict = _lookup_grating_in_bom(part_number)
    if params is None:
        params = _grating_flexure_params_from_bom(mount_dict)
    mount = build_grating_flexure_mount_cad(
        grating_size_mm=size_mm,
        grating_thickness_mm=thickness_mm,
        params=params,
    )
    grating = _load_vendor_grating_at_front(part_number)
    grating.label = part_number
    grating.color = _GRATING_COLOR
    mount.color = _MOUNT_COLOR
    children = [mount, grating]

    insert_flange_r = 0.5 * mfg.insert_flange_dia_mm
    half_size = 0.5 * size_mm
    u_vertex = 0.0
    u_wall_rear = -(thickness_mm + params.rear_wall_mm)
    u_setscrew = -0.5 * thickness_mm
    w_top = half_size + params.head_clearance_mm
    w_bot = -(half_size + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm

    def _trim_offset(u):
        return (u + insert_flange_r - u_wall_rear) * math.tan(
            math.radians(params.trim_angle_deg))

    raw_insert = _load_or_procedural("94459A110")
    ins_bottom = raw_insert.rotate(Axis.X, 180.0)
    ins_bottom = ins_bottom.translate((0, -ins_bottom.bounding_box().min.Y, 0))
    ins_top = raw_insert.translate((0, -raw_insert.bounding_box().max.Y, 0))

    children.append(ins_top.translate((0, w_top, u_setscrew)))

    w_opening_upper = half_size + 0.5 * (mfg.assembly_clearance_mm + mfg.print_tolerance_mm + params.optic_clearance_mm)
    bump_center_w = w_opening_upper + params.contact_radius_mm - params.contact_offset_mm
    w_bump_top = bump_center_w + params.contact_radius_mm

    raw_ss = _load_or_procedural("92605A043")
    ss_top = raw_ss.rotate(Axis.Z, 90.0)
    ss_top = ss_top.translate((0, -ss_top.bounding_box().min.Y, 0))
    children.append(ss_top.translate((0, w_bump_top, u_setscrew)))

    off_pusher = _trim_offset(0.0)
    children.append(ins_bottom.translate((0, w_foot_bot + off_pusher, 0.0)))

    raw_pusher = _load_or_procedural("92605A044")
    pusher = raw_pusher.rotate(Axis.X, 180.0)
    pusher = pusher.translate((0, -pusher.bounding_box().max.Y, 0))
    children.append(pusher.translate((0, w_shelf_bot, 0.0)))

    if params.roll_flexure_height_mm > 0:
        u_roll_actuate = 0.5 * (u_wall_rear + 0.0)
        v_insert = params.roll_insert_spacing_mm
        w_ped_bot = w_roll_bot + 1.0
        raw_roll_ss = _load_or_procedural("92605A912")
        roll_ss = raw_roll_ss.rotate(Axis.X, 180.0)
        roll_ss = roll_ss.translate((0, -roll_ss.bounding_box().max.Y, 0))
        for sign in (+1, -1):
            children.append(ins_bottom.translate((
                sign * v_insert, w_foot_bot + off_pusher, u_roll_actuate)))
            children.append(roll_ss.translate((
                sign * v_insert, w_ped_bot, u_roll_actuate)))

    asm = Compound(children=children)
    asm.label = f"assembly_{part_number}"
    return asm


def export_grating_flexure_assembly_step(
    part_number: str,
    output_path: Path | None = None,
    params: GratingFlexureMountParams | None = None,
) -> Path:
    """Export grating flexure mount + vendor grating assembly as STEP."""
    assembly = build_grating_flexure_assembly(part_number, params)
    if output_path is None:
        output_path = Path(f"output/assembly_{part_number}_flexure.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(assembly, str(output_path))
    return output_path


_FIXTURE_COLOR = Color(0.75, 0.15, 0.15)


_DEBOSS_DEPTH_MM = 0.5




def build_mirror_assembly_fixture(
    part_number: str,
    *,
    wall_mm: float = 3.0,
    lip_pocket_mm: float = 2.0,
    label: str | None = None,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Assembly fixture for placing a mirror into its flexure mount.

    Envelope box around the mount+mirror assembly with:
      - Mount pocket (outer hull steps 1-5, inflated by clearance)
      - Mirror pocket (mirror solid, inflated by clearance)
      - Lip pocket (optic_diameter - lip_pocket_mm, through fixture)
      - Setscrew access slot (rounded, open to -z edge for removal clearance)

    Frame: same as mount — x=v, y=w, z=u.
    """
    if mfg is None:
        mfg = _load_manufacturing()

    focal_mm, diameter_mm, thickness_mm, mount_dict = _lookup_mirror_in_bom(part_number)
    params = _flexure_params_from_bom(mount_dict)

    t = mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    optic_radius_mm = 0.5 * diameter_mm
    u_vertex = 0.0
    u_shoulder = -thickness_mm
    u_wall_rear = u_shoulder - params.rear_wall_mm
    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm
    w_top = optic_radius_mm + params.head_clearance_mm
    w_bot = -(optic_radius_mm + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    v_half_mm = optic_radius_mm + params.wall_margin_mm
    slab_total_u_mm = u_foot_front - u_wall_rear
    slab_depth_mm = u_vertex - u_wall_rear
    slab_depth_mm = u_vertex - u_wall_rear
    full_height_mm = w_top - w_foot_bot
    w_center = 0.5 * (w_top + w_foot_bot)
    u_front_bolt = params.front_bolt_offset_mm
    u_setscrew = 0.5 * (u_wall_rear + u_vertex)

    # Assembly for bounding box
    asm = build_mirror_flexure_assembly(part_number, params)
    ab = asm.bounding_box()
    z_lo = u_wall_rear
    z_hi = u_foot_front + wall_mm
    lip_pocket_r = 0.5 * (diameter_mm - lip_pocket_mm)

    # 1. Envelope + lip pocket
    with BuildPart() as fixture_bp:
        with Locations(Pos(0.5 * (ab.min.X + ab.max.X),
                           0.5 * (ab.min.Y + ab.max.Y),
                           0.5 * (z_lo + z_hi))):
            Box(ab.max.X - ab.min.X + 2 * wall_mm,
                ab.max.Y - ab.min.Y + 2 * wall_mm,
                z_hi - z_lo)
        with BuildSketch(Plane(origin=(0, 0, z_lo),
                               x_dir=(1, 0, 0), z_dir=(0, 0, 1))):
            Circle(lip_pocket_r)
        extrude(until=Until.LAST, mode=Mode.SUBTRACT)

    # 2. Mount hull (steps 1-5, inflated by t in v and w only)
    rear_plane = Plane(origin=(0, 0, u_wall_rear),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    with BuildPart() as hull:
        with BuildSketch(rear_plane):
            with Locations(Pos(0, w_center)):
                Rectangle(2 * v_half_mm + 2 * t, full_height_mm + 2 * t)
        extrude(amount=slab_total_u_mm)
        side_excess_mm = v_half_mm - optic_radius_mm
        if side_excess_mm > 0:
            for sign in (+1, -1):
                v_cut = sign * (optic_radius_mm + t + 0.5 * side_excess_mm)
                with Locations(Pos(v_cut, w_center,
                                   u_wall_rear + 0.5 * slab_total_u_mm)):
                    Box(side_excess_mm, full_height_mm + 2 * t,
                        slab_total_u_mm, mode=Mode.SUBTRACT)
        front_face = hull.faces().sort_by(Axis.Z)[-1]
        mill_depth_mm = u_foot_front - u_vertex
        w_slab_center = 0.5 * (w_top + w_shelf_bot)
        with BuildSketch(front_face):
            with Locations(Pos(0, w_slab_center - w_center)):
                Rectangle(2 * v_half_mm + 2 * t, w_top - w_shelf_bot + 2 * t)
        extrude(amount=-mill_depth_mm, mode=Mode.SUBTRACT)
        tongue_width_mm = 2 * params.foot_bolt_spacing_mm + boss_width_mm
        foot_face = hull.faces().filter_by(Axis.Y).sort_by(Axis.Y)[0]
        foot_u_mid = foot_face.center().Z
        with BuildSketch(foot_face):
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_vertex) - foot_u_mid)):
                Rectangle(2 * v_half_mm + 2 * t, slab_depth_mm)
            tongue_u_len = u_front_bolt + 0.5 * boss_width_mm - u_wall_rear
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_front_bolt
                               + 0.5 * boss_width_mm) - foot_u_mid)):
                Rectangle(tongue_width_mm + 2 * t, tongue_u_len)
            fillet_verts = [
                v for v in hull.vertices()
                if (abs(v.Y - (u_vertex - foot_u_mid)) < 0.2
                    and abs(abs(v.X) - 0.5 * (tongue_width_mm + 2 * t)) < 0.2)
                or (abs(v.Y - (u_front_bolt + 0.5 * boss_width_mm
                               - foot_u_mid)) < 0.2
                    and abs(abs(v.X) - 0.5 * (tongue_width_mm + 2 * t)) < 0.2)
            ]
            if fillet_verts:
                try:
                    fillet2d(fillet_verts, radius=mfg.fillet_radius_mm)
                except Exception:
                    pass
        extrude(amount=-(w_top - w_foot_bot + 2 * t), mode=Mode.INTERSECT)

    # Pitch pusher shelf hull
    _shelf_plane = Plane(origin=(0, w_shelf_bot - t, 0),
                         x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    with BuildPart() as _shelf_bp:
        with BuildSketch(_shelf_plane):
            Circle(0.5 * boss_width_mm + t)
        extrude(amount=params.pusher_shelf_mm + 2 * t)
    mount_hull = hull.part.fuse(_shelf_bp.part)

    # Roll flexure band hull (slab depth, no overhang)
    if params.roll_flexure_height_mm > 0:
        w_roll_mid = 0.5 * (w_roll_bot + w_bot)
        _roll_plane = Plane(origin=(0, 0, u_wall_rear),
                            x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildPart() as _roll_hull_bp:
            with BuildSketch(_roll_plane):
                with Locations(Pos(0, w_roll_mid)):
                    Rectangle(2 * optic_radius_mm + 2 * t,
                              params.roll_flexure_height_mm + 2 * t)
            extrude(amount=slab_depth_mm)
        mount_hull = mount_hull.fuse(_roll_hull_bp.part)

    # 3. Mirror (inflated by t in v and w only)
    mt, co = _lookup_mirror_shape(part_number)
    mirror = _build_generic_concave_mirror(
        focal_mm, diameter_mm + 2 * t, thickness_mm,
        mirror_type=mt, cylindrical_orientation=co)

    # 4. Subtract mount hull + mirror
    fixture = fixture_bp.part - mount_hull - mirror

    # 5. Foot/shelf relief + setscrew access slot
    _fr = mfg.fillet_radius_mm
    with BuildPart() as fix_final:
        fix_final._obj = fixture
        # Through-cut from pitch shelf down so the foot can't bottom out.
        with Locations(Pos(0,
                           0.5 * (w_foot_bot + w_shelf_bot),
                           0.5 * (u_vertex - _fr + z_hi))):
            Box(tongue_width_mm + 2 * t + 2 * _fr,
                w_shelf_bot - w_foot_bot + 2 * t,
                z_hi - u_vertex + _fr + 2 * t,
                mode=Mode.SUBTRACT)
        # Through-cut for the pitch pusher shelf pocket.
        with Locations(Pos(0,
                           w_shelf_bot + 0.5 * params.pusher_shelf_mm,
                           0.5 * (u_vertex - _fr + z_hi))):
            Box(tongue_width_mm + 2 * t + 2 * _fr,
                params.pusher_shelf_mm + 4 * t,
                z_hi - u_vertex + _fr + 2 * t,
                mode=Mode.SUBTRACT)
        ss_plane = Plane(origin=(0, 0, 0),
                         x_dir=(1, 0, 0), z_dir=(0, 1, 0))
        with BuildSketch(ss_plane):
            _r = 0.5 * mfg.insert_bore_dia_mm
            with Locations(Pos(0, -u_setscrew)):
                Circle(_r)
            _slot_h = u_setscrew - z_lo + _r
            with Locations(Pos(0, 0.5 * (-u_setscrew - z_lo + _r))):
                Rectangle(mfg.insert_bore_dia_mm, _slot_h)
        extrude(until=Until.LAST, mode=Mode.SUBTRACT)
        # Debossed label on top face (+w)
        if label:
            top_y = ab.max.Y + wall_mm
            text_plane = Plane(
                origin=Vector(0, top_y, 0.5 * (z_lo + z_hi)),
                x_dir=Vector(1, 0, 0),
                z_dir=Vector(0, 1, 0))
            with BuildSketch(text_plane):
                Text(label, font_size=8,
                     font="DejaVu Sans", font_style=FontStyle.BOLD)
            extrude(amount=-_DEBOSS_DEPTH_MM, mode=Mode.SUBTRACT)

    result = fix_final.part.solids()[0]
    result.color = _FIXTURE_COLOR
    result.label = f"fixture_{part_number}"
    return result


def export_mirror_assembly_fixture_step(
    part_number: str,
    output_path: Path | None = None,
    **kwargs,
) -> Path:
    """Export a mirror assembly fixture as STEP."""
    fixture = build_mirror_assembly_fixture(part_number, **kwargs)
    if output_path is None:
        output_path = Path(f"output/fixture_{part_number}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(fixture, str(output_path))
    return output_path


def build_grating_assembly_fixture(
    part_number: str = "GR25-0605",
    *,
    wall_mm: float = 3.0,
    lip_pocket_mm: float = 2.0,
    label: str | None = "GR",
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Assembly fixture for placing a grating into its flexure mount.

    Same pattern as mirror fixture: envelope, lip pocket, mount hull
    (steps 1-5 inflated by clearance), grating subtracted, setscrew
    access slot (rounded, open to -z edge).
    """
    if mfg is None:
        mfg = _load_manufacturing()

    size_mm, thickness_mm, mount_dict = _lookup_grating_in_bom(part_number)
    params = _grating_flexure_params_from_bom(mount_dict)

    t = mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    half_size_mm = 0.5 * size_mm
    u_vertex = 0.0
    u_wall_rear = -(thickness_mm + params.rear_wall_mm)
    slab_depth_mm = u_vertex - u_wall_rear
    w_top = half_size_mm + params.head_clearance_mm
    w_bot = -(half_size_mm + params.foot_clearance_mm)
    w_roll_bot = w_bot - params.roll_flexure_height_mm
    w_shelf_bot = w_roll_bot - params.pusher_shelf_mm
    w_foot_bot = w_shelf_bot - params.pitch_flexure_gap_mm - params.foot_thickness_mm
    v_half_mm = half_size_mm
    bolt_head_mm = mfg.bolt_dims[params.foot_bolt_thread]["head_dia_mm"]
    boss_width_mm = bolt_head_mm + params.bolt_safety_mm
    foot_length_mm = params.front_bolt_offset_mm + 0.5 * boss_width_mm
    u_foot_front = u_vertex + foot_length_mm
    slab_total_u_mm = u_foot_front - u_wall_rear
    full_height_mm = w_top - w_foot_bot
    w_center = 0.5 * (w_top + w_foot_bot)
    u_front_bolt = params.front_bolt_offset_mm
    u_setscrew = -0.5 * thickness_mm

    # Assembly for bounding box
    asm = build_grating_flexure_assembly(part_number, params)
    ab = asm.bounding_box()
    z_lo = u_wall_rear
    z_hi = u_foot_front + wall_mm

    # Lip pocket: full grating width in v, 0.5mm lip on top/bottom (w)
    lip_v = size_mm
    lip_w = size_mm - lip_pocket_mm

    # 1. Envelope + lip pocket
    with BuildPart() as fixture_bp:
        with Locations(Pos(0.5 * (ab.min.X + ab.max.X),
                           0.5 * (ab.min.Y + ab.max.Y),
                           0.5 * (z_lo + z_hi))):
            Box(ab.max.X - ab.min.X + 2 * wall_mm,
                ab.max.Y - ab.min.Y + 2 * wall_mm,
                z_hi - z_lo)
        with BuildSketch(Plane(origin=(0, 0, z_lo),
                               x_dir=(1, 0, 0), z_dir=(0, 0, 1))):
            Rectangle(lip_v, lip_w)
        extrude(until=Until.LAST, mode=Mode.SUBTRACT)

    # 2. Mount hull (steps 1-5, inflated by t in v and w only)
    rear_plane = Plane(origin=(0, 0, u_wall_rear),
                       x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    with BuildPart() as hull:
        with BuildSketch(rear_plane):
            with Locations(Pos(0, w_center)):
                Rectangle(2 * v_half_mm + 2 * t, full_height_mm + 2 * t)
        extrude(amount=slab_total_u_mm)

        front_face = hull.faces().sort_by(Axis.Z)[-1]
        mill_depth_mm = u_foot_front - u_vertex
        w_slab_center = 0.5 * (w_top + w_shelf_bot)
        with BuildSketch(front_face):
            with Locations(Pos(0, w_slab_center - w_center)):
                Rectangle(2 * v_half_mm + 2 * t, w_top - w_shelf_bot + 2 * t)
        extrude(amount=-mill_depth_mm, mode=Mode.SUBTRACT)

        tongue_width_mm = 2 * params.foot_bolt_spacing_mm + boss_width_mm
        foot_face = hull.faces().filter_by(Axis.Y).sort_by(Axis.Y)[0]
        foot_u_mid = foot_face.center().Z
        with BuildSketch(foot_face):
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_vertex) - foot_u_mid)):
                Rectangle(2 * v_half_mm + 2 * t, slab_depth_mm)
            tongue_u_len = u_front_bolt + 0.5 * boss_width_mm - u_wall_rear
            with Locations(Pos(0, 0.5 * (u_wall_rear + u_front_bolt
                               + 0.5 * boss_width_mm) - foot_u_mid)):
                Rectangle(tongue_width_mm + 2 * t, tongue_u_len)
            fillet_verts = [
                v for v in hull.vertices()
                if (abs(v.Y - (u_vertex - foot_u_mid)) < 0.2
                    and abs(abs(v.X) - 0.5 * (tongue_width_mm + 2 * t)) < 0.2)
                or (abs(v.Y - (u_front_bolt + 0.5 * boss_width_mm
                               - foot_u_mid)) < 0.2
                    and abs(abs(v.X) - 0.5 * (tongue_width_mm + 2 * t)) < 0.2)
            ]
            if fillet_verts:
                try:
                    fillet2d(fillet_verts, radius=mfg.fillet_radius_mm)
                except Exception:
                    pass
        extrude(amount=-(w_top - w_foot_bot + 2 * t), mode=Mode.INTERSECT)

    # Pitch pusher shelf hull
    _shelf_plane = Plane(origin=(0, w_shelf_bot - t, 0),
                         x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    with BuildPart() as _shelf_bp:
        with BuildSketch(_shelf_plane):
            Circle(0.5 * boss_width_mm + t)
        extrude(amount=params.pusher_shelf_mm + 2 * t)
    mount_hull = hull.part.fuse(_shelf_bp.part)

    # Roll flexure band hull
    if params.roll_flexure_height_mm > 0:
        w_roll_mid = 0.5 * (w_roll_bot + w_bot)
        _roll_plane = Plane(origin=(0, 0, u_wall_rear),
                            x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildPart() as _roll_hull_bp:
            with BuildSketch(_roll_plane):
                with Locations(Pos(0, w_roll_mid)):
                    Rectangle(2 * half_size_mm + 2 * t,
                              params.roll_flexure_height_mm + 2 * t)
            extrude(amount=slab_depth_mm)
        mount_hull = mount_hull.fuse(_roll_hull_bp.part)

    # 3. Grating solid (inflated by t in v and w only)
    grating = _build_generic_grating(size_mm + 2 * t, thickness_mm)

    # 4. Subtract mount hull + grating
    fixture = fixture_bp.part - mount_hull - grating

    # 5. Foot/shelf relief + setscrew access slot
    _fr = mfg.fillet_radius_mm
    with BuildPart() as fix_final:
        fix_final._obj = fixture
        with Locations(Pos(0,
                           0.5 * (w_foot_bot + w_shelf_bot),
                           0.5 * (u_vertex - _fr + z_hi))):
            Box(tongue_width_mm + 2 * t + 2 * _fr,
                w_shelf_bot - w_foot_bot + 2 * t,
                z_hi - u_vertex + _fr + 2 * t,
                mode=Mode.SUBTRACT)
        with Locations(Pos(0,
                           w_shelf_bot + 0.5 * params.pusher_shelf_mm,
                           0.5 * (u_vertex - _fr + z_hi))):
            Box(tongue_width_mm + 2 * t + 2 * _fr,
                params.pusher_shelf_mm + 4 * t,
                z_hi - u_vertex + _fr + 2 * t,
                mode=Mode.SUBTRACT)
        ss_plane = Plane(origin=(0, 0, 0),
                         x_dir=(1, 0, 0), z_dir=(0, 1, 0))
        with BuildSketch(ss_plane):
            _r = 0.5 * mfg.insert_bore_dia_mm
            with Locations(Pos(0, -u_setscrew)):
                Circle(_r)
            _slot_h = u_setscrew - z_lo + _r
            with Locations(Pos(0, 0.5 * (-u_setscrew - z_lo + _r))):
                Rectangle(mfg.insert_bore_dia_mm, _slot_h)
        extrude(until=Until.LAST, mode=Mode.SUBTRACT)
        if label:
            top_y = ab.max.Y + wall_mm
            text_plane = Plane(
                origin=Vector(0, top_y, 0.5 * (z_lo + z_hi)),
                x_dir=Vector(1, 0, 0),
                z_dir=Vector(0, 1, 0))
            with BuildSketch(text_plane):
                Text(label, font_size=8,
                     font="DejaVu Sans", font_style=FontStyle.BOLD)
            extrude(amount=-_DEBOSS_DEPTH_MM, mode=Mode.SUBTRACT)

    result = fix_final.part.solids()[0]
    result.color = _FIXTURE_COLOR
    result.label = f"fixture_{part_number}"
    return result


def export_grating_assembly_fixture_step(
    part_number: str = "GR25-0605",
    output_path: Path | None = None,
    **kwargs,
) -> Path:
    """Export a grating assembly fixture as STEP."""
    fixture = build_grating_assembly_fixture(part_number, **kwargs)
    if output_path is None:
        output_path = Path(f"output/fixture_{part_number}.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(fixture, str(output_path))
    return output_path


def build_hasma_tap_fixture(
    *,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Tap guide fixture for the HASMA 1/4"-36 thread bore.

    A conical plug that seats in the housing's conical flare cut,
    with a 1/4" through bore to guide a 1/4"-36 tap straight.

    Frame: z-axis along bore (= slit optical axis in the housing).
    z=0 at the narrow end (hex clearance plane), cone flares toward +z.
    """
    import tomllib

    if mfg is None:
        mfg = _load_manufacturing()

    bom_path = _get_bom_path()
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)

    slit_mount = bom["slits"]["mount"]
    hex_half_mm = float(slit_mount["hex_half_mm"])
    boundary_mm = float(slit_mount["boundary_mm"])
    tap_drill_dia_mm = float(slit_mount["tap_drill_dia_mm"])

    from optics.housing import _HASMA_FLARE_HALF_ANGLE_DEG
    flare_half_angle_deg = _HASMA_FLARE_HALF_ANGLE_DEG

    from optics.housing_cad import _CAVITY_POCKET_CLEARANCE_MM
    cavity_clearance_mm = _CAVITY_POCKET_CLEARANCE_MM

    t = mfg.print_tolerance_mm

    # Cone geometry: matches the housing conical flare with clearance.
    # Near radius = hex_clearance_r (at z=0), flares at 45° toward +z.
    hex_clearance_r = hex_half_mm + cavity_clearance_mm
    near_r = hex_clearance_r - t
    cone_length = boundary_mm + 5.0
    far_r = near_r + cone_length * math.tan(math.radians(flare_half_angle_deg))

    # Tap clearance bore: 1/4" = 6.35mm + print/assembly tolerance
    tap_bore_r = 0.5 * 6.35 + mfg.print_tolerance_mm + mfg.assembly_clearance_mm
    cone_length = max(cone_length, 20.0)

    with BuildPart() as fix:
        with BuildSketch(Plane.XY):
            Circle(near_r)
        extrude(amount=cone_length, taper=-flare_half_angle_deg)

        # Through bore for tap
        bore_plane = Plane(origin=(0, 0, 0),
                           x_dir=(1, 0, 0), z_dir=(0, 0, 1))
        with BuildSketch(bore_plane):
            Circle(tap_bore_r)
        extrude(until=Until.LAST, mode=Mode.SUBTRACT)

        # Debossed label wrapped around cone surface
        _label = '1/4"-36 TAP'
        _fs = 8
        _alpha = math.radians(flare_half_angle_deg)
        _z_text = cone_length * 0.55
        _r_text = near_r + _z_text * math.tan(_alpha)
        _char_widths = []
        for _ch in _label:
            if _ch == ' ':
                _char_widths.append(_fs * 0.3)
            else:
                with BuildSketch(Plane.XY) as _ms:
                    Text(_ch, font_size=_fs,
                         font="DejaVu Sans", font_style=FontStyle.BOLD)
                _char_widths.append(_ms.sketch.bounding_box().size.X)
        _total_w = sum(_char_widths)
        _x = -_total_w / 2
        for _ch, _cw in zip(_label, _char_widths):
            _cx = _x + _cw / 2
            _x += _cw
            if _ch == ' ':
                continue
            _theta = math.pi - _cx / _r_text
            _ct, _st = math.cos(_theta), math.sin(_theta)
            _origin = Vector(_r_text * _ct, _r_text * _st, _z_text)
            _n = Vector(math.cos(_alpha) * _ct,
                        math.cos(_alpha) * _st,
                        -math.sin(_alpha))
            _xd = Vector(_st, -_ct, 0)
            with BuildSketch(Plane(origin=_origin, x_dir=_xd, z_dir=_n)):
                Text(_ch, font_size=_fs,
                     font="DejaVu Sans", font_style=FontStyle.BOLD)
            extrude(amount=-_DEBOSS_DEPTH_MM, mode=Mode.SUBTRACT)

    result = fix.part
    result.color = _FIXTURE_COLOR
    result.label = "hasma_tap_fixture"
    return result


def export_hasma_tap_fixture_step(
    output_path: Path | None = None,
    **kwargs,
) -> Path:
    """Export the HASMA tap guide fixture as STEP."""
    fixture = build_hasma_tap_fixture(**kwargs)
    if output_path is None:
        output_path = Path("output/hasma_tap_fixture.step")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(fixture, str(output_path))
    return output_path


if __name__ == "__main__":
    # Mirror flexure mounts
    for part in ("CM254-050-G01", "CM254-100-G01"):
        p = export_flexure_mount_step(part_number=part)
        print(f"Mirror mount ({part}): {p}")
        p = export_flexure_assembly_step(part_number=part)
        print(f"Mirror assembly ({part}): {p}")
    # OAP flexure mounts
    p = export_oap_flexure_mount_step(part_number="MPD129-G01")
    print(f"OAP mount: {p}")
    for role in ("M1", "M2"):
        p = export_oap_flexure_assembly_step(
            part_number="MPD129-G01", role=role,
            output_path=Path(f"output/assembly_MPD129-G01_flexure_{role}.step"),
        )
        print(f"OAP assembly ({role}): {p}")
    # Grating flexure mount
    p = export_grating_flexure_mount_step(part_number="GR25-1205")
    print(f"Grating mount: {p}")
    p = export_grating_flexure_assembly_step(part_number="GR25-1205")
    print(f"Grating assembly: {p}")


def _build_mirror_assembly_for_element(
    el, part_number: str, cache: dict,
    *, full_hardware: bool = True,
) -> tuple:
    """Build or retrieve a mirror assembly for scene placement.

    Returns ``(piece, axis)`` where ``axis`` is ``el.axis`` (the optical
    axis) used by ``place_in_scene_frame`` to orient the assembly.
    When *full_hardware* is False, only the mount + optic are included
    (no inserts or setscrews).
    """
    _, _, _, mount_dict = _lookup_mirror_in_bom(part_number)
    mount_type = mount_dict["type"]
    role = "M1" if el.label == "M1" else "M2"

    if mount_type == "oap_flexure":
        cache_key = (part_number, role, full_hardware)
        if cache_key not in cache:
            params = _oap_flexure_params_from_bom(mount_dict)
            if full_hardware:
                cache[cache_key] = build_oap_flexure_full_assembly(
                    part_number, params, role=role)
            else:
                from build123d import Compound
                _, diameter_mm, thickness_mm, _ = _lookup_mirror_in_bom(part_number)
                mount = build_oap_flexure_mount_cad(
                    optic_diameter_mm=diameter_mm,
                    center_thickness_mm=thickness_mm,
                    params=params,
                )
                oap = _load_vendor_oap_at_optical_centre(
                    part_number, rotate_180=(role == "M1"))
                oap.label = part_number
                oap.color = _OPTIC_COLOR
                asm = Compound(children=[mount, oap])
                asm.label = f"assembly_{part_number}_{role}"
                cache[cache_key] = asm
        return cache[cache_key], el.axis
    else:
        cache_key = (part_number, full_hardware)
        if cache_key not in cache:
            if full_hardware:
                cache[cache_key] = build_mirror_flexure_full_assembly(part_number)
            else:
                cache[cache_key] = build_mirror_flexure_assembly(part_number)
        return cache[cache_key], el.axis


def place_all_in_scene_frame(
    scene, m1_part: str, m2_part: str, grating_part: str,
    *, extract_mounts: bool = False,
    full_hardware: bool = True,
) -> tuple[list, dict]:
    """Place all scene elements into their scene-frame poses.

    Returns ``(placed_children, mount_only_parts)``.  When
    *extract_mounts* is True, ``mount_only_parts`` maps element labels
    to the printable mount plate (first child of the assembly) for
    per-part STL export.  Otherwise it is empty.

    When *full_hardware* is False, assemblies contain only mount + optic
    (no inserts, setscrews, or pusher hardware).

    Slit elements are skipped — the HASMA threads directly into the
    housing wall with no separate mount or CAD assembly.
    """
    mirror_asm_cache: dict = {}
    grating_asm: Compound | None = None

    placed: list = []
    placed_by_label: dict[str, list] = {}
    mount_only: dict = {}

    for el in scene.elements:
        normal = el.axis
        position = el.position
        if el.kind == "detector":
            piece = _load_vendor_detector()
            piece.label = "TCD1304"
            piece.color = _DETECTOR_COLOR
        elif el.kind == "slit":
            piece = _load_vendor_fiber_adapter()
            piece.label = "HASMA"
            piece.color = _SLIT_COLOR
        elif el.kind == "mirror":
            if el.label in ("F1", "F2"):
                fm_part = el.params.get("part_number")
                if fm_part is None:
                    import tomllib
                    with _get_bom_path().open("rb") as _f:
                        _bom = tomllib.load(_f)
                    section = "f2_options" if el.label == "F2" else "f1_options"
                    opts = _bom["mirrors"][section]
                    fm_part = next(k for k, v in opts.items() if isinstance(v, dict))
                if full_hardware:
                    piece = build_f_mirror_assembly(
                        fm_part,
                        cylindrical_orientation=el.params.get(
                            "cylindrical_orientation"))
                else:
                    from build123d import Compound as _Cmp
                    _fm_mirror = _load_vendor_f_mirror_at_face(
                        fm_part, cylindrical_orientation=el.params.get(
                            "cylindrical_orientation"))
                    _fm_bb = _fm_mirror.bounding_box()
                    _fm_mirror = _fm_mirror.translate(
                        (-0.5*(_fm_bb.min.X+_fm_bb.max.X),
                         -0.5*(_fm_bb.min.Y+_fm_bb.max.Y), 0))
                    _, _fm_dia, _fm_ct, _fm_md = _lookup_mirror_in_bom(fm_part)
                    _fm_params = _flexure_params_from_bom(_fm_md)
                    _fm_mount = build_mirror_flexure_mount_cad(
                        optic_diameter_mm=_fm_dia,
                        center_thickness_mm=_fm_ct,
                        params=_fm_params)
                    piece = _Cmp(children=[_fm_mount, _fm_mirror])
                    piece.label = f"assembly_{fm_part}"
            else:
                part_number = m1_part if el.label == "M1" else m2_part
                piece, normal = _build_mirror_assembly_for_element(
                    el, part_number, mirror_asm_cache,
                    full_hardware=full_hardware)
                if extract_mounts:
                    mount_only[el.label] = place_in_scene_frame(
                        piece.children[0], position, normal)
        elif el.kind == "grating":
            if grating_asm is None:
                if full_hardware:
                    grating_asm = build_grating_flexure_full_assembly(grating_part)
                else:
                    grating_asm = build_grating_flexure_assembly(grating_part)
            piece = grating_asm
            normal = el.axis
            if extract_mounts:
                mount_only[el.label] = place_in_scene_frame(
                    piece.children[0], position, normal)
        else:
            continue

        # Inject contact bumps into the assembly if present on the mount.
        if hasattr(piece, 'children') and piece.children:
            mount_child = piece.children[0]
            if hasattr(mount_child, '_contact_bumps'):
                from build123d import Compound as _Cmp
                piece = _Cmp(children=list(piece.children)
                             + [mount_child._contact_bumps])
                piece.label = f"assembly_{el.label}"

        placed_piece = place_in_scene_frame(piece, position, normal)
        placed.append(placed_piece)
        placed_by_label.setdefault(el.label, []).append(placed_piece)

    return placed, mount_only, placed_by_label


def build_laser_alignment_holder(
    *,
    laser_bore_dia_mm: float = 10.5,
    laser_bore_depth_mm: float = 20.0,
    hasma_bore_dia_mm: float = 5.5,
    hasma_bore_depth_mm: float = 8.0,
    wall_mm: float = 3.0,
    insert_depth_mm: float = 10.0,
    mfg: ManufacturingParams | None = None,
) -> Part:
    """Laser-to-fiber alignment holder.

    Cuboid with coaxial bores: laser module bore on one end, tappable
    HASMA (1/4-36 UNS) bore on the other.  An M2 heat-set insert on the
    sidewall at *insert_depth_mm* from the laser entry face holds the
    laser module via a set screw.

    Frame: bore axis along Z.  Laser enters from +Z, fiber exits -Z.
    """
    if mfg is None:
        mfg = _load_manufacturing()

    t = mfg.assembly_clearance_mm + mfg.print_tolerance_mm
    laser_bore_r = 0.5 * laser_bore_dia_mm + t
    hasma_bore_r = 0.5 * hasma_bore_dia_mm + t
    cross_mm = 2 * laser_bore_r + 2 * wall_mm
    total_length_mm = laser_bore_depth_mm + hasma_bore_depth_mm

    with BuildPart() as bp:
        # Outer cuboid — centered at XY, Z from 0 (HASMA face) to total_length
        with BuildSketch(Plane.XY):
            Rectangle(cross_mm, cross_mm)
        extrude(amount=total_length_mm)

        # Laser bore — from +Z face, 20mm deep
        laser_entry_plane = Plane(
            origin=(0, 0, total_length_mm),
            x_dir=(1, 0, 0),
            z_dir=(0, 0, -1),
        )
        with BuildSketch(laser_entry_plane):
            Circle(radius=laser_bore_r)
        extrude(amount=laser_bore_depth_mm, mode=Mode.SUBTRACT)

        # HASMA tap bore — from -Z face (Z=0), coaxial
        hasma_entry_plane = Plane(
            origin=(0, 0, 0),
            x_dir=(1, 0, 0),
            z_dir=(0, 0, 1),
        )
        with BuildSketch(hasma_entry_plane):
            Circle(radius=hasma_bore_r)
        extrude(amount=hasma_bore_depth_mm, mode=Mode.SUBTRACT)

        # M2 heat-set insert bore — sidewall at insert_depth from laser entry
        insert_z = total_length_mm - insert_depth_mm
        insert_plane = Plane(
            origin=(0.5 * cross_mm, 0, insert_z),
            x_dir=(0, 0, -1),
            z_dir=(-1, 0, 0),
        )
        wall_to_bore_mm = 0.5 * cross_mm - laser_bore_r
        with BuildSketch(insert_plane):
            Circle(radius=0.5 * mfg.insert_bore_dia_mm)
        extrude(amount=mfg.insert_length_mm, mode=Mode.SUBTRACT)

        # M2 screw through-hole — full depth from outer wall to laser bore
        screw_dia_mm = mfg.bolt_dims["M2"]["thread_dia_mm"]
        with BuildSketch(insert_plane):
            Circle(radius=0.5 * screw_dia_mm)
        extrude(amount=0.5 * cross_mm, mode=Mode.SUBTRACT)

        # Insert entry chamfer
        chamfer_plane = Plane(
            origin=(0.5 * cross_mm, 0, insert_z),
            x_dir=(0, 0, -1),
            z_dir=(-1, 0, 0),
        )
        chamfer_top_r = 0.5 * mfg.insert_bore_dia_mm + mfg.insert_chamfer_mm
        chamfer_bot_r = 0.5 * mfg.insert_bore_dia_mm
        with BuildSketch(chamfer_plane):
            Circle(radius=chamfer_top_r)
        extrude(amount=mfg.insert_chamfer_mm, mode=Mode.SUBTRACT)

    result = bp.part
    result.label = "laser_alignment_holder"
    result.color = _MOUNT_COLOR
    return result


def build_cfl_shade(
    *,
    inner_dia_mm: float = 65.0,
    height_mm: float = 135.0,
    wall_mm: float = 5.0,
    cap_mm: float = 5.0,
    tap_drill_dia_mm: float = 5.5,
) -> Part:
    """CFL lamp shade with HASMA bore for fiber coupling.

    Open cylinder that slips over a lamp fixture base, closed on the
    opposite end with a 1/4"-36 tappable pilot bore at the centre.

    Frame: Z-axis along cylinder axis.  Open end at Z=0, closed cap at
    Z = height_mm + cap_mm.
    """
    inner_r = 0.5 * inner_dia_mm
    outer_r = inner_r + wall_mm
    total_h = height_mm + cap_mm

    with BuildPart() as bp:
        Cylinder(
            radius=outer_r, height=total_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        with BuildSketch(Plane.XY):
            Circle(inner_r)
        extrude(amount=height_mm, mode=Mode.SUBTRACT)

        cap_top = Plane.XY.offset(total_h)
        with BuildSketch(cap_top):
            Circle(0.5 * tap_drill_dia_mm)
        extrude(amount=-cap_mm, mode=Mode.SUBTRACT)

    result = bp.part
    result.label = "cfl_shade"
    result.color = _MOUNT_COLOR
    return result


def build_laser_alignment_screen(
    *,
    centerline_height_mm: float = 22.5,
    screen_dia_mm: float = 20.0,
    slab_width_mm: float = 25.0,
    slab_thickness_mm: float = 3.0,
    slab_margin_mm: float = 8.0,
    base_width_mm: float = 25.0,
    base_depth_mm: float = 12.0,
    base_thickness_mm: float = 5.0,
    disk_thickness_mm: float = 1.5,
    mark_width_mm: float = 0.8,
    mark_gap_mm: float = 3.0,
    tongue_notch_width_mm: float = 0.0,
    tongue_notch_clearance_mm: float = 0.3,
    ridge_width_mm: float = 0.0,
    ridge_clearance_mm: float = 0.15,
    ridge_height_mm: float = 0.8,
    corner_fillet_mm: float = 0.0,
) -> tuple[Part, Part, Part]:
    """Laser alignment screen — three bodies for multi-color printing.

    Returns (frame, disk, reticle):

    - **frame**: tombstone slab on a base (inverted-T profile) with a
      circular through-hole.  Print in any color.
    - **disk**: circular insert occupying the front half of the
      aperture, with through-holes for the crosshair marks.
      Print in white for visibility.
    - **reticle**: absorbing backing disk occupying the rear half of
      the aperture, with four radial marks that protrude forward
      through the white disk's holes, flush with its front face.
      Print in black for contrast.  No overhangs when printed
      standing on the base.

    The screen center sits at *centerline_height_mm* above the base
    bottom.  *mark_gap_mm* is the clearance between the mark ends
    and both the disk center and rim.

    Frame: slab face normal along +Y, base on XZ plane, Z is up.
    """
    slab_top_mm = centerline_height_mm + 0.5 * screen_dia_mm + slab_margin_mm
    slab_height_mm = slab_top_mm - base_thickness_mm
    disk_r = 0.5 * screen_dia_mm
    backing_thickness_mm = slab_thickness_mm - disk_thickness_mm
    mark_len = disk_r - 2 * mark_gap_mm

    # ── Frame (tombstone + base, with aperture hole) ─────────────
    with BuildPart() as bp_frame:
        base_plane = Plane(
            origin=(0, 0, 0.5 * base_thickness_mm),
            x_dir=(1, 0, 0),
            z_dir=(0, 0, 1),
        )
        with BuildSketch(base_plane):
            Rectangle(base_width_mm, base_depth_mm)
        extrude(amount=0.5 * base_thickness_mm, both=True)

        if tongue_notch_width_mm > 0:
            notch_w = tongue_notch_width_mm + 2 * tongue_notch_clearance_mm
            with BuildSketch(base_plane):
                Rectangle(notch_w, base_depth_mm + 1.0)
            extrude(amount=0.5 * base_thickness_mm, both=True,
                    mode=Mode.SUBTRACT)

        if ridge_width_mm > 0 and tongue_notch_width_mm > 0:
            boss_half = 0.5 * tongue_notch_width_mm
            rc = ridge_clearance_mm
            rw = ridge_width_mm
            cut_h = ridge_height_mm + tongue_notch_clearance_mm
            cut_plane = Plane(
                origin=(0, 0, 0),
                x_dir=(1, 0, 0),
                z_dir=(0, 0, 1),
            )
            # Tongue ridge slots: alongside the tongue, outboard of
            # the tongue notch.  Shallow pocket from the base bottom.
            slot_inner = boss_half + rc - tongue_notch_clearance_mm
            slot_outer = boss_half + rc + rw + tongue_notch_clearance_mm
            slot_w = slot_outer - slot_inner
            slot_cx = 0.5 * (slot_inner + slot_outer)
            for sign in (+1, -1):
                with BuildSketch(cut_plane):
                    with Locations([(sign * slot_cx, 0)]):
                        Rectangle(slot_w, base_depth_mm + 1.0)
                extrude(amount=cut_h, mode=Mode.SUBTRACT)
            # Back ridge slot: across the back (-Y) edge of the base,
            # clearing the arm ridges at u=0.
            arm_slot_depth = rc + rw + tongue_notch_clearance_mm
            arm_slot_cy = -0.5 * base_depth_mm + 0.5 * arm_slot_depth
            with BuildSketch(cut_plane):
                with Locations([(0, arm_slot_cy)]):
                    Rectangle(base_width_mm + 1.0, arm_slot_depth)
            extrude(amount=cut_h, mode=Mode.SUBTRACT)

            # Fillet the vertical edges at the inner back corners of
            # the tongue notch where it abuts the mount corner.
            # Must clear both the foot fillet (fillet_radius_mm from
            # BOM) and the L-ridge inner arc.
            corner_r = corner_fillet_mm + tongue_notch_clearance_mm
            half_notch = 0.5 * notch_w
            back_y = -0.5 * base_depth_mm
            _corner_edges = [
                e for e in bp_frame.edges()
                if e.geom_type == GeomType.LINE
                and e.length > 1.0
                and abs(abs(e.center().X) - half_notch) < 0.2
                and abs(e.center().Y - back_y) < 0.5
            ]
            if _corner_edges:
                from build123d import fillet as fillet3d
                fillet3d(_corner_edges, radius=corner_r)

            # Fillet the vertical edges where the tongue ridge slots
            # meet the back ridge slot — clearing the L-ridge arc.
            ridge_arc_r = corner_fillet_mm - ridge_clearance_mm
            if ridge_arc_r > 0.1:
                ridge_fillet_r = ridge_arc_r + tongue_notch_clearance_mm
                arm_inner_y = (-0.5 * base_depth_mm
                               + rc + rw + tongue_notch_clearance_mm)
                _ridge_edges = [
                    e for e in bp_frame.edges()
                    if e.geom_type == GeomType.LINE
                    and abs(e.length - cut_h) < 0.2
                    and abs(abs(e.center().X) - slot_outer) < 0.2
                    and abs(e.center().Y - arm_inner_y) < 0.2
                ]
                if _ridge_edges:
                    fillet3d(_ridge_edges, radius=ridge_fillet_r)

        slab_z_center = base_thickness_mm + 0.5 * slab_height_mm
        slab_plane = Plane(
            origin=(0, 0, slab_z_center),
            x_dir=(1, 0, 0),
            z_dir=(0, 0, 1),
        )
        with BuildSketch(slab_plane):
            Rectangle(slab_width_mm, slab_thickness_mm)
        extrude(amount=0.5 * slab_height_mm, both=True)

        aperture_plane = Plane(
            origin=(0, 0.5 * slab_thickness_mm, centerline_height_mm),
            x_dir=(1, 0, 0),
            z_dir=(0, -1, 0),
        )
        with BuildSketch(aperture_plane):
            Circle(radius=disk_r)
        extrude(amount=slab_thickness_mm, mode=Mode.SUBTRACT)

    frame = bp_frame.part
    frame.label = "laser_alignment_screen"
    frame.color = _MOUNT_COLOR

    # ── Disk (white, front half, with through-holes for marks) ───
    disk_front_y = 0.5 * slab_thickness_mm
    disk_plane = Plane(
        origin=(0, disk_front_y, centerline_height_mm),
        x_dir=(1, 0, 0),
        z_dir=(0, -1, 0),
    )
    with BuildPart() as bp_disk:
        with BuildSketch(disk_plane):
            Circle(radius=disk_r)
        extrude(amount=disk_thickness_mm)

        for angle_deg in (0, 90, 180, 270):
            rot_plane = disk_plane.rotated((0, angle_deg, 0))
            with BuildSketch(rot_plane):
                with Locations([(0.5 * mark_len + mark_gap_mm, 0)]):
                    Rectangle(mark_len, mark_width_mm)
            extrude(amount=disk_thickness_mm, mode=Mode.SUBTRACT)

    disk = bp_disk.part
    disk.label = "laser_alignment_disk"
    disk.color = Color(0.95, 0.95, 0.95)

    # ── Reticle (black backing disk + marks protruding through) ──
    backing_front_y = disk_front_y - disk_thickness_mm
    backing_plane = Plane(
        origin=(0, backing_front_y, centerline_height_mm),
        x_dir=(1, 0, 0),
        z_dir=(0, -1, 0),
    )
    mark_plane = Plane(
        origin=(0, backing_front_y, centerline_height_mm),
        x_dir=(1, 0, 0),
        z_dir=(0, 1, 0),
    )
    with BuildPart() as bp_reticle:
        with BuildSketch(backing_plane):
            Circle(radius=disk_r)
        extrude(amount=backing_thickness_mm)

        for angle_deg in (0, 90, 180, 270):
            rot_plane = mark_plane.rotated((0, angle_deg, 0))
            with BuildSketch(rot_plane):
                with Locations([(0.5 * mark_len + mark_gap_mm, 0)]):
                    Rectangle(mark_len, mark_width_mm)
            extrude(amount=disk_thickness_mm)

    reticle = bp_reticle.part
    reticle.label = "laser_alignment_reticle"
    reticle.color = Color(0.1, 0.1, 0.1)

    return frame, disk, reticle
