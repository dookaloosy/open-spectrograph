"""Unified export: PNG render, STEP assembly, or full CAD from one scene.

One ``assemble_scene()`` call builds the Scene; every output mode
consumes that same object, guaranteeing no raysect/CAD drift.

Usage::

    # From a GA run (auto-detects design and parts):
    python export.py --run output/optimizer_20260426_205033 --render
    python export.py --run output/optimizer_20260426_205033 --candidate gen01_cand00 --cad

    # Manual / debug (default design=czerny; --L_a, --Dv, --L_b all required together):
    python export.py --render --L_a 50 --L_b 50 --Dv 30 --L_m1 100 --L_m2 100 --theta1 20 --theta2 20
    python export.py --render                              # baseline genome
"""


import argparse
import csv
import dataclasses
import json
import sys
from dataclasses import asdict
from pathlib import Path

from designs import get_design
from designs.czerny_bom import load_parts
from optics.scene import InfeasibleGeometry


# ── GA run loader ──────────────────────────────────────────────────────


def _load_winner(run_dir: str, candidate_id: str | None):
    """Load genome from optimizer output.

    Returns ``(design_mod, genome, parts, label, run_flags)``.
    ``run_flags`` is a dict with ``optics_only`` and ``point_source``
    booleans read from ``fixed_params``.
    """
    from optimizer_engine import load_state

    run_path = Path(run_dir)
    state = load_state(str(run_path))
    if state is None:
        raise SystemExit(f"No optimizer state in {run_dir}")

    fixed = state["fixed_params"]

    from designs.czerny_bom import set_bom_path
    set_bom_path(fixed["bom_path"])
    try:
        from optics.mounts_cad import set_bom_path as set_cad_bom_path
        set_cad_bom_path(fixed["bom_path"])
    except ImportError:
        pass

    problem_name = state.get("problem_name", "czerny")
    design = get_design(problem_name)

    candidates = state["candidates"]
    if candidate_id is not None:
        cand = next((c for c in candidates if c["id"] == candidate_id), None)
        if cand is None:
            raise SystemExit(f"Candidate {candidate_id!r} not found in {run_dir}")
        label = candidate_id
    else:
        scored = [c for c in candidates
                  if c.get("fine_fitness") is not None
                  and c["fine_fitness"] < 999_999
                  and c.get("best_point") is not None]
        if not scored:
            scored = [c for c in candidates
                      if c.get("coarse_fitness") is not None
                      and c["coarse_fitness"] < 999_999
                      and c.get("best_point") is not None]
        if not scored:
            raise SystemExit(f"No viable candidates in {run_dir}")
        cand = min(scored, key=lambda c: c.get("fine_fitness")
                   or c["coarse_fitness"])
        label = cand["id"]

    fp = cand["full_params"]
    bp = cand.get("best_point", {})

    # best_point carries NM-refined values (L_m1, L_m2, theta_f1, …)
    # that override the coarse evolved params in full_params.  Merge
    # with bp last so refined values win on shared keys.
    merged = {**fp, **bp}
    genome, parts = design.reconstruct_genome(merged, merged, fixed)

    run_flags = {
        "optics_only": fixed["optics_only"],
        "point_source": fixed["point_source"],
        "bom_path": fixed.get("bom_path"),
        "center": fixed.get("center"),
        "min_bw": fixed.get("min_bw"),
        "max_rld": fixed.get("max_rld"),
        "max_fnum": fixed.get("max_fnum"),
        "fold_mode": fixed.get("fold_mode"),
    }
    return design, genome, parts, label, run_flags


# ── Argument parsing ───────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output modes (at least one required).
    modes = ap.add_argument_group("output modes")
    modes.add_argument("--render", action="store_true",
                       help="Render a PNG via raysect")
    modes.add_argument("--step", action="store_true",
                       help="Export vendor-optics STEP assembly")
    modes.add_argument("--cad", action="store_true",
                       help="Full CAD: scene STEP + housing STEP + per-part STEP")
    modes.add_argument("--layout", action="store_true",
                       help="2-D top-down layout diagram (matplotlib PNG)")
    modes.add_argument("--save-baseline", type=str, default=None,
                       metavar="PATH",
                       help="Save winning genome as a baseline TOML")

    # GA run source.
    run_group = ap.add_argument_group("GA run (primary)")
    run_group.add_argument("--run", type=str, default=None,
                           help="Optimizer run dir")
    run_group.add_argument("--candidate", type=str, default=None,
                           help="Candidate ID (default: best winner)")

    # Manual / debug source.
    debug = ap.add_argument_group("manual / debug")
    debug.add_argument("--design", type=str, default="czerny",
                       help="Design name for manual mode (default: czerny)")

    # Register design-specific genome args for manual mode.
    # (Ignored when --run is used.)
    design = get_design("czerny")
    design.add_genome_args(debug)

    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Destination directory (default: output/export_<label>/)")
    ap.add_argument("--bom", type=str, default=None,
                    help="Path to BOM TOML (default: data/czerny_bom_v0_design.toml)")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Path to baseline genome TOML (overrides CLI genome args)")

    # Housing.
    housing = ap.add_argument_group("housing")
    housing.add_argument("--wall-albedo", type=float, default=None,
                         help="Override BOM wall_albedo (0..1)")

    # Render options.
    render = ap.add_argument_group("render (--render only)")
    render.add_argument("--pixels", type=int, nargs=2, default=(1280, 960),
                        metavar=("W", "H"))
    render.add_argument("--samples", type=int, default=100,
                        help="pixel_samples per pixel")
    render.add_argument("--spectral-bins", type=int, default=24)
    render.add_argument("--spectral-rays", type=int, default=None,
                        help="Rays per pixel-sample per spectral bin")
    render.add_argument("--camera-distance", type=float, default=0.6,
                        help="Camera distance from origin (metres)")
    render.add_argument("--camera-azimuth", type=float, default=225.0)
    render.add_argument("--camera-elevation", type=float, default=30.0)
    render.add_argument("--camera-fov", type=float, default=40.0)
    render.add_argument("--cam-at", type=str, default=None,
                        help="Place camera at this element's normal")
    render.add_argument("--cam-look", type=str, default=None,
                        help="Aim camera at this element (default: grating)")
    render.add_argument("--cam-offset-mm", type=float, default=2.0,
                        help="Offset along --cam-at element normal (mm)")
    render.add_argument("--no-checkerboard", action="store_true")
    render.add_argument("--unsaturated-fraction", type=float, default=0.995)
    render.add_argument("--orthographic", action="store_true")
    render.add_argument("--ortho-width-mm", type=float, default=30.0)

    args = ap.parse_args()
    if not (args.render or args.step or args.cad or args.layout
            or args.save_baseline):
        ap.error("at least one of --render, --step, --cad, --layout, "
                 "--save-baseline is required")
    return args


# ── Shared assembly ─────────────────────────────────────────────────────


def _assemble(design, genome, parts, args, *, optics_only: bool = False):
    """Shared scene assembly — single source of truth for all output modes."""
    if args.wall_albedo is not None:
        parts = dataclasses.replace(parts, wall_albedo=args.wall_albedo)

    if optics_only:
        scene = design.build_optics_only_scene(genome, parts)
        return scene, False

    try:
        scene = design.assemble_scene(
            genome, parts,
            scene_builder=design.build_optics_only_scene,
        )
        has_housing = True
    except InfeasibleGeometry as exc:
        print(f"[warn] assembly infeasible ({exc}); "
              f"falling back to optics-only scene.",
              file=sys.stderr)
        scene = design.build_optics_only_scene(genome, parts)
        has_housing = False

    return scene, has_housing


# ── Render (PNG) ────────────────────────────────────────────────────────


def _do_render(args, genome, parts, scene, has_housing, label, out_dir):
    from raysect.core.math import Point3D
    from optics.world_builder import build_world, _placement_transform, MM_TO_M
    from optics.render import (
        build_checkerboard_room,
        camera_look_from_to,
        camera_transform,
        element_normal_vec,
        element_position_m,
        render_scene_png,
    )

    input_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    built = build_world(scene, input_fnum=input_fnum)

    lid_removed = False
    if has_housing:
        plate_top = built.primitives.get("plate_top")
        if plate_top is not None:
            plate_top.parent = None
            lid_removed = True

    # Hide simulation-only primitives.
    hidden: list[str] = []
    for name in ("entrance_slit",):
        prim = built.primitives.get(name)
        if prim is not None:
            prim.parent = None
            hidden.append(name)

    # Place exit-slit plate (visible jaw, no emitter).
    from raysect.primitive import Box, Subtract
    from raysect.optical import ConstantSF
    from raysect.optical.material.lambert import Lambert

    exit_el = next((e for e in scene.elements if e.label == "exit_slit"), None)
    has_exit = False
    if exit_el is not None:
        half_w = 0.5 * exit_el.params["width_um"] * 1.0e-6
        half_h = 0.5 * exit_el.params["height_mm"] * MM_TO_M
        jaw_half = 0.025
        t = 1e-3
        plate = Box(lower=Point3D(-jaw_half, -jaw_half, -t),
                    upper=Point3D(+jaw_half, +jaw_half, 0.0))
        slot = Box(lower=Point3D(-half_w, -half_h, -2 * t),
                   upper=Point3D(+half_w, +half_h, t))
        jaw = Subtract(plate, slot)
        jaw.material = Lambert(ConstantSF(parts.wall_albedo))
        jaw.transform = _placement_transform(exit_el)
        jaw.name = "exit_slit"
        jaw.parent = built.world
        has_exit = True

    if not args.no_checkerboard:
        build_checkerboard_room(built.world)

    # Camera placement.
    if args.cam_at is not None:
        base = element_position_m(scene, args.cam_at)
        normal = element_normal_vec(scene, args.cam_at)
        offset_m = args.cam_offset_mm * 1e-3
        cam_pos = Point3D(base.x + normal.x * offset_m,
                          base.y + normal.y * offset_m,
                          base.z + normal.z * offset_m)
        aim_label = args.cam_look or "grating"
        aim = element_position_m(scene, aim_label)
        cam_xform = camera_look_from_to(cam_pos, aim)
    else:
        cam_xform = camera_transform(
            args.camera_distance,
            args.camera_azimuth,
            args.camera_elevation,
        )

    png_path = out_dir / "scene.png"
    print(f"Rendering ({args.pixels[0]}x{args.pixels[1]}, "
          f"{args.samples} spp, {args.spectral_bins} bins)...")
    render_scene_png(
        built.world, png_path,
        pixels=tuple(args.pixels),
        samples=args.samples,
        spectral_bins=args.spectral_bins,
        spectral_rays=args.spectral_rays,
        fov_deg=args.camera_fov,
        camera_transform_matrix=cam_xform,
        unsaturated_fraction=args.unsaturated_fraction,
        orthographic=args.orthographic,
        ortho_width_mm=args.ortho_width_mm,
    )

    manifest = {
        "label": label,
        "genome": asdict(genome),
        "camera": {
            "distance_m": args.camera_distance,
            "azimuth_deg": args.camera_azimuth,
            "elevation_deg": args.camera_elevation,
            "fov_deg": args.camera_fov,
            "pixels": list(args.pixels), "samples": args.samples,
            "spectral_bins": args.spectral_bins,
        },
        "housing": has_housing, "lid_removed": lid_removed,
        "checkerboard_room": not args.no_checkerboard,
        "wall_albedo": parts.wall_albedo,
        "hidden_non_bom": hidden, "has_exit_slit": has_exit,
        "orthographic": args.orthographic,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {png_path}")


# ── STEP (vendor optics) ───────────────────────────────────────────────


def _do_step(args, genome, parts, scene, label, out_dir):
    from build123d import Compound, export_step
    from optics.mounts_cad import place_all_in_scene_frame

    placed, _, _ = place_all_in_scene_frame(
        scene, parts.m1_part, parts.m2_part, parts.grating_part,
        full_hardware=False,
    )
    if not placed:
        print("No placeable elements found.", file=sys.stderr)
        return None

    step_path = out_dir / "scene.step"
    asm = Compound(children=placed)
    export_step(asm, str(step_path))
    print(f"Wrote {step_path}")
    return placed


# ── CAD (housing STEP) ──────────────────────────────────────────────────


def _do_cad(args, design, genome, parts, scene, has_housing, placed, out_dir,
            *, design_wavelength_nm: float = 550.0,
            min_bw_nm: float = 300.0,
            bom_path: str | None = None):
    from build123d import Compound, export_step
    from optics.housing_cad import (
        build_solid_housing_cad,
        printability_warnings,
    )

    housing_compound = None
    housing_body = None
    housing_spec = None
    top_cover = None
    det_cover = None
    bottom_cover = None
    placed_by_label = None

    if has_housing:
        # Solid mode: build the pure-geometry spec, then CAD.
        from optics.housing import build_solid_housing_spec
        from designs.czerny_assembly import CzernyAssembly
        from optics.assembly import SpectrographAssembly

        optics_scene = design.build_optics_only_scene(genome, parts)
        asm_helper = SpectrographAssembly()
        asm_helper.mount_bom_key_by_label = {
            "M1": "m1_mount", "grating": "grating_mount", "M2": "m2_mount",
            **({"F1": "f1_mount"} if parts.f1_mount else {}),
            **({"F2": "f2_mount"} if parts.f2_mount else {}),
        }
        mounts = asm_helper.build_mounts(optics_scene, parts)
        ct = CzernyAssembly()
        beam_path = ct.resolve_beam_path(optics_scene)
        from optics.housing_cad import (
            _CAVITY_POCKET_CLEARANCE_MM,
            _CAVITY_POCKET_FILLET_RADIUS_MM,
            _PCBA_POCKET_CLEARANCE_MM,
        )
        housing_spec = build_solid_housing_spec(
            optics_scene, mounts, parts, beam_path,
            cavity_clearance_mm=_CAVITY_POCKET_CLEARANCE_MM,
            cavity_endmill_radius_mm=_CAVITY_POCKET_FILLET_RADIUS_MM,
            pcba_pocket_clearance_mm=_PCBA_POCKET_CLEARANCE_MM,
        )
        (housing_compound, housing_body, top_cover, det_cover,
         bottom_cover, placed_by_label) = build_solid_housing_cad(
            housing_spec, optics_scene, parts,
            design_wavelength_nm=design_wavelength_nm,
            min_bw_nm=min_bw_nm,
        )

    warnings = printability_warnings(housing_spec)
    for w in warnings:
        print(f"[printability] {w}")

    if housing_compound is not None:
        full_path = out_dir / "full_assembly.step"
        export_step(housing_compound, str(full_path))
        print(f"Wrote {full_path}")

    if housing_body is not None:
        housing_path = out_dir / "housing.step"
        export_step(housing_body, str(housing_path))
        print(f"Wrote {housing_path}")

    if top_cover is not None:
        top_cover_path = out_dir / "top_cover.step"
        export_step(top_cover, str(top_cover_path))
        print(f"Wrote {top_cover_path}")

    if det_cover is not None:
        det_cover_path = out_dir / "detector_cover.step"
        export_step(det_cover, str(det_cover_path))
        print(f"Wrote {det_cover_path}")

    if bottom_cover is not None:
        bottom_cover_path = out_dir / "bottom_cover.step"
        export_step(bottom_cover, str(bottom_cover_path))
        print(f"Wrote {bottom_cover_path}")

    # Per-role flexure mount STEP files (F1, M1, grating, M2, etc.)
    # Export mount + captive TPU contact bumps as a single Compound.
    if placed_by_label is not None:
        for role, parts_list in placed_by_label.items():
            if role in ("entrance_slit", "detector"):
                continue
            if not parts_list:
                continue
            asm = parts_list[0]
            mount = asm.children[0] if asm.children else asm
            mount_children = [mount]
            if hasattr(mount, "_contact_bumps"):
                mount_children.append(mount._contact_bumps)
            mount_compound = Compound(children=mount_children)
            mount_path = out_dir / f"{role.lower()}_mount.step"
            export_step(mount_compound, str(mount_path))
            print(f"Wrote {mount_path}")

    # Assembly fixtures
    from optics.mounts_cad import (
        build_mirror_assembly_fixture, build_grating_assembly_fixture,
        build_hasma_tap_fixture,
    )
    mirror_parts = {
        "m1": getattr(parts, "m1_part", None),
        "m2": getattr(parts, "m2_part", None),
        "f1": getattr(parts, "f1_part", None),
    }
    for role, pn in mirror_parts.items():
        if pn is None:
            continue
        try:
            fixture = build_mirror_assembly_fixture(pn, label=role.upper())
            fixture_path = out_dir / f"{role}_fixture.step"
            export_step(fixture, str(fixture_path))
            print(f"Wrote {fixture_path}")
        except Exception as e:
            print(f"[fixture] {role} skipped: {e}")
    grating_pn = getattr(parts, "grating_part", None)
    if grating_pn is not None:
        try:
            fixture = build_grating_assembly_fixture(grating_pn)
            fixture_path = out_dir / "grating_fixture.step"
            export_step(fixture, str(fixture_path))
            print(f"Wrote {fixture_path}")
        except Exception as e:
            print(f"[fixture] grating skipped: {e}")

    # HASMA tap guide fixture
    try:
        tap_fixture = build_hasma_tap_fixture()
        tap_path = out_dir / "hasma_tap_fixture.step"
        export_step(tap_fixture, str(tap_path))
        print(f"Wrote {tap_path}")
    except Exception as e:
        print(f"[fixture] hasma tap skipped: {e}")

    # Laser alignment screen (standalone STEP)
    try:
        from optics.mounts_cad import build_laser_alignment_screen
        import tomllib as _tomllib
        _bom_p = Path(bom_path) if bom_path else Path("data/czerny_bom_v0_design.toml")
        with _bom_p.open("rb") as _f:
            _bom = _tomllib.load(_f)
        _mfg = _bom["manufacturing"]
        _bolt_head = float(_mfg["bolts"]["M2"]["head_dia_mm"])
        _mount0 = _bom["mirrors"]["m1_options"][
            list(_bom["mirrors"]["m1_options"])[0]]["mount"]
        _tongue_w = _bolt_head + float(_mount0["bolt_safety_mm"])
        _fillet_r = float(_mfg["fillet_radius_mm"])
        frame, disk, reticle = build_laser_alignment_screen(
            tongue_notch_width_mm=_tongue_w,
            ridge_width_mm=1.0,
            corner_fillet_mm=_fillet_r)
        screen_asm = Compound(children=[frame, disk, reticle])
        screen_asm.label = "laser_alignment_screen"
        screen_path = out_dir / "alignment_screen.step"
        export_step(screen_asm, str(screen_path))
        print(f"Wrote {screen_path}")
    except Exception as e:
        print(f"[fixture] alignment screen skipped: {e}")

    if housing_compound is not None:
        _write_bom_csv(parts, bom_path, housing_compound, out_dir)

    return warnings


# ── Layout diagram (2-D top-down) ──────────────────────────────────────


def _do_layout(args, genome, parts, scene, label, out_dir):
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from designs.czerny_assembly import CzernyAssembly
    from optics.collision import polygon_for_element, _axis_2d

    ct = CzernyAssembly()
    beam_path = ct.resolve_beam_path(scene)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_title(f"Layout: {label}", fontsize=13)

    _ELEMENT_COLORS = {
        "slit": "#4477AA",
        "mirror": "#EE6677",
        "grating": "#228833",
        "detector": "#CCBB44",
    }

    for el in scene.elements:
        poly = polygon_for_element(el)
        if poly is None:
            continue
        color = _ELEMENT_COLORS.get(el.kind, "#BBBBBB")
        patch = mpatches.Polygon(
            poly, closed=True,
            facecolor=color, edgecolor="black",
            alpha=0.55, linewidth=1.2,
        )
        ax.add_patch(patch)

        fwd, right = _axis_2d(el)
        x, y = el.position[0], el.position[1]
        ax.annotate(
            el.label, (x, y),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, fontweight="bold",
            color=color,
        )

    for i in range(len(beam_path) - 1):
        ea = beam_path[i]
        eb = beam_path[i + 1]
        ax.plot(
            [ea.position[0], eb.position[0]],
            [ea.position[1], eb.position[1]],
            color="#888888", linewidth=1.0, linestyle="--", zorder=1,
        )

    _SEGMENT_NAMES = {
        ("entrance_slit", "F1"): "L_a−L_f1",
        ("F1", "M1"): "L_f1",
        ("entrance_slit", "M1"): "L_a",
        ("M1", "grating"): "L_m1",
        ("grating", "M2"): "L_m2",
        ("M2", "F2"): "L_b−L_f2",
        ("F2", "detector"): "L_f2",
        ("M2", "detector"): "L_b",
    }

    for i in range(len(beam_path) - 1):
        ea = beam_path[i]
        eb = beam_path[i + 1]
        x0, y0 = ea.position[0], ea.position[1]
        x1, y1 = eb.position[0], eb.position[1]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dist = math.hypot(x1 - x0, y1 - y0)

        seg_key = (ea.label, eb.label)
        seg_name = _SEGMENT_NAMES.get(seg_key, "")
        dist_label = f"{dist:.1f} mm"
        if seg_name:
            dist_label = f"{seg_name} = {dist_label}"

        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        if ln > 1e-6:
            nx, ny = -dy / ln, dx / ln
        else:
            nx, ny = 0.0, 1.0
        off = 5.0
        ax.annotate(
            dist_label, (mx + off * nx, my + off * ny),
            fontsize=7, color="#555555", ha="center", va="center",
            rotation=math.degrees(math.atan2(dy, dx)),
            rotation_mode="anchor",
        )

    _ANGLE_NAMES = {
        "alpha_deg": ("α", genome.alpha_deg),
        "beta_deg": ("β", genome.beta_deg),
        "theta_m1_deg": ("θ_M1", genome.theta_m1_deg),
        "theta_m2_deg": ("θ_M2", genome.theta_m2_deg),
        "theta_d_deg": ("θ_D", genome.theta_d_deg),
    }
    if genome.theta_f1_deg is not None:
        _ANGLE_NAMES["theta_f1_deg"] = ("θ_F1", genome.theta_f1_deg)
    if genome.theta_f2_deg is not None:
        _ANGLE_NAMES["theta_f2_deg"] = ("θ_F2", genome.theta_f2_deg)

    angle_lines = [f"{sym} = {val:.2f}°" for sym, val in _ANGLE_NAMES.values()]
    genome_lines = [
        f"L_a = {genome.L_a_mm:.2f} mm",
        f"L_b = {genome.L_b_mm:.2f} mm",
        f"L_m1 = {genome.L_m1_mm:.2f} mm",
        f"L_m2 = {genome.L_m2_mm:.2f} mm",
        f"Dv = {abs(genome.alpha_deg - genome.beta_deg):.2f}°",
    ]
    if genome.L_f1_mm is not None:
        genome_lines.append(f"L_f1 = {genome.L_f1_mm:.2f} mm")
    if genome.L_f2_mm is not None:
        genome_lines.append(f"L_f2 = {genome.L_f2_mm:.2f} mm")

    info_text = "\n".join(angle_lines + [""] + genome_lines)
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9),
    )

    grating = scene.by_label("grating")
    gx, gy = grating.position[0], grating.position[1]
    n_grating = grating.axis
    arrow_len = 8.0
    ax.annotate(
        "", xy=(gx + n_grating[0] * arrow_len, gy + n_grating[1] * arrow_len),
        xytext=(gx, gy),
        arrowprops=dict(arrowstyle="->", color="#228833", lw=1.5),
    )
    ax.text(gx + n_grating[0] * (arrow_len + 2),
            gy + n_grating[1] * (arrow_len + 2),
            "n̂", fontsize=9, color="#228833", ha="center")

    for mirror_label in ("M1", "M2", "F1", "F2"):
        if mirror_label not in {el.label for el in scene.elements}:
            continue
        mel = scene.by_label(mirror_label)
        mx, my = mel.position[0], mel.position[1]
        fwd, _ = _axis_2d(mel)
        ax.annotate(
            "", xy=(mx + fwd[0] * arrow_len, my + fwd[1] * arrow_len),
            xytext=(mx, my),
            arrowprops=dict(arrowstyle="->", color="#EE6677", lw=1.2),
        )

    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.autoscale()
    pad = 15
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    ax.set_xlim(xl - pad, xr + pad)
    ax.set_ylim(yl - pad, yr + pad)

    png_path = out_dir / "layout.png"
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


# ── Baseline export ───────────────────────────────────────────────────


def _do_save_baseline(args, genome, parts, label, run_flags):
    """Write the winning genome + part selections as a baseline TOML."""
    import time
    from dataclasses import fields as dc_fields

    out_path = Path(args.save_baseline)

    run_info = args.run or "manual"
    candidate_info = args.candidate or label

    lines = [
        f"# Baseline exported from {run_info}",
        f"# Candidate: {candidate_info}",
        f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# ── Run parameters ───────────────────────────────────────────────────",
    ]

    bom_path = run_flags.get("bom_path") or args.bom
    if bom_path:
        lines.append(f'bom_path         = "{bom_path}"')

    _RUN_KEYS = [
        ("center_nm",    "center"),
        ("min_bw_nm",    "min_bw"),
        ("max_rld",      "max_rld"),
        ("max_fnum",     "max_fnum"),
        ("fold_mode",    "fold_mode"),
        ("point_source", "point_source"),
        ("optics_only",  "optics_only"),
    ]
    for toml_key, flag_key in _RUN_KEYS:
        val = run_flags.get(flag_key)
        if val is None:
            continue
        if isinstance(val, bool):
            lines.append(f"{toml_key:<16} = {'true' if val else 'false'}")
        elif isinstance(val, str):
            lines.append(f'{toml_key:<16} = "{val}"')
        else:
            lines.append(f"{toml_key:<16} = {val}")

    lines.append("")
    lines.append("# ── Part selections ──────────────────────────────────────────────────")

    for attr in ("m1_part", "m2_part", "grating_part", "f1_part", "f2_part"):
        val = getattr(parts, attr, None)
        if val is not None:
            lines.append(f'{attr:<16} = "{val}"')

    lines.append("")
    lines.append("# ── Genome parameters ────────────────────────────────────────────────")

    for f in dc_fields(genome):
        val = getattr(genome, f.name)
        if val is None:
            continue
        if isinstance(val, float):
            if abs(val) < 0.01:
                lines.append(f"{f.name:<20} = {val}")
            else:
                lines.append(f"{f.name:<20} = {val:.6f}".rstrip("0").rstrip("."))
        else:
            lines.append(f"{f.name:<20} = {val}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote baseline: {out_path}")


# ── BOM CSV ─────────────────────────────────────────────────────────────


def _count_labels(compound) -> tuple[dict[str, int], dict[str, float]]:
    """Recursively walk a build123d Compound and tally labels and volumes (mm³)."""
    from collections import Counter, defaultdict
    counts: Counter[str] = Counter()
    volumes: defaultdict[str, float] = defaultdict(float)
    for child in compound.children:
        lbl = getattr(child, "label", "") or ""
        if lbl:
            counts[lbl] += 1
            vol = getattr(child, "volume", 0.0) or 0.0
            volumes[lbl] += vol
        if hasattr(child, "children") and child.children:
            sub_counts, sub_volumes = _count_labels(child)
            counts.update(sub_counts)
            for k, v in sub_volumes.items():
                volumes[k] += v
    return dict(counts), dict(volumes)


def _write_bom_csv(parts, bom_path: str | None, compound,
                   out_dir: Path) -> None:
    """Write a CSV bill-of-materials by tallying placed parts in the CAD assembly."""
    from designs.czerny_bom import load_bom

    bom = load_bom(bom_path)
    counts, volumes = _count_labels(compound)
    rows: list[dict[str, str]] = []

    optic_roles = [
        ("M1 (collimating mirror)", "m1", "m1_options"),
        ("M2 (focusing mirror)", "m2", "m2_options"),
        ("F1 (fold mirror)", "f1", "f1_options"),
        ("F2 (fold mirror)", "f2", "f2_options"),
    ]
    for role, prefix, section_key in optic_roles:
        pn = getattr(parts, f"{prefix}_part", None)
        if pn is None:
            continue
        entry = bom["mirrors"][section_key][pn]
        mirror_type = entry.get("mirror_type", "")
        orientation = entry.get("cylindrical_orientation", "")
        desc = mirror_type
        if orientation:
            desc += f" ({orientation})"
        rows.append({
            "role": role,
            "part_number": pn,
            "vendor": entry.get("vendor", ""),
            "description": desc,
            "qty": "1",
            "cost_usd": f"{entry['cost_usd']:.2f}",
        })

    grating_pn = parts.grating_part
    g_entry = bom["grating_options"][grating_pn]
    rows.append({
        "role": "Grating",
        "part_number": grating_pn,
        "vendor": g_entry.get("vendor", ""),
        "description": (f"{g_entry['groove_density_per_mm']:.0f} g/mm"
                        + (f" blazed at {g_entry['blaze_nm']:.0f} nm"
                           if "blaze_nm" in g_entry else "")),
        "qty": "1",
        "cost_usd": f"{g_entry['cost_usd']:.2f}",
    })

    for key, entry in bom.get("slits", {}).get("entrance_options", {}).items():
        rows.append({
            "role": "Fiber adapter (HASMA)",
            "part_number": key,
            "vendor": entry.get("vendor", ""),
            "description": entry.get("type", ""),
            "qty": "1",
            "cost_usd": f"{entry['cost_usd']:.2f}",
        })

    det = bom.get("detector", {})
    if det:
        cost = det.get("cost_usd")
        rows.append({
            "role": "Detector",
            "part_number": det.get("part_number", ""),
            "vendor": det.get("vendor", ""),
            "description": f"{det.get('board', '')} ({det.get('n_pixels', '')} px)",
            "qty": "1",
            "cost_usd": f"{cost:.2f}" if cost else "",
        })

    ctrl = bom.get("controller", {})
    if ctrl:
        cost = ctrl.get("cost_usd")
        rows.append({
            "role": "Controller",
            "part_number": "",
            "vendor": ctrl.get("vendor", ""),
            "description": ctrl.get("board", ""),
            "qty": "1",
            "cost_usd": f"{cost:.2f}" if cost else "",
        })

    hw = bom.get("hardware", {})
    for pn, entry in hw.items():
        qty = counts.get(pn, 0)
        pn_proc = f"{pn}_procedural"
        qty += counts.get(pn_proc, 0)
        if qty == 0:
            continue
        rows.append({
            "role": "Hardware",
            "part_number": pn,
            "vendor": entry.get("vendor", ""),
            "description": entry.get("description", ""),
            "qty": str(qty),
            "cost_usd": f"{entry['cost_usd'] * qty:.2f}",
        })

    housing_section = bom.get("housing", {})
    housing_parts = housing_section.get("parts", {})
    density = housing_section.get("density_g_per_cm3", 1.24)
    infill = housing_section.get("infill_fraction", 1.0)
    cost_per_g = housing_section.get("cost_per_gram_usd", 0.0)
    for key, entry in housing_parts.items():
        qty = entry.get("qty")
        if qty is None:
            if key == "flexure_mount":
                qty = counts.get("flexure_mount", 0)
            else:
                qty = counts.get(key, 0)
        if qty == 0:
            continue
        vol_mm3 = volumes.get(key, 0.0)
        if key == "flexure_mount":
            vol_mm3 = sum(v for k, v in volumes.items()
                         if k == "flexure_mount")
        mass_g = vol_mm3 / 1000.0 * density * infill
        cost = mass_g * cost_per_g if cost_per_g else 0.0
        rows.append({
            "role": "Housing",
            "part_number": key,
            "vendor": entry.get("vendor", ""),
            "description": entry.get("description", ""),
            "qty": str(qty),
            "cost_usd": f"{cost:.2f}" if cost else "",
        })

    out_path = out_dir / "bom.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["role", "part_number", "vendor", "description", "qty", "cost_usd"])
        writer.writeheader()
        writer.writerows(rows)
        total = sum(float(r["cost_usd"]) for r in rows if r["cost_usd"])
        writer.writerow({
            "role": "TOTAL",
            "part_number": "",
            "vendor": "",
            "description": "",
            "qty": "",
            "cost_usd": f"{total:.2f}",
        })
    print(f"Wrote {out_path}")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    run_flags = {"optics_only": False, "point_source": False}
    if args.run:
        design, genome, parts, label, run_flags = _load_winner(
            args.run, args.candidate)
    elif args.baseline:
        from designs.czerny_base import CzernyGenome
        from designs.czerny_bom import load_parts, load_bom
        from optics._config import read_toml
        data = read_toml(Path(args.baseline))
        genome_keys = {f.name for f in __import__("dataclasses").fields(CzernyGenome)}
        genome = CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})
        design = get_design(args.design)
        bom_path = args.bom or data.get("bom_path")
        bom = load_bom(bom_path)
        def _first_key(section):
            return next(k for k, v in section.items() if isinstance(v, dict))
        m1_pn = data.get("m1_part") or _first_key(bom["mirrors"]["m1_options"])
        m2_pn = data.get("m2_part") or _first_key(bom["mirrors"]["m2_options"])
        g_pn = data.get("grating_part") or _first_key(bom["grating_options"])
        f1_pn = data.get("f1_part")
        if f1_pn is None and "f1_options" in bom["mirrors"]:
            f1_pn = _first_key(bom["mirrors"]["f1_options"])
        f2_pn = data.get("f2_part")
        if f2_pn is None and "f2_options" in bom["mirrors"]:
            f2_pn = _first_key(bom["mirrors"]["f2_options"])
        optic_size = float(bom["mirrors"]["m1_options"][m1_pn]["diameter_mm"])
        parts = load_parts(
            m1_part=m1_pn, m2_part=m2_pn, grating_part=g_pn,
            f1_part=f1_pn, f2_part=f2_pn,
            optic_size_mm=optic_size, bom_path=bom_path,
        )
        label = Path(args.baseline).stem
        if data.get("center_nm") is not None:
            run_flags["center"] = data["center_nm"]
        if data.get("min_bw_nm") is not None:
            run_flags["min_bw"] = data["min_bw_nm"]
        if bom_path:
            run_flags["bom_path"] = bom_path
    else:
        design = get_design(args.design)
        genome, m1, m2, grating, label = design.resolve_genome_from_cli(args)
        from designs.czerny_bom import _load_baseline_toml
        bl = _load_baseline_toml()
        bom = design.load_bom(bom_path=args.bom)
        m1_entry = bom["mirrors"]["m1_options"][m1]
        optic_size = float(m1_entry["diameter_mm"])
        parts = design.load_parts(m1_part=m1, m2_part=m2, grating_part=grating,
                                  f1_part=bl.get("f1_part"), f2_part=bl.get("f2_part"),
                                  optic_size_mm=optic_size, bom_path=args.bom)

    if args.save_baseline:
        _do_save_baseline(args, genome, parts, label, run_flags)
        if not (args.render or args.step or args.cad or args.layout):
            return

    _resolved_bom = args.bom or run_flags.get("bom_path")
    if _resolved_bom:
        from optics.mounts_cad import set_bom_path
        set_bom_path(_resolved_bom)

    scene, has_housing = _assemble(design, genome, parts, args,
                                    optics_only=run_flags["optics_only"])

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.run:
        run_stem = Path(args.run).name.replace("optimizer_", "")
        out_dir = Path(f"output/export_{run_stem}_{label}")
    else:
        out_dir = Path(f"output/export_{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_parts = [f"housing={'yes' if has_housing else 'no'}"]
    if run_flags["optics_only"]:
        mode_parts.append("optics_only")
    if run_flags["point_source"]:
        mode_parts.append("point_source")
    print(f"Genome: {label}  {'  '.join(mode_parts)}")

    placed = None
    if args.step or args.cad:
        placed = _do_step(
            args, genome, parts, scene, label, out_dir)

    _center = run_flags.get("center")
    _design_wl = float(_center) if _center is not None else 550.0
    _min_bw = run_flags.get("min_bw")
    _min_bw_nm = float(_min_bw) if _min_bw is not None else 300.0

    warnings = []
    if args.cad and placed:
        warnings = _do_cad(
            args, design, genome, parts, scene, has_housing,
            placed, out_dir, design_wavelength_nm=_design_wl,
            min_bw_nm=_min_bw_nm, bom_path=_resolved_bom)

    if args.layout:
        _do_layout(args, genome, parts, scene, label, out_dir)

    if args.render:
        _do_render(args, genome, parts, scene, has_housing, label, out_dir)

    manifest = {
        "label": label,
        "parts": {k: v for k, v in [
                      ("m1", parts.m1_part), ("m2", parts.m2_part),
                      ("grating", parts.grating_part),
                      ("f1", getattr(parts, "f1_part", None)),
                      ("f2", getattr(parts, "f2_part", None)),
                  ] if v is not None},
        "genome": asdict(genome),
        "housing": has_housing,
        "wall_albedo": parts.wall_albedo,
        "outputs": {
            "render": args.render,
            "step": args.step or args.cad,
            "cad": args.cad,
            "layout": args.layout,
        },
        "printability_warnings": warnings or [],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
