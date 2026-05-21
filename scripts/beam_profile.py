"""Forward-trace beam profile at each optical surface.

Loads BOM, resolves genome, then delegates ray generation, tracing,
and aperture analysis to ``optics.forward_trace``.

Single-wavelength mode (default): traces at --wavelength (550nm) with
the grating rotated to put that wavelength on-axis.

Multi-wavelength mode (--wavelengths): traces at each wavelength with
the grating held fixed at the --design-wavelength angle.  Shows how
wavelengths disperse across the detector in a fixed-grating instrument.

Per-wavelength output has one panel per optical surface plus three
detector analysis panels: spot zoom (rms/σx/σy), tangential profile
(FWHM in µm), and sagittal profile (σy, sag/tan ratio).

Usage::

    python scripts/beam_profile.py
    python scripts/beam_profile.py --wavelength 400 --rays 20000
    python scripts/beam_profile.py --run output/optimizer_... --candidate gen04_cand42
    python scripts/beam_profile.py --run output/optimizer_... --wavelengths 450,550,650
    python scripts/beam_profile.py --run output/optimizer_... --optics_only
    python scripts/beam_profile.py --baseline data/czerny_baseline_v0_design.toml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
from pathlib import Path

import numpy as np

from designs.czerny_bom import load_parts, load_bom
from designs.czerny_bom import add_genome_args, resolve_genome_from_cli
from optics.forward_trace import (
    _OVERSIZE_MM,
    aperture_overlay as _aperture_overlay,
    element_reflectance_at as _element_reflectance,
    is_inside_aperture as _is_inside,
    make_cone_rays as _make_rays,
    spot_decomposition,
    overlay_big_recorder as _overlay_big_recorder,
    trace_rays as _trace_rays,
)
from designs.czerny_bom import build_optics_only_scene as _build_scene
from optics.elements.hit_recorder import HitRecorder
from optics.world_builder import build_world, _placement_transform, MM_TO_M
from optics.grating_math import grating_rotation_deg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_genome_args(ap)
    ap.add_argument("--run", type=str, default=None,
                    help="Optimizer run dir (loads winner genome)")
    ap.add_argument("--candidate", type=str, default=None,
                    help="Candidate ID within --run (default: best winner)")
    ap.add_argument("--wavelength", type=float, default=550.0,
                    help="Wavelength in nm (default: %(default)s)")
    ap.add_argument("--wavelengths", type=str, default=None,
                    help="Comma-separated wavelengths for fixed-grating multi-λ "
                         "trace (e.g. 450,550,650). Grating held at --design-wavelength.")
    ap.add_argument("--design-wavelength", type=float, default=550.0,
                    help="Grating design wavelength in nm (default: %(default)s)")
    ap.add_argument("--rays", type=int, default=10000,
                    help="Number of forward-trace rays (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for ray generation (default: %(default)s)")
    ap.add_argument("--fnum", type=float, default=None,
                    help="Input f/# (default: M1 f/# from BOM)")
    ap.add_argument("--point_source", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Use a point source (default: match run setting, or extended)")
    ap.add_argument("--optics_only", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Optics only, no mounts (default: match run setting, or False)")
    ap.add_argument("--no-title", action="store_true",
                    help="Hide the title line")
    ap.add_argument("--output", type=str, default=None,
                    help="Output PNG path (default: output/beam_profile.png)")
    ap.add_argument("--bom", type=str, default=None,
                    help="Path to BOM TOML (default: data/czerny_bom_v0_design.toml)")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Path to baseline genome TOML (overrides CLI genome args)")
    args = ap.parse_args()

    if args.bom:
        from designs.czerny_bom import set_bom_path
        set_bom_path(args.bom)

    run_flags = {"optics_only": False, "point_source": False}
    if args.run:
        from export import _load_winner
        _, genome, parts, label, run_flags = _load_winner(
            args.run, args.candidate)
    elif args.baseline:
        from designs.czerny_base import CzernyGenome
        from designs.czerny_bom import load_parts, load_bom
        from optics._config import read_toml
        data = read_toml(Path(args.baseline))
        genome_keys = {f.name for f in __import__("dataclasses").fields(CzernyGenome)}
        genome = CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})
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
            optic_size_mm=optic_size, bom_path=bom_path)
        label = Path(args.baseline).stem
    else:
        genome, m1_part, m2_part, grating_part, label = resolve_genome_from_cli(args)
        from designs.czerny_bom import _load_baseline_toml
        bl = _load_baseline_toml()
        bom = load_bom(bom_path=args.bom)
        m1_entry = bom["mirrors"]["m1_options"][m1_part]
        optic_size = float(m1_entry["diameter_mm"])
        parts = load_parts(m1_part=m1_part, m2_part=m2_part,
                           grating_part=grating_part,
                           f1_part=bl["f1_part"], f2_part=bl["f2_part"],
                           optic_size_mm=optic_size,
                           bom_path=args.bom)
    input_fnum = args.fnum if args.fnum is not None else (
        parts.m1_focal_length_mm / parts.m1_diameter_mm)

    # Multi-wavelength fixed-grating mode.
    multi_wavelengths = None
    if args.wavelengths:
        multi_wavelengths = [float(w.strip()) for w in args.wavelengths.split(",")]
        design_wl = args.design_wavelength
    else:
        design_wl = args.wavelength

    if args.point_source is not None:
        point_source = args.point_source
    else:
        point_source = run_flags["point_source"]

    if args.optics_only is not None:
        optics_only = args.optics_only
    else:
        optics_only = run_flags["optics_only"]

    build_optics_only_scene = _build_scene
    has_mounts = False
    if not optics_only:
        from designs.czerny_assembly import assemble_scene
        try:
            scene = assemble_scene(genome, parts,
                                   scene_builder=build_optics_only_scene)
            has_mounts = True
        except Exception as e:
            print(f"Mount assembly failed ({e}), falling back to optics only")
            scene = build_optics_only_scene(genome, parts)
    else:
        scene = build_optics_only_scene(genome, parts)

    slit_el = next(e for e in scene.elements if e.label == "entrance_slit")

    # Optical train for spot diagrams. The last panel shows either the
    # detector (default) or exit_slit — not both. Use --exit-slit to
    # show the slit plane instead of the sensor.
    # Show detector instead of exit_slit for the last panel when
    # the scene has a detector element.
    has_detector = any(e.kind == "detector" for e in scene.elements)
    if has_detector:
        train_labels = [el.label for el in scene.elements
                        if el.label != "exit_slit"]
    else:
        train_labels = [el.label for el in scene.elements
                        if el.kind != "detector"]

    # Map label -> raysect primitive name used in build_world
    prim_names = {
        "entrance_slit": "entrance_slit",  # emitter box
    }
    for lbl in train_labels:
        if lbl not in prim_names:
            prim_names[lbl] = lbl

    n_panels = len(train_labels)
    print(f"Optical train: {' -> '.join(train_labels)}")
    print(f"Genome: {label}")
    if multi_wavelengths:
        print(f"Wavelengths: {multi_wavelengths} nm (grating fixed at {design_wl} nm)")
    else:
        print(f"Wavelength: {args.wavelength} nm")
    print(f"{args.rays} rays, seed {args.seed}")

    def _trace_all_elements(trace_wl):
        """Trace at a single wavelength through all elements, return footprints and stats."""
        footprints_wl: dict[str, tuple[list[float], list[float]]] = {}

        # Material throughput at this wavelength.
        cumulative_material = 1.0
        material_factor_wl: dict[str, float] = {}
        for lbl in train_labels:
            el = next(e for e in scene.elements if e.label == lbl)
            r = _element_reflectance(el, trace_wl, parts)
            if r is not None:
                cumulative_material *= r
            material_factor_wl[lbl] = cumulative_material

        for lbl in train_labels:
            if lbl == "entrance_slit":
                rng = np.random.default_rng(args.seed)
                trace_rays = _make_rays(slit_el, parts, args.rays, rng,
                                        input_fnum=input_fnum, point_source=point_source)
                built = build_world(scene, input_fnum=input_fnum)
                emitter_prim = built.primitives["entrance_slit"]
                w2p = emitter_prim.transform.inverse()
                es_hx, es_hy = [], []
                for origin, _ in trace_rays:
                    lp = origin.transform(w2p)
                    es_hx.append(lp.x * 1000)
                    es_hy.append(lp.y * 1000)
                footprints_wl[lbl] = (es_hx, es_hy)
                print(f"  {lbl}: {len(es_hx)} hits")
                continue

            prim_name = prim_names.get(lbl, lbl)
            el = next(e for e in scene.elements if e.label == lbl)
            built = build_world(scene, input_fnum=input_fnum)

            # Grating held fixed at the design wavelength.
            theta = grating_rotation_deg(
                genome.alpha_deg, genome.beta_deg,
                parts.grating_groove_density_per_mm, design_wl)
            built.set_grating_rotation_deg(theta)

            is_focal_plane = lbl in ("exit_slit", "detector")
            if is_focal_plane:
                from raysect.primitive import Box
                from raysect.core.math import Point3D as _P3D
                rec_half = _OVERSIZE_MM * 1e-3
                rec_box = Box(
                    lower=_P3D(-rec_half, -rec_half, -1e-3),
                    upper=_P3D(+rec_half, +rec_half, 0.0),
                )
                rec_box.transform = _placement_transform(el)
                rec_box.material = HitRecorder()
                rec_box.parent = built.world
                rec_box.name = lbl
                built.primitives[lbl] = rec_box
                prim_name = lbl

            if prim_name not in built.primitives:
                print(f"  Warning: primitive {prim_name!r} not found, skipping {lbl}")
                continue

            if not is_focal_plane:
                _, rec_name = _overlay_big_recorder(built, prim_name, el)
                prim_name = rec_name

            rng = np.random.default_rng(args.seed)
            trace_rays = _make_rays(slit_el, parts, args.rays, rng,
                                    input_fnum=input_fnum, point_source=point_source)
            _trace_rays(trace_rays, built.world, trace_wl)
            rec_prim = built.primitives[prim_name]
            w2p = rec_prim.transform.inverse()
            from raysect.core.math import Point3D as _P3D
            hx_mm, hy_mm = [], []
            for h in rec_prim.material.hits:
                lp = _P3D(*h.point).transform(w2p)
                hx_mm.append(lp.x * 1000)
                hy_mm.append(lp.y * 1000)
            footprints_wl[lbl] = (hx_mm, hy_mm)
            print(f"  {lbl}: {len(hx_mm)} hits")

        # Compute stats.
        stats_wl: dict[str, tuple[int, int, int, float]] = {}
        for lbl in train_labels:
            el = next(e for e in scene.elements if e.label == lbl)
            hx, hy = footprints_wl[lbl]
            otype, oparam = _aperture_overlay(el, parts)
            n = len(hx)
            inside = sum(1 for x, y in zip(hx, hy) if _is_inside(x, y, otype, oparam))
            mf = material_factor_wl[lbl]
            stats_wl[lbl] = (n, inside, args.rays, mf)
        return footprints_wl, stats_wl, material_factor_wl

    # Run traces — one per wavelength in multi-λ mode, else single.
    trace_wavelengths = multi_wavelengths if multi_wavelengths else [args.wavelength]
    all_results: dict[float, tuple] = {}
    for wl in trace_wavelengths:
        print(f"\n--- {wl:.0f} nm ---")
        fp, st, mf = _trace_all_elements(wl)
        all_results[wl] = (fp, st, mf)
        print()
        for lbl in train_labels:
            n, inside, launched, mfv = st[lbl]
            pwr = 100 * mfv
            geo = 100 * inside / launched if launched > 0 else 0
            print(f"  {lbl:20s}  {inside:>6} rays @ {pwr:.1f}% power  (geo {geo:.1f}%)")

    # 2D spot decomposition at the detector.
    det_lbl_decomp = "detector" if has_detector else "exit_slit"
    decompositions: dict[float, dict] = {}
    print("\n--- Spot decomposition (detector plane, all hits) ---")
    print(f"  {'wl':>6s}  {'sig_x':>8s}  {'sig_y':>8s}  {'RMS':>8s}  {'y/x':>6s}  {'skew_x':>7s}")
    print(f"  {'(nm)':>6s}  {'(um)':>8s}  {'(um)':>8s}  {'(um)':>8s}  {'':>6s}  {'':>7s}")
    for wl in trace_wavelengths:
        fp, st, mf = all_results[wl]
        if det_lbl_decomp in fp:
            det_hx, det_hy = fp[det_lbl_decomp]
            sd = spot_decomposition(det_hx, det_hy)
            if sd:
                decompositions[wl] = sd
                print(f"  {wl:6.0f}  {sd['sigma_x_mm']*1000:8.1f}  {sd['sigma_y_mm']*1000:8.1f}  "
                      f"{sd['rms_spot_mm']*1000:8.1f}  {sd['sag_tan_ratio']:6.2f}  {sd['tan_skewness']:7.3f}")
            else:
                print(f"  {wl:6.0f}  (too few hits)")
        else:
            print(f"  {wl:6.0f}  (no detector data)")
    print()
    print("  sig_x = tangential RMS (dispersion axis) -- drives ILF broadening")
    print("  sig_y = sagittal RMS (cross-dispersion) -- astigmatism signature")
    print("  y/x >> 1: astigmatism-dominated | y/x << 1: coma/SA-dominated")
    print("  skew_x != 0: coma present | skew_x ~ 0: spherical aberration")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    _WL_CMAPS = {450: "Blues", 500: "Greens", 550: "Greens",
                 600: "Oranges", 650: "Reds", 700: "Reds"}
    def _wl_cmap(wl):
        if wl in _WL_CMAPS:
            return _WL_CMAPS[wl]
        if wl < 500:
            return "Blues"
        if wl < 600:
            return "Greens"
        return "Reds"

    housing_tag = "optics only" if not has_mounts else ""
    source_tag = ("point source" if point_source
                  else f"extended source (Ø{parts.slit_width_um:.0f} µm fiber)")
    src_suffix = "_point" if point_source else ""
    if args.output:
        out_base = args.output
    elif args.run:
        run_stem = Path(args.run).name.replace("optimizer_", "")
        out_dir_default = Path(f"output/export_{run_stem}_{label}")
        out_dir_default.mkdir(parents=True, exist_ok=True)
        out_base = str(out_dir_default / f"beam_profile{src_suffix}.png")
    else:
        out_base = f"output/beam_profile{src_suffix}.png"
    out_stem = Path(out_base).stem
    out_dir = Path(out_base).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for wl in trace_wavelengths:
        footprints, stats, material_factor = all_results[wl]
        cmap = _wl_cmap(wl)

        total_panels = n_panels + 3  # +1 spot zoom, +1 tangential, +1 sagittal
        ncols = min(total_panels, 3)
        nrows = math.ceil(total_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        if total_panels == 1:
            axes = [axes]
        else:
            axes = list(np.array(axes).flatten())

        for idx, lbl in enumerate(train_labels):
            ax = axes[idx]
            el = next(e for e in scene.elements if e.label == lbl)
            hx, hy = footprints[lbl]
            otype, oparam = _aperture_overlay(el, parts)

            if otype == "circle":
                aperture_extent = oparam
            elif otype == "ellipse":
                r_minor, r_major = oparam[0], oparam[1]
                cx_off = oparam[2] if len(oparam) > 2 else 0.0
                aperture_extent = max(abs(cx_off) + r_major, r_minor)
            elif otype == "rect":
                aperture_extent = oparam
            elif otype == "rect_wh":
                aperture_extent = max(oparam[0], oparam[1])
            else:
                aperture_extent = 12.7
            margin = aperture_extent + max(1.0, aperture_extent * 0.25)
            if lbl == "entrance_slit" and hx:
                dx = max(hx) - min(hx)
                dy = max(hy) - min(hy)
                span = max(dx, dy, aperture_extent * 2)
                pad = span * 0.3
                cx_data = 0.5 * (max(hx) + min(hx))
                cy_data = 0.5 * (max(hy) + min(hy))
                half = 0.5 * span + pad
                rng_xy = [[cx_data - half, cx_data + half],
                           [cy_data - half, cy_data + half]]
            else:
                rng_xy = [[-margin, margin], [-margin, margin]]

            n_total, n_inside, n_launched, mf = stats[lbl]
            tp = 100 * n_inside / n_launched * mf if n_launched > 0 else 0
            ax.set_title(f"{lbl}\n{tp:.1f}% throughput", fontsize=10)
            if hx:
                h, xe, ye = np.histogram2d(hx, hy, bins=80, range=rng_xy)
                if lbl == "entrance_slit":
                    color = {"Greens": "green", "Blues": "blue",
                             "Reds": "red"}.get(cmap, "green")
                    ax.scatter(hx, hy, s=0.3, alpha=0.6, c=color,
                               edgecolors="none", rasterized=True)
                else:
                    masked = np.ma.masked_equal(h.T, 0)
                    ax.imshow(
                        masked, origin="lower",
                        extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        aspect="equal", cmap=cmap,
                    )
            else:
                ax.set_xlim(rng_xy[0])
                ax.set_ylim(rng_xy[1])

            if otype == "circle":
                ax.add_patch(Circle((0, 0), oparam, fill=False, ec="cyan", ls="--", lw=1.5))
            elif otype == "ellipse":
                from matplotlib.patches import Ellipse
                r_minor, r_major = oparam[0], oparam[1]
                cx = oparam[2] if len(oparam) > 2 else 0.0
                ax.add_patch(Ellipse((cx, 0), 2 * r_major, 2 * r_minor,
                                     fill=False, ec="cyan", ls="--", lw=1.5))
            elif otype == "rect":
                ax.add_patch(Rectangle((-oparam, -oparam), 2 * oparam, 2 * oparam,
                                       fill=False, ec="cyan", ls="--", lw=1.5))
            elif otype == "rect_wh":
                hw, hh = oparam
                ax.add_patch(Rectangle((-hw, -hh), 2 * hw, 2 * hh,
                                       fill=False, ec="cyan", ls="--", lw=1.5))
            ax.set_xlabel("local x (mm)")
            ax.set_ylabel("local y (mm)")

        # Zoomed spot panel: scatter of detector hits, auto-scaled to spot extent.
        det_lbl = "detector" if "detector" in footprints else "exit_slit"
        if det_lbl in footprints and n_panels < len(axes):
            det_hx_z, det_hy_z = footprints[det_lbl]
            if det_hx_z:
                ax_zoom = axes[n_panels]
                hx_z = np.array(det_hx_z)
                hy_z = np.array(det_hy_z)
                color = {"Blues": "tab:blue", "Greens": "tab:green",
                         "Reds": "tab:red"}.get(cmap, "white")
                ax_zoom.scatter(hx_z, hy_z, s=0.3, alpha=0.4, c=color,
                                edgecolors="none")
                sd = decompositions.get(wl)
                if sd:
                    rms = sd['rms_spot_mm'] * 1000
                    ax_zoom.set_title(
                        f"spot zoom\n"
                        f"rms={rms:.1f}um, "
                        f"σ$_x$={sd['sigma_x_mm']*1000:.1f}um, "
                        f"σ$_y$={sd['sigma_y_mm']*1000:.1f}um",
                        fontsize=10)
                else:
                    ax_zoom.set_title("spot zoom", fontsize=10)
                cx = np.median(hx_z)
                cy = np.median(hy_z)
                extent = max(np.percentile(np.abs(hx_z - cx), 99),
                             np.percentile(np.abs(hy_z - cy), 99),
                             0.01)
                pad = extent * 1.3
                ax_zoom.set_xlim(cx - pad, cx + pad)
                ax_zoom.set_ylim(cy - pad, cy + pad)
                ax_zoom.set_aspect("equal")
                ax_zoom.set_xlabel("local x (mm)")
                ax_zoom.set_ylabel("local y (mm)")

        # Detector cross-section profile along dispersion axis (local x).
        if det_lbl in footprints and len(axes) > n_panels:
            det_hx, det_hy = footprints[det_lbl]
            if det_hx:
                ax_prof = axes[n_panels + 1] if n_panels + 1 < len(axes) else None
                if ax_prof is not None:
                    hx_arr = np.array(det_hx)
                    margin_x = 18
                    bin_width = 0.008  # 8um pixel pitch (TCD1304)
                    bins = np.arange(-margin_x, margin_x + bin_width, bin_width)
                    counts, edges = np.histogram(hx_arr, bins=bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    color = {"Blues": "tab:blue", "Greens": "tab:green",
                             "Reds": "tab:red"}.get(cmap, "white")
                    ax_prof.bar(centers, counts, width=edges[1]-edges[0],
                                color=color, alpha=0.8)
                    ax_prof.set_xlabel("local x (mm)")
                    ax_prof.set_ylabel("counts")

                    # FWHM extraction from the profile.
                    if counts.max() > 0:
                        half_max = counts.max() * 0.5
                        above = counts >= half_max
                        if above.any():
                            idxs = np.where(above)[0]
                            fwhm = centers[idxs[-1]] - centers[idxs[0]]
                            peak_x = centers[counts.argmax()]
                            ax_prof.axhline(half_max, color="cyan", ls="--",
                                            lw=0.8, alpha=0.7)
                            sd = decompositions.get(wl)
                            sigma_tag = ""
                            if sd:
                                sigma_tag = (f", σ$_x$={sd['sigma_x_mm']*1000:.1f}um"
                                             f", skew={sd['tan_skewness']:.2f}")
                            ax_prof.set_title(
                                f"tangential profile\n"
                                f"FWHM={fwhm*1000:.1f}um{sigma_tag}, "
                                f"n={len(hx_arr)}",
                                fontsize=10)
                        else:
                            ax_prof.set_title(f"tangential profile\nn={len(hx_arr)}",
                                              fontsize=10)
                    if counts.max() > 0:
                        thresh = counts.max() * 0.05
                        above_thresh = np.where(counts >= thresh)[0]
                        lo = centers[above_thresh[0]]
                        hi = centers[above_thresh[-1]]
                        pad = max(0.2 * (hi - lo), 0.1)
                        ax_prof.set_xlim(lo - pad, hi + pad)
                    n_panels_used = n_panels + 2
                else:
                    n_panels_used = n_panels + 1
            else:
                n_panels_used = n_panels + 1
        else:
            n_panels_used = n_panels + 1

        # Sagittal profile: cross-dispersion (y) distribution.
        if det_lbl in footprints and n_panels + 2 < len(axes):
            det_hx_s, det_hy_s = footprints[det_lbl]
            if det_hy_s:
                ax_sag = axes[n_panels + 2]
                hy_arr = np.array(det_hy_s)
                sag_bin = 0.010  # 10 um bins
                sag_margin = max(0.3, np.abs(hy_arr).max() + 0.05)
                sag_bins = np.arange(-sag_margin, sag_margin + sag_bin, sag_bin)
                sag_counts, sag_edges = np.histogram(hy_arr, bins=sag_bins)
                sag_centers = 0.5 * (sag_edges[:-1] + sag_edges[1:])
                color = {"Blues": "tab:blue", "Greens": "tab:green",
                         "Reds": "tab:red"}.get(cmap, "white")
                ax_sag.bar(sag_centers, sag_counts,
                           width=sag_edges[1] - sag_edges[0],
                           color=color, alpha=0.8)
                for yb in (-0.1, 0.1):
                    ax_sag.axvline(yb, color="cyan", ls="--", lw=1.0,
                                   alpha=0.8)
                ax_sag.set_xlabel("local y (mm)")
                ax_sag.set_ylabel("counts")
                sd = decompositions.get(wl)
                if sd:
                    ax_sag.set_title(
                        f"sagittal profile\n"
                        f"σ$_y$={sd['sigma_y_mm']*1000:.1f}um, "
                        f"y/x={sd['sag_tan_ratio']:.2f}",
                        fontsize=10)
                else:
                    ax_sag.set_title("sagittal profile", fontsize=10)
                if sag_counts.max() > 0:
                    thresh_y = sag_counts.max() * 0.05
                    above_y = np.where(sag_counts >= thresh_y)[0]
                    lo_y = sag_centers[above_y[0]]
                    hi_y = sag_centers[above_y[-1]]
                    pad_y = max(0.2 * (hi_y - lo_y), 0.1)
                    ax_sag.set_xlim(lo_y - pad_y, hi_y + pad_y)
                n_panels_used = max(n_panels_used, n_panels + 3)

        for idx in range(n_panels_used, len(axes)):
            fig.delaxes(axes[idx])

        if not args.no_title:
            grating_tag = f"grating @ {design_wl:.0f}nm" if multi_wavelengths else ""
            line2 = f"genome: {label}"
            if housing_tag:
                line2 += f", {housing_tag}"
            fig.suptitle(
                f"Beam footprint at {wl:.0f}nm{', ' + grating_tag if grating_tag else ''}, "
                f"f/{input_fnum:.1f} cone, {source_tag}\n{line2}",
                fontsize=12,
            )
        fig.tight_layout()

        if len(trace_wavelengths) > 1:
            out_path = out_dir / f"{out_stem}_{wl:.0f}nm.png"
        else:
            out_path = Path(out_base)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
