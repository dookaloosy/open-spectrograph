"""Spot diagrams in the style of Xia 2017 Figs 3 & 4.

Each panel shows two wavelengths (λ and λ+Δλ) in green and blue,
demonstrating spectral resolution.  RMS/σx/σy annotation uses the
primary wavelength only (not inflated by the dispersion separation).

Supports both traditional and modified CT (cylindrical collimators),
point source and Ø25 µm extended source, with optional overrides
for detector tilt, focus distance, and arm lengths.

Usage:
    # Traditional CT (default validation BOM)
    python scripts/spot_panel.py --point_source
    python scripts/spot_panel.py  # Ø25 µm source

    # Custom wavelengths and pair offset
    python scripts/spot_panel.py --wavelengths 400,500,600,700 --delta 0.5

    # Traditional CT with layout parameters in title
    python scripts/spot_panel.py --point_source \\
        --tilt 3.5 --lb 108.751 --la 111.30 --l_f1 100.774

    # Modified CT — use modified BOM
    python scripts/spot_panel.py --bom data/czerny_bom_xia2017_corrected.toml \\
        --point_source --tilt 3.70 --lb 107.292 --la 112.650 --l_f1 102.107

    # Custom output path
    python scripts/spot_panel.py --point_source \\
        --output output/my_panel.png

CLI overrides map to layout variables:
    --wavelengths  comma-separated primary wavelengths (default: 350,450,550,650,750)
    --delta        uniform Δλ offset for each pair in nm (default: 0.5)
    --tilt         θ_D       detector tilt (degrees)
    --lb           l_M3D     focuser-to-detector distance (mm)
    --la           L_a       total collimator arm length (mm)
    --l_f1         L_f1      fold-to-collimator distance (mm)
                              l_SM1 = L_a - L_f1 (slit to fold)
                              l_M1M2 = L_f1 (fold to collimator)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from designs.czerny_base import CzernyGenome
from designs.czerny_bom import load_parts, build_optics_only_scene
from optics._config import read_toml
from optics.forward_trace import make_cone_rays, trace_rays
from optics.world_builder import build_world, _placement_transform, MM_TO_M
from optics.grating_math import grating_rotation_deg
from optics.elements.hit_recorder import HitRecorder
from raysect.core.math import Point3D
from raysect.primitive import Box

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BOM = DATA_DIR / "czerny_bom_v0_design.toml"
DEFAULT_BASELINE = DATA_DIR / "czerny_baseline_v0_design.toml"
FIELD_RADIUS_UM = 500.0
SCALE_BAR_UM = 100.0
DEFAULT_WAVELENGTHS = [400.0, 550.0, 700.0]
DEFAULT_DELTA_NM = 0.5


def build_scene(bom_path=None, baseline_path=None,
                tilt_override=None, lb_override=None,
                la_override=None, l_f1_override=None):
    from designs.czerny_bom import load_bom
    baseline = Path(baseline_path) if baseline_path else DEFAULT_BASELINE
    data = read_toml(baseline)
    if bom_path:
        bom_p = Path(bom_path)
    elif data.get("bom_path"):
        bom_p = Path(data["bom_path"])
    else:
        bom_p = DEFAULT_BOM
    if tilt_override is not None:
        data["theta_d_deg"] = tilt_override
    if lb_override is not None:
        data["L_b_mm"] = lb_override
    if la_override is not None:
        data["L_a_mm"] = la_override
    if l_f1_override is not None:
        data["L_f1_mm"] = l_f1_override
    genome_keys = {f.name for f in __import__("dataclasses").fields(CzernyGenome)}
    genome = CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})
    bom = load_bom(bom_p)
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
    parts = load_parts(m1_part=m1_pn, m2_part=m2_pn, grating_part=g_pn,
                       f1_part=f1_pn, f2_part=f2_pn,
                       bom_path=bom_p, optic_size_mm=optic_size)
    scene = build_optics_only_scene(genome, parts, bom_path=bom_p)
    return scene, genome, parts, data


def trace_wavelength(scene, genome, parts, wavelength_nm, n_rays,
                     seed, point_source=False, *, input_fnum):
    det_el = next(
        (e for e in scene.elements if e.kind == "detector"), None)
    if det_el is None:
        det_el = next(e for e in scene.elements if e.label == "exit_slit")
    slit_el = next(e for e in scene.elements if e.label == "entrance_slit")

    built = build_world(scene, input_fnum=input_fnum)
    theta = grating_rotation_deg(
        genome.alpha_deg, genome.beta_deg,
        parts.grating_groove_density_per_mm, 550.0)
    built.set_grating_rotation_deg(theta)

    rec_half = 0.020
    rec_box = Box(
        lower=Point3D(-rec_half, -rec_half, -1e-3),
        upper=Point3D(+rec_half, +rec_half, 0.0),
    )
    rec_box.transform = _placement_transform(det_el)
    recorder = HitRecorder()
    rec_box.material = recorder
    rec_box.parent = built.world
    rec_box.name = det_el.label
    built.primitives[det_el.label] = rec_box

    rng = np.random.default_rng(seed + int(wavelength_nm * 10))
    rays = make_cone_rays(slit_el, parts, n_rays, rng,
                          input_fnum=input_fnum, point_source=point_source)
    trace_rays(rays, built.world, wavelength_nm)

    w2p = rec_box.transform.inverse()
    hx, hy = [], []
    for hit in recorder.hits:
        lp = Point3D(*hit.point).transform(w2p)
        hx.append(lp.x * 1e6)
        hy.append(lp.y * 1e6)

    return np.array(hx), np.array(hy)


def build_scene_from_run(run_dir, candidate=None):
    """Load genome + parts from an optimizer run."""
    from export import _load_winner
    _, genome, parts, label, run_flags = _load_winner(run_dir, candidate)
    scene = build_optics_only_scene(genome, parts)
    return scene, genome, parts, label, run_flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bom", type=str, default=None,
                    help="BOM TOML path (default: validation traditional CT)")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Baseline genome TOML path")
    ap.add_argument("--run", type=str, default=None,
                    help="Optimizer run directory (uses winner genome + parts)")
    ap.add_argument("--candidate", type=str, default=None,
                    help="Candidate ID within --run (default: best winner)")
    ap.add_argument("--point_source", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Use a point source (default: match run setting, or extended)")
    ap.add_argument("--wavelengths", type=str, default=None,
                    help="Comma-separated wavelengths in nm (default: 350,450,550,650,750)")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA_NM,
                    help="Wavelength pair offset in nm (default: %(default)s)")
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--tilt", type=float, default=None,
                    help="Override detector tilt (degrees)")
    ap.add_argument("--lb", type=float, default=None,
                    help="Override L_b_mm (focuser-to-detector distance)")
    ap.add_argument("--la", type=float, default=None,
                    help="Override L_a_mm (collimator arm length)")
    ap.add_argument("--l_f1", type=float, default=None,
                    help="Override L_f1_mm (fold-to-collimator distance)")
    ap.add_argument("--no-title", action="store_true",
                    help="Hide the title line")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if args.bom:
        from designs.czerny_bom import set_bom_path
        set_bom_path(args.bom)

    run_flags = {"optics_only": False, "point_source": False}
    baseline_data = {}

    if args.run:
        scene, genome, parts, label, run_flags = build_scene_from_run(
            args.run, args.candidate)
        mode = label
    else:
        scene, genome, parts, baseline_data = build_scene(
            bom_path=args.bom, baseline_path=args.baseline,
            tilt_override=args.tilt, lb_override=args.lb,
            la_override=args.la, l_f1_override=args.l_f1)
        mode = Path(args.baseline).stem if args.baseline else (
            Path(args.bom).stem if args.bom else "traditional")
        label = None
        if baseline_data.get("point_source"):
            run_flags["point_source"] = True

    if args.point_source is not None:
        point_source = args.point_source
    else:
        point_source = run_flags["point_source"]

    src = "point source" if point_source else f"Ø{parts.slit_width_um:.0f} µm fiber"

    overrides = []
    if args.tilt is not None:
        overrides.append(f"$\\theta_D$={args.tilt:.1f}°")
    if args.lb is not None:
        overrides.append(f"$l_{{M3D}}$={args.lb:.3f} mm")
    if args.la is not None and args.l_f1 is not None:
        l_sm1 = args.la - args.l_f1
        overrides.append(f"$l_{{SM1}}$={l_sm1:.3f} mm")
        overrides.append(f"$l_{{M1M2}}$={args.l_f1:.3f} mm")
    elif args.la is not None:
        overrides.append(f"$L_a$={args.la:.3f} mm")
    elif args.l_f1 is not None:
        overrides.append(f"$L_{{fold}}$={args.l_f1:.3f} mm")
    over_str = f", {', '.join(overrides)}" if overrides else ""
    print(f"Spot diagram panel — {mode}, {src}, {args.rays} rays{over_str}")

    input_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm

    if args.wavelengths:
        wavelengths = [float(w.strip()) for w in args.wavelengths.split(",")]
    elif baseline_data.get("center_nm") and baseline_data.get("min_bw_nm"):
        center = baseline_data["center_nm"]
        bw = baseline_data["min_bw_nm"]
        lo = center - bw / 2
        hi = center + bw / 2
        wavelengths = [lo, center, hi]
    else:
        wavelengths = [DEFAULT_WAVELENGTHS[0], DEFAULT_WAVELENGTHS[2],
                       DEFAULT_WAVELENGTHS[-1]]
    wl_pairs = [(wl, wl + args.delta) for wl in wavelengths]
    n_panels = len(wl_pairs)

    # ── Pass 1: trace all wavelengths ─────────────────────────────────
    panel_data = []
    for wl1, wl2 in wl_pairs:
        print(f"  Tracing {wl1} nm ...", end="", flush=True)
        hx1, hy1 = trace_wavelength(scene, genome, parts, wl1, args.rays,
                                     seed=42, point_source=point_source,
                                     input_fnum=input_fnum)
        print(f" {len(hx1)} hits", end="")

        print(f"  |  {wl2} nm ...", end="", flush=True)
        hx2, hy2 = trace_wavelength(scene, genome, parts, wl2, args.rays,
                                     seed=99, point_source=point_source,
                                     input_fnum=input_fnum)
        print(f" {len(hx2)} hits")

        if len(hx1) > 0 and len(hx2) > 0:
            cx = 0.5 * (np.median(hx1) + np.median(hx2))
            cy = 0.5 * (np.median(hy1) + np.median(hy2))
        elif len(hx1) > 0:
            cx, cy = np.median(hx1), np.median(hy1)
        elif len(hx2) > 0:
            cx, cy = np.median(hx2), np.median(hy2)
        else:
            cx, cy = 0, 0
        hx1 -= cx; hy1 -= cy
        hx2 -= cx; hy2 -= cy
        panel_data.append((wl1, wl2, hx1, hy1, hx2, hy2))

    # ── Fixed scale across all panes ──────────────────────────────────
    half = FIELD_RADIUS_UM
    bar_um = SCALE_BAR_UM

    # ── Pass 2: plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, (wl1, wl2, hx1, hy1, hx2, hy2) in zip(axes, panel_data):
        ms = 0.15
        if len(hx1) > 0:
            ax.scatter(hx1, hy1, s=ms, c='#22aa22', alpha=0.4,
                      edgecolors='none', rasterized=True)
        if len(hx2) > 0:
            ax.scatter(hx2, hy2, s=ms, c='#2244cc', alpha=0.4,
                      edgecolors='none', rasterized=True)

        bar_y = -half * 0.85
        ax.plot([-bar_um/2, bar_um/2], [bar_y, bar_y],
                'k-', linewidth=1.5)
        ax.text(0, bar_y - half * 0.06,
                f'{bar_um:.0f} µm', ha='center', va='top', fontsize=7)

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        rms_text = ""
        if len(hx1) > 2:
            sx = np.sqrt(np.mean((hx1 - hx1.mean())**2))
            sy = np.sqrt(np.mean((hy1 - hy1.mean())**2))
            rms = np.sqrt(sx**2 + sy**2)
            rms_text = f'\nRMS {rms:.1f} µm  σ$_x$ {sx:.1f}  σ$_y$ {sy:.1f}'
        ax.set_xlabel(f'{wl1:.0f} / {wl2:.1f} nm{rms_text}', fontsize=8)

    title = f"{mode} — {src}{over_str}"
    if not args.no_title:
        fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95 if not args.no_title else 1.0])

    src_suffix = "_point" if point_source else ""
    if args.output:
        out = Path(args.output)
    elif args.run:
        cand_label = label or "winner"
        run_name = Path(args.run).name
        export_dir = Path("output") / f"export_{run_name}_{cand_label}"
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"spot_panel{src_suffix}.png"
    else:
        file_src = "point" if point_source else "extended"
        out = Path(f"output/spot_panel_{mode}_{file_src}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
