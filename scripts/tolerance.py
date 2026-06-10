"""Assembly tolerance analysis.

OAT (one-at-a-time) sweep of each design parameter around its nominal
value.  Each parameter is perturbed independently — no downstream
positions are recomputed — to measure the ILF (encircled-energy width,
EE76) degradation caused by that single assembly error.

For each axis the script reports the maximum deviation from nominal
that keeps the ILF below a configurable target (default 1 nm).

Example::

    python scripts/tolerance.py
    python scripts/tolerance.py --rays 100000 --n_points 11 --ilf_target 1.0
    python scripts/tolerance.py --baseline data/czerny_baseline_v0_design.toml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from designs.czerny_base import CzernyGenome
from designs.czerny_bom import (
    build_optics_only_scene, load_parts, load_bom,
    fitness_wavelengths_from_bw,
)
from optics._config import read_toml
from optics.forward_trace import forward_trace_metrics
from optics.scene import Scene


_ROTATION_AXES = ("phi_M1", "phi_F1")
_INCIDENCE_AXES = ("theta_F1", "theta_M1", "alpha", "theta_M2")
_INCIDENCE_LABEL_MAP = {
    "theta_F1": "F1",
    "theta_M1": "M1",
    "alpha": "grating",
    "theta_M2": "M2",
}
_LAYOUT_PARAMS = (
    "L_a_mm", "L_b_mm", "L_f1_mm",
    "L_m1_mm", "L_m2_mm",
    "theta_d_deg",
)

@dataclass
class AxisSpec:
    name: str
    lo: float
    hi: float
    nominal: float
    unit: str


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", type=str, default=None,
                   help="Baseline TOML path (default: v0 design)")
    p.add_argument("--bom", type=str, default=None,
                   help="BOM TOML path (default: from baseline)")
    p.add_argument("--max_rotation", type=float, default=2.0,
                   help="Cylinder axis rotation (phi) sweep ± degrees (default: %(default)s)")
    p.add_argument("--max_tilt", type=float, default=2.0,
                   help="Incidence angle (theta/alpha) sweep ± degrees (default: %(default)s)")
    p.add_argument("--tolerance_pct", type=float, default=5.0,
                   help="Layout sweep range as %% of nominal (default: %(default)s)")
    p.add_argument("--n_points", type=int, default=21,
                   help="Points per axis (default: %(default)s)")
    p.add_argument("--rays", type=int, default=5000,
                   help="Rays per wavelength (default: %(default)s)")
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Fitness wavelengths in nm (default: from targets)")
    p.add_argument("--ilf_target", type=float, default=1.0,
                   help="ILF threshold for budget (nm, default: %(default)s)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: output/tolerance_<ts>/)")
    return p.parse_args()


def _load_design(args):
    if args.baseline:
        bl = read_toml(Path(args.baseline))
        genome_keys = {f.name for f in dataclasses.fields(CzernyGenome)}
        genome = CzernyGenome(**{k: v for k, v in bl.items() if k in genome_keys})
        bom_path = args.bom or bl.get("bom_path")
        bom = load_bom(bom_path=bom_path)
        m1_pn = bl["m1_part"]
        parts = load_parts(
            m1_part=m1_pn, m2_part=bl["m2_part"],
            grating_part=bl["grating_part"],
            f1_part=bl.get("f1_part"),
            bom_path=bom_path,
            optic_size_mm=float(
                bom["mirrors"]["m1_options"][m1_pn]["diameter_mm"]),
        )
        label = Path(args.baseline).stem
        if args.wavelengths:
            wls = tuple(float(w) for w in args.wavelengths.split(","))
        else:
            wls = fitness_wavelengths_from_bw(
                bl["min_bw_nm"], bl["center_nm"])
    else:
        from designs.czerny_bom import BASELINE, load_baseline_parts
        genome = BASELINE
        parts = load_baseline_parts()
        label = "v0_design"
        if args.wavelengths:
            wls = tuple(float(w) for w in args.wavelengths.split(","))
        else:
            bl = read_toml(
                Path("data/czerny_baseline_v0_design.toml"))
            wls = fitness_wavelengths_from_bw(
                bl["min_bw_nm"], bl["center_nm"])
    return genome, parts, wls, label


def _build_axes(genome, parts, max_rotation, max_tilt, tolerance_pct):
    axes = []
    for name in _ROTATION_AXES:
        axes.append(AxisSpec(name, -max_rotation, max_rotation,
                             0.0, "deg"))

    for name in _INCIDENCE_AXES:
        label = _INCIDENCE_LABEL_MAP[name]
        if label == "F1" and parts.f1_focal_length_mm is None:
            continue
        axes.append(AxisSpec(name, -max_tilt, max_tilt, 0.0, "deg"))

    pct = tolerance_pct / 100.0
    for name in _LAYOUT_PARAMS:
        nominal = getattr(genome, name, None)
        if nominal is None:
            continue
        half = max(abs(nominal) * pct, 1.0)
        unit = "deg" if "deg" in name else "mm"
        axes.append(AxisSpec(name, nominal - half, nominal + half,
                             nominal, unit))
    return axes


def _trace(genome, parts, wls, n_rays, fnum, element_overrides=None):
    scene = build_optics_only_scene(genome, parts,
                                    element_overrides=element_overrides)
    return _trace_scene(scene, genome, parts, wls, n_rays, fnum)


def _trace_scene(scene, genome, parts, wls, n_rays, fnum):
    metrics = forward_trace_metrics(
        scene, genome, parts,
        wavelengths_nm=wls,
        design_wavelength_nm=550.0,
        n_rays=n_rays,
        input_fnum=fnum,
        target_fnum=fnum,
        seed=42,
        point_source=True,
    )
    rms = metrics.get("rms_spot_um", float("nan"))
    ilf = metrics.get("ilf_fwhm_nm", float("nan"))
    return rms, ilf


# ── Scene perturbation helpers ─────────────────────────────────────────

def _rotate_axis_z(axis, angle_deg):
    """Rotate a 3-tuple axis vector about the z-axis by angle_deg."""
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    ax, ay, az = axis
    return (ax * c - ay * s, ax * s + ay * c, az)


def _normalize_vec(v):
    n = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0] / n, v[1] / n, v[2] / n)


def _perturb_element_angle(scene, label, angle_deg):
    """Return a new Scene with one element's axis rotated in-plane by angle_deg."""
    new_elements = []
    for elem in scene.elements:
        if elem.label == label:
            new_elements.append(dataclasses.replace(
                elem, axis=_rotate_axis_z(elem.axis, angle_deg)))
        else:
            new_elements.append(elem)
    return Scene(elements=new_elements, fixtures=scene.fixtures)


def _shift_element_position(scene, label, direction, delta_mm):
    """Return a new Scene with one element displaced along direction."""
    new_elements = []
    for elem in scene.elements:
        if elem.label == label:
            new_pos = (
                elem.position[0] + delta_mm * direction[0],
                elem.position[1] + delta_mm * direction[1],
                elem.position[2] + delta_mm * direction[2],
            )
            new_elements.append(dataclasses.replace(elem, position=new_pos))
        else:
            new_elements.append(elem)
    return Scene(elements=new_elements, fixtures=scene.fixtures)


# Layout param → element label.
# Position params shift the element along its arm direction.
# Angle params rotate the element's axis in-plane.
_LAYOUT_POSITION_PARAMS = {
    "L_m1_mm": "M1",
    "L_m2_mm": "M2",
    "L_a_mm": "entrance_slit",
    "L_b_mm": "detector",
    "L_f1_mm": "F1",
}
_LAYOUT_TILT_PARAMS = {
    "theta_d_deg": "detector",
}


def _layout_displacement_direction(scene, param_name):
    """Unit vector along which a position layout param shifts its element."""
    if param_name == "L_m1_mm":
        anchor = scene.by_label("grating").position
        target = scene.by_label("M1").position
    elif param_name == "L_m2_mm":
        anchor = scene.by_label("grating").position
        target = scene.by_label("M2").position
    elif param_name == "L_a_mm":
        try:
            anchor = scene.by_label("F1").position
        except KeyError:
            anchor = scene.by_label("M1").position
        target = scene.by_label("entrance_slit").position
    elif param_name == "L_b_mm":
        try:
            anchor = scene.by_label("F2").position
        except KeyError:
            anchor = scene.by_label("M2").position
        target = scene.by_label("detector").position
    elif param_name == "L_f1_mm":
        anchor = scene.by_label("M1").position
        target = scene.by_label("F1").position
    else:
        raise ValueError(f"Unknown position param: {param_name}")
    return _normalize_vec((
        target[0] - anchor[0],
        target[1] - anchor[1],
        target[2] - anchor[2],
    ))


# ── Sweep functions ────────────────────────────────────────────────────

def _fit_slopes(values, rms_list, ilf_list, nominal, v_shaped=False):
    """Fit linear slopes (used internally by sweep functions)."""
    vals = np.array(values)
    delta = np.abs(vals - nominal) if v_shaped else vals - nominal
    rms_arr = np.array(rms_list)
    ilf_arr = np.array(ilf_list)
    ok_r = np.isfinite(rms_arr)
    ok_i = np.isfinite(ilf_arr)
    rms_slope = (abs(np.polyfit(delta[ok_r], rms_arr[ok_r], 1)[0])
                 if ok_r.sum() >= 2 else float("nan"))
    ilf_slope = (abs(np.polyfit(delta[ok_i], ilf_arr[ok_i], 1)[0])
                 if ok_i.sum() >= 2 else float("nan"))
    return rms_slope, ilf_slope


def _threshold_budget(values, metric_list, nominal, threshold):
    """Max |perturbation| from nominal where metric stays below threshold.

    Linearly interpolates between sweep points to find the crossing.
    Returns the tighter (smaller) of the two sides for asymmetric
    responses, or inf if the metric never exceeds the threshold.
    """
    vals = np.array(values)
    arr = np.array(metric_list, dtype=float)
    deltas = vals - nominal

    budgets = []
    for sign in (-1, +1):
        mask = (deltas * sign >= 0) & np.isfinite(arr)
        if mask.sum() < 2:
            budgets.append(float("inf"))
            continue
        side_d = np.abs(deltas[mask])
        side_m = arr[mask]
        order = np.argsort(side_d)
        side_d, side_m = side_d[order], side_m[order]

        crossed = float("inf")
        for i in range(len(side_m) - 1):
            if side_m[i] <= threshold < side_m[i + 1]:
                frac = ((threshold - side_m[i])
                        / (side_m[i + 1] - side_m[i]))
                crossed = side_d[i] + frac * (side_d[i + 1] - side_d[i])
                break
            elif side_m[i] > threshold:
                crossed = 0.0
                break
        budgets.append(crossed)

    return min(budgets)


def _sweep_layout_axis(spec, genome, parts, wls, n_rays, fnum, n_points):
    """OAT sweep — perturbs only the target element's position or normal."""
    nominal_scene = build_optics_only_scene(genome, parts)
    grid = np.linspace(spec.lo, spec.hi, n_points)
    values = sorted({float(v) for v in np.append(grid, spec.nominal)})

    is_position = spec.name in _LAYOUT_POSITION_PARAMS
    if is_position:
        label = _LAYOUT_POSITION_PARAMS[spec.name]
        direction = _layout_displacement_direction(nominal_scene, spec.name)
    else:
        label = _LAYOUT_TILT_PARAMS[spec.name]

    rms_list, ilf_list = [], []
    for v in values:
        delta = v - spec.nominal
        try:
            if is_position:
                perturbed = _shift_element_position(
                    nominal_scene, label, direction, delta)
            else:
                perturbed = _perturb_element_angle(
                    nominal_scene, label, delta)
            r, i = _trace_scene(perturbed, genome, parts, wls, n_rays, fnum)
        except Exception:
            r, i = float("nan"), float("nan")
        rms_list.append(r)
        ilf_list.append(i)

    rms_slope, ilf_slope = _fit_slopes(values, rms_list, ilf_list,
                                       spec.nominal)
    return values, rms_list, ilf_list, rms_slope, ilf_slope


def _sweep_incidence_axis(spec, genome, parts, wls, n_rays, fnum,
                            n_points):
    """OAT sweep of an incidence angle (theta_M1, theta_M2, theta_F1, alpha)."""
    label = _INCIDENCE_LABEL_MAP[spec.name]
    nominal_scene = build_optics_only_scene(genome, parts)
    grid = np.linspace(spec.lo, spec.hi, n_points)
    values = sorted({float(v) for v in np.append(grid, spec.nominal)})

    rms_list, ilf_list = [], []
    for v in values:
        try:
            tilted = _perturb_element_angle(nominal_scene, label, v)
            r, i = _trace_scene(tilted, genome, parts, wls, n_rays, fnum)
        except Exception:
            r, i = float("nan"), float("nan")
        rms_list.append(r)
        ilf_list.append(i)

    rms_slope, ilf_slope = _fit_slopes(values, rms_list, ilf_list,
                                       spec.nominal, v_shaped=True)
    return values, rms_list, ilf_list, rms_slope, ilf_slope


_ROTATION_LABEL_MAP = {"phi_M1": "M1", "phi_F1": "F1"}


def _sweep_rotation_axis(spec, genome, parts, wls, n_rays, fnum,
                         n_points):
    """OAT sweep of cylinder axis orientation (phi_M1, phi_F1)."""
    label = _ROTATION_LABEL_MAP[spec.name]
    grid = np.linspace(spec.lo, spec.hi, n_points)
    values = sorted({float(v) for v in np.append(grid, spec.nominal)})

    rms_list, ilf_list = [], []
    for v in values:
        try:
            overrides = {label: {"cylindrical_axis_rotation_deg": v}}
            r, i = _trace(genome, parts, wls, n_rays, fnum,
                          element_overrides=overrides)
        except Exception:
            r, i = float("nan"), float("nan")
        rms_list.append(r)
        ilf_list.append(i)

    rms_slope, ilf_slope = _fit_slopes(values, rms_list, ilf_list,
                                       spec.nominal, v_shaped=True)
    return values, rms_list, ilf_list, rms_slope, ilf_slope


def _plot_axis(out_dir, spec, values, rms_list, ilf_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(values, rms_list, "o-", markersize=4)
    ax1.axvline(spec.nominal, color="k", linestyle="--", alpha=0.5)
    ax1.set_xlabel(f"{spec.name} ({spec.unit})")
    ax1.set_ylabel("mean RMS spot (µm)")
    ax1.set_title(f"{spec.name} — RMS")
    ax1.grid(alpha=0.3)

    ax2.plot(values, ilf_list, "o-", markersize=4, color="tab:orange")
    ax2.axvline(spec.nominal, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel(f"{spec.name} ({spec.unit})")
    ax2.set_ylabel("mean ILF EE76 (nm)")
    ax2.set_title(f"{spec.name} — ILF")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    safe_name = spec.name.replace(":", "_")
    fig.savefig(out_dir / f"tol_{safe_name}.png", dpi=110)
    plt.close(fig)


def _plot_ilf_panels(out_dir, spec, values, genome, parts, wls, n_rays,
                     fnum, perturb_fn):
    """Per-wavelength histogram panels showing the ILF shape at each sweep point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optics.forward_trace import make_cone_rays, trace_rays
    from optics.world_builder import build_world, _placement_transform
    from optics.grating_math import grating_rotation_deg
    from optics.elements.hit_recorder import HitRecorder
    from raysect.core.math import Point3D
    from raysect.primitive import Box

    pixel_pitch_mm = 0.008
    zoom_half_mm = 0.15
    safe_name = spec.name.replace(":", "_")

    for wl in wls:
        fig, axes_arr = plt.subplots(len(values), 1,
                                     figsize=(10, 2.0 * len(values)))
        if len(values) == 1:
            axes_arr = [axes_arr]

        for row, v in enumerate(values):
            scene = perturb_fn(v)
            det_el = next(e for e in scene.elements if e.kind == "detector")
            slit_el = next(e for e in scene.elements
                           if e.label == "entrance_slit")

            built = build_world(scene, input_fnum=fnum)
            theta = grating_rotation_deg(
                genome.alpha_deg, genome.beta_deg,
                parts.grating_groove_density_per_mm, 550.0)
            built.set_grating_rotation_deg(theta)

            rec_half = 50.0 * 1e-3
            rec_box = Box(
                lower=Point3D(-rec_half, -rec_half, -1e-3),
                upper=Point3D(+rec_half, +rec_half, 0.0),
            )
            rec_box.transform = _placement_transform(det_el)
            recorder = HitRecorder()
            rec_box.material = recorder
            rec_box.parent = built.world

            rng = np.random.default_rng(42 + int(wl))
            rays = make_cone_rays(slit_el, parts, n_rays, rng,
                                  input_fnum=fnum, point_source=True)
            trace_rays(rays, built.world, wl)

            w2p = rec_box.transform.inverse()
            hx = np.array([Point3D(*h.point).transform(w2p).x * 1000
                           for h in recorder.hits])

            ax = axes_arr[row]
            if len(hx) > 2:
                bins = np.arange(hx.min() - pixel_pitch_mm,
                                 hx.max() + 2 * pixel_pitch_mm,
                                 pixel_pitch_mm)
                if len(bins) >= 3:
                    counts, edges = np.histogram(hx, bins=bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.bar(centers, counts, width=pixel_pitch_mm * 0.9,
                           alpha=0.7, color="steelblue")
                    peak_idx = np.argmax(counts)
                    ax.set_xlim(centers[peak_idx] - zoom_half_mm,
                                centers[peak_idx] + zoom_half_mm)
                    if counts.max() > 0:
                        ax.axhline(counts.max() * 0.5, color="r",
                                   linestyle="--", alpha=0.5, lw=0.8)

            v_str = f"{v:.2f}" if isinstance(v, float) else str(v)
            ax.set_title(f"{spec.name} = {v_str} {spec.unit}    "
                         f"({len(hx)} hits)", fontsize=9)
            ax.set_ylabel("counts", fontsize=7)
            ax.tick_params(labelsize=6)

        axes_arr[-1].set_xlabel("dispersion axis x (mm)", fontsize=8)
        fig.suptitle(f"{spec.name} — λ = {int(wl)} nm", fontsize=11,
                     y=1.001)
        fig.tight_layout()
        fig.savefig(out_dir / f"ilf_{safe_name}_{int(wl)}nm.png", dpi=120)
        plt.close(fig)


def _print_table(results, ilf_target):
    print()
    print(f"{'#':>2}  {'axis':<22s}  "
          f"{'ILF budget':>14s}  "
          f"{'nom ILF':>8s}  {'max ILF':>8s}")
    print(f"     {'':22s}  "
          f"{'(<' + f'{ilf_target:.1f}' + ' nm)':>14s}  "
          f"{'(nm)':>8s}  {'(nm)':>8s}")
    print("-" * 70)
    for rank, r in enumerate(results, start=1):
        u = r["unit"]
        ib = (f"±{r['ilf_budget']:.2f} {u}"
              if math.isfinite(r["ilf_budget"]) and r["ilf_budget"] < 1e3
              else "> sweep")
        ilf_arr = np.array(r["ilf_list"], dtype=float)
        finite = ilf_arr[np.isfinite(ilf_arr)]
        nom_ilf = finite[len(finite) // 2] if len(finite) else float("nan")
        max_ilf = float(np.max(finite)) if len(finite) else float("nan")
        print(f"{rank:>2d}  {r['param']:<22s}  "
              f"{ib:>14s}  "
              f"{nom_ilf:>8.2f}  {max_ilf:>8.2f}")
    print()


def main():
    args = _parse_args()
    genome, parts, wls, label = _load_design(args)
    fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm

    axes = _build_axes(genome, parts, args.max_rotation, args.max_tilt,
                       args.tolerance_pct)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("output") / f"tolerance_{label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Assembly tolerance analysis — {label}")
    print(f"  phi ±{args.max_rotation}°, theta/alpha ±{args.max_tilt}°, "
          f"distances ±{args.tolerance_pct}%")
    print(f"  {args.n_points} points, {args.rays} rays, λ={wls}")
    print(f"  output: {out_dir}")
    print()

    nom_rms, nom_ilf = _trace(genome, parts, wls, args.rays, fnum)
    print(f"  Nominal: RMS={nom_rms:.1f} µm, ILF={nom_ilf:.3f} nm")
    print()

    t0 = time.time()
    results = []

    for spec in axes:
        half = (spec.hi - spec.lo) / 2.0
        print(f"  {spec.name:<20s}  nominal={spec.nominal:.1f} ±{half:.1f} {spec.unit}"
              f" ... ", end="", flush=True)
        ts0 = time.time()

        if spec.name in _ROTATION_AXES:
            values, rms_list, ilf_list, _, _ = \
                _sweep_rotation_axis(
                    spec, genome, parts, wls, args.rays, fnum,
                    args.n_points)
            rot_label = _ROTATION_LABEL_MAP[spec.name]
            def _perturb_rot(v, _l=rot_label):
                return build_optics_only_scene(
                    genome, parts,
                    element_overrides={
                        _l: {"cylindrical_axis_rotation_deg": v}})
            perturb_fn = _perturb_rot
        elif spec.name in _INCIDENCE_AXES:
            values, rms_list, ilf_list, _, _ = \
                _sweep_incidence_axis(
                    spec, genome, parts, wls, args.rays, fnum,
                    args.n_points)
            tilt_label = _INCIDENCE_LABEL_MAP[spec.name]
            nominal_scene = build_optics_only_scene(genome, parts)
            def _perturb_tilt(v, _s=nominal_scene, _l=tilt_label):
                return _perturb_element_angle(_s, _l, v)
            perturb_fn = _perturb_tilt
        else:
            values, rms_list, ilf_list, _, _ = \
                _sweep_layout_axis(
                    spec, genome, parts, wls, args.rays, fnum,
                    args.n_points)
            nominal_scene = build_optics_only_scene(genome, parts)
            is_pos = spec.name in _LAYOUT_POSITION_PARAMS
            if is_pos:
                pos_label = _LAYOUT_POSITION_PARAMS[spec.name]
                pos_dir = _layout_displacement_direction(
                    nominal_scene, spec.name)
                def _perturb_pos(v, _s=nominal_scene, _l=pos_label,
                                 _d=pos_dir, _n=spec.nominal):
                    return _shift_element_position(_s, _l, _d, v - _n)
            else:
                tilt_label = _LAYOUT_TILT_PARAMS[spec.name]
                def _perturb_pos(v, _s=nominal_scene, _l=tilt_label,
                                 _n=spec.nominal):
                    return _perturb_element_angle(_s, _l, v - _n)
            perturb_fn = _perturb_pos

        _plot_axis(out_dir, spec, values, rms_list, ilf_list)
        _plot_ilf_panels(out_dir, spec, values, genome, parts, wls,
                         min(args.rays, 20000), fnum, perturb_fn)

        ilf_budget = _threshold_budget(
            values, ilf_list, spec.nominal, args.ilf_target)

        results.append(dict(
            param=spec.name, nominal=spec.nominal, unit=spec.unit,
            ilf_budget=ilf_budget,
            values=values, rms_list=rms_list, ilf_list=ilf_list,
        ))
        bstr = (f"±{ilf_budget:.2f} {spec.unit}"
                if math.isfinite(ilf_budget) and ilf_budget < 1e3
                else "> sweep range")
        print(f"done {time.time() - ts0:.1f}s  "
              f"ILF budget (<{args.ilf_target}nm): {bstr}")

    results.sort(key=lambda r: r["ilf_budget"])

    _print_table(results, args.ilf_target)

    # CSV output
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "axis", "nominal", "unit",
                     "ilf_budget", "ilf_target_nm"])
        for rank, r in enumerate(results, start=1):
            w.writerow([
                rank, r["param"], f"{r['nominal']:.6g}", r["unit"],
                f"{r['ilf_budget']:.3f}", f"{args.ilf_target:.1f}",
            ])

    with open(out_dir / "raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "value", "rms_um", "ilf_nm"])
        for r in results:
            for v, rms, ilf in zip(r["values"], r["rms_list"],
                                   r["ilf_list"]):
                w.writerow([r["param"], f"{v:.6g}",
                            f"{rms:.3f}", f"{ilf:.6f}"])

    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "label": label,
            "nominal_rms_um": nom_rms,
            "nominal_ilf_nm": nom_ilf,
            "ilf_target_nm": args.ilf_target,
            "max_rotation_deg": args.max_rotation,
            "max_tilt_deg": args.max_tilt,
            "tolerance_pct": args.tolerance_pct,
            "n_points": args.n_points,
            "rays": args.rays,
            "wavelengths_nm": list(wls),
        }, f, indent=2)

    print(f"Total: {time.time() - t0:.1f}s")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
