"""Assembly tolerance analysis.

Sweeps each physical dimension and alignment parameter around its nominal
value, measures mean RMS spot size and ILF FWHM, and ranks by sensitivity.
Answers the question: "which measurements matter most to get right during
assembly, and what tolerances do they need?"

Three categories of tolerance axes:

1. **Axis rotation** — cylinder axis rotation on M1 and F1.
2. **Layout distances** — arm lengths, mirror distances, fold position.
3. **Layout angles** — detector tilt, fold mirror angle.

Each axis is swept ± a configurable range around the baseline nominal.
The output table shows the sensitivity slope (µm of spot and nm of ILF
per unit of error) so tolerance budgets can be read directly.

Axis rotation sensitivity is computed from one-sided perturbation
(the response is symmetric / U-shaped, so a linear fit through both
sides gives a misleadingly low slope).

Example::

    python scripts/tolerance.py
    python scripts/tolerance.py --baseline data/czerny_baseline_v0_design.toml
    python scripts/tolerance.py --tolerance_pct 3 --max_rotation 3 --n_points 21
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


_ROTATION_AXES = ("M1_axis_rotation", "F1_axis_rotation")
_LAYOUT_PARAMS = (
    "L_a_mm", "L_b_mm", "L_f1_mm",
    "L_m1_mm", "L_m2_mm",
    "theta_f1_deg", "theta_d_deg",
)
_ROTATION_PERTURBATIONS = (0.25, 0.5, 1.0, 2.0)


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
    p.add_argument("--max_rotation", type=float, default=5.0,
                   help="Axis rotation sweep range ± degrees (default: %(default)s)")
    p.add_argument("--tolerance_pct", type=float, default=5.0,
                   help="Layout sweep range as %% of nominal (default: %(default)s)")
    p.add_argument("--n_points", type=int, default=21,
                   help="Points per axis (default: %(default)s)")
    p.add_argument("--rays", type=int, default=5000,
                   help="Rays per wavelength (default: %(default)s)")
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Fitness wavelengths in nm (default: from targets)")
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


def _build_axes(genome, max_rotation, tolerance_pct):
    axes = []
    for name in _ROTATION_AXES:
        nominal = 90.0 if "F1" in name else 0.0
        axes.append(AxisSpec(name, -max_rotation, max_rotation,
                             nominal, "deg"))

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


def _sweep_layout_axis(spec, genome, parts, wls, n_rays, fnum, n_points):
    grid = np.linspace(spec.lo, spec.hi, n_points)
    values = sorted({float(v) for v in np.append(grid, spec.nominal)})

    rms_list, ilf_list = [], []
    for v in values:
        try:
            g = dataclasses.replace(genome, **{spec.name: v})
            r, i = _trace(g, parts, wls, n_rays, fnum)
        except Exception:
            r, i = float("nan"), float("nan")
        rms_list.append(r)
        ilf_list.append(i)

    vals = np.array(values)
    rms_arr = np.array(rms_list)
    ilf_arr = np.array(ilf_list)
    ok_r = np.isfinite(rms_arr)
    ok_i = np.isfinite(ilf_arr)

    rms_slope = (abs(np.polyfit(vals[ok_r] - spec.nominal,
                                rms_arr[ok_r], 1)[0])
                 if ok_r.sum() >= 2 else float("nan"))
    ilf_slope = (abs(np.polyfit(vals[ok_i] - spec.nominal,
                                ilf_arr[ok_i], 1)[0])
                 if ok_i.sum() >= 2 else float("nan"))

    return values, rms_list, ilf_list, rms_slope, ilf_slope


def _sweep_rotation_axis(spec, genome, parts, wls, n_rays, fnum):
    nom_rms, nom_ilf = _trace(genome, parts, wls, n_rays, fnum)
    label = spec.name.split("_")[0]

    rms_slopes, ilf_slopes = [], []
    for v in _ROTATION_PERTURBATIONS:
        try:
            overrides = {label: {"cylindrical_axis_rotation_deg": v}}
            r, i = _trace(genome, parts, wls, n_rays, fnum,
                          element_overrides=overrides)
            rms_slopes.append((r - nom_rms) / v)
            ilf_slopes.append(
                (i - nom_ilf) / v
                if math.isfinite(i) and math.isfinite(nom_ilf)
                else float("nan"))
        except Exception:
            rms_slopes.append(float("nan"))
            ilf_slopes.append(float("nan"))

    # Use smallest valid perturbation for linear-regime slope
    rms_slope = next((s for s in rms_slopes if math.isfinite(s)),
                     float("nan"))
    ilf_slope = next((s for s in ilf_slopes if math.isfinite(s)),
                     float("nan"))

    return rms_slope, ilf_slope


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
    ax2.set_ylabel("mean ILF FWHM (nm)")
    ax2.set_title(f"{spec.name} — ILF")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    safe_name = spec.name.replace(":", "_")
    fig.savefig(out_dir / f"tol_{safe_name}.png", dpi=110)
    plt.close(fig)


def _print_table(results):
    print()
    print(f"{'#':>2}  {'axis':<20s}  {'nominal':>8s} {'unit':>4s}  "
          f"{'RMS slope':>14s}  {'ILF slope':>14s}  "
          f"{'±1px RMS':>10s}  {'±0.1nm ILF':>10s}")
    print("-" * 100)
    for rank, r in enumerate(results, start=1):
        u = r["unit"]
        rs = (f"{r['rms_slope']:.1f} µm/{u}"
              if math.isfinite(r["rms_slope"]) else "—")
        ils = (f"{r['ilf_slope']:.4f} nm/{u}"
               if math.isfinite(r["ilf_slope"]) else "—")
        rb = (f"±{r['rms_budget']:.2f} {u}"
              if math.isfinite(r["rms_budget"]) and r["rms_budget"] < 1e3
              else "—")
        ib = (f"±{r['ilf_budget']:.2f} {u}"
              if math.isfinite(r["ilf_budget"]) and r["ilf_budget"] < 1e3
              else "—")
        print(f"{rank:>2d}  {r['param']:<20s}  {r['nominal']:>8.1f} {u:>4s}  "
              f"{rs:>14s}  {ils:>14s}  {rb:>10s}  {ib:>10s}")
    print()


def main():
    args = _parse_args()
    genome, parts, wls, label = _load_design(args)
    fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm

    axes = _build_axes(genome, args.max_rotation, args.tolerance_pct)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("output") / f"tolerance_{label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Assembly tolerance analysis — {label}")
    print(f"  axis rotation ±{args.max_rotation}°, layout ±{args.tolerance_pct}%")
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
            rms_slope, ilf_slope = _sweep_rotation_axis(
                spec, genome, parts, wls, args.rays, fnum)
            values, rms_list, ilf_list = [], [], []
        else:
            values, rms_list, ilf_list, rms_slope, ilf_slope = \
                _sweep_layout_axis(
                    spec, genome, parts, wls, args.rays, fnum,
                    args.n_points)
            _plot_axis(out_dir, spec, values, rms_list, ilf_list)

        rms_budget = (8.0 / rms_slope
                      if rms_slope > 0 and math.isfinite(rms_slope)
                      else float("inf"))
        ilf_budget = (0.1 / ilf_slope
                      if ilf_slope > 0 and math.isfinite(ilf_slope)
                      else float("inf"))

        results.append(dict(
            param=spec.name, nominal=spec.nominal, unit=spec.unit,
            rms_slope=rms_slope, ilf_slope=ilf_slope,
            rms_budget=rms_budget, ilf_budget=ilf_budget,
            values=values, rms_list=rms_list, ilf_list=ilf_list,
        ))
        print(f"done {time.time() - ts0:.1f}s  "
              f"RMS={rms_slope:.1f} µm/{spec.unit}  "
              f"ILF={ilf_slope:.4f} nm/{spec.unit}")

    results.sort(key=lambda r: (
        -r["rms_slope"] if math.isfinite(r["rms_slope"]) else 0))

    _print_table(results)

    # CSV output
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "axis", "nominal", "unit",
                     "rms_slope_um_per_unit", "ilf_slope_nm_per_unit",
                     "rms_budget", "ilf_budget"])
        for rank, r in enumerate(results, start=1):
            w.writerow([
                rank, r["param"], f"{r['nominal']:.6g}", r["unit"],
                f"{r['rms_slope']:.3f}", f"{r['ilf_slope']:.6f}",
                f"{r['rms_budget']:.3f}", f"{r['ilf_budget']:.3f}",
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
            "max_rotation_deg": args.max_rotation,
            "tolerance_pct": args.tolerance_pct,
            "n_points": args.n_points,
            "rays": args.rays,
            "wavelengths_nm": list(wls),
        }, f, indent=2)

    print(f"Total: {time.time() - t0:.1f}s")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
