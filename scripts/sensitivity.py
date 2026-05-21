"""1-D OAT (One-At-a-Time) sensitivity diagnostic.

Pivots on BASELINE or a GA winner, sweeps each genome axis across its
full BOM-resolved range holding all others at the pivot, records fitness
per point, and prints a ranked table with sweep-vs-evolve recommendations.

Rank axes by ``delta_fitness = max - min``. Highest delta = most sensitive.

Recommendation rule:
    rel < 0.3                    → "fix"    (low impact; pin to reduce DoF)
    discrete                     → "evolve (discrete)"
    monotonic                    → "evolve" (GA pushes to boundary)
    n_basins >= 2                → "sweep"  (multi-modal; brute-force wins)
    else                         → "evolve" (single-basin smooth)

Alternatives for when OAT is insufficient:

- Morris elementary-effects — random-start OAT trajectories; catches
  first-order interactions OAT misses. ~350 evals at 100k SPP via
  ``SALib.sample.morris`` + ``SALib.analyze.morris``. Worth adding if
  OAT rankings shift materially between pivots.
- Sobol variance decomposition — total-order indices including all
  interactions. ~1000+ evals for convergence. Run once for a definitive
  ranking, not per-tuning-iteration.

For assembly tolerance analysis (axis rotation, arm lengths, fold position),
see ``scripts/tolerance.py``.

Example::

    python scripts/sensitivity.py --max_rld 17 --min_bw 300 --max_fnum 4 --fold_mode F1
    python scripts/sensitivity.py --max_rld 17 --min_bw 300 --max_fnum 4 --run output/optim_czerny_...
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.setrecursionlimit(10_000)

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from optimizer_engine import find_winner, load_state

from optics.problem import SpectrographProblem

from designs.czerny_bom import BASELINE, _resolve_geometry
from optics._config import load_defaults
from optics.fitness import scalar_fitness

from designs.czerny_bom import (
    FOLD_MODES,
    bounds_from_ranges,
    build_czerny_problem,
    fitness_wavelengths_from_bw,
    fold_axes_for_mode,
    fold_config,
    load_targets,
    max_footprint_xy_mm,
    resolve_from_specs,
)


_CONTINUOUS_AXES = ("Dv_deg", "theta_m1_deg", "theta_d_deg")
_FOLD_AXES = ("theta_f1_deg",)
_REL_FIX_THRESHOLD = 0.3  # axes below this fraction of max Δ are "fix"
_MONOTONIC_TOL = 1e-6


@dataclass
class AxisSpec:
    name: str
    kind: str             # "continuous" | "discrete"
    lo: float | None      # None for discrete
    hi: float | None
    choices: list | None  # None for continuous
    pivot: Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--max_rld", type=float, default=None,
                   help="Maximum reciprocal linear dispersion (nm/mm). "
                        "Default: from czerny_targets.toml.")
    p.add_argument("--min_bw", type=float, default=None,
                   help="Minimum bandwidth on detector (nm). "
                        "Default: from czerny_targets.toml.")
    p.add_argument("--max_fnum", type=float, default=None,
                   help="Maximum f-number (slowest acceptable). "
                        "Default: from czerny_targets.toml.")
    p.add_argument("--center", type=float, default=None,
                   help="Design center wavelength (nm). "
                        "Default: from czerny_targets.toml.")
    p.add_argument("--fold_mode", choices=FOLD_MODES, default=None,
                   help="Fold mode (default: from defaults.toml)")
    p.add_argument("--run", type=str, default=None,
                   help="Optimizer run dir (pivot on winner; default: baseline)")
    p.add_argument("--candidate", type=str, default=None,
                   help="Candidate ID within --run (default: best winner)")
    p.add_argument("--n_points", type=int, default=20,
                   help="Points per continuous axis (default: %(default)s)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: output/sensitivity_<ts>/).")
    return p.parse_args()


def _resolve_pivot(run_dir: str | None, candidate_id: str | None,
                   resolved: dict, fold_axes: tuple[str, ...],
                   bounds: dict[str, tuple[float, float]]) -> tuple[dict[str, Any], str]:
    """Build a pivot params dict covering every genome axis."""
    if run_dir is None:
        combo = resolved["combo_list"][0]
        params = {
            "Dv_deg":        float(BASELINE.alpha_deg - BASELINE.beta_deg),
            "theta_m1_deg":  float(BASELINE.theta_m1_deg),
            "theta_d_deg":   float(BASELINE.theta_d_deg),
            "m1_part":       combo["m1"],
            "m2_part":       combo["m2"],
            "grating_part":  combo["grating"],
            "optic_size_mm": resolved["optic_size_mm"],
        }
        if "f1" in combo:
            params["f1_part"] = combo["f1"]
        for axis in fold_axes:
            if axis in bounds:
                lo, hi = bounds[axis]
                params[axis] = 0.5 * (lo + hi)
        if hasattr(BASELINE, "theta_f1_deg") and "theta_f1_deg" in bounds:
            params["theta_f1_deg"] = float(BASELINE.theta_f1_deg)
        return params, "baseline"

    state = load_state(run_dir)
    if state is None:
        raise SystemExit(f"No optimizer_state.json in {run_dir}")
    if candidate_id:
        cand = next((c for c in state["candidates"] if c["id"] == candidate_id), None)
        if cand is None:
            raise SystemExit(f"Candidate {candidate_id!r} not found in {run_dir}")
    else:
        cand = find_winner(state)
        if cand is None:
            raise SystemExit(f"No fine_done winner in {run_dir}")

    fp = cand["full_params"]
    bp = cand.get("best_point")
    if not bp:
        raise SystemExit(
            f"Candidate {cand['id']} has no best_point — "
            f"cannot pivot on a candidate with all-infeasible sweeps")

    params: dict[str, Any] = {"optic_size_mm": resolved["optic_size_mm"]}
    for key in ("Dv_deg", "theta_m1_deg", "theta_d_deg",
                "m1_part", "m2_part", "grating_part", "f1_part"):
        if key in bp:
            val = bp[key]
        elif key in fp:
            val = fp[key]
        else:
            continue
        params[key] = float(val) if isinstance(val, (int, float)) else val
    for axis in fold_axes:
        if axis in fp and fp[axis] is not None and not _is_nan(fp[axis]):
            params[axis] = float(fp[axis])
        else:
            lo, hi = bounds[axis]
            params[axis] = 0.5 * (lo + hi)

    return params, cand["id"]


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _build_sensitivity_problem(
    fitness_wavelengths_nm,
    max_footprint_xy_mm,
    bounds: dict[str, tuple[float, float]],
    target_fnum: float,
    optic_size_mm: float,
    design_wavelength_nm: float,
    combo_list: list[dict],
    fold_mode: str,
) -> SpectrographProblem:
    """Build a SpectrographProblem with zero searched axes so every axis lives
    in ``_params`` and one ``evaluate([])`` call measures one genome."""
    defaults = load_defaults()
    geometry = _resolve_geometry()
    return build_czerny_problem(
        searched_axis_names=[],
        fitness_wavelengths_nm=fitness_wavelengths_nm,
        max_footprint_xy_mm=max_footprint_xy_mm,
        bounds=bounds,
        target_fnum=target_fnum,
        optic_size_mm=optic_size_mm,
        optics_only=True,
        geometry=geometry,
        design_wavelength_nm=design_wavelength_nm,
        combo_list=combo_list,
        fold_mode=fold_mode,
        forward_rays=defaults["sweep"]["forward_rays"],
        coarse_forward_rays=defaults["sweep"]["coarse_forward_rays"],
        point_source=True,
    )


def _build_axis_catalogue(
    pivot: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    resolved: dict,
    fold_axes: tuple[str, ...],
) -> list[AxisSpec]:
    axes: list[AxisSpec] = []
    for name in _CONTINUOUS_AXES:
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        pv = pivot.get(name)
        if pv is None:
            pv = 0.5 * (lo + hi)
        if not (lo <= pv <= hi):
            raise SystemExit(
                f"Pivot {name}={pv:g} outside BOM-resolved bounds [{lo:g}, {hi:g}]. "
                f"Check product_class / usage_type."
            )
        axes.append(AxisSpec(name, "continuous", lo, hi, None, pv))

    for axis in _FOLD_AXES:
        if axis in bounds:
            lo, hi = bounds[axis]
            pv = pivot.get(axis, 0.5 * (lo + hi))
            axes.append(AxisSpec(axis, "continuous", lo, hi, None, pv))

    for axis in fold_axes:
        if axis in bounds and not any(a.name == axis for a in axes):
            lo, hi = bounds[axis]
            pv = pivot.get(axis, 0.5 * (lo + hi))
            axes.append(AxisSpec(axis, "continuous", lo, hi, None, pv))

    gratings = list({c["grating"] for c in resolved["combo_list"]})
    if len(gratings) >= 2:
        axes.append(AxisSpec("grating_part", "discrete",
                             None, None, gratings, pivot.get("grating_part", gratings[0])))

    return axes


def _sweep_axis(
    problem: SpectrographProblem,
    pivot: dict[str, Any],
    spec: AxisSpec,
    n_points: int,
) -> tuple[list, list[float], list[str]]:
    if spec.kind == "continuous":
        grid = np.linspace(spec.lo, spec.hi, n_points)
        pv = float(spec.pivot)
        if spec.lo <= pv <= spec.hi:
            values = sorted({float(v) for v in np.append(grid, pv)})
        else:
            values = [float(v) for v in grid]
    else:
        values = list(spec.choices)

    fitnesses: list[float] = []
    statuses: list[str] = []
    for v in values:
        params = {**pivot, spec.name: v}
        problem.prepare(params)
        metrics, status = problem.evaluate(axis_values=[], seed=0, timeout=None)
        if status == "ok" and metrics is not None:
            raw = dict(zip(problem.result_names, metrics))
            fitnesses.append(float(scalar_fitness(raw)))
        else:
            fitnesses.append(float("nan"))
        statuses.append(status)
    return values, fitnesses, statuses


def _compute_axis_metrics(
    spec: AxisSpec,
    fitnesses: list[float],
    statuses: list[str],
) -> dict[str, Any]:
    arr = np.array(fitnesses, dtype=float)
    valid = arr[np.isfinite(arr)]
    n_ok = sum(1 for s in statuses if s == "ok")
    status_summary = f"{n_ok}/{len(statuses)} ok"
    if n_ok < len(statuses):
        status_summary += f"; {len(statuses) - n_ok} infeasible"

    if valid.size == 0:
        return {
            "fitness_min": float("nan"),
            "fitness_max": float("nan"),
            "delta_fitness": float("nan"),
            "monotonic_bool": False,
            "n_basins": None,
            "status": status_summary,
        }

    fmin = float(np.min(valid))
    fmax = float(np.max(valid))
    delta = fmax - fmin

    monotonic = False
    n_basins: int | None = None
    if spec.kind == "continuous" and valid.size >= 2:
        diffs = np.diff(valid)
        non_trivial = diffs[np.abs(diffs) > _MONOTONIC_TOL]
        if non_trivial.size == 0:
            monotonic = True
        else:
            monotonic = bool(np.all(non_trivial > 0) or np.all(non_trivial < 0))
        signs = np.sign(non_trivial)
        sign_changes = int(np.sum(np.abs(np.diff(signs)) > 0)) if signs.size > 1 else 0
        n_basins = sign_changes + 1

    return {
        "fitness_min": fmin,
        "fitness_max": fmax,
        "delta_fitness": delta,
        "monotonic_bool": monotonic,
        "n_basins": n_basins,
        "status": status_summary,
    }


def _recommendation(spec: AxisSpec, metrics: dict[str, Any], rel: float) -> str:
    if not math.isfinite(metrics["delta_fitness"]):
        return "INFEASIBLE AT PIVOT"
    if rel < _REL_FIX_THRESHOLD:
        return "fix"
    if spec.kind == "discrete":
        return "evolve (discrete)"
    if metrics["monotonic_bool"]:
        return "evolve"
    if metrics["n_basins"] and metrics["n_basins"] >= 2:
        return "sweep"
    return "evolve"


def _plot_axis(out_dir: Path, spec: AxisSpec, values: list,
               fitnesses: list[float], pivot_value: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if spec.kind == "continuous":
        ax.plot(values, fitnesses, "o-")
        try:
            ax.axvline(float(pivot_value), color="k", linestyle="--",
                       alpha=0.6, label=f"pivot={pivot_value:.3g}")
        except (TypeError, ValueError):
            pass
        ax.set_xlabel(spec.name)
    else:
        x = np.arange(len(values))
        ax.bar(x, fitnesses)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right")
        try:
            pi = values.index(pivot_value)
            ax.axvline(pi, color="k", linestyle="--", alpha=0.6,
                       label=f"pivot={pivot_value}")
        except ValueError:
            pass
        ax.set_xlabel(spec.name)
    ax.set_ylabel("mean RMS spot (µm)")
    ax.set_title(spec.name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"sens_{spec.name}.png", dpi=110)
    plt.close(fig)


def _write_summary_csv(out_dir: Path, results: list[dict]) -> None:
    import csv
    path = out_dir / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "axis", "kind", "range_lo", "range_hi", "n_points",
            "pivot_value", "fitness_min", "fitness_max", "delta_fitness",
            "rel_sensitivity", "n_basins", "monotonic_bool",
            "recommendation", "status",
        ])
        for rank, r in enumerate(results, start=1):
            spec = r["spec"]
            m = r["metrics"]
            w.writerow([
                rank, spec.name, spec.kind,
                spec.lo if spec.lo is not None else "",
                spec.hi if spec.hi is not None else "",
                len(r["values"]),
                spec.pivot if not isinstance(spec.pivot, float)
                else f"{spec.pivot:.6g}",
                f"{m['fitness_min']:.6g}",
                f"{m['fitness_max']:.6g}",
                f"{m['delta_fitness']:.6g}",
                f"{r['rel_sensitivity']:.4f}",
                m["n_basins"] if m["n_basins"] is not None else "",
                m["monotonic_bool"],
                r["recommendation"],
                m["status"],
            ])


def _write_raw_csv(out_dir: Path, results: list[dict]) -> None:
    import csv
    path = out_dir / "raw.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "kind", "value", "fitness", "status", "is_pivot"])
        for r in results:
            spec = r["spec"]
            for v, fit, st in zip(r["values"], r["fitnesses"], r["statuses"]):
                is_pivot = v == spec.pivot
                w.writerow([
                    spec.name, spec.kind,
                    v if isinstance(v, str) else f"{v:.6g}",
                    f"{fit:.6g}", st, is_pivot,
                ])


def _print_ranked_table(results: list[dict]) -> None:
    print()
    print(f"{'#':>2}  {'axis':<20s}  {'Δ fit':>8s}  {'rel':>5s}  "
          f"{'basins':>6s}  {'mono':>5s}  recommendation")
    print("-" * 80)
    for rank, r in enumerate(results, start=1):
        spec = r["spec"]
        m = r["metrics"]
        basins = m["n_basins"] if m["n_basins"] is not None else "-"
        mono = "Y" if m["monotonic_bool"] else "N"
        delta = m["delta_fitness"]
        delta_str = f"{delta:.4f}" if math.isfinite(delta) else "   nan"
        print(f"{rank:>2d}  {spec.name:<20s}  {delta_str:>8s}  "
              f"{r['rel_sensitivity']:>5.2f}  {str(basins):>6s}  "
              f"{mono:>5s}  {r['recommendation']}")
    print()


def main() -> None:
    args = _parse_args()
    defaults = load_defaults()
    targets = load_targets()

    max_rld = args.max_rld if args.max_rld is not None else targets["max_rld_nm_per_mm"]
    min_bw = args.min_bw if args.min_bw is not None else targets["min_bw_nm"]
    max_fnum = args.max_fnum if args.max_fnum is not None else targets["max_fnum"]
    center_wl = args.center if args.center is not None else targets["center_nm"]

    default_mode = fold_config(defaults)
    fold_mode = args.fold_mode if args.fold_mode is not None else default_mode
    resolved = resolve_from_specs(max_rld, min_bw, max_fnum, center_wl, defaults, fold_mode)
    fold_axes = fold_axes_for_mode(fold_mode)

    bounds = bounds_from_ranges(resolved["ranges"], fold_axes)
    fitness_wavelengths = fitness_wavelengths_from_bw(min_bw, center_wl)
    max_footprint = max_footprint_xy_mm()

    pivot, pivot_label = _resolve_pivot(args.run, args.candidate,
                                        resolved, fold_axes, bounds)

    problem = _build_sensitivity_problem(
        fitness_wavelengths_nm=fitness_wavelengths,
        max_footprint_xy_mm=max_footprint,
        bounds=bounds,
        target_fnum=resolved["target_fnum"],
        optic_size_mm=resolved["optic_size_mm"],
        design_wavelength_nm=center_wl,
        combo_list=resolved["combo_list"],
        fold_mode=fold_mode,
    )

    axes = _build_axis_catalogue(pivot, bounds, resolved, fold_axes)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("output") / f"sensitivity_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sensitivity analysis — pivot={pivot_label}")
    print(f"  max_rld={max_rld} min_bw={min_bw} max_fnum={max_fnum} center={center_wl} "
          f"fold_mode={fold_mode}")
    print(f"  n_points={args.n_points}")
    print(f"  output: {out_dir}")
    print()

    t0 = time.time()
    results: list[dict] = []
    for spec in axes:
        print(f"  sweeping {spec.name} ({spec.kind}) "
              f"{'[' + str(spec.lo) + ', ' + str(spec.hi) + ']' if spec.kind == 'continuous' else str(spec.choices)}"
              f" ... ", end="", flush=True)
        ts0 = time.time()
        values, fitnesses, statuses = _sweep_axis(
            problem, pivot, spec, args.n_points)
        metrics = _compute_axis_metrics(spec, fitnesses, statuses)
        _plot_axis(out_dir, spec, values, fitnesses, spec.pivot)
        results.append({
            "spec": spec,
            "values": values,
            "fitnesses": fitnesses,
            "statuses": statuses,
            "metrics": metrics,
        })
        print(f"done in {time.time() - ts0:.1f}s  "
              f"Δ={metrics['delta_fitness']:.3f}")

    finite_deltas = [r["metrics"]["delta_fitness"] for r in results
                     if math.isfinite(r["metrics"]["delta_fitness"])]
    max_delta = max(finite_deltas) if finite_deltas else float("nan")
    for r in results:
        d = r["metrics"]["delta_fitness"]
        r["rel_sensitivity"] = d / max_delta if math.isfinite(d) and max_delta > 0 \
            else float("nan")
        r["recommendation"] = _recommendation(r["spec"], r["metrics"],
                                              r["rel_sensitivity"])

    results.sort(key=lambda r: (
        -r["metrics"]["delta_fitness"]
        if math.isfinite(r["metrics"]["delta_fitness"]) else float("inf")
    ))

    _write_summary_csv(out_dir, results)
    _write_raw_csv(out_dir, results)
    with (out_dir / "pivot.json").open("w") as f:
        json.dump(pivot, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump({
            "max_rld":        max_rld,
            "min_bw":         min_bw,
            "max_fnum":       max_fnum,
            "fold_mode":      fold_mode,
            "n_points":       args.n_points,
            "pivot":          pivot_label,
            "run":            args.run,
            "candidate":      args.candidate,
            "bounds":         {k: list(v) for k, v in bounds.items()},
            "fitness_wavelengths_nm": list(fitness_wavelengths),
        }, f, indent=2)

    _print_ranked_table(results)
    print(f"Total runtime: {time.time() - t0:.1f} s")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
