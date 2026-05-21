"""GA run via `optimizer_engine.run_optimizer`.

Spec-driven: ``--max_rld/--min_bw/--max_fnum/--center`` pre-filters BOM into a
small set of valid combos (combo_id). GA evolves combo_id (discrete)
+ Dv, θ_M1, θ_F1 (continuous). Physics derives
θ_M2 (coma cancellation), L_m1/L_m2, L_a/L_b tangential focus. Basin refinement via
Nelder-Mead over (L_a, [L_f1], L_b, θ_D).

Fitness = mean(rms_spot_λ) (lower is better).

Two-tier evaluation pipeline:

- **Tier 1 — Analytical pre-filter**: CCD coverage gate, pure math.
- **Tier 2 — Forward trace**: rays from entrance slit, hits on
  oversized detector-plane recorder. Coarse: 1k rays/λ; fine: 5k.

Configuration sourced from TOML files (no silent fallbacks):
- ``data/defaults.toml`` — optimizer hyperparams, ray counts.
- ``data/czerny_bom_*.toml`` — BOM catalogs per design class.

Examples::

    python run_optimizer.py czerny --max_rld 17 --min_bw 400 --max_fnum 5 --center 550
    python run_optimizer.py czerny --fold_mode F1 --optics_only
    python run_optimizer.py --resume output/optim_czerny_20260509_223024
"""


# Raise recursion limit for housed ray tracing. Raysect's recursive
# daughter.trace(world) can exceed Python's default 1000 when rays
# bounce many times inside Lambert housing walls.
import sys
sys.setrecursionlimit(10_000)

import argparse
import time
from pathlib import Path

from optimizer_engine import find_winner, load_state, run_optimizer

from designs import GEOMETRY_REGISTRY
from designs.czerny_bom import load_baseline_parts, load_parts
from optics._config import load_defaults

from designs.czerny_bom import (
    FOLD_MODES,
    _effective_bom_path,
    build_czerny_problem,
    fitness_wavelengths_from_bw,
    fold_config,
    load_bom,
    load_targets,
    max_footprint_xy_mm,
    peak_from_csv,
    resolve_from_specs,
)


def _describe_parts(full_params: dict) -> list[str]:
    if not full_params:
        return []
    bom = load_bom()
    lines: list[str] = []

    def _mirror_line(label, section, pn):
        opts = bom["mirrors"][section][pn]
        mtype = opts.get("mirror_type", "spherical")
        orient = opts.get("cylindrical_orientation", "")
        if orient:
            mtype = f"{mtype}/{orient}"
        d_mm = opts["diameter_mm"]
        f_mm = opts.get("focal_length_mm")
        peak = peak_from_csv(opts["reflectance_file"])
        parts = [f"  {label:<8s}{pn}  {mtype}  D={d_mm:g}mm"]
        if f_mm:
            parts.append(f"f={f_mm:g}mm  f/{f_mm / d_mm:.2f}")
        parts.append(f"R={peak:.3f}")
        return "  ".join(parts)

    lines.append(_mirror_line("M1", "m1_options", full_params["m1_part"]))
    lines.append(_mirror_line("M2", "m2_options", full_params["m2_part"]))

    g_pn = full_params["grating_part"]
    opts = bom["grating_options"][g_pn]
    peak = peak_from_csv(opts["efficiency_file"])
    lines.append(f"  Grating {g_pn}  {opts['groove_density_per_mm']:g} g/mm"
                 f"  blaze={opts['blaze_nm']:g}nm  {opts['size_mm']:g}mm  E={peak:.3f}")

    for label, section in [("F1", "f1_options"), ("F2", "f2_options")]:
        pn = full_params.get(f"{label.lower()}_part")
        if pn:
            lines.append(_mirror_line(label, section, pn))

    return lines





def _fine_step_defaults() -> dict[str, float]:
    grid = load_defaults()["sweep"]["grid"]
    return {
        "L_m1_mm": float(grid.get("fine_step_L_m1_mm", 1.0)),
        "L_m2_mm": float(grid.get("fine_step_L_m2_mm", 1.0)),
        "f1_fraction": float(grid.get("fine_step_f1_fraction", 0.05)),
    }


_AXIS_RANGE_KEYS = {
    "Dv_deg": "Dv_range",
    "theta_m1_deg": "theta_m1_range",
    "theta_m2_deg": "theta_m2_range",
    "L_m1_mm": "L_m1_range",
    "L_m2_mm": "L_m2_range",
    "f1_fraction": "f1_fraction_range",
    "f2_fraction": "f2_fraction_range",
    "theta_f1_deg": "theta_f_range",
    "theta_f2_deg": "theta_f_range",
}


def _adaptive_grid(
    ranges: dict,
    target_coarse_pts: int,
    searched_axes: tuple[str, ...],
    override_steps: dict[str, float | None],
    override_coarse_factor: int | None,
) -> tuple[dict[str, float], int]:
    """Fix per-axis fine_step; derive coarse_factor to hit target_coarse_pts.

    Fine step defaults from ``defaults.toml`` [sweep.grid].
    CLI overrides in ``override_steps`` (axis_name → value or None).

    Coarse factor is derived per-axis from the range width:

        coarse_factor_ax = round(width[ax] / (target_coarse_pts × fine_step[ax]))

    Take the max across axes so all hit at least ``target_coarse_pts``
    coarse points. ``--coarse_factor`` CLI flag (non-None) short-
    circuits the derivation.

    Returns ``(fine_steps, coarse_factor)``.
    """
    if target_coarse_pts < 2:
        raise ValueError(
            f"target_coarse_pts must be >= 2, got {target_coarse_pts}"
        )

    defaults = _fine_step_defaults()
    fine_steps = {}
    for ax in searched_axes:
        override = override_steps.get(ax)
        if override is not None:
            fine_steps[ax] = float(override)
        elif ax in defaults:
            fine_steps[ax] = defaults[ax]
        else:
            raise ValueError(f"no fine_step default for axis {ax!r}")

    for ax, step in fine_steps.items():
        if step <= 0.0:
            raise ValueError(f"fine_step[{ax}] must be > 0, got {step}")

    if override_coarse_factor is not None:
        if override_coarse_factor < 1:
            raise ValueError(
                f"coarse_factor must be >= 1, got {override_coarse_factor}"
            )
        return fine_steps, int(override_coarse_factor)

    candidates: list[int] = []
    for ax in searched_axes:
        range_key = _AXIS_RANGE_KEYS.get(ax)
        if range_key is None or range_key not in ranges:
            continue
        rng = ranges[range_key]
        if len(rng) < 2:
            raise ValueError(f"{range_key!r} malformed in ranges: {rng!r}")
        width = float(rng[1]) - float(rng[0])
        if width == 0.0:
            continue
        if width < 0.0:
            raise ValueError(f"{range_key}: negative width {width}")
        cf = int(round(width / (target_coarse_pts * fine_steps[ax])))
        if cf < 1:
            raise ValueError(
                f"{ax}: derived coarse_factor {cf} < 1. "
                f"Range width {width:g} too narrow for "
                f"target_coarse_pts={target_coarse_pts} × "
                f"fine_step={fine_steps[ax]:g}. Either widen the range, "
                f"reduce target_coarse_pts, or use a smaller fine_step."
            )
        candidates.append(cf)
    coarse_factor = max(candidates) if candidates else 1
    return fine_steps, coarse_factor




def _parse_args() -> argparse.Namespace:
    defaults = load_defaults()
    sweep = defaults["sweep"]
    grid = sweep["grid"]
    optim = defaults["optimizer"]
    p = argparse.ArgumentParser(
        description="GA optimizer for standard Czerny-Turner spectrographs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run_optimizer.py czerny --bom data/czerny_bom_xia2017_corrected.toml --fold_mode F1 --optics_only --point_source
  python run_optimizer.py --resume output/optim_czerny_20260512_195020
  python run_optimizer.py --winners 10
  python run_optimizer.py --status output/optim_czerny_20260512_195020
""",
    )

    p.add_argument("geometry", nargs="?", default=None,
                    choices=list(GEOMETRY_REGISTRY),
                    help="Geometry variant (required for fresh runs)")

    spec = p.add_argument_group("design targets (default: czerny_targets.toml)")
    spec.add_argument("--max_rld", type=float, default=None,
                       help="Maximum reciprocal linear dispersion (nm/mm). "
                            "Rejects M1 with f_coll < d_grating / max_rld.")
    spec.add_argument("--min_bw", type=float, default=None,
                       help="Minimum bandwidth on detector (nm). "
                            "Rejects M2 with f_cam > L_det * d_grating / min_bw.")
    spec.add_argument("--max_fnum", type=float, default=None,
                       help="Maximum f-number (slowest acceptable). "
                            "Rejects M1 with D < f_coll / max_fnum.")
    spec.add_argument("--center", type=float, default=None,
                       help="Design center wavelength (nm)")

    modes = p.add_argument_group("run modes")
    modes.add_argument("--resume", metavar="RUN_DIR",
                        help="Resume a previous run")
    modes.add_argument("--status", metavar="RUN_DIR",
                        help="Print run status and exit")
    modes.add_argument("--winners", type=int, nargs="?", const=5,
                        default=None, metavar="N",
                        help="Show top N candidates across all runs (default: 5)")
    modes.add_argument("--best", type=int, nargs="?", const=6,
                        default=None, metavar="N",
                        help="Show N diverse top designs via clustering (default: 6)")

    ga = p.add_argument_group("GA parameters")
    ga.add_argument("--pop_size", type=int, default=optim["pop_size"],
                     help="Candidates per generation (default: %(default)s)")
    ga.add_argument("--max_generations", type=int,
                     default=optim["max_generations"],
                     help="Max GA generations (default: %(default)s)")
    ga.add_argument("--n_basins", type=int, default=None,
                     help="Basins per candidate (default: 1)")
    ga.add_argument("--n_workers", type=int, default=None,
                     help="Total CPU budget for parallel evaluation "
                          "(default: 50%% of cores)")
    ga.add_argument("--max_concurrent", type=int, default=None,
                     help="Candidates to evaluate in parallel "
                          "(default: n_workers)")
    ga.add_argument("--seed", action="store_true",
                     help="Seed population with exploration genomes")

    grid_g = p.add_argument_group("coarse grid")
    grid_g.add_argument("--target_coarse_pts", type=int,
                         default=grid["target_coarse_pts"],
                         help="Coarse points per swept axis (default: %(default)s)")
    grid_g.add_argument("--coarse_factor", type=int, default=None,
                         help="Override coarse_factor (default: derived from target_coarse_pts)")
    grid_g.add_argument("--grid_resolution", type=float,
                         default=sweep["grid_resolution"],
                         help="Fine-grid quantization step (default: %(default)s)")
    grid_g.add_argument("--acceptance_threshold", type=float,
                         default=sweep["acceptance_threshold"],
                         help="Min throughput to accept a grid point (default: %(default)s)")

    phys = p.add_argument_group("physics")
    phys.add_argument("--bom", type=str, default=None,
                       help="BOM TOML path (default: czerny_bom_v0_design.toml)")
    phys.add_argument("--fold_mode", choices=FOLD_MODES, default=None,
                       help="Fold configuration: none, F1, F2, both")
    phys.add_argument("--optics_only", action="store_true",
                       help="Skip mounts/housing, bare optic collision only")
    phys.add_argument("--point_source", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Point source (default: True; --no-point_source for fiber convolution)")

    misc = p.add_argument_group("misc")
    misc.add_argument("--run_name", type=str, default=None,
                       help="Custom run directory name")
    return p.parse_args()


def _mirror_focal_mm(part_number: str) -> float:
    bom = load_bom()
    for section in ("m1_options", "m2_options"):
        opt = bom["mirrors"][section].get(part_number)
        if isinstance(opt, dict) and "focal_length_mm" in opt:
            return float(opt["focal_length_mm"])
    return float("nan")


def _candidate_row(c: dict, run_name: str, spec_tag: str,
                   flux_keys: list[str], throughput_keys: list[str]) -> dict:
    """Build a display row dict from a candidate's state entry.

    f1_frac / f2_frac are NaN when the run's fold mode doesn't
    evolve the corresponding axis (i.e. the axis key is absent from the
    candidate's full_params).
    """
    import math
    fp = c["full_params"]
    bp = c["best_point"]
    if bp is None:
        return None
    m1_pn = fp.get("m1_part") or bp.get("m1_part")
    m2_pn = fp.get("m2_part") or bp.get("m2_part")
    L_a = _mirror_focal_mm(m1_pn) if m1_pn else math.nan
    L_b = _mirror_focal_mm(m2_pn) if m2_pn else math.nan
    ilf_keys = [k.replace("throughput_", "ilf_fwhm_") for k in throughput_keys]
    row = {
        "run": run_name,
        "id": c["id"],
        "spec": spec_tag,
        "fitness": c["fine_fitness"],
        "flux_watts": bp.get("flux_watts", math.nan),
        "throughput": bp.get("throughput", math.nan),
        "ilf_fwhm_nm": bp.get("ilf_fwhm_nm", math.nan),
        "Dv": fp.get("Dv_deg", bp.get("Dv_deg", math.nan)),
        "L_a": L_a,
        "L_b": L_b,
        "theta1": fp.get("theta_m1_deg", bp.get("theta_m1_deg", math.nan)),
        "theta2": bp.get("theta_m2_deg", math.nan),
        "f1_frac": float(fp["f1_fraction"])
                       if "f1_fraction" in fp else math.nan,
        "f2_frac": float(fp["f2_fraction"])
                       if "f2_fraction" in fp else math.nan,
        "m1_part": m1_pn or "",
        "m2_part": m2_pn or "",
        "grating_part": fp.get("grating_part", bp.get("grating_part", "")),
    }
    for k in flux_keys:
        row[k] = bp.get(k, math.nan)
    for k in throughput_keys:
        row[k] = bp.get(k, math.nan)
    for k in ilf_keys:
        row[k] = bp.get(k, math.nan)
    return row


def _spec_tag(fp: dict) -> str:
    """One-line summary of a run's spec inputs for display."""
    return f"max_rld={fp['max_rld']} min_bw={fp['min_bw']} max_fnum={fp['max_fnum']}"


def _show_winners(top_n: int) -> None:
    import glob
    import os

    state_files = sorted(glob.glob("output/optim_*/optimizer_state.json")
                         + glob.glob("output/optimizer_*/optimizer_state.json"))
    if not state_files:
        print("No optimizer runs found in output/optimizer_*/")
        return

    all_candidates = []
    for sf in state_files:
        run_dir = os.path.dirname(sf)
        state = load_state(run_dir)
        if state is None:
            continue
        run_name = state["run_name"]
        n_gens = len(state["generation_history"])
        done = [c for c in state["candidates"]
                if c["fine_fitness"] is not None]
        fp = state["fixed_params"]
        from designs.czerny_bom import set_bom_path
        set_bom_path(fp["bom_path"])
        wls = fp["fitness_wavelengths_nm"]
        fk = [f"flux_{int(round(l))}_nm" for l in wls]
        tk = [f"throughput_{int(round(l))}_nm" for l in wls]
        spec_tag = _spec_tag(fp)
        geo_tag = fp["geometry"]
        print(f"  {run_name}: {n_gens} gens, {len(done)} evaluated, "
              f"spec={spec_tag} geometry={geo_tag}")
        for c in done:
            row = _candidate_row(c, run_name, spec_tag, fk, tk)
            if row is None:
                continue
            row["geometry"] = geo_tag
            row["_wavelengths"] = wls
            row["_flux_keys"] = fk
            row["_throughput_keys"] = tk
            all_candidates.append(row)

    if not all_candidates:
        print("\nNo evaluated candidates found.")
        return

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in all_candidates:
        key = (r["spec"], r["geometry"])
        groups.setdefault(key, []).append(r)

    for key in sorted(groups):
        spec, geo = key
        rows = groups[key]
        ranked = sorted(rows, key=lambda c: c["fitness"])[:top_n]
        wls = ranked[0]["_wavelengths"]
        fk = ranked[0]["_flux_keys"]
        tk = ranked[0]["_throughput_keys"]
        print(f'\n══ spec={spec}  geometry={geo}  '
              f"({len(rows)} candidates) ══")
        _print_winner_table(ranked, fk, tk, wls)


def _fold_cell(v: float) -> str:
    """Format a fold_fraction value; NaN renders as an empty slot."""
    import math
    return f"{v:5.2f}" if not math.isnan(v) else "  -- "


def _print_winner_table(ranked, flux_keys, throughput_keys, wavelengths) -> None:
    """Render a single product-class group's top-N table."""
    import math
    if not ranked:
        return
    # Only show BOM-choice columns when at least one ranked candidate has
    # them populated — keeps pre-BOM-evolve output clean.
    show_bom = any(c["m1_part"] or c["m2_part"] or c["grating_part"]
                   for c in ranked)
    show_fold = any(not math.isnan(c["f1_frac"])
                    or not math.isnan(c["f2_frac"]) for c in ranked)
    run_w = max((len(c["run"]) for c in ranked), default=3)

    print(f"\nTop {len(ranked)} candidates:")
    t_lam_hdr = "  ".join(f"{int(round(l)):>5d}" for l in wavelengths)
    ilf_lam_hdr = "  ".join(f"{int(round(l)):>5d}" for l in wavelengths)
    ilf_keys = [f"ilf_fwhm_{int(round(l))}_nm" for l in wavelengths]
    hdr = (f"{'Rank':>4s}  {'Fitness':>7s}  {'T%':>5s}  "
           f"{t_lam_hdr}  "
           f"{'ILF':>5s}  "
           f"{ilf_lam_hdr}  "
           f"{'Dv':>5s}  "
           f"{'L_a':>5s}  {'L_b':>5s}  "
           f"{'θ₁':>5s}  {'θ₂':>5s}")
    if show_fold:
        hdr += f"  {'f_a':>5s}  {'f_b':>5s}"
    if show_bom:
        hdr += (f"  {'M1':>15s}  {'M2':>15s}  {'grating':>10s}")
    hdr += f"  {'run':>{run_w}s}  ID"
    print(hdr)
    print("-" * (len(hdr) + 20))
    for rank, c in enumerate(ranked, 1):
        t_mean = c["throughput"]
        t_lam_cells = "  ".join(f"{c[k]*100:4.1f}%" for k in throughput_keys)
        ilf_mean = c["ilf_fwhm_nm"]
        ilf_cells = "  ".join(f"{c[k]:5.1f}" for k in ilf_keys)
        line = (f"{rank:4d}  {c['fitness']:7.1f}  {t_mean*100:4.1f}%  "
                f"{t_lam_cells}  "
                f"{ilf_mean:5.1f}  "
                f"{ilf_cells}  "
                f"{c['Dv']:5.1f}  "
                f"{c['L_a']:5.1f}  {c['L_b']:5.1f}  "
                f"{c['theta1']:5.1f}  {c['theta2']:5.1f}")
        if show_fold:
            line += f"  {_fold_cell(c['f1_frac'])}  {_fold_cell(c['f2_frac'])}"
        if show_bom:
            line += (f"  {c['m1_part']:>15s}  {c['m2_part']:>15s}  "
                     f"{c['grating_part']:>10s}")
        line += f"  {c['run']:>{run_w}s}  {c['id']}"
        print(line)


def _show_best(n_clusters: int) -> None:
    """Show N diverse top designs per product-class group by clustering
    each group's top decile."""
    import glob
    import os

    state_files = sorted(glob.glob("output/optim_*/optimizer_state.json")
                         + glob.glob("output/optimizer_*/optimizer_state.json"))
    if not state_files:
        print("No optimizer runs found in output/optimizer_*/")
        return

    all_candidates = []
    for sf in state_files:
        run_dir = os.path.dirname(sf)
        state = load_state(run_dir)
        if state is None:
            continue
        run_name = state["run_name"]
        done = [c for c in state["candidates"]
                if c["fine_fitness"] is not None]
        fp = state["fixed_params"]
        from designs.czerny_bom import set_bom_path
        set_bom_path(fp["bom_path"])
        wls = fp["fitness_wavelengths_nm"]
        fk = [f"flux_{int(round(l))}_nm" for l in wls]
        tk = [f"throughput_{int(round(l))}_nm" for l in wls]
        spec_tag = _spec_tag(fp)
        geo_tag = fp["geometry"]
        print(f"  {run_name}: {len(done)} evaluated, spec={spec_tag} geometry={geo_tag}")
        for c in done:
            row = _candidate_row(c, run_name, spec_tag, fk, tk)
            if row is None:
                continue
            row["geometry"] = geo_tag
            row["_wavelengths"] = wls
            row["_flux_keys"] = fk
            row["_throughput_keys"] = tk
            all_candidates.append(row)

    if not all_candidates:
        print("\nNo evaluated candidates found.")
        return

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in all_candidates:
        key = (r["spec"], r["geometry"])
        groups.setdefault(key, []).append(r)

    for key in sorted(groups):
        spec, geo = key
        rows = groups[key]
        wls = rows[0]["_wavelengths"]
        fk = rows[0]["_flux_keys"]
        tk = rows[0]["_throughput_keys"]
        print(f'\n══ spec={spec}  geometry={geo}  '
              f"({len(rows)} candidates) ══")
        _cluster_and_print_best(rows, n_clusters, fk, tk, wls)


def _cluster_and_print_best(rows, n_clusters, flux_keys, throughput_keys, wavelengths) -> None:
    """Top-decile cluster pick + print, scoped to a single product-class group."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    ranked = sorted(rows, key=lambda c: c["fitness"])
    cutoff_idx = max(1, len(ranked) // 10)
    viable = ranked[:cutoff_idx]
    print(f"Top decile: {len(viable)} candidates "
          f"(fitness ≤ {viable[-1]['fitness']:.3f})")

    if len(viable) < 2:
        print("Too few candidates for clustering.")
        return

    features = ["Dv", "theta1", "theta2"]
    X = np.array([[c[f] for f in features] for c in viable])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    n_cl = min(n_clusters, len(viable))
    Z = linkage((X - mu) / sigma, method="ward")
    cl = fcluster(Z, t=n_cl, criterion="maxclust")

    bests = []
    for cid in sorted(set(cl)):
        members = [viable[i] for i, label in enumerate(cl) if label == cid]
        best = min(members, key=lambda c: c["fitness"])
        best["n_members"] = len(members)
        bests.append(best)
    bests.sort(key=lambda c: c["fitness"])

    import math
    show_bom = any(c["m1_part"] or c["m2_part"] or c["grating_part"]
                   for c in bests)
    show_fold = any(not math.isnan(c["f1_frac"])
                    or not math.isnan(c["f2_frac"]) for c in bests)
    run_w = max((len(c["run"]) for c in bests), default=3)
    print(f"{len(bests)} diverse designs from {len(viable)} top-decile candidates:")
    t_lam_hdr = "  ".join(f"{int(round(l)):>5d}" for l in wavelengths)
    ilf_lam_hdr = "  ".join(f"{int(round(l)):>5d}" for l in wavelengths)
    ilf_keys = [f"ilf_fwhm_{int(round(l))}_nm" for l in wavelengths]
    hdr = (f"{'#':>2s}  {'Fitness':>7s}  {'T%':>5s}  "
           f"{t_lam_hdr}  "
           f"{'ILF':>5s}  "
           f"{ilf_lam_hdr}  "
           f"{'Dv':>5s}  "
           f"{'L_a':>5s}  {'L_b':>5s}  "
           f"{'θ₁':>5s}  {'θ₂':>5s}")
    if show_fold:
        hdr += f"  {'f_a':>5s}  {'f_b':>5s}"
    if show_bom:
        hdr += (f"  {'M1':>15s}  {'M2':>15s}  {'grating':>10s}")
    hdr += f"  {'N':>3s}  {'run':>{run_w}s}  ID"
    print(hdr)
    print("-" * (len(hdr) + 20))
    for rank, c in enumerate(bests, 1):
        t_mean = c["throughput"]
        t_lam_cells = "  ".join(f"{c[k]*100:4.1f}%" for k in throughput_keys)
        ilf_mean = c["ilf_fwhm_nm"]
        ilf_cells = "  ".join(f"{c[k]:5.1f}" for k in ilf_keys)
        line = (f"{rank:2d}  {c['fitness']:7.1f}  {t_mean*100:4.1f}%  "
                f"{t_lam_cells}  "
                f"{ilf_mean:5.1f}  "
                f"{ilf_cells}  "
                f"{c['Dv']:5.1f}  "
                f"{c['L_a']:5.1f}  {c['L_b']:5.1f}  "
                f"{c['theta1']:5.1f}  {c['theta2']:5.1f}")
        if show_fold:
            line += f"  {_fold_cell(c['f1_frac'])}  {_fold_cell(c['f2_frac'])}"
        if show_bom:
            line += (f"  {c['m1_part']:>15s}  {c['m2_part']:>15s}  "
                     f"{c['grating_part']:>10s}")
        line += f"  {c['n_members']:3d}  {c['run']:>{run_w}s}  {c['id']}"
        print(line)


def _show_status(run_dir: str) -> None:
    state = load_state(run_dir)
    if state is None:
        print(f"No optimizer state found in {run_dir}")
        return

    settings = state["settings"]
    fp = state["fixed_params"]
    print(f"Run: {state['run_name']}")
    from designs.czerny_bom import set_bom_path
    set_bom_path(fp["bom_path"])
    geo = fp["geometry"]
    print(f"Spec: {_spec_tag(fp)} | center={fp['center']}nm | Geometry: {geo}")
    flags = []
    if fp["optics_only"]:
        flags.append("optics_only")
    if fp["point_source"]:
        flags.append("point_source")
    print(f"Fold: {fp['fold_mode']} | BOM: {fp['bom_path']}"
          + (f" | {' '.join(flags)}" if flags else ""))
    coarse_factor = settings["coarse_factor"]
    print(f"Pop: {settings['pop_size']} | Max gens: {settings['max_generations']}"
          f" | Coarse pts: {fp['target_coarse_pts']} | Coarse factor: {coarse_factor}"
          f" | Basins: {settings['n_basins']}")
    print(f"Generations completed: {len(state['generation_history'])}")
    print(f"Total candidates: {len(state['candidates'])}")

    status_counts: dict[str, int] = {}
    for c in state["candidates"]:
        s = c["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"Status: {status_counts}")

    print(f"\nGeneration history:")
    for gi in state["generation_history"]:
        best_f = gi["best_fine"] if "best_fine" in gi else gi["best_coarse"]
        fitness_str = f"{best_f:.4f}" if isinstance(best_f, float) else "?"
        print(f"  Gen {gi['generation']}: "
              f"best_fitness={fitness_str}  "
              f"({gi['best_id']})")

    all_done = [c for c in state["candidates"]
                if c["fine_fitness"] is not None]
    if all_done:
        best = min(all_done, key=lambda c: c["fine_fitness"])
        bp = best["best_point"]
        print(f"\nOverall best: {best['id']} = {best['fine_fitness']:.4f}")
        if bp:
            for line in _describe_parts(best["full_params"]):
                print(line)
            for k, v in bp.items():
                print(f"  {k:<22s} {v}")
        print(f"  Dir: {best['output_dir']}")


def _print_winner(state):
    winner = find_winner(state) if state else None
    if winner is None:
        print("no winner found")
        return
    print(f"\nwinner: {winner['id']}")
    print(f"fine fitness: {winner['fine_fitness']:.6g}")
    bp = winner["best_point"]
    if bp is None:
        print("best sub-sweep point: none (all candidates infeasible)")
        return
    for line in _describe_parts(winner["full_params"]):
        print(line)
    print(f"best sub-sweep point:")
    for k, v in bp.items():
        print(f"  {k:<22s} {v}")


def _run_resume(args) -> None:
    state = load_state(args.resume)
    if state is None:
        print(f"Error: no state file in {args.resume}", file=sys.stderr)
        sys.exit(1)

    fp = state["fixed_params"]
    from designs.czerny_bom import set_bom_path
    set_bom_path(fp["bom_path"])

    print(f"Resuming {state['run_name']} from generation "
          f"{state['current_gen']}"
          f"  optics_only={fp['optics_only']}"
          f"  point_source={fp['point_source']}")

    # Reconstruct evolved_params from saved state.
    evolved_params = {}
    for k, v in state["evolved_params"].items():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, float)):
            evolved_params[k] = tuple(v)
        else:
            evolved_params[k] = v

    saved = state["settings"]
    resume_ranges = fp["ranges"]
    target_fnum = fp["target_fnum"]
    geo_name = fp["geometry"]
    geometry = GEOMETRY_REGISTRY[geo_name]()
    defaults = load_defaults()

    fold_mode = fp["fold_mode"]
    if args.fold_mode is not None:
        fold_mode = args.fold_mode

    # Detect old-format runs (no combo_id in evolved_params).
    if "combo_id" not in evolved_params:
        print("Error: this run uses the old optimizer architecture. "
              "Please start a fresh run.", file=sys.stderr)
        sys.exit(1)

    combo_list = fp["combo_list"]

    # Reconstruct searched axes from fine_step keys in settings.
    searched = tuple(
        k.removeprefix("fine_step_") for k in saved
        if k.startswith("fine_step_")
    )
    bounds = {}
    for ax in searched:
        range_key = _AXIS_RANGE_KEYS[ax]
        bounds[ax] = tuple(resume_ranges[range_key])

    center_wl = fp["center"]
    fit_wls = fitness_wavelengths_from_bw(fp["min_bw"], center_wl)

    problem = build_czerny_problem(
        searched_axis_names=list(searched),
        fitness_wavelengths_nm=fit_wls,
        max_footprint_xy_mm=max_footprint_xy_mm(),
        bounds=bounds,
        target_fnum=target_fnum,
        optic_size_mm=fp["optic_size_mm"],
        optics_only=fp["optics_only"],
        geometry=geometry,
        design_wavelength_nm=center_wl,
        combo_list=combo_list,
        fold_mode=fold_mode,
        forward_rays=defaults["sweep"]["forward_rays"],
        coarse_forward_rays=defaults["sweep"]["coarse_forward_rays"],
        point_source=fp["point_source"],
        acceptance_threshold=saved["acceptance_threshold"],
    )

    default_max_gens = defaults["optimizer"]["max_generations"]
    max_gens = args.max_generations if args.max_generations != default_max_gens else saved["max_generations"]

    if searched:
        fine_steps, resume_coarse_factor = _adaptive_grid(
            resume_ranges,
            target_coarse_pts=args.target_coarse_pts,
            searched_axes=searched,
            override_steps={},
            override_coarse_factor=args.coarse_factor,
        )
        fine_margins = {
            ax: fine_steps[ax] * resume_coarse_factor / 2 for ax in fine_steps
        }
    else:
        fine_steps = {}
        fine_margins = {}
        resume_coarse_factor = 1

    t0 = time.perf_counter()
    state = run_optimizer(
        fixed_params=state["fixed_params"],
        evolved_params=evolved_params,
        grid_resolution=saved["grid_resolution"],
        fine_steps=fine_steps,
        fine_margins=fine_margins,
        run_name=state["run_name"],
        pop_size=saved["pop_size"],
        max_generations=max_gens,
        coarse_factor=saved.get("coarse_factor", resume_coarse_factor),
        acceptance_threshold=saved["acceptance_threshold"],
        max_attempts=1,
        n_workers=1,
        max_concurrent=1,
        n_basins=(args.n_basins if args.n_basins is not None
                  else saved.get("n_basins", 1)),
        problem=problem,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nGA finished in {elapsed:.1f} s")
    _print_winner(state)


def _run_fresh(args) -> None:
    if args.geometry is None:
        print("Error: geometry (positional) required for fresh runs.",
              file=sys.stderr)
        sys.exit(1)

    if args.bom:
        from designs.czerny_bom import set_bom_path
        set_bom_path(args.bom)

    targets = load_targets()
    max_rld = args.max_rld if args.max_rld is not None else targets["max_rld_nm_per_mm"]
    min_bw = args.min_bw if args.min_bw is not None else targets["min_bw_nm"]
    max_fnum = args.max_fnum if args.max_fnum is not None else targets["max_fnum"]
    center_wl = args.center if args.center is not None else targets["center_nm"]

    geometry = GEOMETRY_REGISTRY[args.geometry]()
    defaults = load_defaults()

    fold_mode = fold_config(defaults)
    if args.fold_mode is not None:
        fold_mode = args.fold_mode

    try:
        resolved = resolve_from_specs(
            max_rld, min_bw, max_fnum, center_wl, defaults,
            fold_mode=fold_mode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    ranges = resolved["ranges"]
    combo_list = resolved["combo_list"]
    f1_cylindrical = resolved.get("f1_cylindrical", False)

    # ── Evolved params: combo_id (discrete) + continuous ───────────────
    evolved_params: dict = {
        "combo_id": list(range(len(combo_list))),
        "Dv_deg": tuple(ranges["Dv_range"]),
        "theta_m1_deg": tuple(ranges["theta_m1_range"]),
    }
    if fold_mode in ("F1", "both"):
        evolved_params["theta_f1_deg"] = tuple(
            ranges.get("theta_f_range", (5.0, 85.0)))

    # ── Searched (swept) axes: r1, r2 ─────────────────────────────────────
    # Fold position (L_f1) is derived analytically for both flat and
    # cylindrical F1: eq. 14 for cylindrical, feasible-band midpoint for
    # flat.  Neither requires a swept axis.
    searched: tuple[str, ...] = ("L_m1_mm", "L_m2_mm")

    bounds: dict = {
        "L_m1_mm": tuple(ranges["L_m1_range"]),
        "L_m2_mm": tuple(ranges["L_m2_range"]),
    }

    fit_wls = fitness_wavelengths_from_bw(min_bw, center_wl)

    problem = build_czerny_problem(
        searched_axis_names=list(searched),
        fitness_wavelengths_nm=fit_wls,
        max_footprint_xy_mm=max_footprint_xy_mm(),
        bounds=bounds,
        target_fnum=resolved["target_fnum"],
        optic_size_mm=resolved["optic_size_mm"],
        optics_only=args.optics_only,
        geometry=geometry,
        design_wavelength_nm=center_wl,
        combo_list=combo_list,
        fold_mode=fold_mode,
        forward_rays=defaults["sweep"]["forward_rays"],
        coarse_forward_rays=defaults["sweep"]["coarse_forward_rays"],
        point_source=args.point_source,
        acceptance_threshold=args.acceptance_threshold,
    )

    from designs.czerny_bom import _effective_bom_path
    bom = load_bom()
    fixed_params = {
        "bom_path": str(_effective_bom_path()),
        "slit_width_um": float(bom["slits"]["width_um"]),
        "slit_height_mm": float(bom["slits"]["height_mm"]),
        "fitness_wavelengths_nm": list(fit_wls),
        "ranges": dict(resolved["ranges"]),
        "target_fnum": resolved["target_fnum"],
        "optic_size_mm": float(resolved["optic_size_mm"]),
        "fold_mode": fold_mode,
        "target_coarse_pts": args.target_coarse_pts,
        "geometry": args.geometry,
        "max_rld": max_rld,
        "min_bw": min_bw,
        "max_fnum": max_fnum,
        "center": center_wl,
        "optics_only": args.optics_only,
        "point_source": args.point_source,
        "combo_list": resolved["combo_list"],
        "f2_part": resolved["f2_part"],
    }

    if args.run_name:
        run_name = args.run_name
    else:
        bom_stem = _effective_bom_path().stem.removeprefix("czerny_bom_")
        tags = [f"optim_{args.geometry}_{bom_stem}"]
        if fold_mode and fold_mode != "none":
            tags.append(fold_mode)
        if not args.point_source:
            tags.append("nopt")
        tags.append(time.strftime("%Y%m%d_%H%M%S"))
        run_name = "_".join(tags)

    # Grid params: only needed when there are searched axes.
    if searched:
        fine_steps, coarse_factor = _adaptive_grid(
            ranges,
            target_coarse_pts=args.target_coarse_pts,
            searched_axes=searched,
            override_steps={},
            override_coarse_factor=args.coarse_factor,
        )
        fine_margins = {
            ax: fine_steps[ax] * coarse_factor / 2 for ax in fine_steps
        }
    else:
        fine_steps = {}
        fine_margins = {}
        coarse_factor = 1
    cf_tag = "user" if args.coarse_factor is not None else "auto"

    print(f"  max_rld: {max_rld} nm/mm, min_bw: {min_bw} nm, max_fnum: {max_fnum}, center: {center_wl} nm")
    print(f"  fitness wavelengths: {fit_wls} nm")
    print(f"  combos: {len(combo_list)} pre-validated")
    cyl_tag = " (cylindrical F1 — astigmatism-free)" if f1_cylindrical else ""
    print(f"  fold mode: {fold_mode}{cyl_tag}")
    print(f"  evolved params:")
    for name, value in evolved_params.items():
        if isinstance(value, list):
            print(f"    {name:<22s} {len(value)} options")
        else:
            print(f"    {name:<22s} [{value[0]}, {value[1]}]")
    if searched:
        print(f"  searched (swept): {searched}")
    else:
        print(f"  searched (swept): — (no grid)")
    print(f"  NM refinement: {'(L_m1, L_m2, L_a, θ_f1, L_b, θ_d)' if fold_mode in ('F1', 'both') else '(L_m1, L_m2, L_a, L_b, θ_d)'}")
    print(f"  pop_size={args.pop_size}, max_generations={args.max_generations}")
    if searched:
        step_parts = [f"coarse_factor={coarse_factor} ({cf_tag})"]
        for ax in searched:
            step_parts.append(f"{ax}={fine_steps[ax]:.3f}")
        print(f"  grid: " + "  ".join(step_parts))
    if args.optics_only:
        print(f"  mode: optics-only (no mounts, no housing)")
    print(f"  run_name: {run_name}")

    run_dir = Path("output") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    optim = load_defaults()["optimizer"]
    n_basins = args.n_basins if args.n_basins is not None else optim["n_basins"]

    import multiprocessing
    n_workers = args.n_workers or max(1, multiprocessing.cpu_count() // 2)
    max_concurrent = args.max_concurrent if args.max_concurrent is not None else n_workers
    print(f"  workers: {n_workers} total, {max_concurrent} concurrent candidates")

    t0 = time.perf_counter()
    state = run_optimizer(
        fixed_params=fixed_params,
        evolved_params=evolved_params,
        grid_resolution=args.grid_resolution,
        fine_steps=fine_steps,
        fine_margins=fine_margins,
        run_name=run_name,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        coarse_factor=coarse_factor,
        acceptance_threshold=args.acceptance_threshold,
        max_attempts=1,
        n_workers=n_workers,
        max_concurrent=max_concurrent,
        n_basins=n_basins,
        problem=problem,
        seed_candidates=None,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nGA finished in {elapsed:.1f} s")
    _print_winner(state)



def main() -> None:
    args = _parse_args()
    if args.winners is not None:
        _show_winners(args.winners)
    elif args.best is not None:
        _show_best(args.best)
    elif args.status:
        _show_status(args.status)
    elif args.resume:
        _run_resume(args)
    else:
        _run_fresh(args)


if __name__ == "__main__":
    import multiprocessing as _mp
    import os as _os
    import signal as _signal

    def _sigint_handler(sig, frame):
        """SIGKILL all children and hard-exit on Ctrl-C.

        Previous version used SIGTERM + join(timeout=2) + SIGKILL, but
        raysect Cython workers hold the GIL and can't process SIGTERM
        until the current ray batch finishes — on WSL2 that's long
        enough to saturate all cores and freeze the machine.

        Strategy: SIGKILL (uncatchable, no GIL needed) via two paths,
        then os._exit without joining.
        """
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
        for proc in _mp.active_children():
            try:
                proc.kill()
            except OSError:
                pass
        # /proc walk catches children multiprocessing doesn't track
        _pid = _os.getpid()
        try:
            for entry in _os.listdir('/proc'):
                if entry.isdigit():
                    try:
                        with open(f'/proc/{entry}/stat') as f:
                            parts = f.read().split()
                            if int(parts[3]) == _pid:
                                _os.kill(int(entry), _signal.SIGKILL)
                    except (FileNotFoundError, PermissionError,
                            ProcessLookupError, IndexError, ValueError,
                            OSError):
                        pass
        except FileNotFoundError:
            pass
        _os._exit(1)

    _signal.signal(_signal.SIGINT, _sigint_handler)
    main()
