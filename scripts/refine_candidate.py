"""Nelder-Mead refinement of a GA candidate or baseline genome.

Loads a candidate from a GA run (or a baseline TOML) and optimizes a
configurable set of layout parameters to minimize mean RMS spot size.
Defaults to tilt-only refinement; ``--params all`` refines the full
basin set.

Usage::

    # Tilt-only from GA run (fast, ~15 evals)
    python scripts/refine_candidate.py --run output/optim_... --candidate gen00_cand03

    # Full basin refinement
    python scripts/refine_candidate.py --run output/optim_... --params all

    # From baseline TOML
    python scripts/refine_candidate.py --baseline data/czerny_baseline_xia2017.toml \\
        --bom data/czerny_bom_xia2017_corrected.toml --params all

    # Custom param subset
    python scripts/refine_candidate.py --run output/optim_... --params theta_d_deg,L_b_mm

    # Fewer rays for exploration
    python scripts/refine_candidate.py --run output/optim_... --rays 5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import dataclasses
import numpy as np
from scipy.optimize import minimize

from designs.czerny_bom import build_optics_only_scene
from optics.forward_trace import forward_trace_metrics
from optics.scene import InfeasibleGeometry

BASIN_PARAMS = [
    "L_m1_mm", "L_m2_mm", "L_a_mm", "L_b_mm",
    "L_f1_mm", "theta_f1_deg", "theta_d_deg",
]


def _load_baseline(baseline_path, bom_path):
    from designs.czerny_base import CzernyGenome
    from designs.czerny_bom import load_parts, load_bom
    from optics._config import read_toml

    data = read_toml(Path(baseline_path))
    genome_keys = {f.name for f in dataclasses.fields(CzernyGenome)}
    genome = CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})

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
    label = Path(baseline_path).stem
    return genome, parts, label


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("source (one required)")
    src.add_argument("--run", type=str, default=None)
    src.add_argument("--candidate", type=str, default=None)
    src.add_argument("--baseline", type=str, default=None)
    src.add_argument("--bom", type=str, default=None)

    ap.add_argument("--rays", type=int, default=10000)
    ap.add_argument("--params", type=str, default="theta_d_deg",
                    help="Comma-separated param names, or 'all' for full basin set")
    ap.add_argument("--wavelengths", type=str, default=None,
                    help="Fitness wavelengths (default: from run config, or 350,450,550,650,750)")
    ap.add_argument("--point-source", action="store_true", default=None)
    ap.add_argument("--optics-only", action="store_true", default=False,
                    help="Skip mount/collision checks (default: full assembly)")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if not args.run and not args.baseline:
        ap.error("one of --run or --baseline is required")

    if args.run:
        from export import _load_winner
        _, genome, parts, label, run_flags = _load_winner(args.run, args.candidate)
        point_source = run_flags["point_source"]

        fixed = None
        try:
            from optimizer_engine import load_state
            state = load_state(str(Path(args.run)))
            if state:
                fixed = state["fixed_params"]
        except Exception:
            pass

        if args.wavelengths:
            wls = tuple(float(w) for w in args.wavelengths.split(","))
        elif fixed:
            wls = tuple(fixed["fitness_wavelengths_nm"])
        else:
            wls = (350.0, 550.0, 750.0)
    else:
        if args.bom:
            from designs.czerny_bom import set_bom_path
            set_bom_path(args.bom)
        genome, parts, label = _load_baseline(args.baseline, args.bom)
        point_source = True
        if args.wavelengths:
            wls = tuple(float(w) for w in args.wavelengths.split(","))
        else:
            wls = (350.0, 450.0, 550.0, 650.0, 750.0)

    if args.point_source is not None:
        point_source = args.point_source

    if args.params == "all":
        param_names = [p for p in BASIN_PARAMS if getattr(genome, p) is not None]
    else:
        param_names = [p.strip() for p in args.params.split(",")]

    input_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    x0 = np.array([getattr(genome, p) for p in param_names])

    if not args.optics_only:
        from designs.czerny_assembly import assemble_scene as _assemble
        from designs.czerny_bom import _resolve_geometry
        _geometry = _resolve_geometry(bom_path=args.bom)

        def _build_scene(g, parts):
            return _assemble(
                g, parts,
                scene_builder=_geometry.build_optics_only_scene,
            )
    else:
        _build_scene = build_optics_only_scene

    rows = []
    call_count = [0]
    best_so_far = [float("inf")]

    def evaluate(x):
        call_count[0] += 1
        updates = {p: float(v) for p, v in zip(param_names, x)}
        g = dataclasses.replace(genome, **updates)

        try:
            scene = _build_scene(g, parts)
        except InfeasibleGeometry as exc:
            param_str = "  ".join(f"{p}={v:.3f}" for p, v in zip(param_names, x))
            print(f"  [{call_count[0]:3d}] {param_str}  INFEASIBLE: {exc}")
            return 1e6
        except Exception:
            return 1e6

        metrics = forward_trace_metrics(
            scene, g, parts,
            wavelengths_nm=wls,
            design_wavelength_nm=550.0,
            n_rays=args.rays,
            input_fnum=input_fnum,
            target_fnum=input_fnum,
            seed=42,
            point_source=point_source,
        )

        rms = [metrics[f"rms_spot_{int(w)}_um"] for w in wls]
        mean_rms = sum(rms) / len(rms)
        tput = [metrics[f"throughput_{int(w)}_nm"] for w in wls]
        min_tput = min(tput)

        row = {"eval": call_count[0], "mean_rms": mean_rms, "min_tput": min_tput}
        for p, v in zip(param_names, x):
            row[p] = v
        for w, r in zip(wls, rms):
            row[f"rms_{int(w)}"] = r
        for w, t in zip(wls, tput):
            row[f"tput_{int(w)}"] = t
        rows.append(row)

        tag = " *" if mean_rms < best_so_far[0] else ""
        best_so_far[0] = min(best_so_far[0], mean_rms)
        param_str = "  ".join(f"{p}={v:.3f}" for p, v in zip(param_names, x))
        rms_str = " ".join(f"{r:.0f}" for r in rms)
        print(f"  [{call_count[0]:3d}] {param_str}  mean={mean_rms:.1f}  [{rms_str}]{tag}")
        return mean_rms

    print(f"Nelder-Mead refinement — {label}, {args.rays} rays, "
          f"{'point source' if point_source else 'extended source'}")
    print(f"Params: {param_names}")
    seed_str = "  ".join(f"{p}={v:.3f}" for p, v in zip(param_names, x0))
    print(f"Seed: {seed_str}")
    print()

    n = len(x0)
    simplex = np.empty((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        vertex = x0.copy()
        step = max(abs(vertex[i]) * 0.05, 1.0)
        vertex[i] += step
        simplex[i + 1] = vertex

    result = minimize(
        evaluate, x0,
        method="Nelder-Mead",
        options={"xatol": 0.01, "fatol": 0.1, "maxiter": 400,
                 "adaptive": True, "initial_simplex": simplex},
    )

    if args.output:
        out_path = Path(args.output)
    elif args.run:
        out_path = Path(f"output/refine_{Path(args.run).name}_{label}.csv")
    else:
        out_path = Path(f"output/refine_{label}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} evaluations to {out_path}")
    print(f"Converged: {result.success} ({result.message})")
    for p, v in zip(param_names, result.x):
        print(f"  {p} = {v:.4f}")
    print(f"  mean RMS = {result.fun:.1f} µm")

    best = min(rows, key=lambda r: r["mean_rms"])
    if best["eval"] != call_count[0]:
        print(f"\nBest eval [{best['eval']}]:")
        for p in param_names:
            print(f"  {p} = {best[p]:.4f}")
        print(f"  mean RMS = {best['mean_rms']:.1f} µm")


if __name__ == "__main__":
    main()
