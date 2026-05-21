"""Smoke tests for the Czerny-Turner design pipeline.

Two checks, run together:
  1. metrics — raw_metrics on the BASELINE genome; "do the numbers look sane?"
  2. fitness — BASELINE vs a deliberately-broken config; "does the fitness
     function rank correctly?"

Run: `python tests/smoke.py`
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from designs.czerny_bom import build_optics_only_scene
from optics.metrics import raw_metrics

from designs.czerny_bom import (
    BASELINE,
    load_baseline_parts,
)
from optics.fitness import INFEASIBLE_PENALTY, scalar_fitness


def _print_metrics(metrics: dict) -> None:
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"    {key:<{width}}  {value:.6g}")


def _score(label, genome, parts, *, target_fnum):
    scene = build_optics_only_scene(genome, parts)
    metrics = raw_metrics(scene, genome, parts,
                          fitness_wavelengths_nm=(450.0, 550.0, 650.0),
                          target_fnum=target_fnum,
                          design_wavelength_nm=550.0,
                          forward_rays=5_000)
    fitness = scalar_fitness(metrics)
    print(f"[{label}] {genome}")
    _print_metrics(metrics)
    print(f"    -> scalar_fitness = {fitness:.6g}"
          + ("  (INFEASIBLE)" if fitness >= INFEASIBLE_PENALTY else ""))
    return fitness


def smoke_metrics() -> None:
    """Print raw metrics for the BASELINE genome."""
    print("── metrics ──")
    parts = load_baseline_parts()
    scene = build_optics_only_scene(BASELINE, parts)
    input_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    metrics = raw_metrics(scene, BASELINE, parts,
                          fitness_wavelengths_nm=(450.0, 550.0, 650.0),
                          target_fnum=input_fnum,
                          design_wavelength_nm=550.0,
                          forward_rays=5_000)
    _print_metrics(metrics)


def smoke_fitness() -> None:
    """Score BASELINE vs broken genome; assert BASELINE wins."""
    print("\n── fitness ──")
    parts = load_baseline_parts()
    target_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    baseline_fitness = _score("BASELINE", BASELINE, parts, target_fnum=target_fnum)
    broken = replace(BASELINE, L_a_mm=150.0, L_b_mm=150.0)
    broken_fitness = _score("BROKEN", broken, parts, target_fnum=target_fnum)

    if broken_fitness <= baseline_fitness:
        raise SystemExit(
            f"fitness function is not self-consistent: "
            f"broken fitness {broken_fitness} <= baseline {baseline_fitness}"
        )
    print(f"\nOK — BASELINE ({baseline_fitness:.4g}) beats "
          f"BROKEN ({broken_fitness:.4g}).")


def main() -> None:
    smoke_metrics()
    smoke_fitness()


if __name__ == "__main__":
    main()
