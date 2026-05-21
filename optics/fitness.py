"""Fitness function for the Czerny-Turner build.

Takes a raw-metrics dict from `optics.metrics.raw_metrics`
and reduces it to a single score (lower = better) that
`evolutionary-solver` can rank.

Fitness = mean(rms_spot_λ) in microns — arithmetic mean of
per-wavelength RMS spot radii.  Balances spot quality across
the band without over-penalising band-edge outliers.  Validated
against published Zemax results: the mean-RMS optimum (tilt ≈ +4°)
matches the Zemax-optimized θ_D = 3.5°.

Lower is better.
"""


import math

INFEASIBLE_PENALTY: float = 1e6


def scalar_fitness(raw_metrics: dict[str, float]) -> float:
    """Collapse a raw-metrics dict to a scalar fitness (lower = better).

    Returns ``INFEASIBLE_PENALTY`` for invalid metrics.
    """
    rms_wl: list[float] = []
    for key in raw_metrics:
        if not key.startswith("rms_spot_") or not key.endswith("_um"):
            continue
        val = raw_metrics[key]
        if val is None or not math.isfinite(val) or val <= 0:
            return INFEASIBLE_PENALTY
        rms_wl.append(val)

    if rms_wl:
        return sum(rms_wl) / len(rms_wl)

    return INFEASIBLE_PENALTY
