"""Extended validation against Xia 2017: NM refinement and tilt sweep.

Heavier validation that complements the fast tests in
``tests/test_xia2017.py``.  Results feed into paper §3.10.

Usage::

    python scripts/validate_xia2017.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize

from designs.czerny_base import CzernyGenome
from designs.czerny_bom import load_parts, build_optics_only_scene
from optics._config import read_toml
from optics.forward_trace import forward_trace_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XIA_STANDARD_BOM = DATA_DIR / "czerny_bom_xia2017_standard.toml"
XIA_CORRECTED_BOM = DATA_DIR / "czerny_bom_xia2017_corrected.toml"
XIA_BASELINE = DATA_DIR / "czerny_baseline_xia2017.toml"

XIA_STANDARD_RMS_UM = {
    350: 101.5,
    450: 132.6,
    550: 180.8,
    650: 248.9,
    750: 336.8,
}

XIA_CORRECTED_RMS_UM = {
    350: 16.7,
    450: 8.2,
    550: 4.4,
    650: 9.0,
    750: 16.9,
}


def _parts_standard():
    return load_parts(
        m1_part="XIA-M1", m2_part="XIA-M2", grating_part="GR25-0605",
        f1_part="XIA-F1",
        bom_path=XIA_STANDARD_BOM, optic_size_mm=25.4,
    )


def _parts_corrected():
    return load_parts(
        m1_part="XIA-M1-CYL", m2_part="XIA-M2", grating_part="GR25-0605",
        f1_part="XIA-F1-CYL",
        bom_path=XIA_CORRECTED_BOM, optic_size_mm=25.4,
    )


def _trace(genome, parts, wavelengths, *, n_rays=20_000, point_source=False):
    fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    scene = build_optics_only_scene(genome, parts)
    return forward_trace_metrics(
        scene, genome, parts,
        wavelengths_nm=tuple(float(l) for l in wavelengths),
        design_wavelength_nm=550.0,
        n_rays=n_rays,
        input_fnum=fnum,
        target_fnum=fnum,
        seed=42,
        point_source=point_source,
    )


def _mean_rms(metrics, wavelengths):
    vals = [metrics[f"rms_spot_{wl}_um"] for wl in wavelengths]
    return sum(vals) / len(vals)


def _print_rms(label, metrics, wavelengths, reference=None):
    print(f"\n{label}:")
    for wl in wavelengths:
        our = metrics[f"rms_spot_{wl}_um"]
        if reference:
            ref = reference[wl]
            pct = abs(our - ref) / ref * 100
            print(f"  {wl} nm: {our:7.1f} µm  (ref: {ref} µm, diff: {pct:.0f}%)")
        else:
            print(f"  {wl} nm: {our:7.1f} µm")
    print(f"  mean: {_mean_rms(metrics, wavelengths):.1f} µm")


def standard_ct():
    """Standard CT forward trace at Xia's exact parameters vs Fig 5a."""
    data = read_toml(Path(XIA_BASELINE))
    genome = CzernyGenome(**data)
    parts = _parts_standard()
    wavelengths = sorted(XIA_STANDARD_RMS_UM.keys())
    metrics = _trace(genome, parts, wavelengths)
    _print_rms("Standard CT (spherical, 3.5° tilt)", metrics, wavelengths,
               XIA_STANDARD_RMS_UM)


def corrected_ct():
    """Aberration-corrected CT at Xia's exact parameters vs Fig 5b.

    Shows the asymmetry at the band edges before NM refinement.
    """
    data = read_toml(Path(XIA_BASELINE))
    genome = CzernyGenome(**data)
    parts = _parts_corrected()
    wavelengths = sorted(XIA_CORRECTED_RMS_UM.keys())
    metrics = _trace(genome, parts, wavelengths, point_source=True)
    _print_rms("Aberration-corrected CT (cylindrical, 3.5° tilt)",
               metrics, wavelengths, XIA_CORRECTED_RMS_UM)


def nm_refinement():
    """NM refinement of cylindrical CT over (L_f1, L_m1, L_b, θ_D).

    Holds α, β, θ_M1, θ_M2, θ_F1, L_a fixed at Xia's values.
    """
    data = read_toml(Path(XIA_BASELINE))
    parts = _parts_corrected()
    wavelengths = sorted(XIA_CORRECTED_RMS_UM.keys())

    # NM with 5k rays for speed.
    x0 = [data["L_f1_mm"], data["L_m1_mm"], data["L_b_mm"],
          data["theta_d_deg"]]

    n_eval = [0]
    def objective(x):
        n_eval[0] += 1
        gd = dict(data)
        gd["L_f1_mm"] = x[0]
        gd["L_m1_mm"] = x[1]
        gd["L_b_mm"] = x[2]
        gd["theta_d_deg"] = x[3]
        genome = CzernyGenome(**gd)
        metrics = _trace(genome, parts, wavelengths,
                         n_rays=5_000, point_source=True)
        return _mean_rms(metrics, wavelengths)

    print(f"\nRunning NM over (L_f1, L_m1, L_b, θ_D) with 5k rays ...")
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 0.01, "fatol": 0.01,
                               "maxiter": 200, "adaptive": True})

    L_f1, L_m1, L_b, theta_d = result.x
    l_SM1 = data["L_a_mm"] - L_f1

    print(f"\nNM result ({result.nfev} evaluations):")
    print(f"  l_SM1  = {l_SM1:.3f} mm  (Xia: 10.526 mm)   [L_a - L_f1]")
    print(f"  l_M1M2 = {L_f1:.3f} mm  (Xia: 100.774 mm)  [L_f1]")
    print(f"  L_m1   = {L_m1:.3f} mm                      [grating to M1, no Xia equiv.]")
    print(f"  l_M3D  = {L_b:.3f} mm  (Xia: 108.751 mm)  [L_b]")
    print(f"  θ_D    = {theta_d:.2f}°  (Xia: 3.50°)")
    print(f"  mean RMS = {result.fun:.1f} µm (5k rays)")

    # Verify at 20k rays.
    gd = dict(data)
    gd["L_f1_mm"], gd["L_m1_mm"] = L_f1, L_m1
    gd["L_b_mm"], gd["theta_d_deg"] = L_b, theta_d
    genome = CzernyGenome(**gd)
    metrics_after = _trace(genome, parts, wavelengths, point_source=True)
    _print_rms("After NM (20k verification)", metrics_after, wavelengths,
               XIA_CORRECTED_RMS_UM)


def main():
    standard_ct()
    corrected_ct()
    nm_refinement()


if __name__ == "__main__":
    main()
