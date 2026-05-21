"""End-to-end validation against Xia 2017 published results.

Three tests:
  1. test_standard_ct — Xia's exact parameters (spherical optics,
     3.5° tilt) vs digitized Fig 5a.
  2. test_aberration_corrected_ct — Same parameters with cylindrical
     BOM (point source) vs digitized Fig 5b.
  3. test_expand_genome — expand_genome() derives the same layout as
     the explicit baseline for both spherical and cylindrical BOMs.

Run: `python tests/test_xia2017.py`
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XIA_STANDARD_BOM = DATA_DIR / "czerny_bom_xia2017_standard.toml"
XIA_CORRECTED_BOM = DATA_DIR / "czerny_bom_xia2017_corrected.toml"
XIA_BASELINE = DATA_DIR / "czerny_baseline_xia2017.toml"

# Xia 2017 Fig 5a RMS spot radius, digitized from the published curve.
XIA_STANDARD_RMS_UM = {
    350: 101.5,
    450: 132.6,
    550: 180.8,
    650: 248.9,
    750: 336.8,
}

# Xia 2017 Fig 5b RMS spot radius, digitized from the published curve.
XIA_CORRECTED_RMS_UM = {
    350: 16.7,
    450: 8.2,
    550: 4.4,
    650: 9.0,
    750: 16.9,
}


def _xia_parts():
    from designs.czerny_bom import load_parts
    return load_parts(
        m1_part="XIA-M1", m2_part="XIA-M2", grating_part="GR25-0605",
        f1_part="XIA-F1",
        bom_path=XIA_STANDARD_BOM, optic_size_mm=25.4,
    )


def _xia_corrected_parts():
    from designs.czerny_bom import load_parts
    return load_parts(
        m1_part="XIA-M1-CYL", m2_part="XIA-M2", grating_part="GR25-0605",
        f1_part="XIA-F1-CYL",
        bom_path=XIA_CORRECTED_BOM, optic_size_mm=25.4,
    )


def _run_trace(genome, parts, wavelengths, *, point_source=False):
    """Forward trace using the f/# encoded in the BOM (M1 diameter)."""
    from designs.czerny_bom import build_optics_only_scene
    from optics.forward_trace import forward_trace_metrics

    fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    scene = build_optics_only_scene(genome, parts)
    return forward_trace_metrics(
        scene, genome, parts,
        wavelengths_nm=tuple(float(l) for l in wavelengths),
        design_wavelength_nm=550.0,
        n_rays=20_000,
        input_fnum=fnum,
        target_fnum=fnum,
        seed=42,
        point_source=point_source,
    )


def _check_rms(label, metrics, tolerance, reference):
    wavelengths = sorted(reference.keys())
    print(f"{label} — RMS spot (tolerance {tolerance:.0%}):")
    all_ok = True
    for wl in wavelengths:
        key = f"rms_spot_{wl}_um"
        our = metrics.get(key, float("nan"))
        expected = reference[wl]
        pct = abs(our - expected) / expected if expected > 0 else float("inf")
        status = "OK" if pct < tolerance else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {wl} nm: {our:7.1f} µm  (ref: {expected} µm, diff: {pct:.1%}) {status}")
    if all_ok:
        print("  PASS\n")
    else:
        print("  FAIL\n")
    return all_ok


def _expand_xia_genome(parts, *, theta_f1_deg=None):
    """Run expand_genome() with Xia's inputs, return the genome dict."""
    from designs.czerny import CzernyGeometry

    geom = CzernyGeometry()
    genome = {
        "Dv_deg": 30.0,
        "theta_m1_deg": 6.4,
        "L_m1_mm": 110.0,
        "L_m2_mm": 110.0,
    }
    if theta_f1_deg is not None:
        genome["theta_f1_deg"] = theta_f1_deg
    geom.expand_genome(genome, parts, 550.0, band_nm=(350.0, 750.0))
    return genome


def _genome_from_dict(genome_dict):
    from designs.czerny_base import CzernyGenome
    return CzernyGenome(**{
        k: v for k, v in genome_dict.items()
        if k in CzernyGenome.__dataclass_fields__
    })


# ── Test 1: Standard CT ─────────────────────────────────────────────

def test_standard_ct():
    """Xia's exact parameters (spherical, 3.5° tilt) vs Fig 5a."""
    from optics._config import read_toml

    data = read_toml(Path(XIA_BASELINE))
    genome = _genome_from_dict(data)
    parts = _xia_parts()
    wavelengths = sorted(XIA_STANDARD_RMS_UM.keys())
    metrics = _run_trace(genome, parts, wavelengths)
    ok = _check_rms("Standard CT (spherical)", metrics,
                     tolerance=0.05, reference=XIA_STANDARD_RMS_UM)
    assert ok, "Standard CT exceeds 5% tolerance vs Xia Fig 5a"


# ── Test 2: Aberration-corrected CT ─────────────────────────────────

def test_aberration_corrected_ct():
    """Xia's exact parameters (cylindrical, 3.5° tilt, point source) vs Fig 5b."""
    from optics._config import read_toml

    data = read_toml(Path(XIA_BASELINE))
    genome = _genome_from_dict(data)
    parts = _xia_corrected_parts()
    wavelengths = sorted(XIA_CORRECTED_RMS_UM.keys())
    metrics = _run_trace(genome, parts, wavelengths, point_source=True)
    ok = _check_rms("Aberration-corrected CT (cylindrical, point source)",
                     metrics, tolerance=0.70, reference=XIA_CORRECTED_RMS_UM)
    assert ok, "Aberration-corrected CT exceeds 70% tolerance vs Xia Fig 5b"


# ── Test 3: expand_genome ───────────────────────────────────────────

def test_expand_genome():
    """expand_genome() derives the same layout as the explicit baseline.

    Checks both spherical and cylindrical BOMs.
    """
    from optics._config import read_toml
    baseline = read_toml(Path(XIA_BASELINE))

    # Spherical BOM.
    parts_sph = _xia_parts()
    genome_sph = _expand_xia_genome(parts_sph)

    print("expand_genome (spherical) vs Xia Table 2:")
    checks_sph = [
        ("θ_M2", genome_sph["theta_m2_deg"], baseline["theta_m2_deg"], 0.15),
        ("α", genome_sph["alpha_deg"], baseline["alpha_deg"], 0.1),
        ("β", genome_sph["beta_deg"], baseline["beta_deg"], 0.1),
        ("L_a", genome_sph["L_a_mm"], baseline["L_a_mm"], 0.5),
        ("L_b", genome_sph["L_b_mm"], baseline["L_b_mm"], 0.5),
    ]
    all_ok = True
    for name, got, expected, tol in checks_sph:
        diff = abs(got - expected)
        status = "OK" if diff < tol else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:6s} = {got:8.3f}  (baseline: {expected}, diff: {diff:.3f}) {status}")
    if all_ok:
        print("  PASS\n")
    else:
        print("  FAIL\n")
    assert all_ok, "expand_genome (spherical) diverges from baseline"

    # Cylindrical BOM — also derives L_f1 via astigmatism-free condition.
    parts_cyl = _xia_corrected_parts()
    genome_cyl = _expand_xia_genome(parts_cyl, theta_f1_deg=18.0)

    print("expand_genome (cylindrical) vs Xia Table 2:")
    checks_cyl = list(checks_sph)
    if "L_f1_mm" in genome_cyl:
        checks_cyl.append(
            ("L_f1", genome_cyl["L_f1_mm"], baseline["L_f1_mm"], 0.5))
    else:
        print("  WARNING: expand_genome did not derive L_f1_mm")
        all_ok = False

    for name, got, expected, tol in checks_cyl:
        diff = abs(got - expected)
        status = "OK" if diff < tol else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:6s} = {got:8.3f}  (baseline: {expected}, diff: {diff:.3f}) {status}")
    if all_ok:
        print("  PASS\n")
    else:
        print("  FAIL\n")
    assert all_ok, "expand_genome (cylindrical) diverges from baseline"


# ── Test 4: Tilt sweep ──────────────────────────────────────────────

def test_tilt_sweep():
    """Sweep θ_D on corrected CT, verify optimum near 4.3°."""
    from optics._config import read_toml

    data = read_toml(Path(XIA_BASELINE))
    parts = _xia_corrected_parts()
    wavelengths = sorted(XIA_CORRECTED_RMS_UM.keys())

    tilt_grid = [2.0, 3.0, 3.5, 4.0, 4.3, 4.5, 5.0, 6.0]
    best_tilt = None
    best_mean_rms = float("inf")
    best_var = float("inf")

    print("Tilt sweep (corrected CT, point source):")
    for tilt in tilt_grid:
        gd = dict(data)
        gd["theta_d_deg"] = tilt
        genome = _genome_from_dict(gd)
        metrics = _run_trace(genome, parts, wavelengths, point_source=True)
        rms_per_wl = {wl: metrics[f"rms_spot_{wl}_um"] for wl in wavelengths}
        mean_rms = sum(rms_per_wl.values()) / len(rms_per_wl)
        import numpy as _np
        rms_arr = _np.array([rms_per_wl[wl] for wl in wavelengths])
        variance = _np.mean((rms_arr - mean_rms)**2)
        print(f"  θ_D={tilt:+5.1f}°: mean={mean_rms:5.1f}  var={variance:6.1f} µm²  "
              f"[{', '.join(f'{wl}={rms_per_wl[wl]:.1f}' for wl in wavelengths)}]")
        if variance < best_var:
            best_var = variance
            best_mean_rms = mean_rms
            best_tilt = tilt

    print(f"  Best tilt = {best_tilt:+.1f}° (mean RMS = {best_mean_rms:.1f} µm, var = {best_var:.1f} µm²)")
    assert abs(best_tilt - 4.3) <= 1.0, \
        f"Best tilt {best_tilt}° too far from expected 4.3°"
    print("  PASS\n")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_standard_ct()
    test_aberration_corrected_ct()
    test_expand_genome()
    test_tilt_sweep()
    print("All Xia 2017 tests passed.")
