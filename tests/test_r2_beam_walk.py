"""Unit tests for _r2_from_beam_walk.

Validates the beam walk formula against manual calculations using the
grating equation: mλ = d(sinα + sinβ), m = −1, so
sinβ_edge = mλ_edge/d − sinα.

Run: `python tests/test_r2_beam_walk.py`
"""

import sys
from math import asin, cos, degrees, radians, sin, tan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from designs.czerny_base import (
    _r2_from_beam_walk,
    grating_angles,
)
from designs.czerny_bom import load_parts

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TL_STANDARD_BOM = DATA_DIR / "czerny_bom_tl_standard.toml"


def _manual_r2(alpha_deg, beta_deg, groove_density, d_m1, d_m2, band_nm):
    """Reference implementation of beam walk r2."""
    m = -1
    d_nm = 1e6 / groove_density
    sin_a = sin(radians(alpha_deg))
    beta_c = radians(beta_deg)
    max_walk = 0.0
    for lam in band_nm:
        arg = m * lam / d_nm - sin_a
        assert abs(arg) <= 1.0, f"{lam}nm evanescent: sin(β)={arg}"
        walk = abs(tan(asin(arg) - beta_c))
        if walk > max_walk:
            max_walk = walk
    margin = (d_m2 - d_m1) / 2.0
    return margin / max_walk


def test_tl_standard_600gpmm():
    """TL standard BOM (600 g/mm, dev=42°): positive α, valid r2."""
    parts = load_parts(m1_part="CM254-100-G01", m2_part="CM508-100-G01",
                       grating_part="GR25-0605",
                       f1_part="PF05-03-G01",
                       bom_path=TL_STANDARD_BOM, optic_size_mm=25.4)
    alpha, beta = grating_angles(42.0, 550.0,
                                             1e6 / parts.grating_groove_density_per_mm)
    assert alpha > 0, f"α should be positive above Littrow: {alpha}"

    kw = {"alpha_deg": alpha, "beta_deg": beta}
    band = (350.0, 750.0)
    r2 = _r2_from_beam_walk(kw, parts, band_nm=band)
    r2_manual = _manual_r2(alpha, beta, parts.grating_groove_density_per_mm,
                           parts.m1_diameter_mm, parts.m2_diameter_mm, band)

    print(f"TL standard (600 g/mm, dev=42°): α={alpha:.2f}°, β={beta:.2f}°")
    print(f"  r2 function = {r2:.3f} mm")
    print(f"  r2 manual   = {r2_manual:.3f} mm")
    assert abs(r2 - r2_manual) < 0.01, f"r2 mismatch: {r2} vs {r2_manual}"
    assert r2 > 0, "r2 should be positive"
    print("  PASS\n")


def test_v1_below_littrow():
    """1200 g/mm at dev=24°: α < 0, but 750nm should NOT be evanescent."""
    d_nm = 1e6 / 1200.0
    alpha, beta = grating_angles(24.0, 550.0, d_nm)
    assert alpha < 0, f"α should be negative below Littrow: {alpha}"

    # 750nm at this geometry: sin(β_edge) = 750/d + sin(α)
    sin_beta_750 = 750.0 / d_nm + sin(radians(alpha))
    print(f"1200 g/mm dev=24°: α={alpha:.2f}°, sin(β_750)={sin_beta_750:.4f}")
    assert abs(sin_beta_750) <= 1.0, \
        f"750nm should not be evanescent: sin(β)={sin_beta_750}"
    print("  750nm reachable: PASS\n")


def test_equal_diameter_rejected():
    """M2 = M1 diameter → r2 should be -1 (infeasible)."""
    parts = load_parts(m1_part="CM254-100-G01", m2_part="CM254-100-G01",
                       grating_part="GR25-0605",
                       f1_part="PF05-03-G01",
                       bom_path=TL_STANDARD_BOM, optic_size_mm=25.4)
    alpha, beta = grating_angles(42.0, 550.0,
                                             1e6 / parts.grating_groove_density_per_mm)
    kw = {"alpha_deg": alpha, "beta_deg": beta}
    r2 = _r2_from_beam_walk(kw, parts, band_nm=(450.0, 650.0))

    print(f"Equal diameter (D_M1=D_M2={parts.m1_diameter_mm}mm): r2={r2}")
    assert r2 == -1.0, f"Expected -1 for equal diameters, got {r2}"
    print("  PASS\n")


def test_band_width_affects_r2():
    """Wider band → shorter r2 (more beam walk to accommodate)."""
    parts = load_parts(m1_part="CM254-100-G01", m2_part="CM508-100-G01",
                       grating_part="GR25-0605",
                       f1_part="PF05-03-G01",
                       bom_path=TL_STANDARD_BOM, optic_size_mm=25.4)
    alpha, beta = grating_angles(24.0, 550.0,
                                             1e6 / parts.grating_groove_density_per_mm)
    kw = {"alpha_deg": alpha, "beta_deg": beta}

    r2_mid = _r2_from_beam_walk(kw, parts, band_nm=(450.0, 650.0))
    r2_wide = _r2_from_beam_walk(kw, parts, band_nm=(350.0, 750.0))
    r2_narrow = _r2_from_beam_walk(kw, parts, band_nm=(500.0, 600.0))

    print(f"Band width effect (600 g/mm):")
    print(f"  r2_mid(450-650)={r2_mid:.1f}, r2_wide(350-750)={r2_wide:.1f}, r2_narrow(500-600)={r2_narrow:.1f}")
    assert r2_wide < r2_mid, "Wider band should give shorter r2"
    assert r2_narrow > r2_mid, "Narrower band should give longer r2"
    print("  PASS\n")


if __name__ == "__main__":
    test_xia_manual()
    test_v1_1200gpmm()
    test_v1_below_littrow()
    test_equal_diameter_rejected()
    test_band_width_affects_r2()
    print("All beam walk tests passed.")
