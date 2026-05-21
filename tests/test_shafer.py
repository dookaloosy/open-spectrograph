"""Unit tests for the Shafer coma-cancellation condition.

Validates shafer_theta2() against Xia 2017 published parameters
and checks edge cases.

Run: `python tests/test_shafer.py`
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from designs.czerny_base import shafer_theta2, grating_angles


def test_xia2017():
    """Xia 2017 Table 1: φ₂=6.4°, φ₃=8.1°, R₂=224mm, R₃=220mm."""
    alpha, beta = grating_angles(30.0, 550.0, 1667.0)
    theta2 = shafer_theta2(
        theta_m1_deg=6.4,
        R1_mm=224.0,
        R2_mm=220.0,
        alpha_deg=alpha,
        beta_deg=beta,
    )
    print(f"Xia 2017: theta1=6.4°, alpha={alpha:.2f}°, beta={beta:.2f}°")
    print(f"  theta2 = {theta2:.4f}° (expected ≈ 8.1°)")
    assert abs(theta2 - 8.1) < 0.15, f"theta2={theta2}, expected ~8.1"
    print("  PASS")


def test_equal_R():
    """Equal-R mirrors: R terms cancel, pure angle constraint."""
    alpha, beta = grating_angles(30.0, 550.0, 1667.0)
    theta2_equal = shafer_theta2(6.4, 200.0, 200.0, alpha, beta)
    theta2_scaled = shafer_theta2(6.4, 400.0, 400.0, alpha, beta)
    print(f"Equal-R: R=200 → theta2={theta2_equal:.4f}°, "
          f"R=400 → theta2={theta2_scaled:.4f}°")
    assert abs(theta2_equal - theta2_scaled) < 1e-6, \
        "Equal-R results should be independent of R magnitude"
    print("  PASS")


def test_symmetric_at_littrow():
    """At Littrow (α=β) with equal R, θ₂ should equal θ₁."""
    theta2 = shafer_theta2(10.0, 200.0, 200.0, 15.0, 15.0)
    print(f"Littrow + equal-R: theta1=10°, theta2={theta2:.4f}° (expected 10°)")
    assert abs(theta2 - 10.0) < 1e-6, f"theta2={theta2}, expected 10.0"
    print("  PASS")


def test_near_zero_theta1():
    """θ₁ ≈ 0 should return θ₂ = 0."""
    theta2 = shafer_theta2(0.001, 200.0, 220.0, 5.0, 25.0)
    print(f"Near-zero: theta1=0.001°, theta2={theta2:.6f}° (expected 0.0)")
    assert theta2 == 0.0
    print("  PASS")


def test_asymmetric_R():
    """Asymmetric R: larger R₂ should give larger θ₂ for same θ₁."""
    alpha, beta = grating_angles(30.0, 550.0, 1667.0)
    theta2_equal = shafer_theta2(10.0, 200.0, 200.0, alpha, beta)
    theta2_larger_R2 = shafer_theta2(10.0, 200.0, 300.0, alpha, beta)
    print(f"Asymmetric R: R2=200 → theta2={theta2_equal:.4f}°, "
          f"R2=300 → theta2={theta2_larger_R2:.4f}°")
    assert theta2_larger_R2 > theta2_equal, \
        "Larger R2 should require larger theta2 for coma balance"
    print("  PASS")


def test_infeasible_raises():
    """Extreme parameters with no solution should raise ValueError."""
    try:
        shafer_theta2(89.0, 10.0, 1000.0, 5.0, 25.0)
        print("Infeasible: no exception raised — FAIL")
        assert False, "Expected ValueError"
    except ValueError as e:
        print(f"Infeasible: correctly raised ValueError: {e}")
        print("  PASS")


if __name__ == "__main__":
    test_xia2017()
    test_equal_R()
    test_symmetric_at_littrow()
    test_near_zero_theta1()
    test_asymmetric_R()
    test_infeasible_raises()
    print("\nAll Shafer tests passed.")
