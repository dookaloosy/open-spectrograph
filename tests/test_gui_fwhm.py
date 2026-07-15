"""Live 404.7 nm FWHM tracker (_fit_fwhm404) — pure-numpy logic, no Qt.

The method never touches Qt widgets, so it is exercised on a stub
carrying only the tracker attributes.
"""

import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from controller.gui.main_window import MainWindow  # noqa: E402
from controller.protocol import SAT_ADU  # noqa: E402

# v0.5.1 device constants: lambda(p) = a0 + a1 p + a2 p^2
COEFF = (793.56592, -0.12134059, -3.5689109e-07, 0.0)


def make_stub(center=3175.0, seed=3175.0, sigma=3.0, coeff=COEFF):
    return types.SimpleNamespace(_fwhm_center=center, _fwhm_seed=seed,
                                 _fwhm_sigma=sigma, _coeff=coeff)


def make_frame(mu=3169.0, sigma=2.0, amp=9000.0, base=1300.0, n=3694):
    x = np.arange(n, dtype=float)
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit(stub, frame):
    return MainWindow._fit_fwhm404(stub, frame)


def test_fwhm_fit_matches_synthetic_line():
    stub = make_stub()
    text = fit(stub, make_frame(mu=3169.0, sigma=2.0))
    nm_px = abs(COEFF[1] + 2 * COEFF[2] * 3169.0)
    expect = 2.3548 * 2.0 * nm_px
    assert f"{expect:.2f} nm" in text
    assert abs(stub._fwhm_center - 3169.0) < 0.1
    assert abs(stub._fwhm_sigma - 2.0) < 0.05


def test_lost_lock_reseeds_from_toml_center():
    stub = make_stub(center=1500.0, seed=3175.0)  # tracking way off
    text = fit(stub, make_frame(mu=3169.0))
    assert "FWHM" in text
    assert abs(stub._fwhm_center - 3169.0) < 0.5


def test_clipped_peak_fits_from_flanks():
    stub = make_stub()
    frame = np.minimum(make_frame(amp=200000.0, sigma=3.0), SAT_ADU + 1000)
    text = fit(stub, frame)
    assert "FWHM" in text
    assert abs(stub._fwhm_sigma - 3.0) < 0.3