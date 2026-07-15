"""controller package: capture parsing and CFL calibration.

Fixtures are real CFL captures from the assembled v0.5.1 instrument
(2026-07-13): a 20 ms file whose first frame is stale (fully clipped)
and a 25 ms file with two stale + two live frames.
"""

from pathlib import Path

import numpy as np
import pytest

from controller import calibrate as cal
from controller import protocol

DATA = Path(__file__).parent / "data"
CAP_20MS = DATA / "cfl_20ms_2frames.tcd1304"
CAP_25MS = DATA / "cfl_25ms_4frames.tcd1304"


def test_parse_capture_frames_and_header():
    cap = protocol.parse_capture(CAP_20MS)
    assert len(cap.frames) == 2
    assert all(len(f) == 3694 for f in cap.frames)
    assert cap.header["sensor"] == '"TCD1304"'
    assert cap.exposure_s == pytest.approx(0.02, abs=1e-3)


def test_stale_frame_guard():
    cap = protocol.parse_capture(CAP_20MS)
    good, n_dropped = protocol.live_frames(cap)
    assert n_dropped == 1
    assert len(good) == 1
    # the surviving frame is a real spectrum: dark baseline, bright lines
    assert np.median(good[0]) < 2000
    assert good[0].max() > 60000


def test_calibrate_against_known_good():
    """Constants must reproduce the 2026-07-13 bench calibration."""
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    result = cal.calibrate(px, cnt, *cal.load_lines())
    assert result.rms_nm < 0.25
    assert result.a0 == pytest.approx(792.504222, abs=0.5)
    assert result.a1 == pytest.approx(-0.12010671, abs=1e-3)
    assert result.a2 == pytest.approx(-6.268825e-07, abs=5e-7)
    # six lines fit, ascending wavelength
    assert [f.wavelength_nm for f in result.lines] == [
        404.656, 435.833, 542.4, 546.074, 611.6, 631.3]
    assert "store coefficients 792." in result.store_command()
    assert result.store_command().endswith(" 0")


def test_lines_toml_is_static():
    """Calibration must never write to data/cfl_lines.toml."""
    before = cal.SEEDS_TOML.read_bytes()
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    result = cal.calibrate(px, cnt, *cal.load_lines())
    assert result.rms_nm < 0.25
    assert cal.SEEDS_TOML.read_bytes() == before
    assert not hasattr(cal, "save_seeds")


def test_all_clipped_raises():
    cap = protocol.parse_capture(CAP_20MS)
    cap.frames = [cap.frames[0]]          # keep only the stale frame
    with pytest.raises(RuntimeError, match="mostly clipped"):
        protocol.live_frames(cap)


# ── pattern locator: wavelength-only, arbitrary shifts ──────────────

def _shifted(cnt, px_shift):
    """Roll the spectrum, padding the wrapped end with the baseline."""
    out = np.roll(cnt, px_shift)
    base = float(np.median(cnt))
    if px_shift > 0:
        out[:px_shift] = base
    elif px_shift < 0:
        out[px_shift:] = base
    return out


@pytest.mark.parametrize("shift_px", [-300, 400])
def test_locator_tolerates_large_shifts(shift_px):
    """Tens of nm of drift (~0.12 nm/px) must still calibrate."""
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    result = cal.calibrate(px, _shifted(cnt, shift_px), *cal.load_lines())
    assert result.rms_nm < 0.25
    # dispersion is preserved; intercept absorbs the shift
    assert result.a1 == pytest.approx(-0.1201, abs=2e-3)
    hg = next(f for f in result.lines if f.wavelength_nm == 404.656)
    assert hg.center_px == pytest.approx(3177.3 + shift_px, abs=1.0)


def test_locator_tolerates_stretch():
    """A few percent of dispersion change on top of a shift."""
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    stretched = np.interp(px, px * 1.04 - 60.0, cnt)
    result = cal.calibrate(px, stretched, *cal.load_lines())
    assert result.rms_nm < 0.3


def test_locator_rejects_flat_spectrum():
    px = np.arange(3694, dtype=float)
    flat = np.full(3694, 1300.0)
    with pytest.raises(RuntimeError, match="pattern locator"):
        cal.locate_lines(px, flat, *cal.load_lines())


def test_locator_rejects_off_detector_shift():
    """+600 px pushes 404.7 nm off the sensor; the rest still match."""
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    with pytest.raises(RuntimeError, match="off the detector"):
        cal.locate_lines(px, _shifted(cnt, 600), *cal.load_lines())


def test_locator_rejects_wrong_dispersion_sign():
    """A flipped dispersion sign (mirrored orientation) must not lock."""
    cap = protocol.parse_capture(CAP_25MS)
    px, cnt, _, _ = protocol.mean_spectrum(cap)
    lines, dispersion = cal.load_lines()
    with pytest.raises(RuntimeError, match="pattern locator"):
        cal.locate_lines(px, cnt, lines, -dispersion)
