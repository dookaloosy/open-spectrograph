"""CFL wavelength calibration: line fitting and quadratic dispersion.

The fitting engine behind `scripts/cfl_calibrate.py` and (from
milestone 4) `controller calibrate --live`.

Six index lines are fit with saturation-aware Gaussian models:
  404.656, 435.833, 631.3   isolated singles, free baseline
  611.6                     core-flank fit, fixed baseline (dodges the
                            Eu blend on its blue side), clip-masked
  542.4 + 546.074           joint two-Gaussian, fixed baseline, masked

The tracked `data/cfl_lines.toml` holds only the reference
wavelengths and the nominal design dispersion; calibration never
writes to it.  Pixel positions are found at run time by the pattern
locator (`locate_lines`), which matches the expected line pattern —
wavelengths mapped through a candidate dispersion within ±25% of
nominal, at any shift — against peaks detected across the full sensor.
Calibration therefore carries no state between runs and tolerates
arbitrary drift (realignment, mount swaps) as long as the index lines
are on the detector.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from controller.protocol import SAT_ADU

REPO = Path(__file__).resolve().parent.parent
SEEDS_TOML = REPO / "data" / "cfl_lines.toml"

SINGLE_HALF_PX = 14           # fit window half-width, isolated lines
CORE_HALF_PX = 15             # 611.6 core window half-width
BLEND_PAD_PX = 22             # window padding around the 542/546 pair

BLEND_PAIR = (542.4, 546.074)
CORE_LINE = 611.6

# Pattern locator: finds the line positions before the precision fits.
LOCATE_DISPERSION_TOL = 0.25      # allowed deviation from nominal dispersion
LOCATE_SINGLE_TOL_PX = 15         # feature match tolerance, single lines
LOCATE_BLEND_TOL_PX = 25          # tolerance for the 542/546 blend blob
LOCATE_MIN_MATCHES = 4            # of the 5 pattern features
LOCATE_MIN_PROMINENCE = 250.0     # peak detection floor (ADU)
LOCATE_BLEND_TOP_K = 4            # blend must land on a top-K peak


class FitWindowError(RuntimeError):
    """A fit window kept too few usable samples (clipped or starved)."""


@dataclass
class LineFit:
    wavelength_nm: float
    center_px: float
    center_err_px: float
    sigma_px: float
    seed_px: float = float("nan")   # located position that seeded the fit


@dataclass
class Calibration:
    lines: list                  # LineFit, ascending wavelength
    a0: float
    a1: float
    a2: float
    residuals_nm: np.ndarray     # same order as `lines`
    rms_nm: float
    baseline_adu: float
    n_clipped_px: int

    def wavelength(self, px):
        return self.a0 + self.a1 * px + self.a2 * np.asarray(px) ** 2

    def store_command(self) -> str:
        return (f"store coefficients {self.a0:.6f} {self.a1:.8f} "
                f"{self.a2:.6e} 0")


def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_lines(path: Path = SEEDS_TOML) -> tuple[list, float]:
    """Load (wavelengths_nm, dispersion_nm_per_px) from the lines TOML."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    lines = [float(lam) for lam in data["wavelengths_nm"]]
    dispersion = float(data["dispersion_nm_per_px"])
    if len(lines) < 3:
        raise RuntimeError(f"{path}: need at least 3 index lines, "
                           f"got {len(lines)}")
    if dispersion == 0.0:
        raise RuntimeError(f"{path}: dispersion_nm_per_px must be nonzero")
    return lines, dispersion


def locate_lines(px, cnt, lines, dispersion_nm_per_px) -> list:
    """Find the pixel position of each index line from scratch.

    Detects candidate peaks (clipped plateaus reduce to their
    midpoints), then searches every anchor assignment of two single
    lines to two peaks for the affine map px = A + B*lambda whose
    dispersion is within LOCATE_DISPERSION_TOL of nominal and which
    matches the most pattern features: the single lines plus the
    542/546 blend treated as one blob.  Matched singles snap to their
    peak; everything else maps through the affine fit.  Returns
    [(center_px, wavelength_nm), ...] seeds for :func:`fit_lines`.

    Raises RuntimeError when fewer than LOCATE_MIN_MATCHES features
    can be matched, or when a line would map off the detector.
    """
    cnt = np.asarray(cnt, dtype=float)
    base = float(np.median(cnt))
    capped = np.minimum(cnt, float(SAT_ADU))
    prom = max(LOCATE_MIN_PROMINENCE, 0.02 * (capped.max() - base))
    idx, _ = find_peaks(capped, prominence=prom, distance=10)
    if len(idx) < 3:
        raise RuntimeError(
            f"pattern locator: only {len(idx)} peaks detected "
            f"(prominence floor {prom:.0f} ADU) — is the lamp on?")
    idx = idx.astype(float)

    b_nom = 1.0 / dispersion_nm_per_px          # px per nm, signed
    singles = sorted(lam for lam in lines if lam not in BLEND_PAIR)
    blend_mid = 0.5 * (BLEND_PAIR[0] + BLEND_PAIR[1])
    features = ([(lam, LOCATE_SINGLE_TOL_PX) for lam in singles]
                + [(blend_mid, LOCATE_BLEND_TOL_PX)])
    # the 542/546 blend is the brightest feature of every CFL — a
    # candidate map that puts it on a dim peak is a false lock
    heights = capped[idx.astype(int)]
    top_k = set(np.argsort(heights)[::-1][:LOCATE_BLEND_TOP_K])

    best = None                          # (matched, -total_resid, B, A)
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            li, lj = singles[i], singles[j]
            for pi in idx:
                for pj in idx:
                    if pi == pj:
                        continue
                    b = (pj - pi) / (lj - li)
                    if not (1.0 - LOCATE_DISPERSION_TOL
                            <= b / b_nom
                            <= 1.0 + LOCATE_DISPERSION_TOL):
                        continue
                    a = pi - b * li
                    matched, resid = 0, 0.0
                    blend_ok = False
                    for lam, tol in features:
                        d = np.abs(idx - (a + b * lam))
                        k = int(np.argmin(d))
                        if d[k] <= tol:
                            matched += 1
                            resid += float(d[k])
                            if lam == blend_mid:
                                blend_ok = k in top_k
                    if not blend_ok:
                        continue
                    key = (matched, -resid)
                    if best is None or key > best[:2]:
                        best = (matched, -resid, b, a)

    if best is None or best[0] < LOCATE_MIN_MATCHES:
        found = 0 if best is None else best[0]
        raise RuntimeError(
            f"pattern locator: matched only {found}/{len(features)} "
            f"CFL features among {len(idx)} peaks — spectrum does not "
            f"look like the expected lamp")

    b, a = best[2], best[3]
    located = []
    for lam in lines:
        pred = a + b * lam
        if lam not in BLEND_PAIR:
            d = np.abs(idx - pred)
            k = int(np.argmin(d))
            if d[k] <= LOCATE_SINGLE_TOL_PX:
                pred = float(idx[k])
        if not 0.0 <= pred <= float(px[-1]):
            raise RuntimeError(
                f"pattern locator: line {lam} nm maps to pixel "
                f"{pred:.0f}, off the detector — spectrum has shifted "
                f"too far to calibrate on all index lines")
        located.append((float(pred), lam))
    return located


def fit_lines(px, cnt, seeds) -> tuple[dict, float]:
    """Fit each index line.  Returns ({lambda: LineFit}, baseline)."""
    base = float(np.median(cnt))
    results = {}

    def window(lo, hi, mask_sat, lam):
        m = (px >= lo) & (px <= hi)
        x, y = px[m], cnt[m]
        if mask_sat:
            keep = y < SAT_ADU
            x, y = x[keep], y[keep]
        if len(x) < 8:
            raise FitWindowError(
                f"line {lam} nm: only {len(x)} usable samples in window "
                f"[{lo:.0f}, {hi:.0f}] after clip masking — frame is "
                f"saturated or seeds are stale")
        return x, y

    seed_of = dict((lam, p0) for p0, lam in seeds)

    # Isolated singles: free baseline.
    for lam, p0 in seed_of.items():
        if lam in BLEND_PAIR or lam == CORE_LINE:
            continue
        x, y = window(p0 - SINGLE_HALF_PX, p0 + SINGLE_HALF_PX,
                      mask_sat=True, lam=lam)
        off0 = float(np.percentile(y, 10))
        popt, pcov = curve_fit(
            lambda x, A, mu, s, off: gauss(x, A, mu, s) + off,
            x, y, p0=[y.max() - off0, p0, 2.0, off0], maxfev=40000)
        results[lam] = LineFit(lam, popt[1], float(np.sqrt(pcov[1, 1])),
                               abs(popt[2]), seed_px=p0)

    # 611.6: core flanks only, fixed baseline, clipped samples masked.
    p0 = seed_of[CORE_LINE]
    x, y = window(p0 - CORE_HALF_PX, p0 + CORE_HALF_PX,
                  mask_sat=True, lam=CORE_LINE)
    popt, pcov = curve_fit(
        lambda x, A, mu, s: gauss(x, A, mu, s) + base,
        x, y, p0=[y.max() - base, p0, 3.0], maxfev=40000)
    results[CORE_LINE] = LineFit(CORE_LINE, popt[1],
                                 float(np.sqrt(pcov[1, 1])), abs(popt[2]),
                                 seed_px=p0)

    # 542.4 / 546.074 blend: joint two-Gaussian, fixed baseline, masked.
    lam_a, lam_b = BLEND_PAIR                      # a = 542.4, b = 546.074
    pa, pb = seed_of[lam_a], seed_of[lam_b]        # pa > pb (blue = high px)
    x, y = window(min(pa, pb) - BLEND_PAD_PX, max(pa, pb) + BLEND_PAD_PX,
                  mask_sat=True, lam=lam_b)
    popt, pcov = curve_fit(
        lambda x, A1, m1, s1, A2, m2, s2:
            gauss(x, A1, m1, s1) + gauss(x, A2, m2, s2) + base,
        x, y, p0=[2e5, pb, 4.0, 4e4, pa, 8.0], maxfev=200000)
    results[lam_b] = LineFit(lam_b, popt[1], float(np.sqrt(pcov[1, 1])),
                             abs(popt[2]), seed_px=pb)
    results[lam_a] = LineFit(lam_a, popt[4], float(np.sqrt(pcov[4, 4])),
                             abs(popt[5]), seed_px=pa)

    return results, base


def calibrate(px, cnt, lines, dispersion_nm_per_px) -> Calibration:
    """Locate the index lines, fit them, and fit the quadratic dispersion.

    ``lines`` is the list of reference wavelengths (nm) and
    ``dispersion_nm_per_px`` the nominal design dispersion, both from
    :func:`load_lines`.  The pattern locator finds the pixel positions
    from scratch each run, so no state is carried between
    calibrations.
    """
    seeds = locate_lines(px, cnt, lines, dispersion_nm_per_px)
    results, base = fit_lines(px, cnt, seeds)
    lams = np.array(sorted(results))
    cens = np.array([results[l].center_px for l in lams])
    coef = np.polyfit(cens, lams, 2)
    a2, a1, a0 = (float(c) for c in coef)
    resid = lams - np.polyval(coef, cens)
    return Calibration(
        lines=[results[l] for l in lams],
        a0=a0, a1=a1, a2=a2,
        residuals_nm=resid,
        rms_nm=float(np.sqrt(np.mean(resid ** 2))),
        baseline_adu=base,
        n_clipped_px=int((np.asarray(cnt) > SAT_ADU).sum()),
    )


def report(cal: Calibration) -> str:
    """Human-readable fit summary."""
    out = [f"{'lambda':>8} {'located':>8} {'center_px':>10} {'+-':>6} "
           f"{'sigma':>6} {'resid_nm':>9}"]
    for fit, r in zip(cal.lines, cal.residuals_nm):
        out.append(f"{fit.wavelength_nm:8.3f} {fit.seed_px:8.1f} "
                   f"{fit.center_px:10.3f} {fit.center_err_px:6.3f} "
                   f"{fit.sigma_px:6.2f} {r:+9.4f}")
    out += ["",
            "lambda(p) = a0 + a1*p + a2*p^2",
            f"a0 = {cal.a0:.6f}",
            f"a1 = {cal.a1:.8f}",
            f"a2 = {cal.a2:.6e}",
            f"RMS residual = {cal.rms_nm:.4f} nm"]
    return "\n".join(out)
