"""Recalibrate the wavelength axis from a CFL spectrum capture.

Workflow (repeatable after every recapture into the same workbook):
  1. Read Sheet1 (pixel / counts + header metadata) and Sheet2
     (previous index-peak centers used as fit seeds, and reference
     wavelengths).
  2. Gaussian-fit each of the six index lines:
       - 404.656, 435.833, 631.3    isolated singles, free baseline
       - 611.6                      core-flank fit, fixed baseline
                                    (dodges the Eu blend on its blue
                                    side), clipped samples masked
       - 542.4 + 546.074            joint two-Gaussian fit, fixed
                                    baseline, clipped samples masked
  3. Quadratic fit lambda(p) = a0 + a1*p + a2*p^2.
  4. Write results back into the workbook by editing the sheet XML
     inside the xlsx (openpyxl round-trips drop charts):
       Sheet1 B18:B20  constants (drive the NM formula column)
       Sheet2 A1:A6    fitted centers, A8/B8/A9 constants copy
     and set fullCalcOnLoad so Excel refreshes the formula column.

Usage:
    python3 scripts/cfl_calibrate.py [path/to/cfl_spectrum.xlsx]
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = (REPO / "output" / "export_czerny_baseline_v0_design"
                / "cfl_spectrum.xlsx")

SAT_ADU = 63000.0             # treat counts above this as clipped
SINGLE_HALF_PX = 14           # fit window half-width, isolated lines
CORE_HALF_PX = 15             # 611.6 core window half-width
BLEND_PAD_PX = 22             # window padding around the 542/546 pair

BLEND_PAIR = (542.4, 546.074)
CORE_LINE = 611.6


def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load(path: Path):
    wb = openpyxl.load_workbook(path, data_only=True)
    px, cnt = [], []
    exposure = None
    for row in wb["Sheet1"].iter_rows(values_only=True):
        if isinstance(row[0], (int, float)) and isinstance(row[2], (int, float)):
            px.append(row[0])
            cnt.append(row[2])
        elif "frame_exposure" in str(row[2]):
            exposure = float(str(row[2]).split("=")[1])
    seeds = []                # (seed_px, lambda_nm), Sheet2 order
    for row in wb["Sheet2"].iter_rows(values_only=True):
        if (isinstance(row[0], (int, float)) and isinstance(row[1], (int, float))
                and row[1] > 100.0):
            seeds.append((float(row[0]), float(row[1])))
    return np.asarray(px, float), np.asarray(cnt, float), seeds, exposure


def fit_lines(px, cnt, seeds):
    base = float(np.median(cnt))
    results = {}              # lambda -> (center, err, sigma)

    def window(lo, hi, mask_sat):
        m = (px >= lo) & (px <= hi)
        x, y = px[m], cnt[m]
        if mask_sat:
            keep = y < SAT_ADU
            x, y = x[keep], y[keep]
        return x, y

    seed_of = dict((lam, p0) for p0, lam in seeds)

    # Isolated singles: free baseline, no clipping expected.
    for lam, p0 in seed_of.items():
        if lam in BLEND_PAIR or lam == CORE_LINE:
            continue
        x, y = window(p0 - SINGLE_HALF_PX, p0 + SINGLE_HALF_PX, mask_sat=True)
        off0 = float(np.percentile(y, 10))
        popt, pcov = curve_fit(
            lambda x, A, mu, s, off: gauss(x, A, mu, s) + off,
            x, y, p0=[y.max() - off0, p0, 2.0, off0], maxfev=40000)
        results[lam] = (popt[1], np.sqrt(pcov[1, 1]), abs(popt[2]))

    # 611.6: core flanks only, fixed baseline, clipped samples masked.
    p0 = seed_of[CORE_LINE]
    x, y = window(p0 - CORE_HALF_PX, p0 + CORE_HALF_PX, mask_sat=True)
    popt, pcov = curve_fit(
        lambda x, A, mu, s: gauss(x, A, mu, s) + base,
        x, y, p0=[y.max() - base, p0, 3.0], maxfev=40000)
    results[CORE_LINE] = (popt[1], np.sqrt(pcov[1, 1]), abs(popt[2]))

    # 542.4 / 546.074 blend: joint two-Gaussian, fixed baseline, masked.
    lam_a, lam_b = BLEND_PAIR                      # a = 542.4, b = 546.074
    pa, pb = seed_of[lam_a], seed_of[lam_b]        # pa > pb (blue = high px)
    x, y = window(min(pa, pb) - BLEND_PAD_PX, max(pa, pb) + BLEND_PAD_PX,
                  mask_sat=True)
    popt, pcov = curve_fit(
        lambda x, A1, m1, s1, A2, m2, s2:
            gauss(x, A1, m1, s1) + gauss(x, A2, m2, s2) + base,
        x, y, p0=[2e5, pb, 4.0, 4e4, pa, 8.0], maxfev=200000)
    results[lam_b] = (popt[1], np.sqrt(pcov[1, 1]), abs(popt[2]))
    results[lam_a] = (popt[4], np.sqrt(pcov[4, 4]), abs(popt[5]))

    n_clip = int((cnt > SAT_ADU).sum())
    return results, base, n_clip


def write_workbook(path: Path, centers_by_row, a0, a1, a2):
    """Chart-safe write-back: edit sheet XML inside the xlsx package."""
    z = zipfile.ZipFile(path)
    s1 = z.read("xl/worksheets/sheet1.xml").decode()
    s2 = z.read("xl/worksheets/sheet2.xml").decode()
    wbxml = z.read("xl/workbook.xml").decode()

    def setcell(xml, ref, val):
        pat = r'(<c r="%s"[^>]*><v>)[^<]*(</v>)' % ref
        new, n = re.subn(pat, r"\g<1>%s\g<2>" % val, xml)
        if n != 1:
            raise RuntimeError(f"cell {ref}: expected 1 match, got {n}")
        return new

    consts = [f"{a0:.6f}", f"{a1:.8f}", f"{a2:.6E}"]
    for ref, val in zip(("B18", "B19", "B20"), consts):
        s1 = setcell(s1, ref, val)
    for i, center in enumerate(centers_by_row, start=1):
        s2 = setcell(s2, f"A{i}", f"{center:.3f}")
    for ref, val in zip(("A8", "B8", "A9"), consts):
        s2 = setcell(s2, ref, val)

    if "fullCalcOnLoad" in wbxml:
        wbxml = re.sub(r'fullCalcOnLoad="[^"]*"', 'fullCalcOnLoad="1"', wbxml)
    elif "<calcPr" in wbxml:
        wbxml = wbxml.replace("<calcPr", '<calcPr fullCalcOnLoad="1"', 1)
    else:
        wbxml = wbxml.replace(
            "</workbook>", '<calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>')

    tmp = str(path) + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info in z.infolist():
            data = z.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                data = s1.encode()
            elif info.filename == "xl/worksheets/sheet2.xml":
                data = s2.encode()
            elif info.filename == "xl/workbook.xml":
                data = wbxml.encode()
            out.writestr(info, data)
    z.close()
    shutil.move(tmp, str(path))


def main():
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    px, cnt, seeds, exposure = load(xlsx)
    results, base, n_clip = fit_lines(px, cnt, seeds)

    lams = np.array(sorted(results))
    cens = np.array([results[l][0] for l in lams])
    coef = np.polyfit(cens, lams, 2)
    a2, a1, a0 = coef
    resid = lams - np.polyval(coef, cens)

    exp_ms = f"{exposure * 1e3:.0f} ms" if exposure else "unknown"
    print(f"{xlsx.name}: exposure {exp_ms}, baseline {base:.0f} ADU, "
          f"{n_clip} clipped px")
    print(f"\n{'lambda':>8} {'seed':>8} {'center_px':>10} {'+-':>6} "
          f"{'sigma':>6} {'resid_nm':>9}")
    seed_of = dict((lam, p0) for p0, lam in seeds)
    for lam, r in zip(lams, resid):
        mu, err, s = results[lam]
        print(f"{lam:8.3f} {seed_of[lam]:8.1f} {mu:10.3f} {err:6.3f} "
              f"{s:6.2f} {r:+9.4f}")
    print(f"\nlambda(p) = a0 + a1*p + a2*p^2")
    print(f"a0 = {a0:.6f}")
    print(f"a1 = {a1:.8f}")
    print(f"a2 = {a2:.6e}")
    print(f"RMS residual = {np.sqrt(np.mean(resid ** 2)):.4f} nm")

    centers_by_row = [results[lam][0] for _, lam in seeds]
    write_workbook(xlsx, centers_by_row, a0, a1, a2)
    print(f"\nWrote constants and centers back into {xlsx}")


if __name__ == "__main__":
    main()
