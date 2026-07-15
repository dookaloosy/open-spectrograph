# Test fixtures

Real CFL captures from the assembled v0.5.1 instrument (bench,
2026-07-13), in the plain-text `.tcd1304` format (`# key = value`
header, then one integer count per pixel per frame). They are not
regenerable: they encode that day's alignment state and the constants
fitted from it.

| File | Contents | Used by |
|------|----------|---------|
| `cfl_20ms_2frames.tcd1304` | 20 ms capture whose first frame is stale (fully clipped) | header/frame parsing, stale- and clipped-frame guards |
| `cfl_25ms_4frames.tcd1304` | 25 ms capture, two stale + two live frames | known-good calibration regression (a₀ = 792.504…, RMS < 0.25 nm), pattern-locator envelope (shift/stretch/rejection) |

Why real captures instead of synthetic spectra: parsing them proves
compatibility with the device's actual output (a round-trip through
our own writer proves nothing), and the fits face real failure modes —
the clipped 542/546 blend, the structured flank under 611.6 nm, the
phosphor continuum — that clean synthetic Gaussians would hide.
