# 4. Instrument Control Software

*We describe `controller`, our client for the TCD1304 detector: a
desktop application for live viewing, capture, and wavelength
calibration, with a command-line layer for scripting and offline
reprocessing. The instrument requires drmcnelson's open firmware
(drmcnelson, 2024a) on its Teensy controller board; beyond that,
`controller` handles everything on the host.*

## 4.1 Architecture

Running `controller` with no arguments launches the desktop
application, the primary tool. Subcommands expose the same underlying
library for scripting and for reprocessing saved captures, and the
library itself is importable for custom analysis. A single worker
thread owns the serial port, so only one client can talk to the
instrument at a time.

## 4.2 Acquisition

The firmware operates the sensor in two timing modes (drmcnelson,
2024a). In periodic interrupt timer (PIT) or "single pulse" mode, the
exposure time sets the frame interval and frames arrive back to back;
exposure has no practical upper limit. In pulse-loop mode (PLM),
the firmware collects fast frame sets with exposures down to
microseconds and clears residual charge internally between frames;
the frame interval must cover the readout plus the exposure. `controller` selects the mode
automatically:

| Mode | Exposure | Character |
|------|----------|-----------|
| PIT + clearing | ≥ 16 ms | ghost-free; the default for quantitative work |
| PLM | 1–15 ms | bright-line regime only (see below) |

`controller` handles the firmware's quirks automatically: it runs the
clearing engine in PIT mode (frames otherwise carry a residual ghost
charge), pads PLM frame intervals to crash-safe values, discards the
contaminated leading frames of every read, and sets the recommended
sensor clock on every connect. One caveat: PLM sacrifices
photometric fidelity at short exposures (weak signals read low and
backgrounds are not flat), so we suggest using it as a bright-line
mode and recommend doing quantitative work in PIT.

## 4.3 Calibration

We calibrate against six index lines of an ordinary compact
fluorescent lamp, listed in `data/cfl_lines.toml` as reference
wavelengths plus the nominal dispersion:

| Wavelength (nm) | Species | Fit |
|-----------------|---------|-----|
| 404.656 | Hg | single Gaussian |
| 435.833 | Hg | single Gaussian |
| 542.4   | Tb³⁺ | joint two-Gaussian (blend) |
| 546.074 | Hg | joint two-Gaussian (blend) |
| 611.6   | Eu³⁺ | core-weighted, fixed baseline |
| 631.3   | Eu³⁺ | single Gaussian |

A pattern locator finds the six lines from scratch on each run (any
shift, ±25% of nominal dispersion), saturation-masked Gaussian fits
refine each peak center, and a quadratic fit to the six centers
yields the wavelength solution λ(p) = a₀ + a₁p + a₂p².

## 4.4 Interfaces

### 4.4.1 Desktop application

The desktop application presents one window (Figure 4, controls in
the table below): a set of controls along the top row drives live
spectral acquisition, and <kbd>Capture to file</kbd> saves the
current acquisition.
Captures use the `.tcd1304` plain-text format (drmcnelson, 2024a);
`controller` extends the header with the firmware identifier, stored
wavelength coefficients, acquisition mode, clearing pulse count, and
exposure, so anyone can audit the photometric regime of a file after
the fact.

Clicking <kbd>Calibrate…</kbd> opens the live
calibration panel, where every frame receives the line location and
fit of Section 4.3 and the fitted coefficients update on screen.
<kbd>Store to device</kbd> writes the latest fit and verifies it by
readback; <kbd>Cancel</kbd> or <kbd>✕</kbd> closes the panel without
saving.

If the USB connection drops, the application retries automatically;
if the retries fail, pressing the <kbd>Reconnect</kbd> button beside
the device indicator re-establishes the connection.

![controller desktop application — live CFL spectrum](figures/fig_4_controller_gui.png)

| Control | Function |
|---------|----------|
| <kbd>Pause</kbd> / <kbd>Resume</kbd> | freeze or resume the live view |
| Metrics readout | per-frame baseline, clipped pixels, maximum, display rate |
| <kbd>Exposure</kbd> | exposure time; automatically selects the acquisition mode (PIT or PLM) |
| <kbd>Frames</kbd> | frames per capture, and per averaged update when averaging is on |
| <kbd>Avg on</kbd> / <kbd>Avg off</kbd> | toggle averaging of the live view over the frame count |
| <kbd>Capture to file</kbd> | write the current acquisition to a `.tcd1304` file in `output/` |
| <kbd>Calibrate…</kbd> | open the live calibration panel |
| Status bar | shows connection indicator, firmware identity, status messages |

### 4.4.2 Command-line interface

`controller` also exposes a command-line interface with subcommands
to support scripting and offline work:

| Command | Function |
|---------|----------|
| `controller capture` | acquire clean frames to a `.tcd1304` file |
| `controller calibrate` | fit lines and constants (saved file or `--live`) |
| `controller plot` | plot a capture on its recorded wavelength axis |
| `controller info` | report firmware version, configuration, stored constants, temperature |
| `controller ports` | list the serial devices the host can see |
| `controller store` / `erase` | manage on-device coefficients |
