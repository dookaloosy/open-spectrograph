"""Acquisition orchestration: one continuous read, keep the stable tail.

Bench-verified CCD behavior (2026-07-13):

- The first frame of every read command is the integration since the
  previous read — the firmware's built-in flush (`read n` delivers
  n+1 frames).  Separate back-to-back `read 1` commands therefore
  NEVER settle: the inter-command gap re-contaminates each first
  frame.  Acquisition must be one continuous multi-frame read.
- After minutes idle the CCD is deeply saturated and takes several
  frames to drain (medians 64558×5, 1533, 1330, 1312, 1299 …), so a
  fixed flush count is wrong; we request a margin of extra frames and
  keep the stable tail (consecutive medians within tolerance).
"""

from dataclasses import dataclass

import numpy as np

from controller.device import Device, DeviceError
from controller.protocol import SAT_ADU

FLUSH_STABLE_TOL = 0.02      # tail medians within 2% of each other
FLUSH_MARGINS = (6, 18)      # extra frames: normal, then deep-drain retry

# SH clearing pulses before each exposure.  Without them, PIT frames
# carry a residual/ghost charge excess (~+5k ADU on illuminated
# pixels, bench 2026-07-13); 20 pulses is the upstream-validated dose.
CLEARING_PULSES = 20

# At the old sensor clock, short frame periods collided with the
# clearing sequence in the firmware's SH schedule (OOPS storm at
# 10 ms, crash at 9 ms, sporadic diagnostics to 20 ms and beyond;
# bench 2026-07-13/14).  With `set clock 1e-6` (sent on every
# connect) the firmware validates frame timing instead: PIT is
# cleanly rejected below 16 ms, and 16-20 ms ran without a single
# diagnostic (bench 2026-07-14).
CLEARING_MIN_EXPOSURE_S = 0.016

# Below the PIT+clearing floor, acquisition switches to pulse-loop
# (PLM) mode with a near-minimal interval: exposure + readout
# + scheduling margin.  PLM handles clearing internally and
# is upstream-validated above ~4k ADU/frame; weak (sub-knee) features
# read low in this mode, and bright features leave a signal-adjacent
# background excess (up to ~+2k ADU beside strong lines at 25 ms,
# zero at dark pixels; bench 2026-07-14) — PLM backgrounds are not
# flat.  Headers record the mode and interval so the regime is
# auditable.
#
# The margin must grow for very short exposures: with +7 ms the
# firmware crashes below 5 ms exposure (USB re-enumeration); with
# +12 ms even 1 ms runs clean (bench 2026-07-13).
PLM_INTERVAL_MARGIN_S = 0.007
PLM_SHORT_EXPOSURE_S = 0.005
PLM_SHORT_MARGIN_S = 0.012
PLM_MIN_EXPOSURE_S = 0.001           # tested floor; below is unexplored

# `set clock 1e-6` (sent on every connect) slows the pixel readout to
# ~14.9 ms, and the PLM interval must exceed it — the firmware rejects
# shorter frames ("frame is too short for readout time 0.014899").
PLM_MIN_INTERVAL_S = 0.017


@dataclass
class Acquisition:
    frames: list                 # clean frames, in order
    exposure_s: float
    n_flushed: int               # leading frames discarded
    baseline_adu: float          # median of the last kept frame
    clearing_pulses: int = CLEARING_PULSES
    mode: str = "PIT"            # "PIT" (+clearing) or "PLM" (short exposures)


def _stable_tail(frames, nframes: int):
    """Return the last ``nframes`` if they form a stable clean tail."""
    if len(frames) < nframes:
        return None
    tail = frames[-nframes:]
    medians = [float(np.median(f)) for f in tail]
    if max(medians) >= SAT_ADU:
        return None
    if max(medians) - min(medians) > FLUSH_STABLE_TOL * min(medians):
        return None
    return tail


def plan(exposure_s: float, clearing_pulses: int | None = None):
    """Choose the acquisition mode for an exposure.

    Returns ``(mode, clearing_pulses, interval_s)``:
    - >= 16 ms: PIT + clearing pulses (ghost-free, weak lines intact)
    - < 16 ms:  PLM at near-minimal interval (bright-line regime;
                sub-knee features read low)
    """
    if exposure_s >= CLEARING_MIN_EXPOSURE_S:
        pulses = CLEARING_PULSES if clearing_pulses is None else clearing_pulses
        return "PIT", pulses, None
    if clearing_pulses:
        raise DeviceError(
            f"exposure {exposure_s*1e3:g} ms is below the "
            f"{CLEARING_MIN_EXPOSURE_S*1e3:g} ms floor for clearing pulses "
            f"(firmware SH-timing collision: corrupted frames, possible "
            f"crash)")
    if exposure_s < PLM_MIN_EXPOSURE_S:
        raise DeviceError(
            f"exposure {exposure_s*1e3:g} ms is below the tested "
            f"{PLM_MIN_EXPOSURE_S*1e3:g} ms PLM floor")
    margin = (PLM_SHORT_MARGIN_S if exposure_s < PLM_SHORT_EXPOSURE_S
              else PLM_INTERVAL_MARGIN_S)
    return "PLM", 0, max(exposure_s + margin, PLM_MIN_INTERVAL_S)


def acquire(dev: Device, nframes: int, exposure_s: float,
            clearing_pulses: int | None = None) -> Acquisition:
    """One continuous read with flush margin; keep the stable tail."""
    mode, pulses, interval_s = plan(exposure_s, clearing_pulses)
    dev.set_clearing_pulses(pulses)
    for margin in FLUSH_MARGINS:
        frames = dev.read_frames(nframes + margin, exposure_s,
                                 interval_s=interval_s)
        tail = _stable_tail(frames, nframes)
        if tail is not None:
            return Acquisition(
                frames=tail, exposure_s=exposure_s,
                n_flushed=len(frames) - nframes,
                baseline_adu=float(np.median(tail[-1])),
                clearing_pulses=pulses, mode=mode)
    raise DeviceError(
        f"frame baseline did not stabilize within "
        f"{nframes + FLUSH_MARGINS[-1]} frames — is the input massively "
        f"overexposed or changing?")
