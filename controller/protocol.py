"""TCD1304 device data formats.

Two related formats (bench-verified 2026-07-13):

- **File format** (`.tcd1304`): `# key = value` header lines and
  `# DATA int64 <n> COLS 1` … `# END DATA` frame blocks, as saved by
  host tooling.
- **Wire format** (device output in ascii mode): a diagnostic
  preamble, then *unprefixed* `DATA <n>` … `END DATA` blocks, lines
  ending `\r`, and a `DONE` sentinel terminating every response.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Counts at or above this are treated as clipped (device full scale is
# ~64.6k on the 16-bit differential ADC).
SAT_ADU = 63000.0


@dataclass
class Capture:
    """Parsed capture: raw frames plus header metadata."""
    frames: list = field(default_factory=list)      # np.ndarray per frame
    header: dict = field(default_factory=dict)      # raw string values
    exposures: list = field(default_factory=list)   # seconds, per header seen

    @property
    def exposure_s(self) -> float | None:
        return self.exposures[-1] if self.exposures else None


def parse_capture(path: Path) -> Capture:
    """Parse a `.tcd1304` capture file (one or more frame blocks)."""
    cap = Capture()
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if body.startswith("DATA"):
                current = []
            elif body.startswith("END DATA"):
                if current is None:
                    raise RuntimeError(f"{path}: END DATA without DATA")
                cap.frames.append(np.asarray(current, float))
                current = None
            elif "=" in body:
                key, _, value = body.partition("=")
                key, value = key.strip(), value.strip()
                cap.header[key] = value
                if key == "frame_exposure":
                    cap.exposures.append(float(value))
        elif line and current is not None:
            current.append(float(line))
    if current is not None:
        raise RuntimeError(f"{path}: unterminated DATA block")
    if not cap.frames:
        raise RuntimeError(f"no DATA blocks found in {path}")
    lengths = {len(f) for f in cap.frames}
    if len(lengths) != 1:
        raise RuntimeError(f"inconsistent frame lengths {lengths} in {path}")
    return cap


def parse_wire_frames(text: str) -> tuple[list, list]:
    """Extract frames from ascii-mode device output.

    Wire framing is `DATA <n>` … integer lines … `END DATA` (no `#`
    prefix); anything outside blocks is ignored, and incomplete
    trailing blocks are dropped.

    The firmware occasionally interleaves async diagnostics into the
    stream (observed: ``OOPS! pulse sh without off bit set …`` at
    long exposures).  Non-numeric lines inside a block are collected
    and returned as ``noise`` rather than raised — the declared
    sample count decides whether the frame survived intact.  Frames
    whose length differs from their ``DATA <n>`` declaration are
    dropped (a diagnostic that clobbered a pixel line shortens the
    frame).

    Returns ``(frames, noise_lines)``.
    """
    frames, noise = [], []
    current, declared = None, None
    for line in text.splitlines():
        line = line.strip()
        m = re.fullmatch(r"DATA (\d+)", line)
        if m:
            current, declared = [], int(m.group(1))
        elif line == "END DATA":
            if current is not None:
                if len(current) == declared:
                    frames.append(np.asarray(current, float))
                else:
                    noise.append(f"dropped frame: {len(current)} of "
                                 f"{declared} samples")
            current, declared = None, None
        elif current is not None:
            if re.fullmatch(r"-?\d+", line):
                current.append(float(line))
            elif line:
                noise.append(line)
    return frames, noise


def write_capture(path: Path, frames, exposure_s: float,
                  header: dict | None = None) -> None:
    """Write frames as a `.tcd1304` file (file framing, host-side).

    Emits the `# key = value` header and one
    `# DATA int64 <n> COLS 1` … `# END DATA` block per frame, so the
    output round-trips through :func:`parse_capture` and stays
    readable by upstream tooling.  Header values are written as
    Python literals (strings quoted) because the upstream reader
    exec()s each header line; the device firmware quotes its own
    string values the same way.
    """
    out = ["# controller capture"]
    for key, value in (header or {}).items():
        literal = repr(value) if isinstance(value, str) else value
        out.append(f"# {key} = {literal}")
    out.append(f"# frame_exposure = {exposure_s}")
    # the upstream reader needs this sentinel to start frame parsing
    out.append("# header end")
    for frame in frames:
        out.append(f"# DATA int64 {len(frame)} COLS 1")
        out.extend(str(int(v)) for v in frame)
        out.append("# END DATA")
    Path(path).write_text("\n".join(out) + "\n")


def live_frames(cap: Capture) -> tuple[list, int]:
    """Split frames into (live, n_dropped), discarding garbage frames.

    A frame whose *median* is at clip level is not a spectrum — it is a
    stale frame that integrated while the device sat idle (common as
    the first frame of a PULSE_LOOP acquisition).
    """
    good = [f for f in cap.frames if np.median(f) < SAT_ADU]
    if not good:
        raise RuntimeError(f"all {len(cap.frames)} frames are mostly clipped")
    return good, len(cap.frames) - len(good)


def mean_spectrum(cap: Capture) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Average the live frames.  Returns (px, counts, n_used, n_dropped)."""
    good, n_dropped = live_frames(cap)
    cnt = np.mean(good, axis=0)
    px = np.arange(len(cnt), dtype=float)
    return px, cnt, len(good), n_dropped
