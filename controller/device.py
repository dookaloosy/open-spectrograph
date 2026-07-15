"""Serial client for the TCD1304 device.

Clean-room implementation against the published command reference and
bench-verified wire behavior (2026-07-13):

- commands are text + ``\n``; the device echoes the command line back
- responses end with a ``DONE`` sentinel line; lines end ``\r``
- ascii-mode frame data is ``DATA <n>`` … ``END DATA`` per frame
- data format is stateful on the device: sessions must ``set ascii``
  and are expected to restore ``set binary`` (leave as found)

The GUI and CLI both drive this module; nothing above it touches the
serial port.
"""

import time

import numpy as np

from controller.protocol import SAT_ADU, parse_wire_frames

TEENSY_VID = 0x16C0
BAUD = 115200          # USB CDC — rate is nominal
DONE = "DONE"


class DeviceError(RuntimeError):
    pass


def find_port(explicit: str | None = None) -> str:
    """Return the device port, discovering by Teensy USB VID."""
    if explicit:
        return explicit
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    teensys = [p for p in ports if p.vid == TEENSY_VID]
    if len(teensys) == 1:
        return teensys[0].device
    listing = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    if not teensys:
        raise DeviceError(
            f"no Teensy (USB VID {TEENSY_VID:#06x}) found; "
            f"serial ports present: {listing}")
    raise DeviceError(
        f"multiple Teensy devices found ({listing}); select one with --port")


def usbipd_reattach() -> str | None:
    """Best-effort re-attach of the device via usbipd (WSL only).

    A firmware crash drops the usbipd attachment along with the USB
    connection, and no amount of in-WSL retrying brings it back — the
    fix is `usbipd.exe attach` on the Windows side, which is callable
    from WSL.  Returns a status message, or None when usbipd.exe is
    not available (not running under WSL).
    """
    import shutil
    import subprocess
    exe = shutil.which("usbipd.exe")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "list"], capture_output=True,
                             text=True, timeout=15).stdout
        busid = next((ln.split()[0] for ln in out.splitlines()
                      if f"{TEENSY_VID:04x}:" in ln.lower()), None)
        if busid is None:
            return (f"usbipd: no device with VID {TEENSY_VID:#06x} "
                    f"on the Windows side")
        r = subprocess.run([exe, "attach", "--wsl", "--busid", busid],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            detail = (r.stderr or r.stdout).strip().splitlines()
            return f"usbipd attach: {detail[-1] if detail else 'failed'}"
        return f"usbipd: re-attached busid {busid}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"usbipd: {e}"


class Device:
    """One open session with exclusive ownership of the port.

    Pass ``ser`` to inject a fake for tests; otherwise the port is
    opened with pyserial.  Use as a context manager: entry drains the
    buffer and switches to ascii; exit restores binary and closes.
    """

    def __init__(self, port: str | None = None, *, ser=None,
                 timeout_s: float = 0.3):
        if ser is None:
            import serial
            ser = serial.Serial(find_port(port), BAUD, timeout=timeout_s)
            time.sleep(0.2)
        self._ser = ser
        self._ser.read(65536)                       # drain stale bytes
        self.last_noise: list = []                  # diagnostics from last read

    def __enter__(self):
        self.set_format("ascii")
        # known-good sensor timing on every connect.  The upstream
        # author recommends 1e-6 as the better default for the chip,
        # and the timing state can corrupt at runtime (bench
        # 2026-07-14: a stepped baseline at 8k/15k ADU instead of the
        # flat ~1.3k dark level; `set clock 1e-6` restored it).
        self.transact("set clock 1e-6")
        return self

    def __exit__(self, *exc):
        try:
            self.set_format("binary")               # leave as found
        finally:
            self._ser.close()
        return False

    # ── transport ────────────────────────────────────────────────

    def _collect(self, deadline_s: float, done_when) -> str:
        buf = b""
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            buf += self._ser.read(65536)
            if done_when(buf):
                break
        else:
            if not buf:
                raise DeviceError(
                    "device silent — is another client "
                    "(e.g. TCD1304Controller.py) holding the port?")
            raise DeviceError(
                f"response did not complete within {deadline_s:.0f}s "
                f"({len(buf)} bytes buffered)")
        return buf.decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")

    def transact(self, cmd: str, deadline_s: float = 5.0) -> list[str]:
        """Send one command, return response lines (echo + DONE stripped)."""
        self._ser.write(cmd.encode() + b"\n")
        text = self._collect(deadline_s,
                             lambda b: b.rstrip().endswith(DONE.encode()))
        lines = [l for l in text.splitlines() if l.strip()]
        if lines and lines[0].strip() == cmd:
            lines = lines[1:]                       # command echo
        if not lines or lines[-1].strip() != DONE:
            raise DeviceError(f"{cmd!r}: missing DONE sentinel in response")
        return [l.strip() for l in lines[:-1]]

    # ── report commands ──────────────────────────────────────────

    def version(self) -> str:
        return self.transact("version")[0]

    def configuration(self) -> dict:
        """e.g. {'PIXELS': 3694, 'START': 16, ...} from the config line."""
        line = self.transact("configuration")[0]
        tokens = line.split()
        if len(tokens) % 2:
            raise DeviceError(f"cannot parse configuration line: {line!r}")
        out = {}
        for key, value in zip(tokens[::2], tokens[1::2]):
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
        return out

    def temperature_c(self) -> float:
        line = self.transact("temperature")[0]
        return float(line.split()[-1])

    def data_format(self) -> str:
        line = self.transact("format")[0]
        return line.split()[-1]                     # 'ascii' | 'binary'

    def set_format(self, fmt: str) -> None:
        if fmt not in ("ascii", "binary"):
            raise ValueError(f"unknown data format {fmt!r}")
        self.transact(f"set {fmt}")
        actual = self.data_format()
        if actual != fmt:
            raise DeviceError(f"set {fmt} did not stick (device says {actual})")

    # ── coefficients ─────────────────────────────────────────────

    def coefficients(self) -> tuple[float, float, float, float]:
        line = self.transact("coefficients")[0]
        parts = line.split()
        if parts[0] != "coefficients" or len(parts) != 5:
            raise DeviceError(f"cannot parse coefficients line: {line!r}")
        return tuple(float(v) for v in parts[1:])

    def store_coefficients(self, a0, a1, a2, a3=0.0) -> None:
        """Store and verify by readback (float32 rounding tolerated)."""
        sent = (a0, a1, a2, a3)
        self.transact(f"store coefficients {a0:.6f} {a1:.8f} {a2:.6e} {a3:g}")
        stored = self.coefficients()
        for name, s, r in zip("a0 a1 a2 a3".split(), sent, stored):
            ok = (s == r == 0.0) or (s != 0.0 and abs(r - s) <= 1e-5 * abs(s))
            if not ok:
                raise DeviceError(
                    f"store verify failed: {name} sent {s!r}, "
                    f"device reports {r!r}")

    def erase_coefficients(self) -> None:
        self.transact("erase coefficients")

    # ── acquisition configuration ────────────────────────────────

    def set_clearing_pulses(self, n: int) -> None:
        """SH clearing pulses before each PIT exposure (default 0 =
        off at power-up).  20 pulses clears residual/ghost charge per
        the upstream validation; the pulse period is fixed in this
        firmware.  Reply is verified.
        """
        lines = self.transact(f"configure clearing pulses {n}")
        if not any(l.startswith(f"clearing pulses {n}") for l in lines):
            raise DeviceError(
                f"clearing pulses not confirmed (device said {lines!r})")

    # ── acquisition ──────────────────────────────────────────────

    def read_frames(self, nframes: int, exposure_s: float,
                    interval_s: float | None = None) -> list:
        """`read <n> <exposure>` and parse the wire frames.

        Acquisition framing (bench, 2026-07-13): the device replies
        with a setup report ending in an *early* ``DONE``, then
        streams the frames asynchronously — ``DATA n``…``END DATA``
        blocks interleaved with FRAME/FRAMESET metadata — and marks
        the true end with ``FRAMESET END`` / ``COMPLETE``.  It also
        delivers one more frame than asked (``read 1`` → 2 frames).

        Frames come back raw — including any stale/flushing frames the
        CCD emits; callers (see controller.capture) decide what to keep.
        """
        cmd = f"read {nframes} {exposure_s:g}"
        if interval_s is not None:                # PLM (pulse-loop) mode
            cmd += f" {interval_s:g}"
        self._ser.write(cmd.encode() + b"\n")
        deadline = 10.0 + nframes * ((interval_s or exposure_s) + 2.0)
        # A rejected read (e.g. exposure below the firmware's minimum
        # frame interval) replies with an Error line + DONE and never
        # streams frames; recognize that instead of timing out.
        text = self._collect(
            deadline,
            lambda b: ((b.count(b"END DATA") >= nframes
                        and b.rstrip().endswith(b"COMPLETE"))
                       or (b"Error" in b
                           and b.rstrip().endswith(DONE.encode()))))
        if "Error" in text:
            err = next(l.strip() for l in text.splitlines()
                       if "Error" in l)
            raise DeviceError(f"device rejected {cmd!r}: "
                              f"{err.removesuffix(DONE).strip()}")
        frames, noise = parse_wire_frames(text)
        self.last_noise = noise
        if noise:
            import warnings
            warnings.warn(f"device diagnostics during {cmd!r}: "
                          + "; ".join(noise[:3]))
        if len(frames) < nframes:
            raise DeviceError(
                f"asked for {nframes} frames, parsed {len(frames)}"
                + (f" (diagnostics: {'; '.join(noise[:3])})" if noise else ""))
        return frames
