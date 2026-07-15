"""DeviceWorker: the one thread that owns the serial port.

The GUI thread never touches the device.  Communication is
signals-out (frame, connected, captured, calibrated, cal_lost,
stored, status, transient, error) and a job queue in (exposure, live
on/off, live averaging, capture, live calibration on/off, store).

Live view displays the *last* frame of each read command — the first
frame of every read is the integration since the previous read
(firmware flush behavior, bench 2026-07-13) and is discarded; with
live averaging on, the flush frame is dropped and the remaining
frames are averaged.  While live calibration is on, every displayed
frame gets a full pattern-locate + line fit: a good fit emits
``calibrated`` (and becomes what Store writes), a failed one emits
``cal_lost`` — signal loss while optics are being tweaked is a
transient state that recovers on the next good frame, never an
error.  Read errors and firmware diagnostics are reported on the
``transient`` signal and never end the session; a lost USB device
triggers an automatic reconnect loop (in-WSL retries only), and when
that gives up the worker idles until the user presses Reconnect.
Only that explicit press may run the usbipd re-attach, and only where
usbipd exists (WSL); on native Linux it simply retries the port.
"""

import queue
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from controller import calibrate as cal
from controller import protocol
from controller.capture import CLEARING_MIN_EXPOSURE_S, acquire, plan
from controller.device import Device, DeviceError, find_port, usbipd_reattach

# a live fit worse than this is treated as signal loss, not a lock
LIVE_CAL_MAX_RMS_NM = 1.5


class DeviceWorker(QThread):
    frame = Signal(object, dict)      # np.ndarray, stats
    connected = Signal(dict)          # version / coefficients / configuration
    disconnected = Signal()           # device lost; reconnect loop entered
    captured = Signal(str)            # path written
    calibrated = Signal(dict)         # result / sigma404 (a live lock)
    cal_lost = Signal(str)            # live calibration lost this frame
    stored = Signal(tuple)            # coefficients now on the device
    status = Signal(str)              # persistent state (live/paused/…)
    transient = Signal(str)           # diagnostics/warnings (auto-expire)
    error = Signal(str)

    def __init__(self, port=None):
        super().__init__()
        self._port = port
        self._jobs = queue.Queue()
        self._live = True
        self._exposure_s = 0.025
        self._stopping = False
        self._last_cal = None
        self._live_avg = False
        self._live_avg_n = 4
        self._live_cal = False
        self._cal_lines = None            # (lines, dispersion), lazy
        self._reconnect_requested = False # set by the Reconnect button

    # ── GUI-thread API (thread-safe: queue + plain writes) ────────

    def submit_exposure(self, seconds: float):
        self._jobs.put(("exposure", seconds))

    def submit_live(self, on: bool):
        self._jobs.put(("live", on))

    def submit_capture(self, nframes: int, path):
        self._jobs.put(("capture", (nframes, path)))

    def submit_live_cal(self, on: bool):
        self._jobs.put(("live_cal", on))

    def submit_live_avg(self, on: bool, nframes: int):
        self._jobs.put(("live_avg", (on, nframes)))

    def submit_store(self):
        self._jobs.put(("store", None))

    def submit_reconnect(self):
        self._reconnect_requested = True

    def shutdown(self):
        self._stopping = True

    # ── worker thread ────────────────────────────────────────────

    MAX_RECONNECTS = 10

    def run(self):
        reconnects = 0
        while not self._stopping:
            try:
                with Device(self._port) as dev:
                    info = {
                        "version": dev.version(),
                        "coefficients": dev.coefficients(),
                        "configuration": dev.configuration(),
                    }
                    # mode-appropriate clearing for the current exposure
                    _, pulses, _ = plan(self._exposure_s)
                    dev.set_clearing_pulses(pulses)
                    self.connected.emit(info)
                    self.status.emit("connected — flushing first frames…")
                    self._first_frame_pending = True
                    reconnects = 0
                    self._usbipd_tried = False
                    while not self._stopping:
                        self._tick(dev, info)
                    return
            except (DeviceError, OSError) as e:
                # serial I/O errors mean the USB device dropped (the
                # Teensy can reboot on some exposure transitions and
                # the usbipd attachment does not follow it)
                reconnects += 1
                if self._stopping:
                    return
                if reconnects == 1:
                    self.disconnected.emit()
                if reconnects > self.MAX_RECONNECTS:
                    self.error.emit(
                        f"{e} — gave up after {reconnects - 1} reconnect "
                        f"attempts; press Reconnect to try again")
                    self._await_reconnect()
                    reconnects = 0
                    continue
                self.transient.emit(f"device lost ({e}) — reconnect "
                                    f"{reconnects}/{self.MAX_RECONNECTS}…")
                # automatic retries never touch the host side; only an
                # explicit Reconnect press runs the usbipd re-attach
                if self._reconnect_requested:
                    self._reconnect_requested = False
                    self._usbipd_if_missing()
                self.msleep(3000)
            except Exception as e:                # surface, never die silent
                self.error.emit(f"{type(e).__name__}: {e} — press "
                                f"Reconnect to try again")
                self._await_reconnect()
                reconnects = 0

    def _await_reconnect(self):
        """Idle, disconnected, until the user presses Reconnect."""
        while not self._stopping:
            if self._reconnect_requested:
                self._reconnect_requested = False
                self._usbipd_if_missing()
                self.status.emit("reconnecting…")
                return
            self.msleep(200)

    def _usbipd_if_missing(self):
        """User-initiated only: re-attach a device that left WSL.

        A firmware crash drops the usbipd attachment, which no in-WSL
        retry can restore.  ``usbipd_reattach`` is a no-op (None) off
        WSL, so pressing Reconnect on native Linux simply retries the
        port.
        """
        try:
            find_port(self._port)
        except DeviceError:
            msg = usbipd_reattach()
            if msg is not None:
                self.transient.emit(msg)
                self.msleep(2000)         # let the tty node appear

    def _tick(self, dev, info):
        try:
            kind, payload = self._jobs.get_nowait()
        except queue.Empty:
            kind, payload = None, None

        if kind == "exposure":
            self._exposure_s = payload
            mode, pulses, _ = plan(payload)
            dev.set_clearing_pulses(pulses)
            note = ("" if mode == "PIT"
                    else " — PLM mode: weak lines read low")
            self.status.emit(f"exposure {payload*1e3:g} ms [{mode}]{note}")
        elif kind == "live":
            self._live = payload      # button state is the indicator
        elif kind == "live_avg":
            self._live_avg, self._live_avg_n = payload
            self.status.emit(f"live averaging {'on' if self._live_avg else 'off'}"
                             + (f" ({self._live_avg_n} frames)"
                                if self._live_avg else ""))
        elif kind == "capture":
            nframes, path = payload
            self.status.emit(f"capturing {nframes} frames…")
            try:
                acq = acquire(dev, nframes, self._exposure_s)
            except DeviceError as e:
                self.transient.emit(f"capture failed: {e}")
                return
            protocol.write_capture(
                path, acq.frames, acq.exposure_s,
                header={"identifier": info["version"],
                        "coefficients": list(info["coefficients"]),
                        "datalength": len(acq.frames[0]),
                        "acquisition_mode": acq.mode,
                        "clearing_pulses": acq.clearing_pulses})
            self.captured.emit(str(path))
            self.status.emit(
                f"wrote {path} ({nframes} frames, flushed {acq.n_flushed}, "
                f"baseline {acq.baseline_adu:.0f} ADU)")
        elif kind == "live_cal":
            self._live_cal = payload
            if payload and self._exposure_s < CLEARING_MIN_EXPOSURE_S:
                self.transient.emit(
                    f"PLM mode: weak lines read low — set exposure ≥ "
                    f"{CLEARING_MIN_EXPOSURE_S*1e3:g} ms to calibrate")
        elif kind == "store":
            if self._last_cal is None:
                self.transient.emit("nothing to store — calibrate first")
                return
            r = self._last_cal
            try:
                dev.store_coefficients(r.a0, r.a1, r.a2)
            except DeviceError as e:
                self.transient.emit(f"store failed: {e}")
                return
            self.stored.emit((r.a0, r.a1, r.a2, 0.0))
            self.status.emit("coefficients stored and verified by readback")
        elif self._live:
            t0 = time.time()
            navg = self._live_avg_n if self._live_avg else 1
            try:
                _, _, interval_s = plan(self._exposure_s)
                frames = dev.read_frames(navg, self._exposure_s,
                                         interval_s=interval_s)
            except (DeviceError, RuntimeError) as e:
                # a rejected read, parse hiccup, or transient glitch
                # must not kill the session — report and keep ticking
                self.transient.emit(f"read failed: {e}")
                self.msleep(300)
                return
            if dev.last_noise:
                self.transient.emit("device: " + dev.last_noise[0])
            if self._live_avg and len(frames) > 1:
                f = np.mean(frames[1:], axis=0)   # skip flush frame, average
            else:
                f = frames[-1]                    # fresh frame; [0] is flush
            self.frame.emit(f, {
                "baseline": float(np.median(f)),
                "clipped": int((f > protocol.SAT_ADU).sum()),
                "max": float(f.max()),
                "dt_s": time.time() - t0,
                # every update reads navg + 1 frames: the firmware
                # delivers n+1 and the first is the discarded flush
                "frames_read": len(frames),
            })
            if self._live_cal:
                self._live_calibrate(f)
            if getattr(self, "_first_frame_pending", False):
                self._first_frame_pending = False
                self.status.emit("")      # clear the flushing notice
        else:
            self.msleep(50)

    def _live_calibrate(self, f):
        """Full pattern-locate + fit on a displayed frame.

        Success updates the storable fit and emits ``calibrated``;
        any failure (locator lost the pattern, a window starved, an
        unstable fit while the optics are being tweaked) emits
        ``cal_lost`` and the next good frame simply locks again.
        """
        try:
            if self._cal_lines is None:
                self._cal_lines = cal.load_lines()
            px = np.arange(len(f), dtype=float)
            result = cal.calibrate(px, np.asarray(f, float),
                                   *self._cal_lines)
        except RuntimeError as e:
            self.cal_lost.emit(str(e))
            return
        if result.rms_nm > LIVE_CAL_MAX_RMS_NM:
            self.cal_lost.emit(f"fit unstable (RMS {result.rms_nm:.2f} nm)")
            return
        self._last_cal = result
        sigma404 = next(fit.sigma_px for fit in result.lines
                        if fit.wavelength_nm == 404.656)
        self.calibrated.emit({"result": result, "sigma404": sigma404})
