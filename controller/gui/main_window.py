"""Open Spectrograph Controller — main window.

Top bar: Pause/Resume with the per-frame metrics beside it, then
right-aligned: exposure (auto mode switch: PIT+clearing from 16 ms,
PLM below), clean-frame count, live-averaging toggle, Capture, and
Calibrate.  Center: live spectrum on the calibrated wavelength axis
(350–750 nm), with a collapsible live-calibration panel: while it is
open every frame gets a full pattern-locate + line fit, so the
per-line table, the fit coefficients (old vs new), the line markers,
and a Gaussian FWHM of the 404.7 nm line all update live while the
optics are tuned.  Signal loss (a hand in the beam, a line lost
mid-tweak) shows as an amber "no lock" state that recovers on the
next good frame; Store (drift-gated, verified by readback) writes the
latest locked fit.  Bottom: a single line with device identity,
auto-expiring diagnostics, and status.

All device I/O happens in :class:`controller.gui.worker.DeviceWorker`;
this module is layout and wiring only.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSizePolicy, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from scipy.optimize import curve_fit

from controller import calibrate as cal
from controller.gui.worker import DeviceWorker
from controller.protocol import SAT_ADU

WARM_SIGMA_PX = 1.7        # 404.7 nm line width at thermal equilibrium
DRIFT_SIGMA_PX = 3.5       # above this, warn before storing

# Live 404.7 nm FWHM readout (alignment aid in the metrics bar)
FWHM_SEARCH_HALF_PX = 60   # tracking window around the last fitted center
FWHM_RESEED_HALF_PX = 150  # fallback search around the seed center
FWHM_MIN_AMP_ADU = 400.0   # peak prominence required to attempt a fit

pg.setConfigOptions(antialias=False, background="k", foreground="w")


class MainWindow(QMainWindow):
    def __init__(self, port=None):
        super().__init__()
        self.setWindowTitle("Open Spectrograph Controller")
        self.resize(1100, 640)

        self._coeff = None            # wavelength axis polynomial
        self._lam = None              # cached axis for current length

        # live 404.7 nm FWHM tracker; (re)seeded by each successful
        # calibration — the panel that shows it only opens after one
        self._fwhm_seed = None
        self._fwhm_center = None
        self._fwhm_sigma = 3.0

        # ── top bar ──────────────────────────────────────────────
        self.conn_label = QLabel(
            '<span style="color:#000000">●</span> connecting…')
        self.exposure = QDoubleSpinBox()
        # >=16 ms runs PIT + clearing (ghost-free, weak lines intact);
        # below that the worker switches to PLM mode (bright-line
        # regime, weak lines read low) — see controller.capture.plan.
        self.exposure.setRange(1.0, 2000.0)
        self.exposure.setValue(25.0)
        self.exposure.setSuffix(" ms")
        self.exposure.setDecimals(1)
        self.live_btn = QPushButton("Pause")
        self.live_btn.setCheckable(True)

        self.cal_btn = QPushButton("Calibrate…")
        self.cap_frames = QSpinBox()
        self.cap_frames.setRange(1, 100)
        self.cap_frames.setValue(4)
        self.cap_btn = QPushButton("Capture to file")
        # label shows the action a click performs (like Pause/Resume):
        # "Avg on" while single-frame, "Avg off" while averaging
        self.avg_btn = QPushButton("Avg on")
        self.avg_btn.setCheckable(True)
        self.avg_btn.setToolTip(
            "average the live view over 'frames' frames (slower updates)")
        self.metrics = QLabel("—")
        self.metrics.setAlignment(Qt.AlignmentFlag.AlignLeft
                                  | Qt.AlignmentFlag.AlignVCenter)
        top = QHBoxLayout()
        top.addWidget(self.live_btn)
        top.addWidget(self.metrics)
        top.addStretch(1)
        top.addWidget(QLabel("Exposure"))
        top.addWidget(self.exposure)
        top.addWidget(QLabel("Frames"))
        top.addWidget(self.cap_frames)
        top.addWidget(self.avg_btn)
        top.addWidget(self.cap_btn)
        top.addWidget(self.cal_btn)

        # ── plot ─────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "wavelength", units="nm")
        self.plot.setLabel("left", "counts", units="ADU")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        # instrument band (matches the paper's spectral range)
        self.plot.setXRange(350, 750, padding=0)
        self.plot.setLimits(xMin=350, xMax=750)
        self.plot.getViewBox().setAutoVisible(y=True)
        self.plot.enableAutoRange(x=False, y=True)
        self.curve = self.plot.plot(pen=pg.mkPen("#4da6ff", width=1))

        # ── calibration results panel (hidden until first fit) ───
        self.coeff_table = QTableWidget(3, 3)
        self.coeff_table.setHorizontalHeaderLabels(["", "old", "new"])
        self.coeff_table.verticalHeader().setVisible(False)
        self.coeff_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.coeff_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.cal_rms = QLabel("")
        self.cal_rms.setStyleSheet("font-family: monospace;")
        self.cal_table = QTableWidget(0, 6)
        self.cal_table.setHorizontalHeaderLabels(
            ["λ (nm)", "located", "center px", "± px", "σ px", "resid nm"])
        self.cal_table.verticalHeader().setVisible(False)
        self.cal_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.cal_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.cal_title = QLabel("")
        # long "no lock" messages wrap within the panel's width instead
        # of widening it
        self.cal_title.setWordWrap(True)
        self.cal_title.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Preferred)
        self.cal_close = QPushButton("✕")
        self.cal_close.setFixedWidth(28)
        cal_header = QHBoxLayout()
        cal_header.addWidget(self.cal_title)
        cal_header.addStretch(1)
        cal_header.addWidget(self.cal_close)
        self.store_btn = QPushButton("Store to device")
        self.store_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        cal_buttons = QHBoxLayout()
        cal_buttons.addStretch(1)
        cal_buttons.addWidget(self.cancel_btn)
        cal_buttons.addWidget(self.store_btn)
        cal_buttons.addStretch(1)

        cal_layout = QVBoxLayout()
        cal_layout.setContentsMargins(0, 0, 0, 0)
        cal_layout.addLayout(cal_header)
        cal_layout.addWidget(QLabel("fit wavelengths:"))
        cal_layout.addWidget(self.cal_table,
                             alignment=Qt.AlignmentFlag.AlignHCenter)
        cal_layout.addWidget(QLabel("fit coefficients:"))
        cal_layout.addWidget(self.coeff_table,
                             alignment=Qt.AlignmentFlag.AlignHCenter)
        cal_layout.addWidget(self.cal_rms)
        self.cal_fwhm = QLabel("")
        self.cal_fwhm.setStyleSheet("font-family: monospace;")
        cal_layout.addWidget(self.cal_fwhm)
        # no-lock notices live between the results and the buttons,
        # where they wrap into empty space instead of shifting the
        # panel contents
        self.cal_notice = QLabel("")
        self.cal_notice.setWordWrap(True)
        self.cal_notice.setStyleSheet("color: #cc8800;")
        self.cal_notice.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Preferred)
        cal_layout.addWidget(self.cal_notice)
        cal_layout.addStretch(1)
        cal_layout.addLayout(cal_buttons)
        self.cal_panel = QWidget()
        self.cal_panel.setLayout(cal_layout)
        self.cal_panel.setVisible(False)
        self._line_markers: list = []
        self._panel_sized = False             # crop table once per session
        self.cal_close.clicked.connect(self.on_panel_closed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.plot)
        splitter.addWidget(self.cal_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # ── transient diagnostics (auto-expires) ─────────────────
        self.transient_label = QLabel("")
        self.transient_label.setStyleSheet("color: #cc8800;")
        self.transient_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._transient_timer = None          # created after QApplication

        # ── bottom bar: device identity + diagnostics + status ───
        self.status_label = QLabel("")
        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.setVisible(False)
        actions = QHBoxLayout()
        actions.addWidget(self.conn_label)
        actions.addWidget(self.reconnect_btn)
        actions.addStretch(1)
        actions.addWidget(self.transient_label)
        actions.addSpacing(12)
        actions.addWidget(self.status_label)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(splitter, stretch=1)
        root.addLayout(actions)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # ── worker wiring ────────────────────────────────────────
        self.worker = DeviceWorker(port)
        self.worker.connected.connect(self.on_connected)
        self.worker.disconnected.connect(self.on_disconnected)
        self.worker.frame.connect(self.on_frame)
        self.worker.captured.connect(lambda p: None)
        self.worker.calibrated.connect(self.on_calibrated)
        self.worker.cal_lost.connect(self.on_cal_lost)
        self.worker.stored.connect(self.on_stored)
        self.worker.status.connect(self.on_status)
        self.worker.transient.connect(self.on_transient)
        self.worker.error.connect(self.on_error)
        self.exposure.valueChanged.connect(
            lambda ms: self.worker.submit_exposure(ms / 1e3))
        self.live_btn.toggled.connect(self.on_live_toggled)
        self.cap_btn.clicked.connect(self.on_capture)
        self.cal_btn.clicked.connect(self.on_calibrate_clicked)
        self.store_btn.clicked.connect(self.on_store_clicked)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        self.reconnect_btn.clicked.connect(self.on_reconnect_clicked)
        self.avg_btn.toggled.connect(self.on_avg_toggled)
        self.cap_frames.valueChanged.connect(
            lambda n: self.on_avg_toggled(self.avg_btn.isChecked()))
        self._pending_cal = None
        self.worker.start()

    # ── slots (GUI thread) ───────────────────────────────────────

    def on_connected(self, info):
        self.transient_label.setText("")      # stale failure notices
        self.reconnect_btn.setVisible(False)
        self._coeff = info["coefficients"]
        pixels = info["configuration"].get("PIXELS", "?")
        self.conn_label.setText(
            f'<span style="color:#00cc44">●</span> '
            f"{info['version'].split(',')[0]}  ({pixels} px)")

    def on_frame(self, f, stats):
        if self._lam is None or len(self._lam) != len(f):
            px = np.arange(len(f), dtype=float)
            a = self._coeff or (0.0, 1.0, 0.0, 0.0)
            self._lam = (a[0] + a[1] * px + a[2] * px ** 2
                         + a[3] * px ** 3) if self._coeff else px
        self.curve.setData(self._lam, f)
        clip = stats["clipped"]
        clip_str = f"<span style='color:#ff5555'>clip {clip} px</span>" \
            if clip else "clip 0 px"
        self.metrics.setText(
            f"baseline {stats['baseline']:.0f}  |  "
            f"max {stats['max']:.0f}  |  {clip_str}  |  "
            f"{1.0/max(stats['dt_s'],1e-6):.1f} fps")
        # live FWHM in the calibration panel — the one context where
        # the source is known to be a CFL and a 404.7 nm line must exist
        if self.cal_panel.isVisible():
            self.cal_fwhm.setText(self._fit_fwhm404(f))

    def _fit_fwhm404(self, f):
        """Track the 404.7 nm line and return a metrics-bar FWHM string.

        The tracked center follows the fitted mean frame to frame, so
        the readout stays locked while flexure tuning walks the line;
        if lock is lost it falls back to a wide search around the
        404.7 nm center of the last successful calibration.
        """
        base = float(np.median(f))
        for c, half in ((self._fwhm_center, FWHM_SEARCH_HALF_PX),
                        (self._fwhm_seed, FWHM_RESEED_HALF_PX)):
            if c is None:
                continue
            lo = max(int(c) - half, 0)
            hi = min(int(c) + half, len(f))
            p0 = lo + int(np.argmax(f[lo:hi]))
            if f[p0] - base >= FWHM_MIN_AMP_ADU:
                break
        else:
            self._fwhm_center = None
            return "FWHM: — @ 404.7nm"
        half = int(min(max(3.5 * self._fwhm_sigma, 15.0), 45.0))
        lo, hi = max(p0 - half, 0), min(p0 + half, len(f))
        x = np.arange(lo, hi, dtype=float)
        y = np.asarray(f[lo:hi], dtype=float)
        keep = y < SAT_ADU
        if keep.sum() < 8:
            return "FWHM: — @ 404.7nm"
        try:
            popt, _ = curve_fit(
                lambda xx, A, mu, s, off: cal.gauss(xx, A, mu, s) + off,
                x[keep], y[keep],
                p0=(max(f[p0] - base, FWHM_MIN_AMP_ADU), float(p0),
                    self._fwhm_sigma, base),
                maxfev=2000)
        except RuntimeError:
            return "FWHM: — @ 404.7nm"
        amp, mu, sigma = popt[0], popt[1], abs(popt[2])
        if not (0.5 < sigma < 25.0 and amp > FWHM_MIN_AMP_ADU / 2
                and lo < mu < hi):
            return "FWHM: — @ 404.7nm"
        self._fwhm_center, self._fwhm_sigma = float(mu), float(sigma)
        fwhm_px = 2.3548 * sigma
        if not self._coeff:
            return f"FWHM: {fwhm_px:.1f} px @ 404.7nm"
        a = self._coeff
        nm_px = abs(a[1] + 2 * a[2] * mu + 3 * a[3] * mu * mu)
        fwhm_nm = fwhm_px * nm_px
        color = ("#00cc44" if fwhm_nm <= 0.60
                 else "#ffbf00" if fwhm_nm <= 1.2 else "#ff5555")
        return (f"FWHM: <span style='color:{color}'>"
                f"{fwhm_nm:.2f} nm</span> ({fwhm_px:.1f} px) @ 404.7nm")

    def on_live_toggled(self, paused):
        self.live_btn.setText("Resume" if paused else "Pause")
        self.worker.submit_live(not paused)

    def on_capture(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S")
        out = Path("output")
        out.mkdir(exist_ok=True)
        self.worker.submit_capture(self.cap_frames.value(),
                                   out / f"capture.{stamp}.tcd1304")

    def _update_cal_panel(self, result):
        """Refresh the live panel from the latest locked fit."""
        old = self._coeff or (None, None, None)
        new = (result.a0, result.a1, result.a2)
        fmt = ("{:.6f}", "{:.8f}", "{:.6e}")
        for row, name in enumerate(("a0", "a1", "a2")):
            cells = [name,
                     fmt[row].format(old[row]) if old[row] is not None
                     else "—",
                     fmt[row].format(new[row])]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self.coeff_table.setItem(row, col, item)
        self.cal_rms.setText(f"RMS residual: {result.rms_nm:.4f} nm")
        self.cal_table.setRowCount(len(result.lines))
        for row, (fit, resid) in enumerate(
                zip(result.lines, result.residuals_nm)):
            cells = [f"{fit.wavelength_nm:.3f}",
                     f"{fit.seed_px:.1f}",
                     f"{fit.center_px:.3f}",
                     f"{fit.center_err_px:.3f}",
                     f"{fit.sigma_px:.2f}",
                     f"{resid:+.4f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self.cal_table.setItem(row, col, item)
        if not self._panel_sized:
            # crop both tables tightly around their contents (no
            # scrollbars, no blank rows/columns), once — live updates
            # must not make the panel jitter
            width = self._crop_table(self.cal_table)
            self._crop_table(self.coeff_table)
            self.cal_panel.setMinimumWidth(width + 16)
            self._panel_sized = True

    @staticmethod
    def _crop_table(table):
        """Fix a table's size to exactly fit its contents; return width."""
        table.resizeColumnsToContents()
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frame = 2 * table.frameWidth()
        width = frame + sum(table.columnWidth(c)
                            for c in range(table.columnCount()))
        height = frame + table.horizontalHeader().height() + \
            sum(table.rowHeight(r) for r in range(table.rowCount()))
        table.setFixedSize(width, height)
        return width

    def _clear_markers(self):
        for marker in self._line_markers:
            self.plot.removeItem(marker)
        self._line_markers = []

    def on_avg_toggled(self, on):
        self.avg_btn.setText("Avg off" if on else "Avg on")
        self.worker.submit_live_avg(on, self.cap_frames.value())

    def on_calibrate_clicked(self):
        self.cal_title.setText("live calibration — acquiring…")
        self.cal_notice.setText("")
        self.cal_panel.setVisible(True)
        self.worker.submit_live_cal(True)

    def on_panel_closed(self):
        self.worker.submit_live_cal(False)
        self.cal_panel.setVisible(False)
        self._clear_markers()

    def on_cancel_clicked(self):
        self.worker.submit_live_cal(False)
        self.cal_panel.setVisible(False)
        self.store_btn.setEnabled(False)
        self._pending_cal = None
        self._clear_markers()

    def on_cal_lost(self, message):
        if not self.cal_panel.isVisible():
            return                # in-flight signal after close
        # transient while optics are being tweaked — keep the last
        # locked fit on display and recover on the next good frame
        self.cal_title.setText(
            '<span style="color:#ffbf00">no lock</span>')
        self.cal_notice.setText(message)

    def on_calibrated(self, payload):
        from datetime import datetime
        if not self.cal_panel.isVisible():
            return                # in-flight signal after close
        self._pending_cal = payload
        result = payload["result"]
        hg = next(f for f in result.lines if f.wavelength_nm == 404.656)
        self._fwhm_seed = self._fwhm_center = hg.center_px
        self._fwhm_sigma = max(hg.sigma_px, 1.0)
        self.cal_title.setText(
            f'<span style="color:#00cc44">locked</span> — '
            f'{datetime.now().strftime("%H:%M:%S")}')
        self.cal_notice.setText("")
        self._update_cal_panel(result)
        self.store_btn.setEnabled(True)
        # mark the fitted centers on the current (device-coefficient) axis
        if self._lam is None:
            return
        positions = [(fit, self._lam[int(round(fit.center_px))])
                     for fit in result.lines
                     if 0 <= int(round(fit.center_px)) < len(self._lam)]
        if len(positions) == len(self._line_markers):
            # live update: slide the existing markers, no rebuild
            for marker, (_, pos) in zip(self._line_markers, positions):
                marker.setValue(pos)
            return
        for marker in self._line_markers:
            self.plot.removeItem(marker)
        self._line_markers = []
        # which side of its line each label sits on (anchor x: 1 = text
        # ends at the line (left side), 0 = text starts at it (right))
        label_side = {542.4: (1, 0.5), 404.656: (1, 0.5), 631.3: (0, 0.5)}
        # stagger label heights so close lines (542.4/546.1) don't
        # overlap
        for i, (fit, pos) in enumerate(positions):
            opts = {"position": 0.95 - 0.07 * (i % 3),
                    "color": "#ffcc00"}
            anchor = label_side.get(fit.wavelength_nm)
            if anchor:
                opts["anchors"] = [anchor, anchor]
            line = pg.InfiniteLine(
                pos=pos, angle=90,
                pen=pg.mkPen("#ffcc00", style=Qt.PenStyle.DashLine),
                label=f"{fit.wavelength_nm:.1f}",
                labelOpts=opts)
            self.plot.addItem(line)
            self._line_markers.append(line)

    def on_store_clicked(self):
        if self._pending_cal is None:
            return
        result = self._pending_cal["result"]
        sigma = self._pending_cal["sigma404"]
        text = (f"Store on the device (replaces current coefficients):\n\n"
                f"a0 = {result.a0:.6f}\n"
                f"a1 = {result.a1:.8f}\n"
                f"a2 = {result.a2:.6e}\n\n"
                f"RMS residual {result.rms_nm:.3f} nm")
        if sigma > DRIFT_SIGMA_PX:
            text += (f"\n\nWARNING — warm-up drift suspected:\n"
                     f"404.7 nm line width is {sigma:.1f} px "
                     f"(warm norm ~{WARM_SIGMA_PX} px).\n"
                     f"The instrument is likely still drifting; these "
                     f"constants will go stale.\nStore anyway?")
        answer = QMessageBox.question(
            self, "Store coefficients", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.worker.submit_store()

    def on_stored(self, coeff):
        from datetime import datetime
        self._coeff = coeff
        self._lam = None                      # recompute axis on next frame
        self.cal_title.setText(
            f'<span style="color:#00cc44">stored on device</span> — '
            f'{datetime.now().strftime("%H:%M:%S")}')

    def _expiring_text(self, label, timer_attr, message):
        """Set a label's text and clear it again after 30 s."""
        from PySide6.QtCore import QTimer
        timer = getattr(self, timer_attr, None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(30_000)
            timer.timeout.connect(lambda: label.setText(""))
            setattr(self, timer_attr, timer)
        label.setText(message)
        if message:
            timer.start()

    def on_status(self, message):
        self._expiring_text(self.status_label, "_status_timer", message)

    def on_transient(self, message):
        self._expiring_text(self.transient_label, "_transient_timer", message)

    def on_disconnected(self):
        self.conn_label.setText(
            '<span style="color:#000000">●</span> disconnected')
        self.reconnect_btn.setVisible(True)

    def on_error(self, message):
        self.on_disconnected()
        self.on_transient(f"error: {message}")

    def on_reconnect_clicked(self):
        # stays visible until a connection succeeds
        self.worker.submit_reconnect()

    def closeEvent(self, event):
        self.worker.shutdown()
        self.worker.wait(5000)
        super().closeEvent(event)
