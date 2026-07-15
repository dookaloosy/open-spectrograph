"""`controller` — capture, calibrate, and view the Open Spectrograph.

Standalone client for the TCD1304 detector (drmcnelson firmware); no
upstream host software required.  `controller` with no arguments
launches the desktop app — the primary tool: live spectrum, capture,
and live calibration.  The subcommands are the scriptable/offline
layer for batch captures and reprocessing saved files.

Acquisitions run ghost-free (PIT mode + clearing pulses) at 16 ms
exposure and above, switching to pulse-loop mode below; every capture
file records its acquisition regime in the header.

Subcommands:
    gui                      launch the desktop app (the default)
    ports                    list candidate serial devices
    info                     version, configuration, coefficients, temperature
    capture                  acquire clean frames to a .tcd1304 file
    plot                     plot a capture on its calibrated wavelength axis
    calibrate                CFL wavelength calibration from a file or --live
    store                    store constants on the device (readback-verified)
    erase                    erase stored constants
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from controller import calibrate as cal
from controller import protocol
from controller.capture import CLEARING_PULSES, acquire
from controller.device import Device, DeviceError, find_port

LIVE_FRAMES_DEFAULT = 4
LIVE_EXPOSURE_DEFAULT = 0.025
STEPDOWN_EXPOSURES = (0.025, 0.010, 0.005, 0.003)


def parse_exposure(text: str) -> float:
    t = text.strip().lower()
    if t.endswith("ms"):
        return float(t[:-2]) / 1e3
    if t.endswith("s"):
        return float(t[:-1])
    return float(t)


def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def cmd_ports(args):
    from serial.tools import list_ports
    for p in list_ports.comports():
        vid = f"{p.vid:#06x}" if p.vid else "------"
        print(f"{p.device}  vid {vid}  {p.description}")


def cmd_info(args):
    with Device(args.port) as dev:
        print(dev.version())
        cfg = dev.configuration()
        print("  ".join(f"{k} {v}" for k, v in cfg.items()))
        a = dev.coefficients()
        print(f"coefficients: a0={a[0]}  a1={a[1]}  a2={a[2]}  a3={a[3]}")
        print(f"chip temperature: {dev.temperature_c():.1f} C")


def _default_capture_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S")
    out = Path("output")
    out.mkdir(exist_ok=True)
    return out / f"capture.{stamp}.tcd1304"


def _capture(args):
    with Device(args.port) as dev:
        acq = acquire(dev, args.frames, args.exposure,
                      clearing_pulses=args.clearing_pulses)
        header = {
            "identifier": dev.version(),
            "coefficients": list(dev.coefficients()),
            "datalength": len(acq.frames[0]),
            "acquisition_mode": acq.mode,
            "clearing_pulses": acq.clearing_pulses,
        }
    return acq, header


def cmd_capture(args):
    acq, header = _capture(args)
    path = args.output or _default_capture_path()
    protocol.write_capture(path, acq.frames, acq.exposure_s, header)
    print(f"{len(acq.frames)} frames -> {path}  "
          f"(flushed {acq.n_flushed}, baseline {acq.baseline_adu:.0f} ADU, "
          f"max {max(f.max() for f in acq.frames):.0f})")


def cmd_plot(args):
    import matplotlib.pyplot as plt
    cap = protocol.parse_capture(args.file)
    px, cnt, n_used, n_dropped = protocol.mean_spectrum(cap)
    # axis from the constants recorded in the capture header if present
    lam = px
    xlabel = "pixel"
    if "coefficients" in cap.header:
        try:
            a = [float(v) for v in
                 cap.header["coefficients"].strip("[] ").split(",")]
            lam = a[0] + a[1] * px + a[2] * px ** 2 + a[3] * px ** 3
            xlabel = "wavelength (nm)"
        except (ValueError, IndexError):
            pass
    plt.figure(figsize=(12, 5))
    plt.plot(lam, cnt, lw=0.8)
    plt.xlabel(xlabel)
    plt.ylabel("counts (ADU)")
    plt.title(f"{Path(args.file).name}  ({n_used} frames)")
    plt.grid(alpha=0.35)
    if args.png:
        plt.savefig(args.png, dpi=120)
        print(f"wrote {args.png}")
    else:
        plt.show()


def _calibrate_spectrum(px, cnt, lines, dispersion):
    result = cal.calibrate(px, cnt, lines, dispersion)
    print(cal.report(result))
    return result


def cmd_calibrate(args):
    lines, dispersion = (cal.load_lines(args.lines) if args.lines
                         else cal.load_lines())

    if args.live:
        with Device(args.port) as dev:
            result = None
            for exposure in ([args.exposure] if args.exposure
                             else STEPDOWN_EXPOSURES):
                acq = acquire(dev, args.frames, exposure)
                cnt = np.mean(acq.frames, axis=0)
                px = np.arange(len(cnt), dtype=float)
                print(f"exposure {exposure*1e3:g} ms: "
                      f"baseline {acq.baseline_adu:.0f} ADU, "
                      f"{int((cnt > protocol.SAT_ADU).sum())} clipped px")
                try:
                    result = _calibrate_spectrum(px, cnt, lines, dispersion)
                    break
                except cal.FitWindowError as e:
                    # clipped/starved window: a shorter exposure can fix
                    print(f"  -> {e}")
                except RuntimeError as e:
                    # any other fit failure won't improve with less
                    # light — abort instead of stepping into the
                    # short-exposure regime
                    raise DeviceError(f"calibration fit failed: {e}") from e
            if result is None:
                raise DeviceError("no exposure in the step-down ladder "
                                  "produced a fittable spectrum")
            _finish_calibration(result, args, dev=dev)
    else:
        if not args.file:
            raise SystemExit("calibrate: need a capture file or --live")
        cap = protocol.parse_capture(args.file)
        px, cnt, n_used, n_dropped = protocol.mean_spectrum(cap)
        if n_dropped:
            print(f"dropped {n_dropped} mostly-clipped frame(s)")
        result = _calibrate_spectrum(px, cnt, lines, dispersion)
        _finish_calibration(result, args, dev=None)


def _finish_calibration(result, args, dev):
    print()
    print("controller command:")
    print(result.store_command())
    if args.store:
        if not confirm(f"store a0={result.a0:.6f} a1={result.a1:.8f} "
                       f"a2={result.a2:.6e} on the device?"):
            print("not stored")
            return
        if dev is not None:
            dev.store_coefficients(result.a0, result.a1, result.a2)
        else:
            with Device(args.port) as d:
                d.store_coefficients(result.a0, result.a1, result.a2)
        print("stored and verified by readback")


def cmd_store(args):
    with Device(args.port) as dev:
        old = dev.coefficients()
        print(f"device currently: {old}")
        if not confirm(f"replace with a0={args.a0} a1={args.a1} "
                       f"a2={args.a2} a3={args.a3}?"):
            print("not stored")
            return
        dev.store_coefficients(args.a0, args.a1, args.a2, args.a3)
        print("stored and verified by readback")


def cmd_gui(args):
    from controller.gui.app import run
    # WSLg: the wayland socket lives under /mnt/wslg but shells don't
    # always carry an XDG_RUNTIME_DIR that contains it.
    import os
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    wsl_runtime = Path("/mnt/wslg/runtime-dir")
    if (not (Path(runtime) / "wayland-0").exists()
            and (wsl_runtime / "wayland-0").exists()):
        os.environ["XDG_RUNTIME_DIR"] = str(wsl_runtime)
    return run(port=args.port)


def cmd_erase(args):
    with Device(args.port) as dev:
        print(f"device currently: {dev.coefficients()}")
        if not confirm("erase stored coefficients?"):
            print("not erased")
            return
        dev.erase_coefficients()
        print("erased")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="controller", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None,
                    help="serial port (default: auto-discover Teensy)")
    # no subcommand launches the desktop app — the primary tool
    ap.set_defaults(func=cmd_gui)
    sub = ap.add_subparsers(dest="command", required=False,
                            metavar="<command>")

    sub.add_parser(
        "ports", help="list candidate serial devices",
    ).set_defaults(func=cmd_ports)
    sub.add_parser(
        "info",
        help="report device version, configuration, stored "
             "coefficients, and temperature",
    ).set_defaults(func=cmd_info)

    p = sub.add_parser(
        "capture",
        help="acquire clean frames to a .tcd1304 file (raw ADU; stale "
             "flush frames dropped automatically)")
    p.add_argument("-e", "--exposure", type=parse_exposure,
                   default=LIVE_EXPOSURE_DEFAULT, help="e.g. 25ms or 0.025")
    p.add_argument("-n", "--frames", type=int, default=LIVE_FRAMES_DEFAULT,
                   help="clean frames to keep (default 4)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output file (default "
                        "output/capture.<timestamp>.tcd1304)")
    p.add_argument("--clearing-pulses", type=int, default=None,
                   help="SH clearing pulses per frame (default: auto — "
                        "20 in PIT mode at 16ms and above; below that the "
                        "acquisition switches to PLM mode, which clears "
                        "internally)")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser(
        "plot",
        help="plot a capture on the wavelength axis recorded in its "
             "header (interactive window, or --png)")
    p.add_argument("file", type=Path)
    p.add_argument("--png", type=Path, default=None,
                   help="write a PNG instead of opening a window")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser(
        "calibrate",
        help="locate and fit the six CFL index lines (pattern locator "
             "+ saturation-aware Gaussians) and the quadratic "
             "wavelength constants")
    p.add_argument("file", type=Path, nargs="?", default=None,
                   help="a .tcd1304 capture (omit with --live)")
    p.add_argument("--live", action="store_true",
                   help="acquire from the device first")
    p.add_argument("-e", "--exposure", type=parse_exposure, default=None,
                   help="fix the exposure (default: step-down ladder)")
    p.add_argument("-n", "--frames", type=int, default=LIVE_FRAMES_DEFAULT,
                   help="frames to average (default 4)")
    p.add_argument("--lines", type=Path, default=None,
                   help="override the index-line list (default "
                        "data/cfl_lines.toml)")
    p.add_argument("--store", action="store_true",
                   help="store the constants on the device after the fit "
                        "(prompts, then verifies by readback)")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "store",
        help="store wavelength constants on the device (prompts, then "
             "verifies by readback)")
    p.add_argument("a0", type=float)
    p.add_argument("a1", type=float)
    p.add_argument("a2", type=float)
    p.add_argument("a3", type=float, nargs="?", default=0.0)
    p.set_defaults(func=cmd_store)

    sub.add_parser(
        "erase",
        help="erase the constants stored on the device (prompts)",
    ).set_defaults(func=cmd_erase)
    sub.add_parser(
        "gui",
        help="launch the desktop app: live view, capture, live "
             "calibration (the default when no command is given)",
    ).set_defaults(func=cmd_gui)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except DeviceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: serial I/O failed ({e}) — the USB device may have "
              f"dropped; if it left WSL, re-attach from Windows with "
              f"`usbipd attach --wsl --busid <id>`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
