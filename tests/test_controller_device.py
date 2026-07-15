"""controller.device / controller.capture against a fake serial device.

The fake replays the wire behavior verified on the bench 2026-07-13:
command echo, `\r` line endings, `DONE` sentinel, stateful
ascii/binary format, unprefixed `DATA n … END DATA` frame blocks,
and CCD flush behavior (clipped frames, then a decaying baseline).
"""

import numpy as np
import pytest

from controller import protocol
from controller.capture import acquire
from controller.device import Device, DeviceError


class FakeSerial:
    """Minimal stand-in for serial.Serial driven by a responder fn."""

    def __init__(self, respond):
        self.respond = respond
        self.queue = b""
        self.sent = []
        self.closed = False

    def write(self, data):
        cmd = data.decode().strip()
        self.sent.append(cmd)
        self.queue += self.respond(cmd)

    def read(self, n):
        out, self.queue = self.queue[:n], self.queue[n:]
        return out

    def close(self):
        self.closed = True


def wire(*lines):
    """Encode lines the way the device sends them (CR endings)."""
    return "".join(l + "\r" for l in lines).encode()


def make_device_sim(frame_series=None):
    """Stateful responder mimicking the bench-verified firmware."""
    state = {"fmt": "binary", "coeff": (792.50421, -0.12010671,
                                        -6.2688252e-07, 0.0),
             "frame_i": 0, "clearing": 0}
    frame_series = frame_series or []

    def respond(cmd):
        if cmd in ("set ascii", "set binary"):
            state["fmt"] = cmd.split()[1]
            return wire(cmd, "DONE")
        if cmd.startswith("set clock"):
            state["clk"] = cmd.split()[2]
            return wire(cmd,
                        f"timings clk {state['clk']} sh duration 1e-06 "
                        f"offset 1e-06 icg duration 3.25e-06 offset 5e-07 "
                        f"cnvst duration 0 offset 3.35e-06 period 1e-05",
                        "DONE")
        if cmd == "format":
            return wire(cmd, f"format {state['fmt']}", "DONE")
        if cmd == "version":
            return wire(cmd, "TCD1304Device vers 0.6 (fake)", "DONE")
        if cmd == "configuration":
            return wire(cmd, "PIXELS 3694 START 16 DARK 13 STOP 3680 "
                             "BITS 16 VFS 0.819200 SENSOR TCD1304", "DONE")
        if cmd == "temperature":
            return wire(cmd, "CHIPTEMPERATURE 58.02", "DONE")
        if cmd == "coefficients":
            a = state["coeff"]
            return wire(cmd, f"coefficients {a[0]} {a[1]} {a[2]} {a[3]:g}",
                        "DONE")
        if cmd.startswith("configure clearing pulses"):
            state["clearing"] = int(cmd.split()[3])
            return wire(cmd, f"clearing pulses {state['clearing']}", "DONE")
        if cmd.startswith("store coefficients"):
            vals = [float(v) for v in cmd.split()[2:]]
            # emulate float32 storage rounding
            state["coeff"] = tuple(float(np.float32(v)) for v in vals)
            return wire(cmd, "DONE")
        if cmd.startswith("read "):
            exposure = float(cmd.split()[2])
            is_plm = len(cmd.split()) == 4        # read n exp interval
            if exposure < 0.008 and not is_plm:   # PIT floor only
                # real firmware glues DONE onto the error text
                return wire(cmd, "loading sub_module cnvst",
                            "Error: setup_pit exposure too short for "
                            "this pulse configuration, need at least "
                            "0.008DONE")
            # real framing: setup report + EARLY DONE, then frames with
            # metadata interleave, FRAMESET END + COMPLETE.  The device
            # delivers one more frame than asked.
            n = int(cmd.split()[1]) + 1
            out = [cmd, "stop_pit", "#tcd1304 setup pulse, clk 4e-07s",
                   f"FRAME COUNTS {n}", "DONE", "FRAMESET START "]
            for k in range(n):
                i = min(state["frame_i"], len(frame_series) - 1)
                frame = frame_series[i]
                state["frame_i"] += 1
                out += [f"FRAME COUNTER {k}", f"DATA {len(frame)}"]
                out += [str(int(v)) for v in frame]
                out += ["END DATA"]
            out += ["FRAMESET END", "COMPLETE"]
            return wire(*out)
        raise AssertionError(f"fake device got unknown command {cmd!r}")

    return respond, state


def flat(value, n=32):
    return np.full(n, float(value))


def test_transact_strips_echo_and_done():
    respond, _ = make_device_sim()
    dev = Device(ser=FakeSerial(respond))
    assert dev.version().startswith("TCD1304Device vers 0.6")


def test_configuration_parses_typed_values():
    respond, _ = make_device_sim()
    dev = Device(ser=FakeSerial(respond))
    cfg = dev.configuration()
    assert cfg["PIXELS"] == 3694
    assert cfg["VFS"] == pytest.approx(0.8192)
    assert cfg["SENSOR"] == "TCD1304"


def test_context_manager_sets_ascii_and_restores_binary():
    respond, state = make_device_sim()
    fake = FakeSerial(respond)
    with Device(ser=fake) as dev:
        assert state["fmt"] == "ascii"
        # known-good sensor timing on every connect
        assert state["clk"] == "1e-6"
    assert state["fmt"] == "binary"
    assert fake.closed


def test_store_coefficients_verifies_readback():
    respond, _ = make_device_sim()
    dev = Device(ser=FakeSerial(respond))
    dev.store_coefficients(791.552999, -0.11954256, -7.667151e-07)
    assert dev.coefficients()[0] == pytest.approx(791.553, abs=1e-3)


def test_store_verify_failure_raises():
    respond, state = make_device_sim()

    def lying(cmd):
        out = respond(cmd)
        if cmd.startswith("store"):
            state["coeff"] = (1.0, 2.0, 3.0, 0.0)     # device "lies"
        return out

    dev = Device(ser=FakeSerial(lying))
    with pytest.raises(DeviceError, match="store verify failed"):
        dev.store_coefficients(791.55, -0.1195, -7.7e-07)


def test_read_frames_parses_wire_blocks_and_ignores_preamble():
    respond, _ = make_device_sim(frame_series=[flat(100), flat(200),
                                               flat(300)])
    dev = Device(ser=FakeSerial(respond))
    frames = dev.read_frames(2, 0.025)
    assert len(frames) == 3                    # device sends n+1
    assert frames[0][0] == 100 and frames[1][0] == 200


def test_acquire_keeps_stable_tail_after_decay():
    # bench-shaped series: deep saturation, then decaying baselines
    series = [flat(64558), flat(64558), flat(64558), flat(1533),
              flat(1330), flat(1312), flat(1299), flat(1291),
              flat(1283), flat(1281)]
    respond, state = make_device_sim(frame_series=series)
    dev = Device(ser=FakeSerial(respond))
    acq = acquire(dev, nframes=3, exposure_s=0.025)
    # ghost mitigation configured before the read
    assert state["clearing"] == 20
    assert acq.clearing_pulses == 20 and acq.mode == "PIT"
    # single continuous read of 3+6 frames (device sends +1 = 10)
    assert acq.n_flushed == 7
    assert len(acq.frames) == 3
    medians = [np.median(f) for f in acq.frames]
    assert max(medians) <= 1291                # the settled tail
    assert acq.baseline_adu == 1281


def test_read_rejected_exposure_raises_named_error():
    respond, _ = make_device_sim(frame_series=[flat(1200)])
    dev = Device(ser=FakeSerial(respond))
    with pytest.raises(DeviceError, match="too short"):
        dev.read_frames(1, 0.0001)


def test_short_exposure_switches_to_plm():
    respond, state = make_device_sim(frame_series=[flat(1200)])
    fake = FakeSerial(respond)
    dev = Device(ser=fake)
    acq = acquire(dev, nframes=1, exposure_s=0.005)
    assert acq.mode == "PLM" and acq.clearing_pulses == 0
    assert state["clearing"] == 0
    # 3-arg read, interval floored at the ~15 ms pixel readout time
    assert any(cmd.startswith("read ") and cmd.endswith(" 0.017")
               for cmd in fake.sent)


def test_explicit_clearing_below_floor_raises():
    respond, _ = make_device_sim(frame_series=[flat(1200)])
    dev = Device(ser=FakeSerial(respond))
    with pytest.raises(DeviceError, match="clearing pulses"):
        acquire(dev, nframes=1, exposure_s=0.010, clearing_pulses=20)


def test_acquire_gives_up_if_never_stable():
    respond, _ = make_device_sim(frame_series=[flat(64558)])
    dev = Device(ser=FakeSerial(respond))
    with pytest.raises(DeviceError, match="did not stabilize"):
        acquire(dev, nframes=1, exposure_s=0.025)


def test_wire_parser_tolerates_firmware_diagnostics():
    # observed at 100 ms exposure: async OOPS line interleaved mid-block
    text = ("DATA 4\n10\n20\nOOPS! pulse sh without off bit set 0x400026744\n"
            "30\n40\nEND DATA\n"
            "DATA 4\n10\nOOPS! clobbered a pixel line\n20\n30\nEND DATA\n")
    frames, noise = protocol.parse_wire_frames(text)
    assert len(frames) == 1                    # full frame kept
    assert list(frames[0]) == [10, 20, 30, 40]
    assert any("OOPS" in n for n in noise)     # diagnostic surfaced
    assert any("dropped frame: 3 of 4" in n for n in noise)


def test_write_capture_roundtrips_through_parse(tmp_path):
    frames = [flat(1200, 64), flat(1210, 64)]
    out = tmp_path / "cap.tcd1304"
    protocol.write_capture(out, frames, 0.025,
                           header={"identifier": "fake device v0",
                                   "acquisition_mode": "PIT",
                                   "coefficients": [793.5, -0.12, 0.0, 0.0],
                                   "datalength": 64})
    cap = protocol.parse_capture(out)
    assert len(cap.frames) == 2
    assert cap.exposure_s == pytest.approx(0.025)
    assert np.array_equal(cap.frames[0], frames[0])
    # the upstream reader exec()s each header line as Python — every
    # value we write must be a valid literal (strings quoted)
    ns = {}
    for line in out.read_text().splitlines():
        body = line.lstrip("# ")
        if line.startswith("#") and " = " in body and "DATA" not in body:
            exec(body, ns)
    assert ns["acquisition_mode"] == "PIT"
    assert ns["identifier"] == "fake device v0"
    assert ns["coefficients"][0] == pytest.approx(793.5)
    # the sentinel the upstream reader requires before frame blocks
    assert "# header end" in out.read_text()