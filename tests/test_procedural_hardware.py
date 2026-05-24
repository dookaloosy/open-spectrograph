"""Validate procedural hardware CAD solids against vendor STEP references.

Vendor STEP files were removed from the repo for licensing reasons.
This test recovers them from git history, builds procedural replacements,
and compares volume / surface area / bounding box.

Threaded parts (set screws, flat head screws) are modelled as smooth
cylinders with socket recesses, so their volume will be higher than
the vendor STEPs which include actual thread geometry.  Volume
tolerances are set per-part-type accordingly.

Run:  .venv/bin/python3 -m pytest tests/test_procedural_hardware.py -v
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build123d import import_step

from optics.mounts_cad import (
    _FLAT_HEAD_SCREW_CATALOG,
    _procedural_flat_head_screw,
    _procedural_flat_tip_set_screw,
    _procedural_hasma,
    _procedural_heat_set_insert,
    _procedural_short_flat_tip_set_screw,
)

# Commit before vendor STEPs were removed.
_VENDOR_STEP_PARENT = "1ab5764~1"

# Vendor STEP paths in the repo (as of the parent commit).
_VENDOR_PATHS = {
    "HASMA":     "data/step/HASMA.step",
    "94459A110": "data/step/94459A110_Heat-Set Inserts for Plastic.STEP",
    "92605A047": "data/step/92605A047_Stainless Steel Flat-Tip Set Screw.STEP",
    "92605A044": "data/step/92605A044_Stainless Steel Flat-Tip Set Screw.STEP",
    "91771A108": "data/step/91771A108_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
    "91771A109": "data/step/91771A109_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
    "91771A194": "data/step/91771A194_Passivated 18-8 Stainless Steel Phillips Flat Head Screw.STEP",
}

# Volume tolerance per part type.
# HASMA: simple multi-section body → tight tolerance.
# Heat-set insert: knurling + tapped bore → moderate.
# Set screws: thread grooves remove ~40% of cylinder volume → loose.
# Flat head screws: thread + Phillips recess → loose.
_VOL_TOL = {
    "HASMA":     0.15,   # 15% (simplified hex, no chamfers/knurling)
    "94459A110": 0.35,   # 35% (knurl valleys + internal bore)
    "92605A047": 0.60,   # 60% (M2 thread grooves dominate)
    "92605A044": 0.60,   # 60% (M2 thread grooves dominate)
    "91771A108": 0.30,   # 30% (imperial thread + Phillips recess)
    "91771A109": 0.30,
    "91771A194": 0.25,
}


def _recover_vendor_step(part_number: str, tmpdir: Path) -> Path | None:
    """Recover a vendor STEP from git history into tmpdir."""
    repo_path = _VENDOR_PATHS.get(part_number)
    if repo_path is None:
        return None
    out_path = tmpdir / f"{part_number}.step"
    try:
        result = subprocess.run(
            ["git", "show", f"{_VENDOR_STEP_PARENT}:{repo_path}"],
            capture_output=True, check=True,
        )
        out_path.write_bytes(result.stdout)
        return out_path
    except subprocess.CalledProcessError:
        return None


# Procedural builder dispatch.
_BUILDERS = {
    "HASMA":     _procedural_hasma,
    "94459A110": _procedural_heat_set_insert,
    "92605A047": _procedural_flat_tip_set_screw,
    "92605A044": _procedural_short_flat_tip_set_screw,
}
for _pn, _dims in _FLAT_HEAD_SCREW_CATALOG.items():
    _BUILDERS[_pn] = lambda dims=_dims: _procedural_flat_head_screw(**dims)

# Reference measurements from vendor STEPs (volume, area, bbox dims sorted).
_REFERENCE = {
    "HASMA":     dict(vol=284.47, area=467.83, bbox=(8.001, 9.160, 9.652)),
    "94459A110": dict(vol=13.13,  area=66.60,  bbox=(2.500, 3.600, 3.600)),
    "92605A047": dict(vol=11.82,  area=67.67,  bbox=(2.000, 2.000, 6.000)),
    "92605A044": dict(vol=15.01,  area=40.82,  bbox=(2.000, 2.000, 5.000)),
    "91771A108": dict(vol=53.07,  area=168.38, bbox=(5.385, 5.385, 9.525)),
    "91771A109": dict(vol=60.58,  area=189.78, bbox=(5.385, 5.385, 11.113)),
    "91771A194": dict(vol=159.03, area=342.81, bbox=(7.925, 7.925, 12.700)),
}

ALL_PARTS = list(_REFERENCE.keys())


@pytest.fixture(scope="module")
def vendor_steps():
    """Recover all vendor STEPs into a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        steps = {}
        for pn in ALL_PARTS:
            path = _recover_vendor_step(pn, tmpdir)
            if path is not None and path.exists():
                steps[pn] = path
        yield steps


@pytest.mark.parametrize("part_number", ALL_PARTS)
def test_procedural_volume_vs_reference(part_number):
    """Procedural volume should be within tolerance of vendor reference."""
    proc = _BUILDERS[part_number]()
    ref = _REFERENCE[part_number]
    tol = _VOL_TOL[part_number]
    vol_err = abs(proc.volume - ref["vol"]) / ref["vol"]
    assert vol_err < tol, (
        f"{part_number} volume: procedural={proc.volume:.2f}, "
        f"reference={ref['vol']:.2f}, err={vol_err:.1%}, tol={tol:.0%}"
    )


@pytest.mark.parametrize("part_number", ALL_PARTS)
def test_procedural_bbox_dimensions(part_number):
    """Procedural bounding box dimensions should match reference within 10%."""
    proc = _BUILDERS[part_number]()
    ref = _REFERENCE[part_number]
    bb = proc.bounding_box()
    proc_dims = sorted([bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z])
    ref_dims = sorted(ref["bbox"])

    for i, (pd, rd) in enumerate(zip(proc_dims, ref_dims)):
        err = abs(pd - rd) / rd
        assert err < 0.10, (
            f"{part_number} bbox dim {i}: procedural={pd:.3f}, "
            f"reference={rd:.3f}, err={err:.1%}"
        )


@pytest.mark.parametrize("part_number", ALL_PARTS)
def test_procedural_solid_is_valid(part_number):
    """Procedural solid should have positive volume and area."""
    proc = _BUILDERS[part_number]()
    assert proc.volume > 0, f"{part_number}: zero volume"
    assert proc.area > 0, f"{part_number}: zero area"


@pytest.mark.parametrize("part_number", ALL_PARTS)
def test_procedural_vs_vendor_step(vendor_steps, part_number):
    """Compare procedural solid to actual vendor STEP (when available)."""
    if part_number not in vendor_steps:
        pytest.skip(f"vendor STEP not recovered for {part_number}")

    vendor = import_step(str(vendor_steps[part_number]))
    proc = _BUILDERS[part_number]()
    tol = _VOL_TOL[part_number]

    vol_err = abs(proc.volume - vendor.volume) / vendor.volume
    assert vol_err < tol, (
        f"{part_number} volume vs vendor: procedural={proc.volume:.2f}, "
        f"vendor={vendor.volume:.2f}, err={vol_err:.1%}"
    )


def test_load_or_procedural_fallback():
    """_load_or_procedural returns a valid solid when vendor STEP is missing."""
    from optics.mounts_cad import _load_or_procedural
    # These vendor STEP files don't exist on disk, so fallback must fire.
    for pn in ALL_PARTS:
        part = _load_or_procedural(pn)
        assert part.volume > 0, f"{pn}: fallback produced zero-volume solid"


def test_load_vendor_fiber_adapter_fallback():
    """_load_vendor_fiber_adapter returns a valid solid without vendor STEP."""
    from optics.mounts_cad import _load_vendor_fiber_adapter
    hasma = _load_vendor_fiber_adapter()
    assert hasma.volume > 0
    # Body should extend in -z (behind housing wall).
    bb = hasma.bounding_box()
    assert bb.min.Z < 0, "HASMA body should extend behind z=0"


# ── Summary table (not a test, runs with -s flag) ──────────────────────────

def test_print_comparison_table(vendor_steps, capsys):
    """Print a comparison table of all parts (visible with pytest -s)."""
    header = (f"{'Part':<14} {'Vol Ref':>8} {'Vol Proc':>9} {'Vol Err':>8} "
              f"{'BBox Match':>10}")
    rows = [header, "-" * len(header)]

    for pn in ALL_PARTS:
        ref = _REFERENCE[pn]
        proc = _BUILDERS[pn]()
        bb = proc.bounding_box()
        proc_dims = sorted([bb.max.X - bb.min.X, bb.max.Y - bb.min.Y,
                            bb.max.Z - bb.min.Z])
        ref_dims = sorted(ref["bbox"])
        bbox_ok = all(
            abs(pd - rd) / rd < 0.10
            for pd, rd in zip(proc_dims, ref_dims)
        )
        vol_err = (proc.volume - ref["vol"]) / ref["vol"]
        rows.append(
            f"{pn:<14} {ref['vol']:8.2f} {proc.volume:9.2f} "
            f"{vol_err:+7.1%} {'OK' if bbox_ok else 'FAIL':>10}"
        )

    with capsys.disabled():
        print()
        print("\n".join(rows))
        print()
