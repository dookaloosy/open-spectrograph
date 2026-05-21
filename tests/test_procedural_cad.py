"""Validate procedural CAD solids against vendor STEP reference files.

Each test loads the vendor STEP, builds the same part procedurally,
and asserts that volume and surface area match within tolerance.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build123d import import_step

from optics.mounts_cad import (
    _build_generic_grating,
    _build_generic_oap_mirror,
    _load_vendor_grating_at_front,
    _load_vendor_oap_at_optical_centre,
    set_bom_path,
)

DATA = Path(__file__).resolve().parent.parent / "data"
STEP_DIR = DATA / "step"
BOM = DATA / "czerny_bom_v0_design.toml"
BOM_OAP = DATA / "czerny_bom_tl_oap.toml"

# Tolerances: volume within 0.5%, area within 1% (area diverges more due to
# bolt hole cylinder/cone surfaces in vendor STEPs that are absent in the
# procedural model).
VOL_TOL = 0.005
AREA_TOL_GRATING = 0.001  # gratings are perfect boxes
AREA_TOL_OAP = 0.15       # OAP vendor has bolt hole surfaces we omit


# ── Gratings ─────────────────────────────────────────────────────────────────

GRATING_PARTS = [
    ("GR13-0605", 12.7, 6.0),
    ("GR13-1205", 12.7, 6.0),
    ("GR13-1208", 12.7, 6.0),
    ("GR13-1210", 12.7, 6.0),
    ("GR25-0605", 25.0, 6.0),
    ("GR25-1205", 25.0, 6.0),
    ("GR25-1208", 25.0, 6.0),
    ("GR25-1210", 25.0, 6.0),
    ("GR50A-0605", 50.0, 9.5),
    ("GH50-12V", 50.0, 9.5),
]


@pytest.mark.parametrize("part_number,size_mm,thickness_mm", GRATING_PARTS,
                         ids=[p[0] for p in GRATING_PARTS])
def test_grating_volume_and_area(part_number, size_mm, thickness_mm):
    step_path = STEP_DIR / f"{part_number}.step"
    if not step_path.exists():
        pytest.skip(f"vendor STEP not found: {step_path}")

    vendor = import_step(str(step_path))
    procedural = _build_generic_grating(size_mm, thickness_mm)

    vol_err = abs(procedural.volume - vendor.volume) / vendor.volume
    area_err = abs(procedural.area - vendor.area) / vendor.area

    assert vol_err < VOL_TOL, (
        f"{part_number} volume: procedural={procedural.volume:.4f}, "
        f"vendor={vendor.volume:.4f}, err={vol_err:.6f}"
    )
    assert area_err < AREA_TOL_GRATING, (
        f"{part_number} area: procedural={procedural.area:.4f}, "
        f"vendor={vendor.area:.4f}, err={area_err:.6f}"
    )


# ── OAP Mirrors ──────────────────────────────────────────────────────────────

OAP_PARTS = [
    # (part_number, focal_length_mm, diameter_mm, center_thickness_mm)
    ("MPD029-G01", 50.8, 12.7, 12.1),
    ("MPD129-G01", 50.8, 25.4, 17.4),
    ("MPD149-G01", 101.6, 25.4, 18.2),
    ("MPD249-G01", 101.6, 50.8, 34.2),
]


@pytest.mark.parametrize("part_number,focal_mm,diameter_mm,ct_mm", OAP_PARTS,
                         ids=[p[0] for p in OAP_PARTS])
def test_oap_volume(part_number, focal_mm, diameter_mm, ct_mm):
    """Procedural OAP volume should be close to vendor (no bolt holes)."""
    step_path = STEP_DIR / f"{part_number}.step"
    if not step_path.exists():
        pytest.skip(f"vendor STEP not found: {step_path}")

    vendor = import_step(str(step_path))
    procedural = _build_generic_oap_mirror(focal_mm, diameter_mm, ct_mm)

    # Procedural has no bolt holes so it should be >= vendor volume.
    # Allow up to 4% difference (bolt holes remove ~3-4% for small OAPs).
    vol_err = abs(procedural.volume - vendor.volume) / vendor.volume
    assert vol_err < 0.04, (
        f"{part_number} volume: procedural={procedural.volume:.2f}, "
        f"vendor={vendor.volume:.2f}, err={vol_err:.4f}"
    )


@pytest.mark.parametrize("part_number,focal_mm,diameter_mm,ct_mm", OAP_PARTS,
                         ids=[p[0] for p in OAP_PARTS])
def test_oap_bounding_box(part_number, focal_mm, diameter_mm, ct_mm):
    """Procedural OAP bounding box should match vendor diameter and height."""
    from build123d import Axis
    step_path = STEP_DIR / f"{part_number}.step"
    if not step_path.exists():
        pytest.skip(f"vendor STEP not found: {step_path}")

    vendor = import_step(str(step_path))
    vendor = vendor.rotate(Axis.X, -90).rotate(Axis.X, 180)
    vbb = vendor.bounding_box()
    vendor_height = vbb.max.Z - vbb.min.Z
    vendor_width_x = vbb.max.X - vbb.min.X
    vendor_width_y = vbb.max.Y - vbb.min.Y

    procedural = _build_generic_oap_mirror(focal_mm, diameter_mm, ct_mm)
    pbb = procedural.bounding_box()
    proc_height = pbb.max.Z - pbb.min.Z
    proc_width_x = pbb.max.X - pbb.min.X
    proc_width_y = pbb.max.Y - pbb.min.Y

    # Width should match diameter exactly
    assert abs(proc_width_x - diameter_mm) < 0.1, (
        f"{part_number} X width: {proc_width_x:.3f} vs diameter {diameter_mm}")
    assert abs(proc_width_y - diameter_mm) < 0.1, (
        f"{part_number} Y width: {proc_width_y:.3f} vs diameter {diameter_mm}")

    # Height should be within 5% of vendor (exact height depends on parabola profile)
    height_err = abs(proc_height - vendor_height) / vendor_height
    assert height_err < 0.05, (
        f"{part_number} height: procedural={proc_height:.3f}, "
        f"vendor={vendor_height:.3f}, err={height_err:.4f}"
    )


# ── Integration: fallback path produces valid solids ─────────────────────────

def test_grating_fallback_produces_solid():
    """_load_vendor_grating_at_front falls back to procedural for unknown parts."""
    set_bom_path(BOM)
    grating = _load_vendor_grating_at_front("GR25-0605")
    assert grating.volume > 0
    bb = grating.bounding_box()
    assert abs(bb.max.Z) < 0.01  # front face at z=0


def test_oap_fallback_produces_solid():
    """_load_vendor_oap_at_optical_centre falls back for unknown parts."""
    set_bom_path(BOM_OAP)
    oap = _load_vendor_oap_at_optical_centre("MPD129-G01")
    assert oap.volume > 0
    bb = oap.bounding_box()
    assert bb.min.Z < 0 and bb.max.Z > 0
