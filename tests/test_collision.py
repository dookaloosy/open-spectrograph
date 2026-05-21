"""Tests for optics.collision — 2-D polygon collision detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optics.collision import (
    _circle_polygon,
    _rect_polygon,
    beam_cone_polygon,
    check_optic_collisions,
    polygon_for_element,
    polygons_overlap,
)
from optics.scene import ElementPlacement, Scene


# ── SAT primitives ─────────────────────────────────────────────────────

def test_overlapping_squares():
    sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
    sq2 = [(1, 1), (3, 1), (3, 3), (1, 3)]
    assert polygons_overlap(sq1, sq2)


def test_separated_squares():
    sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
    sq2 = [(5, 5), (7, 5), (7, 7), (5, 7)]
    assert not polygons_overlap(sq1, sq2)


def test_touching_edge():
    sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
    sq2 = [(2, 0), (4, 0), (4, 2), (2, 2)]
    assert polygons_overlap(sq1, sq2)


def test_circle_vs_rect_overlap():
    circ = _circle_polygon(0, 0, 5)
    rect = _rect_polygon(4, 0, 2, 2, (1, 0), (0, 1))
    assert polygons_overlap(circ, rect)


def test_circle_vs_rect_separated():
    circ = _circle_polygon(0, 0, 5)
    rect = _rect_polygon(20, 0, 2, 2, (1, 0), (0, 1))
    assert not polygons_overlap(circ, rect)


def test_rotated_rects_no_overlap():
    r1 = _rect_polygon(0, 0, 1, 5, (1, 0), (0, 1))
    r2 = _rect_polygon(0, 0, 1, 5, (0, 1), (-1, 0))
    assert polygons_overlap(r1, r2)


# ── polygon_for_element ───────────────────────────────────────────────

def test_mirror_polygon():
    el = ElementPlacement("M1", "mirror", (10, 20, 0), (0, -1, 0),
                          {"diameter_mm": 25.4, "edge_thickness_mm": 6.0})
    p = polygon_for_element(el)
    assert p is not None
    assert len(p) == 4
    xs = [v[0] for v in p]
    ys = [v[1] for v in p]
    assert min(xs) < 10 - 12 and max(xs) > 10 + 12
    # depth along axis (y) is edge_thickness=6, not diameter
    assert max(ys) - min(ys) < 7


def test_grating_polygon():
    el = ElementPlacement("grating", "grating", (0, 0, 0), (0, 1, 0),
                          {"size_mm": 25.0})
    p = polygon_for_element(el)
    assert p is not None
    assert len(p) == 4


def test_detector_polygon():
    el = ElementPlacement("detector", "detector", (30, -10, 0), (0.2, 0.98, 0),
                          {"package_width_mm": 41.6, "package_height_mm": 10.16,
                           "glass_to_die_mm": 1.7})
    p = polygon_for_element(el)
    assert p is not None
    assert len(p) == 4


def test_slit_with_hasma_polygon():
    el = ElementPlacement("entrance_slit", "slit", (-30, 20, 0), (0.5, -0.87, 0),
                          {"width_um": 400, "height_mm": 3,
                           "optic_center_height_mm": 15,
                           "bore_radius_mm": 3.25, "slit_length_mm": 9.652})
    p = polygon_for_element(el)
    assert p is not None
    assert len(p) == 4


def test_slit_without_hasma_returns_none():
    el = ElementPlacement("entrance_slit", "slit", (-30, 20, 0), (0.5, -0.87, 0),
                          {"width_um": 400, "height_mm": 3,
                           "optic_center_height_mm": 15})
    p = polygon_for_element(el)
    assert p is None


# ── beam cone ──────────────────────────────────────────────────────────

def test_beam_cone_is_quad():
    a = ElementPlacement("M1", "mirror", (0, 0, 0), (1, 0, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6})
    b = ElementPlacement("grating", "grating", (100, 0, 0), (0, 1, 0),
                         {"size_mm": 25.0})
    cone = beam_cone_polygon(a, b)
    assert len(cone) == 4


# ── optic collision scenarios ──────────────────────────────────────────

def _make_scene(*elements):
    return Scene(elements=list(elements))


def test_m2_blocking_grating():
    """M2 placed right next to grating — optic body overlap."""
    els = [
        ElementPlacement("entrance_slit", "slit", (-30, 20, 0), (0.5, -0.87, 0),
                         {"width_um": 400, "height_mm": 3,
                          "optic_center_height_mm": 15}),
        ElementPlacement("M1", "mirror", (-10, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("grating", "grating", (0, 0, 0), (0, 1, 0),
                         {"size_mm": 25.0}),
        ElementPlacement("M2", "mirror", (15, 5, 0), (-0.5, 0.87, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("detector", "detector", (30, -10, 0), (0.2, 0.98, 0),
                         {"package_width_mm": 41.6, "package_height_mm": 10.16,
                          "glass_to_die_mm": 1.7}),
    ]
    scene = _make_scene(*els)
    try:
        check_optic_collisions(scene, els)
        assert False, "should have detected collision"
    except Exception as e:
        assert "optic collision" in str(e)


def test_m2_in_beam_cone():
    """M2 in the M1→grating beam path (not overlapping bodies)."""
    els = [
        ElementPlacement("entrance_slit", "slit", (-30, 20, 0), (0.5, -0.87, 0),
                         {"width_um": 400, "height_mm": 3,
                          "optic_center_height_mm": 15}),
        ElementPlacement("M1", "mirror", (-10, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("grating", "grating", (0, 0, 0), (0, 1, 0),
                         {"size_mm": 25.0}),
        ElementPlacement("M2", "mirror", (-5, 50, 0), (-0.5, 0.87, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("detector", "detector", (30, -10, 0), (0.2, 0.98, 0),
                         {"package_width_mm": 41.6, "package_height_mm": 10.16,
                          "glass_to_die_mm": 1.7}),
    ]
    scene = _make_scene(*els)
    try:
        check_optic_collisions(scene, els)
        assert False, "should have detected beam-cone clip"
    except Exception as e:
        assert "beam-cone clip" in str(e)


def test_valid_layout_passes():
    """Well-separated optics — no collision."""
    els = [
        ElementPlacement("entrance_slit", "slit", (-60, 20, 0), (0.5, -0.87, 0),
                         {"width_um": 400, "height_mm": 3,
                          "optic_center_height_mm": 15}),
        ElementPlacement("M1", "mirror", (-10, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("grating", "grating", (0, 0, 0), (0, 1, 0),
                         {"size_mm": 25.0}),
        ElementPlacement("M2", "mirror", (50, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("detector", "detector", (60, 10, 0), (-0.5, 0.87, 0),
                         {"package_width_mm": 41.6, "package_height_mm": 10.16,
                          "glass_to_die_mm": 1.7}),
    ]
    scene = _make_scene(*els)
    check_optic_collisions(scene, els)


def test_detector_grating_collision():
    """Detector package clips grating — Xia-like close layout."""
    els = [
        ElementPlacement("entrance_slit", "slit", (-30, 20, 0), (0.5, -0.87, 0),
                         {"width_um": 400, "height_mm": 3,
                          "optic_center_height_mm": 15}),
        ElementPlacement("M1", "mirror", (-10, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("grating", "grating", (0, 0, 0), (0, 1, 0),
                         {"size_mm": 25.0}),
        ElementPlacement("M2", "mirror", (50, 100, 0), (0, -1, 0),
                         {"diameter_mm": 25.4, "edge_thickness_mm": 9.6}),
        ElementPlacement("detector", "detector", (20, -5, 0), (0.2, 0.98, 0),
                         {"package_width_mm": 41.6, "package_height_mm": 10.16,
                          "glass_to_die_mm": 1.7}),
    ]
    scene = _make_scene(*els)
    try:
        check_optic_collisions(scene, els)
        assert False, "should have detected detector-grating collision"
    except Exception as e:
        assert "optic collision" in str(e)
        assert "detector" in str(e) or "grating" in str(e)


# ── Xia 2017 integration ──────────────────────────────────────────────

def test_xia_traditional_collision():
    """Xia theoretical layout has physical infeasibilities (HASMA vs F1,
    detector package vs grating) that the collision check must catch."""
    try:
        from designs.czerny_bom import load_parts, build_optics_only_scene
        from designs.czerny_assembly import CzernyAssembly
        from optics._config import read_toml
        from designs.czerny_base import CzernyGenome
    except ImportError:
        print("SKIP — raysect not available")
        return

    bom = Path("data/czerny_bom_xia2017_standard.toml")
    parts = load_parts(m1_part="XIA-M1", m2_part="XIA-M2",
                       grating_part="GR25-0605", f1_part="XIA-F1",
                       f2_part=None, bom_path=bom, optic_size_mm=25.4)
    data = read_toml(Path("data/czerny_baseline_xia2017.toml"))
    genome = CzernyGenome(**{
        k: v for k, v in data.items()
        if k in CzernyGenome.__dataclass_fields__
    })
    scene = build_optics_only_scene(genome, parts, bom_path=bom)
    beam_path = CzernyAssembly().resolve_beam_path(scene)
    try:
        check_optic_collisions(scene, beam_path)
        assert False, "Xia layout should be rejected (physical infeasibilities)"
    except Exception as e:
        assert "optic collision" in str(e) or "beam-cone clip" in str(e)
        print(f"  Xia correctly rejected: {e}")


# ── runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__} ... ", end="", flush=True)
        t()
        print("PASS")
    print(f"\nAll {len(tests)} collision tests passed.")
