"""Test that HitRecorder counts each incoming ray exactly once.

Regression test for double-counting bug: when the recorder box sat in
front of the mirror (not wrapping around it), reflected rays re-entered
the box from behind and triggered a second hit recording.

With the fix, the box wraps around the entire optic body. Reflected
rays EXIT the box (exiting=True) and are not re-counted.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from raysect.core.math import Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World, Ray
from raysect.primitive import Box, Cylinder, Sphere, Subtract

from optics.elements.hit_recorder import HitRecorder
from optics.elements.spherical_mirror import SphericalMirror
from optics.elements.flat_mirror import FlatMirror
from optics.world_builder import MM_TO_M, _orthogonal_up, _placement_transform


class _FakeElement:
    """Minimal stand-in for ElementPlacement."""
    def __init__(self, position, axis, params, label="M1"):
        self.position = position
        self.axis = axis
        self.params = params
        self.label = label
        self.kind = "mirror"


def _build_spherical_mirror(world, position_mm, axis, focal_length_mm=100.0,
                            diameter_mm=25.4, center_thickness_mm=6.0):
    """Build one concave spherical mirror aligned with _placement_transform."""
    r_m = diameter_mm * 0.5 * MM_TO_M
    R_m = 2.0 * focal_length_mm * MM_TO_M
    sag_m = R_m - math.sqrt(R_m**2 - r_m**2)
    substrate_m = center_thickness_mm * MM_TO_M

    cyl = Cylinder(radius=r_m, height=substrate_m + sag_m)
    cyl.transform = translate(0.0, 0.0, -substrate_m)
    sphere = Sphere(radius=R_m)
    sphere.transform = translate(0.0, 0.0, R_m)
    mirror = Subtract(cyl, sphere)

    element = _FakeElement(
        position=position_mm,
        axis=axis,
        params={
            "diameter_mm": diameter_mm,
            "edge_thickness_mm": center_thickness_mm + sag_m / MM_TO_M,
            "center_thickness_mm": center_thickness_mm,
            "focal_length_mm": focal_length_mm,
            "mirror_type": "spherical",
            "reflectance": 1.0,
        },
    )
    mirror.transform = _placement_transform(element)
    mirror.name = "M1"
    mirror.material = SphericalMirror(
        focal_length_m=focal_length_mm * MM_TO_M, reflectance=1.0,
    )
    mirror.parent = world
    return mirror, element


def _build_flat_fold(world, position_mm, axis, diameter_mm=25.4,
                     edge_thickness_mm=3.0):
    """Build one flat fold mirror aligned with _placement_transform."""
    r_m = diameter_mm * 0.5 * MM_TO_M
    thickness_m = edge_thickness_mm * MM_TO_M

    element = _FakeElement(
        position=position_mm,
        axis=axis,
        params={
            "diameter_mm": diameter_mm,
            "edge_thickness_mm": edge_thickness_mm,
            "center_thickness_mm": edge_thickness_mm,
            "mirror_type": "flat",
            "reflectance": 1.0,
        },
        label="F1",
    )
    disk = Cylinder(radius=r_m, height=thickness_m)
    disk.transform = _placement_transform(element) * translate(0.0, 0.0, -thickness_m)
    disk.name = "F1"
    disk.material = FlatMirror(reflectance=1.0)
    disk.parent = world
    return disk, element


class _FakeBuilt:
    def __init__(self, world, name, prim):
        self.world = world
        self.primitives = {name: prim}


def test_no_double_counting_concave():
    """A single ray reflected by a concave mirror should produce exactly one hit."""
    from optics.forward_trace import overlay_big_recorder

    world = World()
    mirror, element = _build_spherical_mirror(
        world, position_mm=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0))
    built = _FakeBuilt(world, "M1", mirror)
    recorder, _ = overlay_big_recorder(built, "M1", element)

    ray = Ray(
        origin=Point3D(0.0, 0.0, 0.05),
        direction=Vector3D(0.0, 0.0, -1.0),
        min_wavelength=549.5, max_wavelength=550.5, max_depth=15,
    )
    ray.trace(world)

    n = len(recorder.hits)
    print(f"  concave mirror: {n} hit(s)")
    for i, h in enumerate(recorder.hits):
        print(f"    hit {i}: point={tuple(f'{c:.6f}' for c in h.point)}, depth={h.ray_depth}")
    assert n == 1, f"expected 1 hit, got {n} — reflected ray re-counted"


def test_no_double_counting_fold():
    """A fold mirror at 45 deg should produce exactly one hit per incoming ray."""
    from optics.forward_trace import overlay_big_recorder

    world = World()
    fwd = Vector3D(1.0, 1.0, 0.0).normalise()
    mirror, element = _build_flat_fold(
        world, position_mm=(0.0, 0.0, 0.0),
        axis=(fwd.x, fwd.y, fwd.z))
    built = _FakeBuilt(world, "F1", mirror)
    recorder, _ = overlay_big_recorder(built, "F1", element)

    ray = Ray(
        origin=Point3D(0.05, 0.0, 0.0),
        direction=Vector3D(-1.0, 0.0, 0.0),
        min_wavelength=549.5, max_wavelength=550.5, max_depth=15,
    )
    ray.trace(world)

    n = len(recorder.hits)
    print(f"  fold mirror: {n} hit(s)")
    for i, h in enumerate(recorder.hits):
        print(f"    hit {i}: point={tuple(f'{c:.6f}' for c in h.point)}, depth={h.ray_depth}")
    assert n == 1, f"expected 1 hit, got {n} — reflected ray re-counted"


if __name__ == "__main__":
    print("test_no_double_counting_concave:")
    test_no_double_counting_concave()
    print("  PASS\n")

    print("test_no_double_counting_fold:")
    test_no_double_counting_fold()
    print("  PASS\n")
