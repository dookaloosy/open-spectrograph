"""Layout-independent scene types.

Shared by all spectrometer topologies (Czerny-Turner, Littrow, Ebert,
etc.). Topology-specific geometry modules (``czerny_base.py``,
``geometry_littrow.py``, …) import from here; downstream modules
(``world_builder``, ``metrics``, ``trace``) depend only
on this module, not on any specific topology.

Grating equation helpers live in ``grating_math.py``.
"""


from dataclasses import dataclass, field
from math import sqrt
from typing import Any

Vec3 = tuple[float, float, float]


class InfeasibleGeometry(Exception):
    """Raised when a candidate geometry is infeasible.

    Covers all pre-trace rejection reasons: unreachable grating angles,
    fold-angle limits, optic body collisions, beam-cone clipping, mount
    overlaps, footprint violations, and housing wall clearance failures.
    """


@dataclass(frozen=True)
class ElementPlacement:
    """A single optical element positioned in the global frame.

    `kind` is one of: 'slit', 'mirror', 'grating', 'detector'.
    `position` is the element's reference point in mm.
    `axis`     is the element's optical axis (unit vector). For round optics
               this is the disk/cylinder axis. Coincides with the surface
               normal at the vertex for spherical mirrors and flat optics;
               for OAPs it is the cylinder axis (collimated-beam direction).
    `params`   carries kind-specific extras (focal length, groove density, …).
    """

    label: str
    kind: str
    position: Vec3
    axis: Vec3
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    """A built spectrometer layout — optical elements + fixtures.

    Topology-specific builders (``build_optics_only_scene`` etc.) return
    a Scene with only ``elements``. A product's scene assembler adds
    ``fixtures`` (mounts, housing walls, baseplate — pre-materialed
    raysect CSG) for tracing.
    """

    elements: list[ElementPlacement]
    fixtures: tuple = ()

    def by_label(self, label: str) -> ElementPlacement:
        for el in self.elements:
            if el.label == label:
                return el
        raise KeyError(label)


@dataclass(frozen=True)
class Mount:
    """A placed fixture: raysect CSG + collision polygons.

    Used for optic mounts, housing walls, and baseplate — anything on
    ``Scene.fixtures``.  ``build_world`` parents each ``csg`` to the
    raysect world.

    ``bbox_xy_mm`` is the axis-aligned bounding box (used for housing
    outline generation and body-vs-body overlap checks).

    ``slab_polygon_xy`` is the oriented rectangle of the mount slab
    cross-section at the optical centre height — tighter than the AABB
    because it excludes the forward foot tongue.  Used for beam-cone
    clipping checks where only the slab (not the foot) matters.

    ``parent_label`` identifies the optic this mount belongs to, so
    self-pairs (optic vs its own mount) can be exempted from collision.
    """

    label: str
    csg: Any
    bbox_xy_mm: tuple[float, float, float, float]
    parent_label: str = ""
    z_range_mm: tuple[float, float] | None = None
    slab_polygon_xy: tuple[tuple[float, float], ...] | None = None
    foot_polygon_xy: tuple[tuple[float, float], ...] | None = None
    foot_outline_xy: tuple[tuple[float, float], ...] | None = None
    tongue_half_mm: float | None = None
    u_front_mm: float | None = None


# ─── 2-D geometry helpers ────────────────────────────────────────────────────

def perpendicular_xy(axis: Vec3) -> tuple[float, float]:
    """Return the normalised direction perpendicular to *axis* in the xy plane."""
    nx, ny = axis[0], axis[1]
    length = sqrt(nx * nx + ny * ny)
    if length < 1e-12:
        return (1.0, 0.0)
    return (-ny / length, nx / length)
