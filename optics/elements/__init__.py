"""Custom raysect `Material` subclasses used by ``optics.world_builder``.

Pure-Python subclasses of the Cython `Material` ABC. They are slower than
the shipped materials (roughly an order of magnitude) but the spectrometer
trace is only a handful of hits per ray so the overhead is irrelevant for
stage (a).
"""

from optics.elements.flat_mirror import FlatMirror, TabulatedFlatMirror
from optics.elements.reflective_grating import (
    BlazedGrating,
    TabulatedBlazedGrating,
)
from optics.elements.hit_recorder import HitRecorder
from optics.elements.cylindrical_mirror import CylindricalMirror, TabulatedCylindricalMirror
from optics.elements.oap_mirror import ParaboloidalMirror, TabulatedParaboloidalMirror
from optics.elements.spherical_mirror import SphericalMirror, TabulatedSphericalMirror

__all__ = [
    "FlatMirror",
    "TabulatedFlatMirror",
    "SphericalMirror",
    "TabulatedSphericalMirror",
    "CylindricalMirror",
    "TabulatedCylindricalMirror",
    "ParaboloidalMirror",
    "TabulatedParaboloidalMirror",
    "BlazedGrating",
    "TabulatedBlazedGrating",
    "HitRecorder",
]
