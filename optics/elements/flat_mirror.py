"""Flat mirror materials on a flat-disk primitive.

Two classes:

`FlatMirror` — flat (wavelength-independent) reflectance.

`TabulatedFlatMirror` — wavelength-dependent reflectance from a CSV
lookup table (e.g. vendor coating curve).

Both implement specular reflection using the true surface normal
delivered by raysect.  Only the front face (local +z normal) reflects;
back face, barrel, and ``exiting`` hits delegate to a Lambert diffuse
reflector (frosted glass substrate, albedo 0.5).

Used for fold mirrors — ideal planar reflectors with no curvature,
no aberration contribution.
"""


from raysect.core.math import Vector3D
from raysect.optical import ConstantSF
from raysect.optical.material import Material
from raysect.optical.material.lambert import Lambert

from optics.elements._reflect import specular_reflect
from optics.elements._tabulated import interp_table, load_two_column_csv

# Frosted glass substrate on barrel and back face.
_FLAT_BACK_ALBEDO = 0.5


class FlatMirror(Material):
    """Flat mirror with scalar reflectance."""

    def __init__(self, reflectance: float = 1.0,
                 back_albedo: float = _FLAT_BACK_ALBEDO):
        super().__init__()
        if not 0.0 < reflectance <= 1.0:
            raise ValueError("reflectance must be in (0, 1]")
        self._reflectance = float(reflectance)
        self._back = Lambert(ConstantSF(back_albedo))

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or abs(1.0 - normal.z) > 1e-6:
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        d_out = specular_reflect(d_local, normal)
        origin_world = outside_point.transform(primitive_to_world)
        d_out_world = d_out.transform(primitive_to_world).normalise()
        result = ray.spawn_daughter(origin_world, d_out_world).trace(world)
        if self._reflectance < 1.0:
            result.samples[:] *= self._reflectance
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum


class TabulatedFlatMirror(Material):
    """Flat mirror with wavelength-dependent reflectance."""

    def __init__(self, reflectance_wavelengths_nm, reflectance_values,
                 back_albedo: float = _FLAT_BACK_ALBEDO):
        super().__init__()
        lam = list(reflectance_wavelengths_nm)
        refl = list(reflectance_values)
        if len(lam) != len(refl):
            raise ValueError("wavelength and reflectance arrays must match")
        if len(lam) < 2:
            raise ValueError("need at least 2 points for interpolation")
        self._lam_table = lam
        self._refl_table = refl
        self._back = Lambert(ConstantSF(back_albedo))

    @staticmethod
    def from_csv(csv_path: str):
        lam, refl = load_two_column_csv(csv_path)
        return TabulatedFlatMirror(lam, refl)

    def _interp(self, wavelength_nm: float) -> float:
        return interp_table(wavelength_nm, self._lam_table, self._refl_table)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or abs(1.0 - normal.z) > 1e-6:
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        d_out = specular_reflect(d_local, normal)
        origin_world = outside_point.transform(primitive_to_world)
        d_out_world = d_out.transform(primitive_to_world).normalise()
        result = ray.spawn_daughter(origin_world, d_out_world).trace(world)
        wavelength_nm = 0.5 * (ray.min_wavelength + ray.max_wavelength)
        result.samples[:] *= self._interp(wavelength_nm)
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum
