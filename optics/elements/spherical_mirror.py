"""Concave spherical mirror materials for a Subtract(Cylinder, Sphere) CSG primitive.

Two classes:

`SphericalMirror` — flat (wavelength-independent) reflectance.

`TabulatedSphericalMirror` — wavelength-dependent reflectance from a
CSV lookup table (e.g. vendor coating curve).

The primitive is a ``Subtract(Cylinder, Sphere)`` built by the scene
builder — a cylinder blank with the concave sphere carved out.  Raysect
delivers the true surface normal and hit point on the curved face, so
reflection is just the standard specular formula — no virtual-sphere
intersection needed.

Face discrimination: a hit is on the reflective sphere face when its
distance from the sphere centre equals R (within tolerance).  All other
hits (barrel, back face, or any ``exiting=True`` case) delegate to a
Lambert diffuse reflector (frosted glass substrate, default albedo 0.5).

Convention (unchanged from the flat-disk era):
- Local +z axis is the primitive's outward normal at the vertex.
- Rays enter from the +z half-space and reflect back toward +z.
- Sphere centre at (0, 0, +R) in the primitive's local frame, radius R.
"""


from raysect.core.math import Vector3D
from raysect.optical import ConstantSF
from raysect.optical.material.lambert import Lambert

from optics.elements._reflect import specular_reflect
from optics.elements._tabulated import interp_table, load_two_column_csv
from raysect.optical.material import Material

# Frosted glass substrate on barrel and back face.
_BACK_ALBEDO = 0.5
# Tolerance in metres; effective surface-distance threshold is tol/2.
_SPHERE_HIT_TOL = 1e-6


def _is_sphere_face(hit_point, R):
    """True when the hit lies on the sphere surface (within tolerance)."""
    x, y, z = hit_point.x, hit_point.y, hit_point.z
    dist_sq = x * x + y * y + (z - R) * (z - R)
    return abs(dist_sq - R * R) < _SPHERE_HIT_TOL * R


def _reflect_off_sphere_face(hit_point, normal, d_local,
                             outside_point, primitive_to_world, ray, world):
    """Specular reflection using the true sphere normal from raysect."""
    d_out_local = specular_reflect(d_local, normal)
    origin_world = outside_point.transform(primitive_to_world)
    d_out_world = d_out_local.transform(primitive_to_world).normalise()
    return ray.spawn_daughter(origin_world, d_out_world).trace(world)


class SphericalMirror(Material):
    """Concave spherical mirror with flat (scalar) reflectance.

    Parameters
    ----------
    focal_length_m : float
        Mirror focal length in metres.  Must be positive (concave).
    reflectance : float
        Fraction of incident power reflected (0–1).  Default 1.0.
    back_albedo : float
        Lambertian albedo for non-reflective faces.
    """

    def __init__(self, focal_length_m: float, reflectance: float = 1.0,
                 back_albedo: float = _BACK_ALBEDO):
        super().__init__()
        if focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive (concave)")
        if not 0.0 < reflectance <= 1.0:
            raise ValueError("reflectance must be in (0, 1]")
        self._f = float(focal_length_m)
        self._R = 2.0 * self._f
        self._reflectance = float(reflectance)
        self._back = Lambert(ConstantSF(back_albedo))

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not _is_sphere_face(hit_point, self._R):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_sphere_face(
            hit_point, normal, d_local, outside_point,
            primitive_to_world, ray, world,
        )
        if self._reflectance < 1.0:
            result.samples[:] *= self._reflectance
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum


class TabulatedSphericalMirror(Material):
    """Concave spherical mirror with wavelength-dependent reflectance.

    Parameters
    ----------
    focal_length_m : float
        Mirror focal length in metres.  Must be positive (concave).
    reflectance_wavelengths_nm : sequence of float
        Wavelength sample points (nm), sorted ascending.
    reflectance_values : sequence of float
        Reflectance at each sample point (fraction, 0–1).
    back_albedo : float
        Lambertian albedo for non-reflective faces.
    """

    def __init__(
        self,
        focal_length_m: float,
        reflectance_wavelengths_nm,
        reflectance_values,
        back_albedo: float = _BACK_ALBEDO,
    ):
        super().__init__()
        if focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive (concave)")
        lam = list(reflectance_wavelengths_nm)
        refl = list(reflectance_values)
        if len(lam) != len(refl):
            raise ValueError("wavelength and reflectance arrays must match")
        if len(lam) < 2:
            raise ValueError("need at least 2 points for interpolation")
        self._f = float(focal_length_m)
        self._R = 2.0 * self._f
        self._lam_table = lam
        self._refl_table = refl
        self._back = Lambert(ConstantSF(back_albedo))

    @staticmethod
    def from_csv(focal_length_m: float, csv_path: str,
                 back_albedo: float = _BACK_ALBEDO):
        """Load from a two-column CSV (wavelength_nm, reflectance)."""
        lam, refl = load_two_column_csv(csv_path)
        return TabulatedSphericalMirror(focal_length_m, lam, refl, back_albedo)

    def _interp(self, wavelength_nm: float) -> float:
        return interp_table(wavelength_nm, self._lam_table, self._refl_table)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not _is_sphere_face(hit_point, self._R):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_sphere_face(
            hit_point, normal, d_local, outside_point,
            primitive_to_world, ray, world,
        )
        wavelength_nm = 0.5 * (ray.min_wavelength + ray.max_wavelength)
        result.samples[:] *= self._interp(wavelength_nm)
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum
