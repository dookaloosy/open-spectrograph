"""Concave cylindrical mirror materials for a Subtract(Cylinder, Cylinder) CSG primitive.

Two classes:

`CylindricalMirror` — flat (wavelength-independent) reflectance.

`TabulatedCylindricalMirror` — wavelength-dependent reflectance from a
CSV lookup table (e.g. vendor coating curve).

The primitive is a ``Subtract(Cylinder_blank, Cylinder_cut)`` built by
the scene builder.  The cut cylinder's axis orientation determines which
plane gets curvature:

- **tangential** (default): axis along local y, curvature in xz plane.
- **sagittal**: axis along local x, curvature in yz plane.

Raysect delivers the true surface normal on the curved face; the
material does standard specular reflection off it.

Face discrimination: a hit is on the reflective cylindrical face when
its distance from the cut-cylinder axis equals R (within tolerance).
All other hits delegate to a Lambert diffuse reflector.

Convention (same as SphericalMirror):
- Local +z axis is the primitive's outward normal at the vertex.
- Rays enter from the +z half-space and reflect back toward +z.
- Cut cylinder centre at (0, 0, +R) in local frame.
"""


import math

from raysect.core.math import Vector3D
from raysect.optical import ConstantSF
from raysect.optical.material.lambert import Lambert

from optics.elements._reflect import specular_reflect
from optics.elements._tabulated import interp_table, load_two_column_csv
from raysect.optical.material import Material

_BACK_ALBEDO = 0.5
_CYL_HIT_TOL = 1e-6


def _is_cylinder_face_tangential(hit_point, R, axis_rotation_rad=0.0):
    """Cut axis along y — check distance in the xz plane."""
    x, y, z = hit_point.x, hit_point.y, hit_point.z
    if axis_rotation_rad != 0.0:
        c, s = math.cos(-axis_rotation_rad), math.sin(-axis_rotation_rad)
        x, y = x * c - y * s, x * s + y * c
    dist_sq = x * x + (z - R) * (z - R)
    return abs(dist_sq - R * R) < _CYL_HIT_TOL * R


def _is_cylinder_face_sagittal(hit_point, R, axis_rotation_rad=0.0):
    """Cut axis along x — check distance in the yz plane."""
    x, y, z = hit_point.x, hit_point.y, hit_point.z
    if axis_rotation_rad != 0.0:
        c, s = math.cos(-axis_rotation_rad), math.sin(-axis_rotation_rad)
        x, y = x * c - y * s, x * s + y * c
    dist_sq = y * y + (z - R) * (z - R)
    return abs(dist_sq - R * R) < _CYL_HIT_TOL * R


def _reflect_off_cylinder_face(hit_point, normal, d_local,
                               outside_point, primitive_to_world, ray, world):
    """Specular reflection using the true cylinder normal from raysect."""
    d_out_local = specular_reflect(d_local, normal)
    origin_world = outside_point.transform(primitive_to_world)
    d_out_world = d_out_local.transform(primitive_to_world).normalise()
    return ray.spawn_daughter(origin_world, d_out_world).trace(world)


class CylindricalMirror(Material):
    """Concave cylindrical mirror with flat (scalar) reflectance.

    Parameters
    ----------
    focal_length_m : float
        Mirror focal length in metres (curved axis).  Must be positive.
    reflectance : float
        Fraction of incident power reflected (0–1).  Default 1.0.
    orientation : str
        ``"tangential"`` (curvature in xz) or ``"sagittal"`` (curvature
        in yz).  Must match the CSG primitive orientation.
    back_albedo : float
        Lambertian albedo for non-reflective faces.
    """

    def __init__(self, focal_length_m: float, reflectance: float = 1.0,
                 orientation: str = "tangential",
                 back_albedo: float = _BACK_ALBEDO,
                 axis_rotation_deg: float = 0.0):
        super().__init__()
        if focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive (concave)")
        if not 0.0 < reflectance <= 1.0:
            raise ValueError("reflectance must be in (0, 1]")
        if orientation not in ("tangential", "sagittal"):
            raise ValueError(f"orientation must be 'tangential' or 'sagittal', got {orientation!r}")
        self._f = float(focal_length_m)
        self._R = 2.0 * self._f
        self._reflectance = float(reflectance)
        self._axis_rotation_rad = math.radians(axis_rotation_deg)
        self._is_face_fn = (_is_cylinder_face_sagittal if orientation == "sagittal"
                            else _is_cylinder_face_tangential)
        self._back = Lambert(ConstantSF(back_albedo))

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not self._is_face_fn(hit_point, self._R, self._axis_rotation_rad):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_cylinder_face(
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


class TabulatedCylindricalMirror(Material):
    """Concave cylindrical mirror with wavelength-dependent reflectance.

    Parameters
    ----------
    focal_length_m : float
        Mirror focal length in metres (curved axis).  Must be positive.
    reflectance_wavelengths_nm : sequence of float
        Wavelength sample points (nm), sorted ascending.
    reflectance_values : sequence of float
        Reflectance at each sample point (fraction, 0–1).
    orientation : str
        ``"tangential"`` or ``"sagittal"``.
    back_albedo : float
        Lambertian albedo for non-reflective faces.
    """

    def __init__(
        self,
        focal_length_m: float,
        reflectance_wavelengths_nm,
        reflectance_values,
        orientation: str = "tangential",
        back_albedo: float = _BACK_ALBEDO,
        axis_rotation_deg: float = 0.0,
    ):
        super().__init__()
        if focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive (concave)")
        if orientation not in ("tangential", "sagittal"):
            raise ValueError(f"orientation must be 'tangential' or 'sagittal', got {orientation!r}")
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
        self._axis_rotation_rad = math.radians(axis_rotation_deg)
        self._is_face_fn = (_is_cylinder_face_sagittal if orientation == "sagittal"
                            else _is_cylinder_face_tangential)
        self._back = Lambert(ConstantSF(back_albedo))

    @staticmethod
    def from_csv(focal_length_m: float, csv_path: str,
                 orientation: str = "tangential",
                 back_albedo: float = _BACK_ALBEDO,
                 axis_rotation_deg: float = 0.0):
        """Load from a two-column CSV (wavelength_nm, reflectance)."""
        lam, refl = load_two_column_csv(csv_path)
        return TabulatedCylindricalMirror(
            focal_length_m, lam, refl, orientation, back_albedo, axis_rotation_deg)

    def _interp(self, wavelength_nm: float) -> float:
        return interp_table(wavelength_nm, self._lam_table, self._refl_table)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not self._is_face_fn(hit_point, self._R, self._axis_rotation_rad):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_cylinder_face(
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
