"""Off-axis paraboloidal mirror materials for raysect CSG.

The OAP is modelled as a cylinder blank with the concave paraboloid
carved out: ``Subtract(Cylinder, Parabola)``, built by the scene
builder. Raysect delivers the true surface normal at each hit point,
so reflection is the standard specular formula.

Face discrimination: a hit on the reflective paraboloid face satisfies
the paraboloid equation (within tolerance). All other hits (barrel,
back face, exiting) use a Lambert diffuse reflector (black-anodized
substrate, albedo 0.05).

Paraboloid equation in the cylinder's local frame::

    (x - a)² + y² = 4f(z - v)

where:
    f = parent focal length = RFL × cos²(OAA/2)
    a = parent axis offset in x = 2f × tan(OAA/2)
    v = vertex z-position = focus_height - f

    RFL = reflected focal length (from BOM focal_length_mm)
    OAA = off-axis angle in vendor convention (90° for a 90° OAP;
          the incidence angle at the surface is OAA/2 = 45°)
    focus_height = height of the parent focal point above the
                   cylinder base (from BOM focus_height_mm)

Local frame convention:
    +z = cylinder axis (toward grating)
    +x = toward the parent parabola axis
    Rays enter from the +x half-space and reflect toward +z.
"""


import math

from raysect.core.math import Vector3D
from raysect.optical import ConstantSF
from raysect.optical.material.lambert import Lambert
from raysect.optical.material import Material

from optics.elements._reflect import specular_reflect
from optics.elements._tabulated import interp_table, load_two_column_csv

# Black-anodized aluminum substrate on barrel and back face.
_PARABOLOIDAL_BACK_ALBEDO = 0.05
# Tolerance in metres; effective surface-distance threshold is tol/2.
_PARAB_HIT_TOL = 1e-6


def _paraboloid_params(reflected_focal_length_m, off_axis_angle_deg,
                       focus_height_m):
    """Compute (a, f, v) for the paraboloid equation in the local frame.

    Parameters:
        reflected_focal_length_m: RFL in metres (BOM focal_length_mm).
        off_axis_angle_deg: vendor-convention OAP angle (e.g. 90° for
            a 90° OAP). Divided by 2 internally to get the incidence
            angle at the mirror surface.
        focus_height_m: height of the parent focal point above the
            cylinder base — equals BOM center_thickness_mm for 90° OAPs.

    Returns (a, f, v) in metres:
        f = RFL × cos²(OAA/2)        parent focal length
        a = 2f × tan(OAA/2)          parent axis x-offset
        v = focus_height - f          vertex z-position
    """
    phi = math.radians(off_axis_angle_deg / 2.0)
    cp = math.cos(phi)
    tp = math.tan(phi)

    f = reflected_focal_length_m * cp * cp
    a = 2.0 * f * tp
    v = focus_height_m - f

    return a, f, v


def _is_paraboloid_face(hit_point, pa, pf, pv):
    """True when the hit lies on the paraboloid surface (within tolerance)."""
    x, y, z = hit_point.x, hit_point.y, hit_point.z
    residual = (x - pa) ** 2 + y * y - 4.0 * pf * (z - pv)
    return abs(residual) < _PARAB_HIT_TOL * pf


def _reflect_off_paraboloid_face(normal, d_local,
                                 outside_point, primitive_to_world, ray, world):
    """Specular reflection using the true paraboloid normal from raysect."""
    d_out_local = specular_reflect(d_local, normal)
    origin_world = outside_point.transform(primitive_to_world)
    d_out_world = d_out_local.transform(primitive_to_world).normalise()
    return ray.spawn_daughter(origin_world, d_out_world).trace(world)


class ParaboloidalMirror(Material):
    """Off-axis paraboloidal mirror with flat (scalar) reflectance.

    Parameters
    ----------
    reflected_focal_length_m : float
        Reflected focal length (RFL) in metres — the distance from the
        paraboloidal surface centre to the parent paraboloid's focus.
    off_axis_angle_deg : float
        Off-axis angle φ in degrees. For a 90° reflection paraboloidal mirror, φ = 45°.
    reflectance : float
        Fraction of incident power reflected (0–1). Default 1.0.
    back_albedo : float
        Lambertian albedo for the back face and barrel edge.
        Default 0.05 (black-anodized aluminum substrate).
    """

    def __init__(self, reflected_focal_length_m: float,
                 off_axis_angle_deg: float,
                 reflectance: float = 1.0,
                 back_albedo: float = _PARABOLOIDAL_BACK_ALBEDO,
                 *,
                 center_thickness_m: float):
        super().__init__()
        if reflected_focal_length_m <= 0:
            raise ValueError("reflected_focal_length_m must be positive")
        if off_axis_angle_deg <= 0 or off_axis_angle_deg > 90:
            raise ValueError("off_axis_angle_deg must be in (0, 90]")
        if not 0.0 < reflectance <= 1.0:
            raise ValueError("reflectance must be in (0, 1]")
        self._reflectance = float(reflectance)
        self._back = Lambert(ConstantSF(back_albedo))
        self._pa, self._pf, self._pv = _paraboloid_params(
            reflected_focal_length_m, off_axis_angle_deg,
            focus_height_m=center_thickness_m)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not _is_paraboloid_face(hit_point, self._pa, self._pf, self._pv):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_paraboloid_face(
            normal, d_local, outside_point,
            primitive_to_world, ray, world,
        )
        if self._reflectance < 1.0:
            result.samples[:] *= self._reflectance
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum


class TabulatedParaboloidalMirror(Material):
    """Off-axis paraboloidal mirror with wavelength-dependent reflectance.

    Parameters
    ----------
    reflected_focal_length_m : float
        Reflected focal length (RFL) in metres.
    off_axis_angle_deg : float
        Off-axis angle φ in degrees.
    reflectance_wavelengths_nm : sequence of float
        Wavelength sample points (nm), sorted ascending.
    reflectance_values : sequence of float
        Reflectance at each sample point (fraction, 0–1).
    back_albedo : float
        Lambertian albedo for the back face and barrel edge.
        Default 0.05 (black-anodized aluminum substrate).
    """

    def __init__(
        self,
        reflected_focal_length_m: float,
        off_axis_angle_deg: float,
        reflectance_wavelengths_nm,
        reflectance_values,
        back_albedo: float = _PARABOLOIDAL_BACK_ALBEDO,
        *,
        center_thickness_m: float,
    ):
        super().__init__()
        if reflected_focal_length_m <= 0:
            raise ValueError("reflected_focal_length_m must be positive")
        if off_axis_angle_deg <= 0 or off_axis_angle_deg > 90:
            raise ValueError("off_axis_angle_deg must be in (0, 90]")
        lam = list(reflectance_wavelengths_nm)
        refl = list(reflectance_values)
        if len(lam) != len(refl):
            raise ValueError("wavelength and reflectance arrays must match")
        if len(lam) < 2:
            raise ValueError("need at least 2 points for interpolation")
        self._lam_table = lam
        self._refl_table = refl
        self._back = Lambert(ConstantSF(back_albedo))
        self._pa, self._pf, self._pv = _paraboloid_params(
            reflected_focal_length_m, off_axis_angle_deg,
            focus_height_m=center_thickness_m)

    @staticmethod
    def from_csv(reflected_focal_length_m: float,
                 off_axis_angle_deg: float,
                 csv_path: str,
                 back_albedo: float = _PARABOLOIDAL_BACK_ALBEDO,
                 *,
                 center_thickness_m: float):
        """Load from a two-column CSV (wavelength_nm, reflectance)."""
        lam, refl = load_two_column_csv(csv_path)
        return TabulatedParaboloidalMirror(
            reflected_focal_length_m, off_axis_angle_deg,
            lam, refl, back_albedo,
            center_thickness_m=center_thickness_m)

    def _interp(self, wavelength_nm: float) -> float:
        return interp_table(wavelength_nm, self._lam_table, self._refl_table)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or not _is_paraboloid_face(hit_point, self._pa, self._pf, self._pv):
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)

        d_local = ray.direction.transform(world_to_primitive).normalise()
        result = _reflect_off_paraboloid_face(
            normal, d_local, outside_point,
            primitive_to_world, ray, world,
        )
        wavelength_nm = 0.5 * (ray.min_wavelength + ray.max_wavelength)
        result.samples[:] *= self._interp(wavelength_nm)
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum
