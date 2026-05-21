"""Reflective diffraction grating materials.

Four classes, one base + three efficiency models:

`_GratingBase` — shared grating equation, importance sampling, and ray
dispatch. Subclasses override `_order_efficiencies()` only.

`BlazedGrating` — scalar Kirchhoff sinc² blaze envelope. Wavelength-
and angle-dependent per-order efficiency. Fallback when no measured
data is available.

`TabulatedBlazedGrating` — linearly-interpolated measured efficiency
from CSV files. Production material for all BOM gratings with
`efficiency_file`.

All implement the grating equation in vector form for reflection:

    sin θ_out = sin θ_in − m · λ / d

Convention for the grating primitive's local frame:
  - local +z is the surface normal (outward)
  - local +y is the groove direction
  - local +x is the dispersion axis (perpendicular to grooves, in-surface)

Rotating the grating (the wavelength-scan DoF) is done by re-posing the
primitive in the world, not by touching this material. The material
itself is stateless and trivially reusable across wavelengths.
"""


import math
import random

# SystemRandom draws from /dev/urandom — no shared state across
# MulticoreEngine fork'd workers (stdlib random.random() inherits
# the parent's Mersenne Twister state, producing correlated samples).
_rng = random.SystemRandom()

from optics.elements._tabulated import interp_table, load_two_column_csv

from raysect.core.math import Vector3D
from raysect.optical import ConstantSF
from raysect.optical.material import Material
from raysect.optical.material.lambert import Lambert


# Default per-order peak scaling relative to the m=+1 efficiency.
# Used by BlazedGrating and TabulatedBlazedGrating. Values are
# absolute power fractions that INCLUDE substrate reflectance (the
# tabulated m=+1 data from vendor charts is measured with the
# aluminum coating in place, so the coating loss is already folded
# into the numbers; the scaling factors for other orders are
# calibrated to be consistent with that convention).
_DEFAULT_ORDER_SCALES: dict[int, float] = {
    +1: 1.00,    # design order — tabulated vendor efficiency
     0: 0.00,    # specular (zeroed — no validated model yet)
    -1: 0.00,    # wrong-side first order (zeroed)
    +2: 0.00,    # second order (zeroed)
    -2: 0.00,    # wrong-side second order (zeroed)
}


_GRATING_BACK_ALBEDO = 0.5


class _GratingBase(Material):
    """Base class for reflective diffraction gratings.

    Implements the grating equation, evanescent-order filtering,
    importance sampling, and daughter-ray dispatch. Subclasses provide
    per-order efficiency via `_order_efficiencies()`.

    Only the front face (local +z) diffracts; back face, edges, and
    any ``exiting`` hit delegate to a Lambert diffuse reflector
    (frosted glass substrate, albedo 0.5).
    """

    def __init__(self, groove_density_per_mm: float,
                 back_albedo: float = _GRATING_BACK_ALBEDO):
        super().__init__()
        if groove_density_per_mm <= 0:
            raise ValueError("groove_density_per_mm must be positive")
        self._groove_period_nm = 1.0e6 / groove_density_per_mm
        self._back = Lambert(ConstantSF(back_albedo))

    def _order_efficiencies(
        self,
        sin_theta_in: float,
        cos_theta_in: float,
        wavelength_nm: float,
    ) -> list[tuple[int, float]]:
        """Return [(order, efficiency), ...] for non-evanescent orders.

        Subclasses must override. Each (m, η) pair must have η > 0.
        Evanescent filtering is done by the caller — subclasses should
        return all orders they model; the base evaluate_surface skips
        any with |sin_theta_out| > 1.
        """
        raise NotImplementedError

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting or abs(1.0 - normal.z) > 1e-6:
            return self._back.evaluate_surface(
                world, ray, primitive, hit_point, exiting,
                inside_point, outside_point, normal,
                world_to_primitive, primitive_to_world, intersection)
        # Move the incoming direction into the grating's local frame.
        d_in_local = ray.direction.transform(world_to_primitive).normalise()
        dx, dy, dz = d_in_local.x, d_in_local.y, d_in_local.z

        # Only the component perpendicular to the grooves (the x-axis and
        # z-axis in local frame) is affected by diffraction. The y component
        # (along the grooves) is preserved.
        in_plane_mag2 = 1.0 - dy * dy
        if in_plane_mag2 <= 0.0:
            return ray.new_spectrum()
        in_plane_mag = math.sqrt(in_plane_mag2)
        sin_theta_in = dx / in_plane_mag
        cos_theta_in = math.sqrt(max(0.0, 1.0 - sin_theta_in * sin_theta_in))

        # Single-wavelength ray: use midpoint of its spectral range.
        wavelength_nm = 0.5 * (ray.min_wavelength + ray.max_wavelength)
        inv_d = 1.0 / self._groove_period_nm

        # Get per-order efficiencies from the subclass, then filter
        # evanescent orders and build the importance-sampling pool.
        order_etas = self._order_efficiencies(
            sin_theta_in, cos_theta_in, wavelength_nm,
        )

        active_sin_out = []
        active_eta = []
        for m, eta in order_etas:
            s_out = sin_theta_in - m * wavelength_nm * inv_d
            if abs(s_out) > 1.0:
                continue
            if eta <= 0.0:
                continue
            active_sin_out.append(s_out)
            active_eta.append(eta)

        if not active_eta:
            return ray.new_spectrum()

        # Importance-sample one order: p_m = η_m / Σ_active.
        # Weight = η_m / p_m = Σ_active (constant, unbiased estimator).
        total_eta = sum(active_eta)
        r = _rng.random() * total_eta
        cumsum = 0.0
        idx = len(active_eta) - 1
        for i, eta in enumerate(active_eta):
            cumsum += eta
            if r < cumsum:
                idx = i
                break

        sin_theta_out = active_sin_out[idx]
        cos_theta_out = math.sqrt(1.0 - sin_theta_out * sin_theta_out)

        # Rebuild d_out in local frame. For reflection, the outgoing ray
        # leaves on the +z side (cos_theta_out > 0 by construction), while
        # the incoming ray was on the −z side (dz < 0). The y component is
        # conserved; x component becomes sin_theta_out * in_plane_mag.
        dx_out = sin_theta_out * in_plane_mag
        dy_out = dy
        dz_out = cos_theta_out * in_plane_mag

        d_out_local = Vector3D(dx_out, dy_out, dz_out).normalise()

        origin_world = outside_point.transform(primitive_to_world)
        d_out_world = d_out_local.transform(primitive_to_world).normalise()

        daughter = ray.spawn_daughter(origin_world, d_out_world)
        result = daughter.trace(world)
        result.samples[:] *= total_eta
        return result

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point, world_to_primitive, primitive_to_world):
        return spectrum


# ─── Multi-order gratings (subclasses of _GratingBase) ───────────────────────

class BlazedGrating(_GratingBase):
    """Reflective blazed grating with scalar-diffraction efficiency.

    Uses the Kirchhoff (scalar) approximation (Palmer, *Diffraction Grating
    Handbook*, ch. 9):

        η_m(α_i, α_d, λ) = ρ_peak · sinc²(π · d · cos(γ_B) / λ ·
                                          [sin(α_i − γ_B) + sin(α_d − γ_B)])

    Peak efficiency is obtained when α_i − γ_B = −(α_d − γ_B), i.e. the
    ray enters and diffracts symmetrically about the facet normal.

    Parameters
    ----------
    groove_density_per_mm : float
    blaze_angle_deg : float
        Blaze angle γ_B in degrees.
    peak_efficiency : float
        Peak Littrow-at-blaze absolute efficiency for m=+1 (~0.80).
    order_peak_scales : dict[int, float] | None
        Per-order peak scaling relative to ``peak_efficiency``.
    """

    def __init__(
        self,
        groove_density_per_mm: float,
        blaze_angle_deg: float,
        peak_efficiency: float = 0.80,
        order_peak_scales: dict[int, float] | None = None,
    ):
        super().__init__(groove_density_per_mm)
        if not 0.0 < peak_efficiency <= 1.0:
            raise ValueError("peak_efficiency must be in (0, 1]")
        self._blaze_angle_rad = math.radians(blaze_angle_deg)
        self._cos_blaze = math.cos(self._blaze_angle_rad)
        self._sin_blaze = math.sin(self._blaze_angle_rad)
        self._facet_width_nm = self._groove_period_nm * self._cos_blaze
        scales = order_peak_scales or _DEFAULT_ORDER_SCALES
        self._orders = list(scales.keys())
        self._order_peaks = [peak_efficiency * scales[m] for m in self._orders]

    def _blaze_envelope(self, sin_in, cos_in, sin_out, cos_out, wavelength_nm):
        """sinc²(π · b / λ · [sin(α_i − γ_B) + sin(α_d − γ_B)]) in [0, 1]."""
        cB, sB = self._cos_blaze, self._sin_blaze
        sin_in_minus_b = sin_in * cB - cos_in * sB
        sin_out_minus_b = sin_out * cB - cos_out * sB
        phase = math.pi * self._facet_width_nm / wavelength_nm * (
            sin_in_minus_b + sin_out_minus_b
        )
        if abs(phase) < 1e-10:
            return 1.0
        s = math.sin(phase) / phase
        return s * s

    def _order_efficiencies(self, sin_theta_in, cos_theta_in, wavelength_nm):
        inv_d = 1.0 / self._groove_period_nm
        result = []
        for m, peak in zip(self._orders, self._order_peaks):
            s_out = sin_theta_in - m * wavelength_nm * inv_d
            if abs(s_out) > 1.0:
                continue
            if m == 0:
                eta = peak
            else:
                c_out = math.sqrt(1.0 - s_out * s_out)
                eta = peak * self._blaze_envelope(
                    sin_theta_in, cos_theta_in, s_out, c_out, wavelength_nm,
                )
            if eta > 0.0:
                result.append((m, eta))
        return result


class TabulatedBlazedGrating(_GratingBase):
    """Reflective blazed grating with measured (tabulated) efficiency.

    Replaces `BlazedGrating`'s sinc² scalar envelope with a linearly-
    interpolated lookup table of average-polarisation absolute efficiency
    vs. wavelength, digitised from vendor data sheets.

    The tabulated data is for m=+1 in Littrow configuration. Angular
    dependence (non-Littrow operation) is a second-order effect at
    the deviation angles this instrument operates at (dev ≈ 30-35°)
    and is currently ignored.

    Parameters
    ----------
    groove_density_per_mm : float
    efficiency_wavelengths_nm : sequence of float
        Wavelength sample points (nm), sorted ascending.
    efficiency_values : sequence of float
        Absolute average-polarisation m=+1 efficiency at each sample
        point (fraction, 0-1).
    order_peak_scales : dict[int, float] | None
        Per-order scaling relative to m=+1 table value.
    """

    def __init__(
        self,
        groove_density_per_mm: float,
        efficiency_wavelengths_nm,
        efficiency_values,
        order_peak_scales: dict[int, float] | None = None,
    ):
        super().__init__(groove_density_per_mm)
        lam = list(efficiency_wavelengths_nm)
        eff = list(efficiency_values)
        if len(lam) != len(eff):
            raise ValueError("wavelength and efficiency arrays must match")
        if len(lam) < 2:
            raise ValueError("need at least 2 points for interpolation")
        self._lam_table = lam
        self._eff_table = eff
        scales = order_peak_scales or _DEFAULT_ORDER_SCALES
        self._orders = list(scales.keys())
        self._order_scales = [scales[m] for m in self._orders]

    @staticmethod
    def from_csv(groove_density_per_mm: float, csv_path: str,
                 order_peak_scales=None):
        """Load efficiency table from a two-column CSV (wavelength_nm, efficiency)."""
        lam, eff = load_two_column_csv(csv_path)
        return TabulatedBlazedGrating(
            groove_density_per_mm, lam, eff,
            order_peak_scales=order_peak_scales,
        )

    def _interp(self, wavelength_nm: float) -> float:
        """Linearly interpolate the m=+1 efficiency table."""
        return interp_table(wavelength_nm, self._lam_table, self._eff_table)

    def _order_efficiencies(self, sin_theta_in, cos_theta_in, wavelength_nm):
        eta_m1 = self._interp(wavelength_nm)
        return [(m, eta_m1 * scale)
                for m, scale in zip(self._orders, self._order_scales)]
