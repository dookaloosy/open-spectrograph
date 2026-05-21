"""Diffraction grating equation helpers.

Pure math — no raysect, no scene types. Takes raw scalars (angles,
groove density, wavelength, focal length, slit width) and returns
derived quantities.
"""

from math import acos, atan2, cos, degrees, hypot, radians, sin, sqrt


def grating_rotation_deg(
    alpha_deg: float,
    beta_deg: float,
    groove_density_per_mm: float,
    wavelength_nm: float,
    diffraction_order: int = -1,
) -> float:
    """Return the grating rotation angle (degrees) for a given wavelength.

    Accepts signed angles (α incidence, β diffraction). Internally
    uses unsigned |β| for the physical grating rotation computation.
    """
    from math import pi

    alpha_rad = radians(alpha_deg)
    beta_unsigned_rad = radians(abs(beta_deg))
    groove_period_nm = 1.0e6 / groove_density_per_mm

    sin_a = sin(alpha_rad)
    sin_b = sin(beta_unsigned_rad)
    cos_a = cos(alpha_rad)
    cos_b = cos(beta_unsigned_rad)

    A = sin_b - sin_a
    B = cos_a + cos_b
    R = hypot(A, B)
    psi = atan2(B, A)

    target = abs(diffraction_order) * wavelength_nm / (groove_period_nm * R)
    if abs(target) > 1.0:
        lambda_max_nm = groove_period_nm * R / abs(diffraction_order)
        raise ValueError(
            f"wavelength {wavelength_nm} nm exceeds the geometry's reach "
            f"(lambda_max = {lambda_max_nm:.0f} nm at "
            f"alpha={alpha_deg:.2f} deg, beta={beta_deg:.2f} deg, "
            f"order={diffraction_order})"
        )
    theta_rad = acos(target) - psi
    while theta_rad > pi:
        theta_rad -= 2.0 * pi
    while theta_rad <= -pi:
        theta_rad += 2.0 * pi
    return degrees(theta_rad)


def analytical_dispersion_nm_per_mm(
    groove_density_per_mm: float,
    focal_length_mm: float,
    wavelength_nm: float,
) -> float:
    """Reciprocal linear dispersion R_D (nm/mm) from the grating equation.

    R_D = 1e6 * cos(beta) / (|m| * n * F)

    where n is groove density (grooves/mm), F is focal length (mm), and
    beta is the diffraction angle (Littrow approximation for |β|).
    """
    sin_beta = wavelength_nm * groove_density_per_mm / 1.0e6
    if sin_beta >= 1.0:
        raise ValueError(
            f"evanescent: sin(beta)={sin_beta:.4f} at {wavelength_nm}nm "
            f"with {groove_density_per_mm} g/mm")
    cos_beta = sqrt(1.0 - sin_beta * sin_beta)
    return 1.0e6 * cos_beta / (groove_density_per_mm * focal_length_mm)


def analytical_bandpass_nm(
    groove_density_per_mm: float,
    focal_length_mm: float,
    slit_width_um: float,
    wavelength_nm: float,
) -> float:
    """Geometric bandpass S_lambda = R_D * W_slit (nm)."""
    R_D = analytical_dispersion_nm_per_mm(
        groove_density_per_mm, focal_length_mm, wavelength_nm)
    return R_D * slit_width_um * 1.0e-3


def wavelength_pixel_position(
    wavelength_nm: float,
    design_wavelength_nm: float,
    groove_density_per_mm: float,
    focal_length_mm: float,
    n_pixels: int = 3648,
    pixel_pitch_mm: float = 0.008,
) -> float:
    """Pixel index where *wavelength_nm* lands on a fixed-grating CCD.

    Returns a float. Values outside [0, n_pixels) mean off-detector.
    """
    R_D = analytical_dispersion_nm_per_mm(
        groove_density_per_mm, focal_length_mm, design_wavelength_nm)
    centre_px = n_pixels // 2
    # Negate: shorter wavelengths diffract to higher pixel indices
    # in the detector's local frame (local +x opposes the grating
    # equation's sign convention for beta).
    offset_mm = -(wavelength_nm - design_wavelength_nm) / R_D
    return centre_px + offset_mm / pixel_pitch_mm


def wavelengths_on_detector(
    wavelengths_nm: tuple[float, ...],
    design_wavelength_nm: float,
    groove_density_per_mm: float,
    focal_length_mm: float,
    n_pixels: int = 3648,
    pixel_pitch_mm: float = 0.008,
    margin_px: int = 10,
) -> bool:
    """True if every wavelength lands within the CCD active area."""
    for lam in wavelengths_nm:
        px = wavelength_pixel_position(
            lam, design_wavelength_nm, groove_density_per_mm,
            focal_length_mm, n_pixels, pixel_pitch_mm)
        if px < margin_px or px > n_pixels - margin_px:
            return False
    return True


