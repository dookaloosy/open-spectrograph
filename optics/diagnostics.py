"""Winner-characterisation diagnostics (backward trace).

These functions use raysect's backward observer pipeline for
measurements that require physically correct power accounting
(absolute flux, spectral rejection). Not used in the GA evaluation
loop — the GA uses forward trace exclusively.
"""


from optics.scene import Scene
from optics.grating_math import (
    analytical_bandpass_nm as _analytical_bandpass_nm,
)


def stray_light_test(
    scene: Scene,
    genome,
    parts,
    *,
    wavelength_nm: float,
    n_bandpasses: int = 8,
    backward_spp: int = 10_000_000,
    target_fnum: float,
) -> dict[str, object]:
    """Industry-standard stray-light test (HORIBA / Jobin Yvon protocol).

    Measure flux at the test wavelength lambda_0, then at lambda_0 +/- N*bandpass.
    The stray-light ratio is I(off) / I(on).  A typical measurement
    uses N = 8 bandpasses offset from a laser line.

    Parameters
    ----------
    scene, genome, parts :
        Full assembled Scene (optics + mounts + housing) — caller must
        use ``czerny_assembly.assemble_scene()`` to populate housing
        before calling this.
    wavelength_nm :
        Test wavelength (the "laser line"). Defaults to mid-band.
    n_bandpasses :
        Number of bandpasses to offset for the off-band measurement.
    backward_spp :
        Samples per pixel for each backward trace.

    Returns
    -------
    dict
        ``wavelength_nm``       : test wavelength
        ``bandpass_nm``         : geometric bandpass S_lambda = R_D * W
        ``flux_on_watts``       : flux at lambda_0 (grating tuned to lambda_0)
        ``flux_off_plus_watts`` : flux at lambda_0 (grating tuned to lambda_0 + N*S_lambda)
        ``flux_off_minus_watts``: flux at lambda_0 (grating tuned to lambda_0 - N*S_lambda)
        ``stray_ratio_plus``    : I(off+) / I(on)
        ``stray_ratio_minus``   : I(off-) / I(on)
        ``stray_ratio``         : average of plus and minus ratios
    """
    from optics.world_builder import build_world, attach_observer
    from optics.grating_math import grating_rotation_deg
    from optics._raysect_wrapper import safe_observe

    bandpass_nm = _analytical_bandpass_nm(
        parts.grating_groove_density_per_mm, parts.m2_focal_length_mm,
        parts.slit_width_um, wavelength_nm)

    offset_nm = n_bandpasses * bandpass_nm

    built = build_world(scene, input_fnum=target_fnum)
    attach_observer(built, scene, target_fnum)
    built.observer.pixel_samples = backward_spp
    built.observer.min_wavelength = wavelength_nm - 0.5
    built.observer.max_wavelength = wavelength_nm + 0.5
    built.observer.spectral_bins = 1

    def _measure_flux(grating_tune_nm: float) -> float:
        theta = grating_rotation_deg(genome.alpha_deg, genome.beta_deg,
                                     parts.grating_groove_density_per_mm, grating_tune_nm)
        built.set_grating_rotation_deg(theta)
        safe_observe(built.observer)
        return float(built.observer_pipeline.value.mean)

    flux_on = _measure_flux(wavelength_nm)
    flux_off_plus = _measure_flux(wavelength_nm + offset_nm)
    flux_off_minus = _measure_flux(wavelength_nm - offset_nm)

    stray_plus = flux_off_plus / flux_on if flux_on > 0 else float("nan")
    stray_minus = flux_off_minus / flux_on if flux_on > 0 else float("nan")
    stray_avg = 0.5 * (stray_plus + stray_minus) if flux_on > 0 else float("nan")

    return {
        "wavelength_nm": wavelength_nm,
        "bandpass_nm": bandpass_nm,
        "offset_nm": offset_nm,
        "n_bandpasses": n_bandpasses,
        "flux_on_watts": flux_on,
        "flux_off_plus_watts": flux_off_plus,
        "flux_off_minus_watts": flux_off_minus,
        "stray_ratio_plus": stray_plus,
        "stray_ratio_minus": stray_minus,
        "stray_ratio": stray_avg,
    }
