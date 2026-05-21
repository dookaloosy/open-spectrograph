"""Raw-metrics dict for a Czerny-Turner `Scene`.

Every number here is a *raw* measurement — no weighting, no
no weighting or fitness combining. The fitness function (``fitness.py``)
decides how to combine them into a single score and which hard gates
to enforce.

Metrics come from forward trace (``forward_trace_metrics`` in
``optics/forward_trace.py``): geometric throughput (hits × material
reflectance product) and ILF FWHM (hit distribution binned at 8 um
pixel pitch).

This module must not encode build-specific constants. Signatures take
a parts object.
"""


import math

from optics.scene import Scene
from optics.grating_math import (
    analytical_bandpass_nm as _analytical_bandpass_nm_raw,
    analytical_dispersion_nm_per_mm as _analytical_dispersion_nm_per_mm_raw,
)


def raw_metrics(
    scene: Scene,
    genome,
    parts,
    *,
    fitness_wavelengths_nm: tuple[float, ...],
    target_fnum: float,
    design_wavelength_nm: float,
    forward_rays: int,
    point_source: bool = False,
) -> dict[str, float]:
    """Compute raw metrics for a Czerny-Turner scene.

    Metrics come from two sources:

    1. **Geometry/BOM** (always, no trace): `f_number`, `dispersion`,
       `bandpass`, `LGC`, `étendue`, `footprint`. These are analytical
       closed-form quantities from the grating equation and BOM.

    2. **Forward trace**: RMS spot radius, throughput, ILF FWHM at each
       fitness wavelength via ``forward_trace_metrics``.
    """
    lam_mid = design_wavelength_nm

    # ── 1. Geometry/BOM metrics (no trace) ────────────────────────────

    # Report f/# using M1 (collection-defining mirror). Commercial
    # spectrometer datasheets quote the camera mirror's f/# instead;
    # adding a separate m2 f/# would be trivial if wanted.
    f_number = parts.m1_focal_length_mm / parts.m1_diameter_mm
    dispersion_nm_per_mm = _analytical_dispersion_nm_per_mm_raw(
        parts.grating_groove_density_per_mm, parts.m2_focal_length_mm, lam_mid)
    bandpass_nm = _analytical_bandpass_nm_raw(
        parts.grating_groove_density_per_mm, parts.m2_focal_length_mm,
        parts.slit_width_um, lam_mid)

    lgc = (parts.slit_height_mm
           / (f_number ** 2 * dispersion_nm_per_mm))

    omega_sr = math.pi / (4.0 * f_number ** 2)
    slit_area_mm2 = (parts.slit_width_um * 1.0e-3) * parts.slit_height_mm
    geometric_extent = slit_area_mm2 * omega_sr

    # BOM optical throughput ceiling at the grating's peak efficiency:
    # T_bom = R_M1 × E_peak × R_M2, where E_peak is the absolute
    # measured grating efficiency at the blaze wavelength (already
    # includes the grating's own Al-coating reflectance — no
    # double-counting with the mirror R values).
    R1 = parts.m1_reflectance
    R2 = parts.m2_reflectance
    E = parts.grating_efficiency
    nominal_throughput = R1 * E * R2

    xs = [el.position[0] for el in scene.elements]
    ys = [el.position[1] for el in scene.elements]
    footprint_width_mm = float(max(xs) - min(xs))
    footprint_height_mm = float(max(ys) - min(ys))

    result: dict[str, float] = {
        "f_number": f_number,
        "dispersion_nm_per_mm": dispersion_nm_per_mm,
        "bandpass_nm": bandpass_nm,
        "light_gathering_capacity": lgc,
        "geometric_extent_mm2_sr": geometric_extent,
        "nominal_throughput": nominal_throughput,
        "footprint_width_mm": footprint_width_mm,
        "footprint_height_mm": footprint_height_mm,
    }

    # ── 2. Forward trace ────────────────────────────────────────────────
    from optics.forward_trace import forward_trace_metrics

    fwd = forward_trace_metrics(
        scene, genome, parts,
        wavelengths_nm=tuple(float(l) for l in fitness_wavelengths_nm),
        design_wavelength_nm=design_wavelength_nm,
        n_rays=forward_rays,
        input_fnum=parts.m1_focal_length_mm / parts.m1_diameter_mm,
        target_fnum=target_fnum,
        seed=42,
        point_source=point_source,
    )
    result.update(fwd)
    return result
