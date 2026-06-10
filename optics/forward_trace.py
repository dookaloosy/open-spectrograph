"""Forward-trace evaluation and beam footprint diagnostic.

Fires rays from the entrance slit through the full optical train and
records where they land at each element's plane. Uses oversized
recording planes so overfill beyond the physical aperture is visible.

``forward_trace_metrics()`` is the Tier 2 GA evaluation for
fixed-grating mode: geometric throughput (hits × material reflectance
product) and ILF FWHM (hit distribution at 8 um pixel pitch) at each
fitness wavelength.

Spot-diagram helpers (``make_cone_rays``, ``trace_rays``,
``overlay_big_recorder``, etc.) are shared between the GA evaluation
and the ``scripts/beam_profile.py`` diagnostic.
"""


import math
from pathlib import Path

import numpy as np
from raysect.core.math import Point3D, Vector3D, rotate_basis, translate
from raysect.optical import Ray
from raysect.primitive import Box

from optics.elements.hit_recorder import HitRecorder
from optics.world_builder import MM_TO_M, _orthogonal_up, _placement_transform

_OVERSIZE_MM = 20.0
def spot_decomposition(hx, hy):
    """Decompose 2D hit distribution into tangential/sagittal components.

    hx: hit positions along dispersion axis (mm)
    hy: hit positions along cross-dispersion axis (mm)

    Returns dict or None (if < 3 hits).
    """
    hx = np.asarray(hx, dtype=float)
    hy = np.asarray(hy, dtype=float)
    if len(hx) < 3:
        return None
    cx, cy = hx.mean(), hy.mean()
    dx, dy = hx - cx, hy - cy
    sigma_x = np.sqrt(np.mean(dx**2))
    sigma_y = np.sqrt(np.mean(dy**2))
    rms_spot = np.sqrt(sigma_x**2 + sigma_y**2)
    sag_tan_ratio = sigma_y / sigma_x if sigma_x > 1e-9 else float('inf')
    tan_skewness = np.mean(dx**3) / sigma_x**3 if sigma_x > 1e-9 else 0.0
    return {
        'centroid_x_mm': cx,
        'centroid_y_mm': cy,
        'sigma_x_mm': sigma_x,
        'sigma_y_mm': sigma_y,
        'rms_spot_mm': rms_spot,
        'sag_tan_ratio': sag_tan_ratio,
        'tan_skewness': tan_skewness,
    }


def make_cone_rays(slit_el, parts, n_rays, rng, *, input_fnum,
                   point_source=False):
    """Generate rays from the entrance slit into an f/#-matched cone."""
    sx, sy, sz = [c * 1e-3 for c in slit_el.position]
    snx, sny, snz = slit_el.axis

    half_angle = math.atan(1.0 / (2.0 * input_fnum))
    cos_min = math.cos(half_angle)

    forward = Vector3D(snx, sny, snz).normalise()
    up = Vector3D(0, 0, 1)
    right = forward.cross(up).normalise()
    up = right.cross(forward).normalise()

    half_w = 0.5 * parts.slit_width_um * 1e-6
    half_h = 0.5 * parts.slit_height_mm * 1e-3

    rays = []
    for _ in range(n_rays):
        if point_source:
            dw = 0.0
            dh = 0.0
        else:
            # Rejection-sample from circular fiber core.
            while True:
                dw = rng.uniform(-half_w, half_w)
                dh = rng.uniform(-half_h, half_h)
                if dw * dw / (half_w * half_w) + dh * dh / (half_h * half_h) <= 1.0:
                    break
        origin = Point3D(
            sx + snx * 1e-6 + dw * right.x + dh * up.x,
            sy + sny * 1e-6 + dw * right.y + dh * up.y,
            sz + snz * 1e-6 + dw * right.z + dh * up.z,
        )
        cos_theta = rng.uniform(cos_min, 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta**2)
        phi = rng.uniform(0, 2 * math.pi)
        ldx = sin_theta * math.cos(phi)
        ldy = sin_theta * math.sin(phi)
        ldz = cos_theta
        d_world = Vector3D(
            ldx * right.x + ldy * up.x + ldz * forward.x,
            ldx * right.y + ldy * up.y + ldz * forward.y,
            ldx * right.z + ldy * up.z + ldz * forward.z,
        ).normalise()
        rays.append((origin, d_world))
    return rays


def overlay_big_recorder(built, label, element):
    """Add an oversized transparent HitRecorder wrapping the optic body.

    The real primitive stays in the scene (keeps its optical behavior).
    A transparent recorder box encloses the entire optic so incoming
    rays are logged on entry and reflected rays exit the box (exiting=True)
    without being re-counted.

    The box origin sits at the optic's rim plane (or surface plane for
    flat elements).  It extends backward past the optic body by
    ``edge_thickness + margin`` and forward by ``margin``.

    Placement by mirror type:

    - **Spherical / cylindrical**: normal to the optical axis, offset
      forward by ``edge_thickness - center_thickness`` (the sag height
      from vertex to rim plane).

    - **Paraboloidal (OAP)**: normal to the bisector of the OAA
      (i.e. OAA/2 off the cylinder axis), shifted forward by
      ``sag * sin(OAA/2)`` from the optic centre.

    - **Flat / grating**: at the element surface (no sag offset).
    """
    recorder = HitRecorder()
    _margin = 1e-9
    d_mm = element.params.get("diameter_mm") or element.params.get("size_mm")
    half_m = d_mm * 0.5 * MM_TO_M + _margin
    et_mm = element.params["edge_thickness_mm"]
    depth_m = et_mm * MM_TO_M
    big = Box(
        lower=Point3D(-half_m, -half_m, -(depth_m + _margin)),
        upper=Point3D(+half_m, +half_m, _margin),
    )

    if element.params.get("mirror_type") == "paraboloidal":
        from optics.elements.oap_mirror import _paraboloid_params
        fl_m = element.params["focal_length_mm"] * MM_TO_M
        oaa = element.params["off_axis_angle_deg"]
        r_m = element.params["diameter_mm"] * 0.5 * MM_TO_M
        ct_m = element.params["center_thickness_mm"] * MM_TO_M
        pa, pf, pv = _paraboloid_params(fl_m, oaa, focus_height_m=ct_m)
        z_oc = pv + pa * pa / (4.0 * pf)
        z_rim_mid = 0.5 * (
            pv + (r_m - pa) ** 2 / (4.0 * pf)
            + pv + (r_m + pa) ** 2 / (4.0 * pf)
        )
        sag_m = z_rim_mid - z_oc
        phi = math.radians(oaa / 2.0)

        gd = element.axis
        fd = element.params["paraboloidal_focus_dir"]
        gx, gy, _ = gd
        fx, fy, _ = fd
        forward = Vector3D(gx, gy, 0.0).normalise()
        right = Vector3D(fx, fy, 0.0).normalise()
        cut_normal = Vector3D(
            math.sin(phi) * right.x + math.cos(phi) * forward.x,
            math.sin(phi) * right.y + math.cos(phi) * forward.y,
            0.0,
        ).normalise()
        cut_up = _orthogonal_up(cut_normal)
        x, y, z = element.position
        fwd_shift = sag_m * math.sin(phi)
        ox = x * MM_TO_M + fwd_shift * forward.x
        oy = y * MM_TO_M + fwd_shift * forward.y
        oz = 0.0
        big.transform = (
            translate(ox, oy, oz)
            * rotate_basis(cut_normal, cut_up)
        )
    else:
        et = element.params.get("edge_thickness_mm", 0.0)
        ct = element.params.get("center_thickness_mm", 0.0)
        sag_mm = et - ct
        if sag_mm > 0:
            nx, ny, nz = element.axis
            fwd = Vector3D(nx, ny, nz).normalise()
            up = _orthogonal_up(fwd)
            x, y, z = element.position
            sag_m = sag_mm * MM_TO_M
            big.transform = (
                translate(
                    x * MM_TO_M + sag_m * fwd.x,
                    y * MM_TO_M + sag_m * fwd.y,
                    z * MM_TO_M + sag_m * fwd.z,
                )
                * rotate_basis(fwd, up)
            )
        else:
            big.transform = _placement_transform(element)

    rec_label = label + "_rec"
    big.material = recorder
    big.name = rec_label
    big.parent = built.world
    built.primitives[rec_label] = big
    return recorder, rec_label


def trace_rays(rays, world, wavelength_nm):
    """Fire all rays into the world at a single wavelength."""
    for origin, d_world in rays:
        Ray(
            origin, d_world,
            min_wavelength=wavelength_nm - 0.5,
            max_wavelength=wavelength_nm + 0.5,
            max_depth=15,
        ).trace(world)


def aperture_overlay(element, parts):
    """Return (type, params) describing the element's physical aperture."""
    if element.kind == "mirror":
        r = element.params["diameter_mm"] / 2
        if element.params.get("mirror_type") == "paraboloidal":
            from optics.elements.oap_mirror import _paraboloid_params
            oaa = element.params["off_axis_angle_deg"]
            phi = math.radians(oaa / 2.0)  # vendor OAA → incidence angle
            fl_m = element.params["focal_length_mm"] * 1e-3
            ct_m = element.params["center_thickness_mm"] * 1e-3
            pa, pf, pv = _paraboloid_params(fl_m, oaa, focus_height_m=ct_m)
            z_oc = pv + pa * pa / (4.0 * pf)
            z_near = pv + (r * 1e-3 - pa) ** 2 / (4.0 * pf)
            z_far = pv + (r * 1e-3 + pa) ** 2 / (4.0 * pf)
            z_mid = 0.5 * (z_near + z_far)
            r_major = r / math.cos(phi)
            if element.label == "M1":
                cx = (z_mid - z_oc) * math.sin(phi) * 1e3
            else:
                cx = 0.0
            return "ellipse", (r, r_major, cx)
        return "circle", r
    if element.kind == "grating":
        half = element.params["size_mm"] / 2
        return "rect", half
    if element.kind == "detector":
        # TCD1304: 3648 px × 8 µm = 29.184 mm array, 200 µm pixel height
        hw = 0.5 * 29.184
        hh = 0.5 * 0.2
        return "rect_wh", (hw, hh)
    if element.kind == "slit":
        r = parts.slit_width_um * 1e-3 / 2
        return "circle", r
    return None, None


def is_inside_aperture(x, y, overlay_type, overlay_param):
    """Test whether (x, y) in mm falls inside the aperture shape."""
    if overlay_type == "circle":
        return math.sqrt(x**2 + y**2) <= overlay_param
    if overlay_type == "ellipse":
        r_minor, r_major = overlay_param[0], overlay_param[1]
        cx = overlay_param[2] if len(overlay_param) > 2 else 0.0
        return ((x - cx) / r_major) ** 2 + (y / r_minor) ** 2 <= 1.0
    if overlay_type == "rect":
        return abs(x) <= overlay_param and abs(y) <= overlay_param
    if overlay_type == "rect_wh":
        hw, hh = overlay_param
        return abs(x) <= hw and abs(y) <= hh
    return True


def forward_trace_metrics(
    scene,
    genome,
    parts,
    *,
    wavelengths_nm: tuple[float, ...],
    design_wavelength_nm: float,
    n_rays: int,
    input_fnum: float,
    target_fnum: float,
    seed: int = 42,
    filter_stray: bool = False,
    point_source: bool = False,
) -> dict[str, float]:
    """Tier 2 forward-trace evaluation for fixed-grating mode.

    Launches rays from the entrance slit through the fixed-grating optical
    train and records hits on an oversized detector plane.  Throughput is
    geometric (hits-in-aperture / launched × material reflectance product).
    ILF FWHM is extracted from the hit distribution binned at 8 um pixel
    pitch via half-max crossing interpolation.
    """
    from optics.world_builder import build_world, _placement_transform, MM_TO_M
    from optics.grating_math import grating_rotation_deg

    slit_el = next(e for e in scene.elements if e.label == "entrance_slit")
    det_el = next(e for e in scene.elements if e.kind == "detector")

    train_labels = [el.label for el in scene.elements
                    if el.label != "entrance_slit"]

    # Bounce count at the detector for stray-light filtering.
    if filter_stray:
        det_bounces = 0
        for lbl in train_labels:
            if lbl == det_el.label:
                break
            el = next(e for e in scene.elements if e.label == lbl)
            if el.kind in ("mirror", "grating"):
                det_bounces += 1
    else:
        det_bounces = None

    # Material throughput at the detector for each wavelength.
    material_factor: dict[float, float] = {}
    for wl in wavelengths_nm:
        cumul = 1.0
        for lbl in train_labels:
            el = next(e for e in scene.elements if e.label == lbl)
            cumul *= element_reflectance_at(el, wl, parts)
        material_factor[wl] = cumul

    # Detector aperture for in-aperture hit test.
    det_atype, det_aparam = aperture_overlay(det_el, parts)

    # Collected power (same normalisation as backward trace).
    theta_max = math.atan(1.0 / (2.0 * target_fnum))
    cos_theta = math.cos(theta_max)
    area_m2 = parts.slit_width_um * 1e-6 * parts.slit_height_mm * 1e-3
    p_collected = area_m2 * 2.0 * math.pi * (1.0 - cos_theta) * 1.0

    # Pixel pitch for ILF binning.
    pixel_pitch_mm = 0.008

    result: dict[str, float] = {}

    for wl in wavelengths_nm:
        built = build_world(scene, input_fnum=input_fnum)

        # Fix grating at design wavelength.
        theta = grating_rotation_deg(
            genome.alpha_deg, genome.beta_deg,
            parts.grating_groove_density_per_mm, design_wavelength_nm)
        built.set_grating_rotation_deg(theta)

        # Add oversized hit recorder at detector position.
        rec_half = _OVERSIZE_MM * 1e-3
        rec_box = Box(
            lower=Point3D(-rec_half, -rec_half, -1e-3),
            upper=Point3D(+rec_half, +rec_half, 0.0),
        )
        rec_box.transform = _placement_transform(det_el)
        recorder = HitRecorder()
        rec_box.material = recorder
        rec_box.parent = built.world
        rec_box.name = det_el.label
        built.primitives[det_el.label] = rec_box

        rng = np.random.default_rng(seed + int(wl))
        rays = make_cone_rays(slit_el, parts, n_rays, rng,
                              input_fnum=input_fnum,
                              point_source=point_source)
        trace_rays(rays, built.world, wl)

        w2p = rec_box.transform.inverse()
        hx_list, hy_list = [], []
        for h in recorder.hits:
            if det_bounces is not None and h.ray_depth != det_bounces:
                continue
            lp = Point3D(*h.point).transform(w2p)
            hx_list.append(lp.x * 1000)
            hy_list.append(lp.y * 1000)
        hx = np.array(hx_list)
        hy = np.array(hy_list)

        n_hits = len(hx)
        if n_hits == 0:
            result[f"flux_{int(wl)}_nm"] = 0.0
            result[f"throughput_{int(wl)}_nm"] = 0.0
            result[f"ilf_fwhm_{int(wl)}_nm"] = float('nan')
            result[f"rms_spot_{int(wl)}_um"] = float('nan')
            result[f"sigma_x_{int(wl)}_um"] = float('nan')
            result[f"sigma_y_{int(wl)}_um"] = float('nan')
            result[f"sag_tan_ratio_{int(wl)}"] = float('nan')
            result[f"tan_skewness_{int(wl)}"] = float('nan')
            continue

        n_inside = sum(
            1 for x, y in zip(hx, hy)
            if is_inside_aperture(x, y, det_atype, det_aparam)
        )

        geo_throughput = n_inside / n_rays
        throughput = geo_throughput * material_factor[wl]
        flux = throughput * p_collected

        result[f"flux_{int(wl)}_nm"] = flux
        result[f"throughput_{int(wl)}_nm"] = throughput

        # 2D spot decomposition (all hits, not aperture-filtered).
        sd = spot_decomposition(hx, hy)
        if sd:
            result[f"rms_spot_{int(wl)}_um"] = sd['rms_spot_mm'] * 1000
            result[f"sigma_x_{int(wl)}_um"] = sd['sigma_x_mm'] * 1000
            result[f"sigma_y_{int(wl)}_um"] = sd['sigma_y_mm'] * 1000
            result[f"sag_tan_ratio_{int(wl)}"] = sd['sag_tan_ratio']
            result[f"tan_skewness_{int(wl)}"] = sd['tan_skewness']
        else:
            result[f"rms_spot_{int(wl)}_um"] = float('nan')
            result[f"sigma_x_{int(wl)}_um"] = float('nan')
            result[f"sigma_y_{int(wl)}_um"] = float('nan')
            result[f"sag_tan_ratio_{int(wl)}"] = float('nan')
            result[f"tan_skewness_{int(wl)}"] = float('nan')

        # ILF from hit distribution along dispersion axis.
        # Uses encircled-energy width: the narrowest contiguous window
        # of pixel bins containing ≥76% of hits (equivalent to a
        # Gaussian FWHM). Robust against split spots and skew tails.
        if n_inside < 3:
            result[f"ilf_fwhm_{int(wl)}_nm"] = float('nan')
        else:
            from optics.grating_math import analytical_dispersion_nm_per_mm
            R_D = analytical_dispersion_nm_per_mm(
                parts.grating_groove_density_per_mm,
                parts.m2_focal_length_mm, wl)
            sorted_hx = np.sort(hx[np.isfinite(hx)])
            n = len(sorted_hx)
            if n < 3:
                result[f"ilf_fwhm_{int(wl)}_nm"] = float('nan')
            else:
                target = int(math.ceil(n * 0.7655))
                target = min(target, n)
                best_width = sorted_hx[-1] - sorted_hx[0]
                for i in range(n - target + 1):
                    w = sorted_hx[i + target - 1] - sorted_hx[i]
                    if w < best_width:
                        best_width = w
                ee_nm = best_width * R_D
                result[f"ilf_fwhm_{int(wl)}_nm"] = max(ee_nm, 0.1)

    # Aggregate metrics.
    fluxes = [result[f"flux_{int(wl)}_nm"] for wl in wavelengths_nm]
    mean_flux = sum(fluxes) / len(fluxes)
    result["flux_watts"] = mean_flux
    result["flux_error"] = 0.0
    result["throughput"] = mean_flux / p_collected if p_collected > 0 else 0.0

    ilf_values = [result[f"ilf_fwhm_{int(wl)}_nm"] for wl in wavelengths_nm]
    finite_ilfs = [v for v in ilf_values if np.isfinite(v)]
    result["ilf_fwhm_nm"] = (sum(finite_ilfs) / len(finite_ilfs)
                              if finite_ilfs else float('nan'))

    rms_values = [result[f"rms_spot_{int(wl)}_um"] for wl in wavelengths_nm]
    finite_rms = [v for v in rms_values if np.isfinite(v)]
    result["rms_spot_um"] = (sum(finite_rms) / len(finite_rms)
                              if finite_rms else float('nan'))

    # Worst-wavelength throughput for acceptance gating.
    per_wl_tp = [result[f"throughput_{int(wl)}_nm"] for wl in wavelengths_nm]
    result["min_throughput"] = min(per_wl_tp)

    return result


def element_reflectance_at(element, wavelength_nm, parts):
    """Look up the element's reflectance/efficiency at a wavelength."""
    from optics.elements._tabulated import interp_table, load_two_column_csv

    if element.kind in ("slit", "detector"):
        return 1.0
    if element.kind == "mirror":
        rf = element.params.get("reflectance_file")
        if rf:
            lam, refl = load_two_column_csv(rf)
            return interp_table(wavelength_nm, lam, refl)
        return element.params["reflectance"]
    if element.kind == "grating":
        ef = element.params.get("efficiency_file")
        if ef:
            lam, eff = load_two_column_csv(ef)
            return interp_table(wavelength_nm, lam, eff)
        return element.params["peak_efficiency"]
    raise ValueError(f"unknown element kind {element.kind!r} for reflectance lookup")
