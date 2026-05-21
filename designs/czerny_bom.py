"""Czerny-Turner design facade.

Re-exports the standard design interface and provides genome
reconstruction from optimizer state or CLI args, plus optimizer
configuration helpers (BOM resolution, fold-mode logic, bounds
construction, CzernyProblem factory).
"""

import argparse
import math
import tomllib
from pathlib import Path

from designs.czerny_assembly import assemble_scene  # noqa: F401
from optics.scene import InfeasibleGeometry  # noqa: F401
from designs.czerny_base import (
    CzernyGenome,
    CzernyParts,
    grating_angles,
    fold_fraction_lower_bound,
    fold_fraction_upper_bound,
)
from designs.czerny import CzernyGeometry
from optics._config import (
    DATA_DIR,
    PROJECT_ROOT,
    load_defaults,
    read_toml,
    resolve_path,
)


_DATA_DIR = DATA_DIR
_DEFAULT_BOM = _DATA_DIR / "czerny_bom_v0_design.toml"
_BOM_PATH_OVERRIDE: Path | None = None


def set_bom_path(path: Path | str | None) -> None:
    """Set a session-wide default BOM path for load_parts() and load_bom()."""
    global _BOM_PATH_OVERRIDE
    _BOM_PATH_OVERRIDE = Path(path) if path is not None else None


def _effective_bom_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if _BOM_PATH_OVERRIDE is not None:
        return _BOM_PATH_OVERRIDE
    return _DEFAULT_BOM


def _resolve_geometry(bom_path=None, parts=None):
    """Return a CzernyGeometry instance (mirror-type logic is internal)."""
    return CzernyGeometry()


def build_optics_only_scene(genome, parts, *, bom_path=None,
                            element_overrides=None):
    """Build the optics-only scene using the geometry variant from the BOM.

    Dispatch order:
    1. ``geometry`` key in the BOM TOML (explicit variant name)
    2. ``parts.m1_mirror_type`` fallback (legacy: infer from mirror type)

    ``element_overrides`` is an optional dict mapping element labels
    (e.g. "M1", "F1") to dicts of extra params merged into the
    element's params before scene building.
    """
    geo = _resolve_geometry(bom_path=bom_path, parts=parts)
    scene = geo.build_optics_only_scene(genome, parts)
    if element_overrides:
        for elem in scene.elements:
            if elem.label in element_overrides:
                elem.params.update(element_overrides[elem.label])
    return scene


# ── Config loaders ─────────────────────────────────────────────────────


def _load_controller_fields(bom: dict) -> dict:
    """Load optional [controller] section from BOM into CzernyParts kwargs."""
    ctrl = bom.get("controller")
    if ctrl is None:
        return {}
    return {
        "controller_board_width_mm": float(ctrl["board_width_mm"]),
        "controller_board_height_mm": float(ctrl["board_height_mm"]),
        "controller_board_thickness_mm": float(ctrl["board_thickness_mm"]),
        "controller_component_height_mm": float(ctrl["component_height_mm"]),
        "controller_solder_protrusion_mm": float(ctrl["solder_protrusion_mm"]),
        "controller_insert_spacing_width_mm": float(ctrl["insert_spacing_width_mm"]),
        "controller_insert_spacing_height_mm": float(ctrl["insert_spacing_height_mm"]),
        "controller_solder_clearance_depth_mm": float(ctrl["solder_clearance_depth_mm"]),
        "controller_corner_boss_radius_mm": float(ctrl["corner_boss_radius_mm"]),
    }


def load_parts(
    m1_part: str,
    m2_part: str,
    grating_part: str,
    f1_part: str | None = None,
    f2_part: str | None = None,
    *,
    optic_size_mm: float,
    bom_path: Path | str | None = None,
):
    """Load the locked BOM and return a ``CzernyParts`` dataclass.

    M1, M2, and grating are required.  F1 and F2 may be None when
    fold mirrors are not active (fold_mode=none).
    """
    bom_path = _effective_bom_path(bom_path)
    with bom_path.open("rb") as f:
        bom = tomllib.load(f)

    m1 = bom["mirrors"]["m1_options"][m1_part]
    m2 = bom["mirrors"]["m2_options"][m2_part]
    g = bom["grating_options"][grating_part]

    och = 0.5 * float(optic_size_mm)

    def _mirror_fields(fm, prefix):
        return {
            f"{prefix}_diameter_mm": float(fm["diameter_mm"]),
            f"{prefix}_edge_thickness_mm": float(fm["edge_thickness_mm"]),
            f"{prefix}_center_thickness_mm": float(fm["center_thickness_mm"]),
            f"{prefix}_reflectance": float(fm["reflectance"]),
            f"{prefix}_reflectance_file": resolve_path(str(fm["reflectance_file"])),
            f"{prefix}_mirror_type": str(fm["mirror_type"]),
            f"{prefix}_focal_length_mm": (
                float(fm["focal_length_mm"])
                if "focal_length_mm" in fm else None),
            f"{prefix}_cylindrical_orientation": (
                str(fm["cylindrical_orientation"])
                if "cylindrical_orientation" in fm else None),
            f"{prefix}_mount": dict(fm["mount"]),
        }

    fold_fields: dict = {}
    if f1_part is not None:
        fm_a = bom["mirrors"]["f1_options"][f1_part]
        fold_fields["f1_part"] = f1_part
        fold_fields.update(_mirror_fields(fm_a, "f1"))
    if f2_part is not None:
        fm_b = bom["mirrors"]["f2_options"][f2_part]
        fold_fields["f2_part"] = f2_part
        fold_fields.update(_mirror_fields(fm_b, "f2"))

    return CzernyParts(
        m1_part=m1_part,
        m2_part=m2_part,
        grating_part=grating_part,
        m1_focal_length_mm=float(m1["focal_length_mm"]),
        m1_diameter_mm=float(m1["diameter_mm"]),
        m1_edge_thickness_mm=float(m1["edge_thickness_mm"]),
        m1_reflectance=float(m1["reflectance"]),
        m1_reflectance_file=resolve_path(str(m1["reflectance_file"])),
        m1_mirror_type=str(m1["mirror_type"]),
        m2_focal_length_mm=float(m2["focal_length_mm"]),
        m2_diameter_mm=float(m2["diameter_mm"]),
        m2_edge_thickness_mm=float(m2["edge_thickness_mm"]),
        m2_reflectance=float(m2["reflectance"]),
        m2_mirror_type=str(m2["mirror_type"]),
        m2_reflectance_file=resolve_path(str(m2["reflectance_file"])),
        m1_off_axis_angle_deg=float(m1["off_axis_angle_deg"]),
        m2_off_axis_angle_deg=float(m2["off_axis_angle_deg"]),
        m1_cylindrical_orientation=(
            str(m1["cylindrical_orientation"])
            if "cylindrical_orientation" in m1 else None),
        m1_center_thickness_mm=float(m1["center_thickness_mm"]),
        m2_center_thickness_mm=float(m2["center_thickness_mm"]),
        grating_size_mm=float(g["size_mm"]),
        grating_thickness_mm=float(g["center_thickness_mm"]),
        grating_groove_density_per_mm=float(g["groove_density_per_mm"]),
        grating_blaze_nm=float(g["blaze_nm"]),
        grating_blaze_angle_deg=float(g["blaze_angle_deg"]),
        grating_peak_efficiency=float(g["peak_efficiency"]),
        grating_efficiency=float(g["peak_efficiency"]),
        grating_efficiency_file=resolve_path(str(g["efficiency_file"])),
        optic_center_height_mm=och,
        slit_width_um=float(bom["slits"]["width_um"]),
        slit_height_mm=float(bom["slits"]["height_mm"]),
        wall_albedo=float(bom["housing"]["wall_albedo"]),
        slit_blade_thickness_mm=float(bom["slits"]["blade_thickness_mm"]),
        detector_board_width_mm=float(bom["detector"]["board_width_mm"]),
        detector_board_height_mm=float(bom["detector"]["board_height_mm"]),
        detector_board_behind_die_mm=float(bom["detector"]["board_behind_die_mm"]),
        detector_glass_to_die_mm=float(bom["detector"]["glass_to_die_mm"]),
        detector_boundary_behind_die_mm=float(bom["detector"]["boundary_behind_die_mm"]),
        detector_package_width_mm=float(bom["detector"]["package_width_mm"]),
        detector_package_height_mm=float(bom["detector"]["package_height_mm"]),
        detector_insert_spacing_width_mm=float(bom["detector"]["insert_spacing_width_mm"]),
        detector_insert_spacing_height_mm=float(bom["detector"]["insert_spacing_height_mm"]),
        detector_solder_clearance_depth_mm=float(bom["detector"]["solder_clearance_depth_mm"]),
        detector_corner_boss_radius_mm=float(bom["detector"]["corner_boss_radius_mm"]),
        detector_array_length_mm=float(bom["detector"]["array_length_mm"]),
        m1_mount=dict(m1["mount"]),
        m2_mount=dict(m2["mount"]),
        grating_mount=dict(g["mount"]),
        slit_mount=dict(bom["slits"]["mount"]),
        print_tolerance_mm=float(bom["manufacturing"]["print_tolerance_mm"]),
        **fold_fields,
        **_load_controller_fields(bom),
    )


def _load_baseline_toml() -> dict:
    return read_toml(_DATA_DIR / "czerny_baseline_v0_design.toml")


def load_baseline():
    """Load the baseline genome from ``czerny_baseline_v0_design.toml``."""
    data = _load_baseline_toml()
    genome_keys = {f.name for f in __import__("dataclasses").fields(CzernyGenome)}
    return CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})


def load_baseline_parts(*, bom_path: Path | str | None = None) -> CzernyParts:
    """Load baseline part selections from ``czerny_baseline_v0_design.toml``.

    Returns a ``CzernyParts`` built from the baseline part numbers
    looked up in the active BOM.
    """
    data = _load_baseline_toml()
    m1 = data["m1_part"]
    bom = load_bom(bom_path)
    optic_size = float(bom["mirrors"]["m1_options"][m1]["diameter_mm"])
    return load_parts(
        m1_part=m1,
        m2_part=data["m2_part"],
        grating_part=data["grating_part"],
        f1_part=data.get("f1_part"),
        f2_part=data.get("f2_part"),
        optic_size_mm=optic_size,
        bom_path=bom_path,
    )


BASELINE = load_baseline()


FOLD_MODES = ("none", "F1", "F2", "both")

# ── From optimizer state ───────────────────────────────────────────────


def reconstruct_genome(full_params, best_point, bom_snapshot):
    """Reconstruct a CzernyGenome from GA optimizer output.

    Uses expand_genome() for the deterministic cascade. Falls back
    to explicit r1/r2 if present in full_params (spec-driven path
    stores them in best_point).

    Returns ``(genome, parts)``.
    """
    from designs.czerny import CzernyGeometry

    parts = load_parts(m1_part=full_params["m1_part"],
                       m2_part=full_params["m2_part"],
                       grating_part=full_params["grating_part"],
                       f1_part=full_params.get("f1_part"),
                       f2_part=full_params.get("f2_part"),
                       optic_size_mm=full_params["optic_size_mm"])

    kw: dict = {}
    for key in ("Dv_deg", "theta_m1_deg", "theta_m2_deg",
                "L_m1_mm", "L_m2_mm",
                "f1_fraction", "f2_fraction",
                "theta_f1_deg", "theta_f2_deg"):
        val = best_point.get(key, full_params.get(key))
        if val is not None:
            kw[key] = float(val)

    wls = bom_snapshot["fitness_wavelengths_nm"]
    band = (min(wls), max(wls))
    center_nm = bom_snapshot["center"]
    geom = CzernyGeometry()
    geom.expand_genome(kw, parts, center_nm, band_nm=band)

    # NM-refined values override the cascade.
    for key in ("L_a_mm", "L_b_mm", "L_f1_mm", "theta_d_deg"):
        val = best_point.get(key)
        if val is not None:
            kw[key] = float(val)

    genome = CzernyGenome(**{
        k: v for k, v in kw.items()
        if k in CzernyGenome.__dataclass_fields__
    })
    return genome, parts


# ── CLI args for debug mode ────────────────────────────────────────────


def add_genome_args(ap: argparse.ArgumentParser) -> None:
    """Register Czerny-specific genome flags on *ap*."""
    ap.add_argument("--L_a", type=float, default=None,
                    help="Entrance arm length in mm (slit to M1)")
    ap.add_argument("--L_b", type=float, default=None,
                    help="Exit arm length in mm (M2 to exit slit)")
    ap.add_argument("--L_m1", type=float, default=None,
                    help="Grating-to-M1 distance in mm")
    ap.add_argument("--L_m2", type=float, default=None,
                    help="Grating-to-M2 distance in mm")
    ap.add_argument("--theta1", type=float, default=None,
                    help="M1 fold half-angle in degrees")
    ap.add_argument("--theta2", type=float, default=None,
                    help="M2 fold half-angle in degrees")
    ap.add_argument("--Dv", type=float, default=None,
                    help="Grating deviation angle in degrees (required with --L_a)")
    ap.add_argument("--center", type=float, default=None,
                    help="Design center wavelength in nm (default: from targets TOML)")
    ap.add_argument("--m1", type=str, default=None,
                    help="M1 part number (default: from baseline)")
    ap.add_argument("--m2", type=str, default=None,
                    help="M2 part number (default: from baseline)")
    ap.add_argument("--grating", type=str, default=None,
                    help="Grating part number (default: from baseline)")
    ap.add_argument("--f1_frac", type=float, default=None,
                    help="F1 fold fraction for entrance arm (0-1)")
    ap.add_argument("--f2_frac", type=float, default=None,
                    help="F2 fold fraction for exit arm (0-1)")


def resolve_genome_from_cli(args):
    """Build a CzernyGenome from explicit CLI flags or baseline.

    Returns ``(genome, m1_part, m2_part, grating_part, label)``.
    """
    bl = _load_baseline_toml()
    m1_part = args.m1 or bl["m1_part"]
    m2_part = args.m2 or bl["m2_part"]
    grating_part = args.grating or bl["grating_part"]
    f1_part = bl.get("f1_part")
    f2_part = bl.get("f2_part")

    targets = load_targets()
    center_nm = getattr(args, "center", None) or targets["center_nm"]

    bom = load_bom()
    m1_entry = bom["mirrors"]["m1_options"][m1_part]
    optic_size = float(m1_entry["diameter_mm"])
    if args.L_a is not None:
        parts = load_parts(m1_part=m1_part, m2_part=m2_part,
                           grating_part=grating_part,
                           f1_part=f1_part, f2_part=f2_part,
                           optic_size_mm=optic_size)
        if args.Dv is None:
            raise SystemExit("--Dv is required when --L_a is specified")
        dev = args.Dv
        d_nm = 1e6 / parts.grating_groove_density_per_mm
        alpha, beta = grating_angles(dev, center_nm, d_nm)

        if args.L_b is None:
            raise SystemExit("--L_b is required when --L_a is specified")
        L_a = args.L_a
        L_b = args.L_b
        f_kw: dict[str, float] = {}
        if getattr(args, "f1_frac", None) is not None:
            f_kw["L_f1_mm"] = L_a * args.f1_frac
        if getattr(args, "f2_frac", None) is not None:
            f_kw["L_f2_mm"] = L_b * args.f2_frac
        genome = CzernyGenome(
            L_a_mm=L_a, L_b_mm=L_b,
            L_m1_mm=args.L_m1, L_m2_mm=args.L_m2,
            theta_m1_deg=args.theta1, theta_m2_deg=args.theta2,
            alpha_deg=alpha, beta_deg=beta,
            theta_d_deg=0.0,
            **f_kw,
        )
        return genome, m1_part, m2_part, grating_part, "cli"

    return BASELINE, m1_part, m2_part, grating_part, "baseline"


# ── Optimizer configuration ────────────────────────────────────────────


def max_footprint_xy_mm() -> tuple[float, float]:
    defaults = load_defaults()
    env = defaults["optimizer"]["max_footprint_mm"]
    return (float(env[0]), float(env[1]))


def fitness_wavelengths_from_bw(
    bw_nm: float, center_nm: float, step: float = 50.0,
) -> tuple[float, ...]:
    """Derive fitness wavelengths from bandwidth + center, rounded to step."""
    half = bw_nm / 2.0
    lam_min = round((center_nm - half) / step) * step
    lam_max = round((center_nm + half) / step) * step
    lam_center = round(center_nm / step) * step
    return tuple(sorted({lam_min, lam_center, lam_max}))


def parts_builder_from_params(params: dict):
    """Rebuild CzernyParts per candidate from discrete BOM choices.

    Must be a module-level function (not a closure) so that multiprocessing
    can pickle it.  F1 and F2 are None when fold mirrors are not active.
    """
    return load_parts(
        m1_part=params["m1_part"],
        m2_part=params["m2_part"],
        grating_part=params["grating_part"],
        f1_part=params.get("f1_part"),
        f2_part=params.get("f2_part"),
        optic_size_mm=params["optic_size_mm"],
    )


def assemble_with_envelope(genome, parts, *, max_footprint_xy_mm, geometry):
    """Module-level assembler so multiprocessing can pickle it."""
    return assemble_scene(
        genome, parts,
        max_footprint_xy_mm=max_footprint_xy_mm,
        scene_builder=geometry.build_optics_only_scene,
    )


def load_targets() -> dict:
    with (_DATA_DIR / "czerny_targets.toml").open("rb") as f:
        return tomllib.load(f)


def load_bom(bom_path: Path | str | None = None) -> dict:
    bom_path = _effective_bom_path(bom_path)
    with bom_path.open("rb") as f:
        return tomllib.load(f)


def peak_from_csv(csv_path: str) -> float:
    """Read the peak value from a 2-column wavelength,value CSV."""
    import csv as _csv
    path = Path(csv_path)
    if not path.is_absolute():
        path = _DATA_DIR.parent / csv_path
    peak = 0.0
    with path.open() as f:
        for row in _csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "wavelength_nm":
                continue
            v = float(row[1])
            if v > peak:
                peak = v
    return peak


def _resolve_mirrors(focal_length_mm: float, diameter_mm: float,
                     bom: dict, section: str,
                     mirror_type: str | None = None) -> list[str]:
    matches = []
    for name, m in bom["mirrors"][section].items():
        if isinstance(m, dict) and "focal_length_mm" in m:
            if (abs(m["focal_length_mm"] - focal_length_mm) < 1.0
                    and abs(m["diameter_mm"] - diameter_mm) < 2.0):
                if mirror_type is not None and m["mirror_type"] != mirror_type:
                    continue
                matches.append(name)
    return sorted(matches)


def resolve_from_specs(
    max_rld: float,
    min_bw: float,
    max_fnum: float,
    center_nm: float,
    defaults: dict,
    fold_mode: str = "none",
) -> dict:
    """Resolve BOM parts and build pre-validated combo list.

    Parameters
    ----------
    max_rld : Maximum reciprocal linear dispersion (nm/mm).
        Rejects M1 with f_coll < d_grating / max_rld.
    min_bw : Minimum bandwidth on detector (nm).
        Rejects M2 with f_cam > L_det * d_grating / min_bw.
    max_fnum : Maximum f-number (slowest acceptable).
        Rejects M1 with D < f_coll / max_fnum.
    center_nm : Design center wavelength (nm).
    defaults : Loaded defaults.toml dict.
    fold_mode : 'none', 'F1', 'F2', or 'both'.

    Applies tight heuristics to produce a small set of valid combos:
      1. max_rld → M1 focal length lower bound (finer dispersion OK)
      2. min_bw  → M2 focal length upper bound (wider bandwidth OK)
      3. max_fnum → M1 diameter lower bound (faster f/# OK)
      4. D_M2 > D_M1 (beam walk)
      5. CCD coverage → exact pixel check for fitness wavelengths
      6. D_F1 ≤ D_M1 (fold mirror fits in collimated beam)
      7. D_F2 > D_detector (fold mirror covers detector array)
      8. f_F1 ≤ f_M2 (astigmatism-free condition, cylindrical F1 only)

    Mirror type filtering is left to the BOM: the TOML file should
    only contain mirrors appropriate for the design class.

    For fold_mode='none': combo = (M1, M2, grating).
    For fold_mode='F1':   combo = (M1, M2, grating, F1), F2 locked.

    Returns dict with keys: combo_list, ranges, target_fnum,
    optic_size_mm, f2_part, f1_cylindrical.
    """
    bom = load_bom()
    all_gratings = bom["grating_options"]
    all_m1 = bom["mirrors"]["m1_options"]
    all_m2 = bom["mirrors"]["m2_options"]

    det = bom["detector"]
    L_det = float(det["array_length_mm"])

    # ── Step 1-4: M1 + M2 + grating triplets ──────────────────────────
    triplets: list[dict] = []
    for gname, g in all_gratings.items():
        if not isinstance(g, dict) or "groove_density_per_mm" not in g:
            continue
        d_nm = 1.0e6 / g["groove_density_per_mm"]
        f_coll_min = d_nm / max_rld
        f_cam_max = L_det * d_nm / min_bw
        g_size = float(g["size_mm"])

        for m1name, m1 in all_m1.items():
            if not isinstance(m1, dict) or "focal_length_mm" not in m1:
                continue
            f1 = m1["focal_length_mm"]
            d1 = m1["diameter_mm"]
            if not (0.8 * d1 <= g_size <= 1.2 * d1):
                continue
            if f1 < f_coll_min:
                continue
            if d1 < f1 / max_fnum:
                continue

            for m2name, m2 in all_m2.items():
                if not isinstance(m2, dict) or "focal_length_mm" not in m2:
                    continue
                f2 = m2["focal_length_mm"]
                if f2 > f_cam_max:
                    continue
                d2 = m2["diameter_mm"]
                if d2 <= d1:
                    continue
                triplets.append({
                    "m1": m1name, "m2": m2name, "grating": gname,
                    "f_coll": f1, "f_cam": f2,
                    "d_m1": d1, "d_m2": d2,
                })

    if not triplets:
        raise ValueError(
            f"No BOM combinations satisfy --max_rld {max_rld} --min_bw {min_bw} --max_fnum {max_fnum}")

    # Exact CCD coverage check: reject triplets where any fitness
    # wavelength falls off the detector.  The BW constraint above uses
    # linear dispersion; this check uses the full grating equation.
    from optics.grating_math import wavelengths_on_detector
    fit_wls = fitness_wavelengths_from_bw(min_bw, center_nm)
    triplets = [
        t for t in triplets
        if wavelengths_on_detector(
            fit_wls, center_nm,
            all_gratings[t["grating"]]["groove_density_per_mm"],
            t["f_cam"])
    ]
    if not triplets:
        raise ValueError(
            f"No BOM combinations pass exact CCD coverage check for "
            f"fitness wavelengths {fit_wls}")

    # ── Step 5-7: F1 and F2 fold mirrors ───────────────────────────────
    all_f1 = bom["mirrors"]["f1_options"] if fold_mode in ("F1", "both") else {}
    all_f2 = bom["mirrors"]["f2_options"] if fold_mode in ("F2", "both") else {}

    # F2: lock to largest with D_F2 > D_detector (only for modes that use F2)
    locked_f2 = None
    if fold_mode in ("F2", "both"):
        D_det = L_det
        f2_candidates = sorted(
            (k for k, v in all_f2.items()
             if isinstance(v, dict) and "diameter_mm" in v
             and v["diameter_mm"] > D_det),
            key=lambda k: all_f2[k]["diameter_mm"])
        if not f2_candidates:
            raise ValueError(
                f"No F2 fold mirror with D > {D_det:.1f}mm (detector array)")
        locked_f2 = f2_candidates[0]

    # Build combo list depending on fold_mode.
    combo_list: list[dict] = []
    if fold_mode in ("F1", "both"):
        f1_names = [k for k, v in all_f1.items()
                    if isinstance(v, dict) and "diameter_mm" in v]
        for tri in triplets:
            d1 = tri["d_m1"]
            f_cam = tri["f_cam"]
            for f1name in f1_names:
                f1_opt = all_f1[f1name]
                # D_F1 ≤ D_M1
                if f1_opt["diameter_mm"] > d1:
                    continue
                # Cylindrical: f_F1 ≤ f_M2 (astigmatism-free feasibility)
                if (f1_opt.get("mirror_type") == "cylindrical"
                        and f1_opt.get("focal_length_mm") is not None
                        and f1_opt["focal_length_mm"] > f_cam):
                    continue
                combo = {
                    "m1": tri["m1"], "m2": tri["m2"],
                    "grating": tri["grating"], "f1": f1name,
                }
                if locked_f2 is not None:
                    combo["f2"] = locked_f2
                combo_list.append(combo)
    else:
        # fold_mode=none: combo is (M1, M2, grating), no fold mirrors.
        seen = set()
        for tri in triplets:
            key = (tri["m1"], tri["m2"], tri["grating"])
            if key not in seen:
                seen.add(key)
                combo_list.append({
                    "m1": tri["m1"], "m2": tri["m2"],
                    "grating": tri["grating"],
                })

    if not combo_list:
        raise ValueError(
            f"No valid combos after fold mirror filtering "
            f"(fold_mode={fold_mode})")

    # ── Ranges ─────────────────────────────────────────────────────────
    global_ranges = dict(defaults["optimizer"]["ranges"])

    # Deviation bounds from the grating equation.
    # min_dev = arcsin(mλ/d) where α=0 (Littrow floor).
    # max_dev = arcsin(1 − mλ/d) + 90° (grazing-exit ceiling).
    grating_densities = {
        all_gratings[c["grating"]]["groove_density_per_mm"]
        for c in combo_list
    }
    from math import asin, degrees as _deg
    min_devs = []
    max_devs = []
    for lpm in grating_densities:
        ratio = center_nm / (1e6 / lpm)
        if abs(ratio) <= 1.0:
            min_devs.append(_deg(asin(ratio)))
        sin_alpha_max = 1.0 - ratio
        if abs(sin_alpha_max) <= 1.0:
            max_devs.append(_deg(asin(sin_alpha_max)) + 90.0)
    if not min_devs or not max_devs:
        raise ValueError(
            f"no valid deviation range: center {center_nm}nm unreachable "
            f"by any grating ({sorted(grating_densities)} g/mm)")
    Dv_phys_lo = min(min_devs)
    Dv_phys_hi = max(max_devs)
    cfg_dv = global_ranges.get("Dv_range")
    if cfg_dv is not None:
        Dv_phys_lo = max(Dv_phys_lo, float(cfg_dv[0]))
        Dv_phys_hi = min(Dv_phys_hi, float(cfg_dv[1]))
    global_ranges["Dv_range"] = [Dv_phys_lo, Dv_phys_hi]

    f_colls = [t["f_coll"] for t in triplets]
    f_cams = [t["f_cam"] for t in triplets]

    # Representative L_m1/L_m2 ranges for adaptive grid sizing.
    # Actual per-candidate ranges are computed in valid_range();
    # these are conservative estimates from the part catalog.
    # Floor: mirror sag (edge_thickness - center_thickness) is the only
    # protrusion past the optical vertex toward the grating surface.
    from math import cos, radians
    m1_sags = [all_m1[t["m1"]]["edge_thickness_mm"]
               - all_m1[t["m1"]]["center_thickness_mm"] for t in triplets]
    m2_sags = [all_m2[t["m2"]]["edge_thickness_mm"]
               - all_m2[t["m2"]]["center_thickness_mm"] for t in triplets]
    global_ranges["L_m1_range"] = [max(m1_sags), max(f_colls)]
    global_ranges["L_m2_range"] = [max(m2_sags), max(f_cams)]

    # Fold fraction range (for 1D sweep over f1_fraction when F1 is flat)
    if fold_mode in ("F1", "both"):
        slit_half_h = float(bom["slits"]["height_mm"]) / 2.0
        m1_choices = sorted({c["m1"] for c in combo_list})
        f1_choices = sorted({c.get("f1") for c in combo_list if c.get("f1")})
        f1_lowers, f1_uppers = [], []
        for name in m1_choices:
            m1_opt = all_m1[name]
            d = float(m1_opt["diameter_mm"])
            L_a = float(m1_opt["focal_length_mm"])
            for f1name in f1_choices:
                f1_opt = all_f1[f1name]
                d_f1 = float(f1_opt["diameter_mm"])
                wall_margin = float(f1_opt["mount"]["wall_margin_mm"])
                f1_plate_radius = 0.5 * d_f1 + wall_margin
                f1_lowers.append(
                    fold_fraction_lower_bound(d, slit_half_h, d_f1, 45.0))
                f1_uppers.append(
                    fold_fraction_upper_bound(L_a, f1_plate_radius))
        global_ranges["f1_fraction_range"] = [
            min(f1_lowers), max(f1_uppers)]

    # OAP mirrors: pin theta1/theta2 to off-axis angle.
    # Check if all M1s in combos are paraboloidal.
    m1_types = {all_m1[c["m1"]].get("mirror_type") for c in combo_list}
    if m1_types == {"paraboloidal"}:
        m1_first = all_m1[combo_list[0]["m1"]]
        m2_first = all_m2[combo_list[0]["m2"]]
        oaa1 = float(m1_first["off_axis_angle_deg"]) / 2.0
        oaa2 = float(m2_first["off_axis_angle_deg"]) / 2.0
        global_ranges["theta_m1_range"] = [oaa1, oaa1]
        global_ranges["theta_m2_range"] = [oaa2, oaa2]

    f1_cylindrical = False
    if fold_mode in ("F1", "both"):
        f1_choices_set = {c.get("f1") for c in combo_list if c.get("f1")}
        f1_cylindrical = any(
            all_f1[n].get("mirror_type") == "cylindrical"
            for n in f1_choices_set if isinstance(all_f1.get(n), dict)
        )

    return {
        "combo_list":       combo_list,
        "ranges":           global_ranges,
        "target_fnum":      max_fnum,
        "optic_size_mm":    max(t["d_m1"] for t in triplets),
        "f2_part":          locked_f2,
        "f1_cylindrical":   f1_cylindrical,
    }


def fold_config(defaults: dict) -> str:
    mode = str(defaults["optimizer"]["fold"]["mode"])
    if mode not in FOLD_MODES:
        raise ValueError(
            f"invalid [optimizer.fold].mode {mode!r}; "
            f"expected one of {FOLD_MODES}"
        )
    return mode


def fold_axes_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "F1":
        return ("f1_fraction",)
    if mode == "F2":
        return ("f2_fraction",)
    if mode == "both":
        return ("f1_fraction", "f2_fraction")
    return ()


def bounds_from_ranges(
    ranges: dict,
    fold_axes: tuple[str, ...] = (),
) -> dict[str, tuple[float, float]]:
    bounds = {
        "alpha_deg":     (0.0, 90.0),
        "beta_deg":      (-90.0, 0.0),
        "Dv_deg": tuple(ranges["Dv_range"]),
        "theta_m1_deg":    tuple(ranges["theta_m1_range"]),
        "theta_d_deg": tuple(ranges.get(
            "theta_d_range", (-10.0, 10.0))),
        "theta_f1_deg": tuple(ranges.get(
            "theta_f_range", (5.0, 85.0))),
        "theta_f2_deg": tuple(ranges.get(
            "theta_f_range", (5.0, 85.0))),
    }
    if "theta_m2_range" in ranges:
        bounds["theta_m2_deg"] = tuple(ranges["theta_m2_range"])
    if "r_total_range" in ranges:
        bounds["r_total_mm"] = tuple(ranges["r_total_range"])
    for axis in fold_axes:
        bounds[axis] = tuple(ranges[f"{axis}_range"])
    return bounds


def build_czerny_problem(
    searched_axis_names: list[str],
    fitness_wavelengths_nm,
    max_footprint_xy_mm,
    bounds: dict[str, tuple[float, float]],
    target_fnum: float,
    optic_size_mm: float,
    optics_only: bool = False,
    *,
    geometry,
    design_wavelength_nm: float,
    combo_list: list[dict] | None = None,
    fold_mode: str = "none",
    forward_rays: int = 5_000,
    coarse_forward_rays: int = 1_000,
    point_source: bool = False,
    acceptance_threshold: float = 0.005,
):
    import functools
    from designs.czerny_problem import CzernyProblem
    from optics.fitness import scalar_fitness

    if optics_only:
        assembler = functools.partial(
            _assemble_optics_only, geometry=geometry)
    else:
        assembler = functools.partial(
            assemble_with_envelope,
            max_footprint_xy_mm=max_footprint_xy_mm,
            geometry=geometry,
        )

    return CzernyProblem(
        parts=None,
        parts_builder=parts_builder_from_params,
        scalarizer=scalar_fitness,
        searched_axis_names=searched_axis_names,
        assembler=assembler,
        bounds=bounds,
        fitness_wavelengths_nm=fitness_wavelengths_nm,
        target_fnum=target_fnum,
        geometry=geometry,
        design_wavelength_nm=design_wavelength_nm,
        forward_rays=forward_rays,
        coarse_forward_rays=coarse_forward_rays,
        point_source=point_source,
        project_root=_DATA_DIR.parent,
        combo_list=combo_list,
        fold_mode=fold_mode,
        acceptance_threshold=acceptance_threshold,
    )


def _assemble_optics_only(genome, parts, *, geometry):
    """Optics-only assembler: no mounts, no housing.

    Runs bare-optic polygon collision and beam-cone clip checks.
    """
    from designs.czerny_assembly import CzernyAssembly
    from optics.collision import check_optic_collisions
    scene = geometry.build_optics_only_scene(genome, parts)
    beam_path = CzernyAssembly().resolve_beam_path(scene)
    check_optic_collisions(scene, beam_path)
    return scene
