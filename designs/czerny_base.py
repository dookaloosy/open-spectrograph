"""Czerny-Turner geometry — fixed-grating CT layout for all mirror types.

Defines the complete Czerny-Turner layout: mirror placement from grating
arm angles, slit and fold-mirror poses, surface normals from the law of
reflection, and element list assembly. Handles spherical, cylindrical,
and off-axis paraboloidal mirrors via internal branching on mirror_type
from the BOM.

All angles are incidence angles (ray-to-normal):

  α (alpha)      — incidence angle at the grating
  β (beta)       — diffraction angle at the grating
  θ_M1 (theta_m1) — incidence angle at M1
  θ_M2 (theta_m2) — incidence angle at M2
  θ_F1 (theta_f1) — incidence angle at fold mirror F1
  θ_F2 (theta_f2) — incidence angle at fold mirror F2
  θ_D  (theta_d) — detector θ_d
  L_a            — entrance slit to M1 distance
  L_b            — M2 to exit slit distance
  L_m1           — grating pivot to M1 distance
  L_m2           — grating pivot to M2 distance
"""


from dataclasses import dataclass
from math import asin, cos, degrees, radians, sin, sqrt
from typing import NamedTuple

from optics.scene import (
    ElementPlacement,
    Scene,
    Vec3,
    perpendicular_xy,
)


# ─── Degrees of freedom (the GA genome) ──────────────────────────────────────

@dataclass(frozen=True)
class CzernyGenome:
    """Free DoF for an asymmetric Czerny-Turner.

    All angles are incidence angles (angle between incoming ray and
    surface normal):

      α (alpha_deg)        — grating incidence angle
      β (beta_deg)         — grating diffraction angle
      θ_M1 (theta_m1_deg) — incidence angle at M1 (half-deflection)
      θ_M2 (theta_m2_deg) — incidence angle at M2
      θ_F1 (theta_f1_deg) — incidence angle at fold mirror F1
      θ_F2 (theta_f2_deg) — incidence angle at fold mirror F2
      θ_D  (theta_d_deg) — detector θ_d (CW viewed from +z)

    Mirror positions are set by (α, L_m1) and (β, L_m2) — the grating
    arm geometry. Slit positions are set by (θ_M1, L_a) and (θ_M2, L_b).
    """

    L_a_mm: float
    L_b_mm: float
    alpha_deg: float
    beta_deg: float
    theta_m1_deg: float
    theta_m2_deg: float
    L_m1_mm: float
    L_m2_mm: float
    theta_d_deg: float
    L_f1_mm: float | None = None
    L_f2_mm: float | None = None
    theta_f1_deg: float | None = None
    theta_f2_deg: float | None = None


GENOME_FIELD_NAMES: frozenset[str] = frozenset(
    f.name for f in __import__("dataclasses").fields(CzernyGenome)
)


# ─── Parts (the BOM inputs, filled by designs.czerny_bom.load_parts()) ──────

@dataclass(frozen=True)
class CzernyParts:
    """Fixed part specs. Populated from the BOM via ``designs.czerny_bom.load_parts()``."""

    m1_part: str
    m2_part: str
    grating_part: str
    m1_focal_length_mm: float
    m1_diameter_mm: float
    m1_reflectance: float
    m2_focal_length_mm: float
    m2_diameter_mm: float
    m2_reflectance: float
    m1_mirror_type: str
    m2_mirror_type: str
    grating_size_mm: float
    grating_thickness_mm: float
    grating_groove_density_per_mm: float
    grating_blaze_nm: float
    grating_blaze_angle_deg: float
    grating_peak_efficiency: float
    grating_efficiency: float
    optic_center_height_mm: float
    slit_width_um: float
    slit_height_mm: float
    wall_albedo: float
    slit_blade_thickness_mm: float
    grating_efficiency_file: str
    m1_reflectance_file: str
    m2_reflectance_file: str
    m1_off_axis_angle_deg: float
    m2_off_axis_angle_deg: float
    m1_edge_thickness_mm: float
    m2_edge_thickness_mm: float
    m1_center_thickness_mm: float
    m2_center_thickness_mm: float
    detector_board_width_mm: float
    detector_board_height_mm: float
    detector_board_behind_die_mm: float
    detector_glass_to_die_mm: float
    detector_boundary_behind_die_mm: float
    detector_package_width_mm: float
    detector_package_height_mm: float
    detector_insert_spacing_width_mm: float
    detector_insert_spacing_height_mm: float
    detector_solder_clearance_depth_mm: float
    detector_corner_boss_radius_mm: float
    detector_array_length_mm: float
    controller_board_width_mm: float | None = None
    controller_board_height_mm: float | None = None
    controller_board_thickness_mm: float | None = None
    controller_component_height_mm: float | None = None
    controller_solder_protrusion_mm: float | None = None
    controller_insert_spacing_width_mm: float | None = None
    controller_insert_spacing_height_mm: float | None = None
    controller_solder_clearance_depth_mm: float | None = None
    controller_corner_boss_radius_mm: float | None = None
    m1_cylindrical_orientation: str | None = None
    f1_part: str | None = None
    f1_diameter_mm: float | None = None
    f1_edge_thickness_mm: float | None = None
    f1_center_thickness_mm: float | None = None
    f1_reflectance: float | None = None
    f1_reflectance_file: str | None = None
    f1_mirror_type: str | None = None
    f1_focal_length_mm: float | None = None
    f1_cylindrical_orientation: str | None = None
    f2_part: str | None = None
    f2_diameter_mm: float | None = None
    f2_edge_thickness_mm: float | None = None
    f2_center_thickness_mm: float | None = None
    f2_reflectance: float | None = None
    f2_reflectance_file: str | None = None
    f2_mirror_type: str | None = None
    f2_focal_length_mm: float | None = None
    f2_cylindrical_orientation: str | None = None
    print_tolerance_mm: float = 0.1
    m1_mount: dict | None = None
    m2_mount: dict | None = None
    grating_mount: dict | None = None
    slit_mount: dict | None = None
    f1_mount: dict | None = None
    f2_mount: dict | None = None


# ─── Utility functions ──────────────────────────────────────────────────────

def _normalize(v: Vec3) -> Vec3:
    n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def _rotate_xy(v: Vec3, angle_rad: float) -> Vec3:
    """Rotate a vector in the xy plane by *angle_rad* (z unchanged)."""
    c, s = cos(angle_rad), sin(angle_rad)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2])


# ─── Fold-mirror feasibility gate constants ─────────────────────────────────

_FOLD_HALF_ANGLE_MIN_DEG = 2.0


def _compute_fold_geometry(
    fold_pos: Vec3,
    fold_angle_deg: float,
    L_slit: float,
    *,
    d_slit_to_m1: Vec3 | None = None,
    d_m2_to_slit: Vec3 | None = None,
) -> tuple[Vec3, Vec3, Vec3]:
    """Compute fold normal and slit position for a folded CT arm.

    The two arms have different ray-direction conventions relative to
    the arm direction, so the caller identifies the arm via keyword:

    * ``d_slit_to_m1`` for the entrance arm (F1).
    * ``d_m2_to_slit`` for the exit arm (F2).

    Exactly one must be provided.

    Returns (fold_normal, slit_pos, u_slit_normal).
    """
    from optics.scene import InfeasibleGeometry

    if (d_slit_to_m1 is None) == (d_m2_to_slit is None):
        raise ValueError(
            "exactly one of d_slit_to_m1 / d_m2_to_slit must be set")

    if d_slit_to_m1 is not None:
        d_fold_to_mirror = d_slit_to_m1
        rot_sign = 1.0
    else:
        assert d_m2_to_slit is not None
        d_fold_to_mirror = (
            -d_m2_to_slit[0], -d_m2_to_slit[1], 0.0)
        rot_sign = -1.0

    if fold_angle_deg <= 0.0 or fold_angle_deg >= 180.0:
        raise InfeasibleGeometry(
            f"fold angle {fold_angle_deg:.1f}° is degenerate"
        )
    half_angle = fold_angle_deg / 2.0
    if half_angle < _FOLD_HALF_ANGLE_MIN_DEG:
        raise InfeasibleGeometry(
            f"fold half-angle {half_angle:.2f}° below "
            f"{_FOLD_HALF_ANGLE_MIN_DEG}° numerical floor"
        )

    phi = radians(fold_angle_deg)
    cp, sp = cos(phi), sin(phi)
    nat_x = -d_fold_to_mirror[0]
    nat_y = -d_fold_to_mirror[1]
    d_fold_to_slit = (
        nat_x * cp + rot_sign * nat_y * sp,
        -rot_sign * nat_x * sp + nat_y * cp,
        0.0,
    )

    slit_pos = (
        fold_pos[0] + L_slit * d_fold_to_slit[0],
        fold_pos[1] + L_slit * d_fold_to_slit[1],
        0.0,
    )
    u_slit_normal = (-d_fold_to_slit[0], -d_fold_to_slit[1], 0.0)

    bx = d_fold_to_mirror[0] + d_fold_to_slit[0]
    by = d_fold_to_mirror[1] + d_fold_to_slit[1]
    fold_normal = _normalize((bx, by, 0.0))
    if (fold_normal[0] * d_fold_to_slit[0]
            + fold_normal[1] * d_fold_to_slit[1] < 0):
        fold_normal = (-fold_normal[0], -fold_normal[1], 0.0)
    return fold_normal, slit_pos, u_slit_normal


# ─── Inter-phase data containers ────────────────────────────────────────────

class _MirrorPositions(NamedTuple):
    m1: Vec3
    m2: Vec3
    d_m1_to_grating: Vec3
    d_m2_to_grating: Vec3
    d_arm_a: Vec3
    d_arm_b: Vec3


class _SlitPoses(NamedTuple):
    entrance_slit: Vec3
    exit_slit: Vec3
    n_entrance_slit: Vec3
    n_exit_slit: Vec3
    f1: tuple[Vec3, Vec3] | None
    f2: tuple[Vec3, Vec3] | None


class _SurfaceNormals(NamedTuple):
    n_m1: Vec3
    n_m2: Vec3
    n_grating: Vec3
    n_entrance_slit: Vec3
    n_exit_slit: Vec3


# ─── Geometry class ─────────────────────────────────────────────────────────

class CzernyGeometry:
    """Fixed-grating Czerny-Turner geometry for all mirror types."""

    _is_oap: bool = False

    def mirror_axis(
        self, label: str,
        mirrors: _MirrorPositions,
        normals: _SurfaceNormals,
    ) -> Vec3:
        """Return the element axis for M1 or M2."""
        if self._is_oap:
            return mirrors.d_m1_to_grating if label == "M1" else mirrors.d_m2_to_grating
        return normals.n_m1 if label == "M1" else normals.n_m2

    def mirror_params(
        self, label: str,
        mirrors: _MirrorPositions,
    ) -> dict:
        """Return extra element params for M1 or M2."""
        if not self._is_oap:
            return {}
        if label == "M1":
            return {"paraboloidal_focus_dir": (
                -mirrors.d_arm_a[0], -mirrors.d_arm_a[1], 0.0)}
        return {"paraboloidal_focus_dir": mirrors.d_arm_b}

    def constrain_genome(
        self, kw: dict,
        parts: CzernyParts,
    ) -> None:
        """Apply mirror-type-specific genome constraints in-place."""
        self._is_oap = parts.m1_mirror_type == "paraboloidal"
        if self._is_oap:
            kw["theta_m1_deg"] = parts.m1_off_axis_angle_deg / 2.0
            kw["theta_m2_deg"] = parts.m2_off_axis_angle_deg / 2.0

    # ── Genome expansion ────────────────────────────────────────────────

    def expand_genome(
        self, kw: dict, parts: CzernyParts, lambda_center_nm: float,
        *, band_nm: tuple[float, float],
    ) -> None:
        """Expand virtual genome parameters into physical values, in-place.

        Handles the deterministic derivation:
          deviation → (α, β)  via grating eq. at λ_c
          constrain_genome()  variant-specific (e.g. OAP θ pinning)
          θ₁ → θ₂            coma cancellation (if not already set)
          L_m1, L_m2          derived from geometry (beam walk / clearance)
          L_a, L_b            tangential focal length
          f1/f2_fraction → L_f1/L_f2

        Parameters
        ----------
        band_nm : (λ_min, λ_max) for beam walk sizing. L_m2 is sized to
            keep the beam on M2 across this band.
        """
        # Deviation gauge: expand Dv_deg → canonical (α, β).
        if "Dv_deg" in kw:
            dev = float(kw.pop("Dv_deg"))
            groove_period_nm = 1.0e6 / parts.grating_groove_density_per_mm
            try:
                alpha_c, beta_c = grating_angles(
                    dev, lambda_center_nm, groove_period_nm,
                    diffraction_order=-1,
                )
            except ValueError:
                return
            kw["alpha_deg"] = alpha_c
            kw["beta_deg"] = beta_c

        # Variant-specific genome constraints (e.g. θ pinning for OAP).
        self.constrain_genome(kw, parts)

        # Derive θ₂ from coma cancellation if not already set (OAP sets it in
        # constrain_genome; explicit baselines may provide it directly).
        if ("theta_m2_deg" not in kw
                and "theta_m1_deg" in kw
                and "alpha_deg" in kw
                and "beta_deg" in kw):
            R1 = 2.0 * parts.m1_focal_length_mm
            R2 = 2.0 * parts.m2_focal_length_mm
            try:
                kw["theta_m2_deg"] = shafer_theta2(
                    float(kw["theta_m1_deg"]), R1, R2,
                    float(kw["alpha_deg"]), float(kw["beta_deg"]),
                )
            except ValueError:
                return

        # L_m1, L_m2 derivation.
        kw.pop("r_total_mm", None)
        kw.pop("r_ratio", None)
        if "L_m1_mm" not in kw and "L_m2_mm" not in kw:
            r2 = _r2_from_beam_walk(kw, parts, band_nm=band_nm)
            if r2 <= 0:
                return  # M2 too small for beam — infeasible
            kw["L_m2_mm"] = r2
            kw["L_m1_mm"] = _r1_from_clearance(kw, parts)

        # Arm lengths from tangential focal length: f_t = f·cos(θ).
        # Sagittal focal length f_s = f/cos(θ) is longer; the difference
        # is astigmatism, corrected by the cylindrical fold when present.
        if parts.m1_mirror_type == "paraboloidal":
            L_a = parts.m1_focal_length_mm
        else:
            L_a = parts.m1_focal_length_mm * cos(radians(float(kw["theta_m1_deg"])))
        if parts.m2_mirror_type == "paraboloidal":
            L_b = parts.m2_focal_length_mm
        else:
            L_b = parts.m2_focal_length_mm * cos(radians(float(kw["theta_m2_deg"])))
        kw["L_a_mm"] = L_a
        kw["L_b_mm"] = L_b

        # Fold position: L_f1 = distance from curved mirror to fold.
        # Cylindrical F1: derive from astigmatism-free condition.
        # Flat F1: derive from feasible-band midpoint for theta_f1.
        if (parts.f1_mirror_type == "cylindrical"
                and parts.f1_focal_length_mm is not None
                and "theta_f1_deg" in kw
                and "theta_m2_deg" in kw):
            kw.pop("f1_fraction", None)
            phi_fold = float(kw["theta_f1_deg"])
            R_fold = 2.0 * parts.f1_focal_length_mm
            R_coll = 2.0 * parts.m1_focal_length_mm
            R_focus = 2.0 * parts.m2_focal_length_mm
            try:
                l_SM1 = astigmatism_free_l_SM1(
                    R_fold, R_coll, R_focus,
                    phi_fold, float(kw["theta_m1_deg"]),
                    float(kw["theta_m2_deg"]))
                if 0 < l_SM1 < L_a:
                    kw["L_f1_mm"] = L_a - l_SM1
                else:
                    return
            except ValueError:
                return
        elif (parts.f1_mirror_type == "flat"
                and parts.f1_diameter_mm is not None
                and "theta_f1_deg" in kw
                and parts.f1_mount is not None):
            kw.pop("f1_fraction", None)
            theta_f1 = float(kw["theta_f1_deg"])
            slit_half_h = parts.slit_height_mm / 2.0
            min_slit_to_fold = (0.5 * parts.f1_diameter_mm
                                + parts.f1_mount["wall_margin_mm"])
            try:
                frac_lo = fold_fraction_lower_bound(
                    parts.m1_diameter_mm, slit_half_h,
                    parts.f1_diameter_mm, theta_f1)
                frac_hi = fold_fraction_upper_bound(L_a, min_slit_to_fold)
            except ValueError:
                return
            if frac_lo >= frac_hi:
                return
            kw["L_f1_mm"] = (frac_lo + frac_hi) / 2.0 * L_a
        elif "f1_fraction" in kw:
            kw["L_f1_mm"] = L_a * float(kw.pop("f1_fraction"))
        if "f2_fraction" in kw:
            kw["L_f2_mm"] = L_b * float(kw.pop("f2_fraction"))

    # ── Phases 1-3 (shared) ─────────────────────────────────────────────

    def _mirror_positions(self, genome: CzernyGenome) -> _MirrorPositions:
        alpha = radians(genome.alpha_deg)
        beta = radians(genome.beta_deg)
        t1 = radians(genome.theta_m1_deg)
        t2 = radians(genome.theta_m2_deg)
        r1 = genome.L_m1_mm
        r2 = genome.L_m2_mm

        m1 = (-r1 * sin(alpha), +r1 * cos(alpha), 0.0)
        m2 = (-r2 * sin(beta),  +r2 * cos(beta),  0.0)

        d_m1_to_grating = (sin(alpha), -cos(alpha), 0.0)
        d_m2_to_grating = (sin(beta), -cos(beta), 0.0)

        d_m1_to_slit = _rotate_xy(d_m1_to_grating, -2.0 * t1)
        d_m2_to_slit = _rotate_xy(d_m2_to_grating, +2.0 * t2)
        d_arm_a = (-d_m1_to_slit[0], -d_m1_to_slit[1], 0.0)
        d_arm_b = d_m2_to_slit

        return _MirrorPositions(
            m1=m1, m2=m2,
            d_m1_to_grating=d_m1_to_grating, d_m2_to_grating=d_m2_to_grating,
            d_arm_a=d_arm_a, d_arm_b=d_arm_b,
        )

    def _slit_poses(
        self, genome: CzernyGenome,
        mirrors: _MirrorPositions,
        parts: CzernyParts,
    ) -> _SlitPoses:
        L_a = genome.L_a_mm
        L_b = genome.L_b_mm
        m1, m2 = mirrors.m1, mirrors.m2
        d_arm_a, d_arm_b = mirrors.d_arm_a, mirrors.d_arm_b

        use_f1 = (genome.L_f1_mm is not None
                      and genome.theta_f1_deg is not None)
        use_f2 = (genome.L_f2_mm is not None
                      and genome.theta_f2_deg is not None)

        entrance_slit_natural = (
            m1[0] - L_a * d_arm_a[0],
            m1[1] - L_a * d_arm_a[1],
            0.0,
        )
        exit_slit_natural = (
            m2[0] + L_b * d_arm_b[0],
            m2[1] + L_b * d_arm_b[1],
            0.0,
        )

        f1 = None
        if use_f1:
            L_f1 = float(genome.L_f1_mm)
            f1_pos = (
                m1[0] - L_f1 * d_arm_a[0],
                m1[1] - L_f1 * d_arm_a[1],
                0.0,
            )
            f1_deflection = 180.0 - 2.0 * float(genome.theta_f1_deg)
            f1_normal, entrance_slit, n_entrance_slit = _compute_fold_geometry(
                f1_pos,
                f1_deflection, L_a - L_f1,
                d_slit_to_m1=d_arm_a,
            )
            f1 = (f1_pos, f1_normal)
        else:
            entrance_slit = entrance_slit_natural
            n_entrance_slit = _normalize((m1[0] - entrance_slit[0],
                                          m1[1] - entrance_slit[1], 0.0))

        f2 = None
        if use_f2:
            L_f2 = float(genome.L_f2_mm)
            f2_pos = (
                m2[0] + L_f2 * d_arm_b[0],
                m2[1] + L_f2 * d_arm_b[1],
                0.0,
            )
            f2_deflection = 180.0 - 2.0 * float(genome.theta_f2_deg)
            f2_normal, exit_slit, n_exit_slit = _compute_fold_geometry(
                f2_pos,
                f2_deflection, L_b - L_f2,
                d_m2_to_slit=d_arm_b,
            )
            f2 = (f2_pos, f2_normal)
        else:
            exit_slit = exit_slit_natural
            n_exit_slit = _normalize((m2[0] - exit_slit[0],
                                      m2[1] - exit_slit[1], 0.0))

        return _SlitPoses(
            entrance_slit=entrance_slit, exit_slit=exit_slit,
            n_entrance_slit=n_entrance_slit, n_exit_slit=n_exit_slit,
            f1=f1, f2=f2,
        )

    def _surface_normals(
        self, genome: CzernyGenome,
        mirrors: _MirrorPositions,
        slits: _SlitPoses,
    ) -> _SurfaceNormals:
        d_m1_to_grating = mirrors.d_m1_to_grating
        d_arm_a = mirrors.d_arm_a
        d_arm_b = mirrors.d_arm_b

        d_in_m1 = d_arm_a
        d_out_m1 = d_m1_to_grating
        n_m1 = _normalize((
            d_out_m1[0] - d_in_m1[0],
            d_out_m1[1] - d_in_m1[1],
            0.0,
        ))
        if n_m1[0] * d_in_m1[0] + n_m1[1] * d_in_m1[1] > 0:
            n_m1 = (-n_m1[0], -n_m1[1], 0.0)

        d_in_m2 = (-mirrors.d_m2_to_grating[0], -mirrors.d_m2_to_grating[1], 0.0)
        d_out_m2 = d_arm_b
        n_m2 = _normalize((
            d_out_m2[0] - d_in_m2[0],
            d_out_m2[1] - d_in_m2[1],
            0.0,
        ))
        if n_m2[0] * d_in_m2[0] + n_m2[1] * d_in_m2[1] > 0:
            n_m2 = (-n_m2[0], -n_m2[1], 0.0)

        n_grating = (0.0, 1.0, 0.0)

        return _SurfaceNormals(
            n_m1=n_m1, n_m2=n_m2, n_grating=n_grating,
            n_entrance_slit=slits.n_entrance_slit,
            n_exit_slit=slits.n_exit_slit,
        )

    # ── Phase 4 (calls variant hooks) ──────────────────────────────────

    def _build_element_list(
        self,
        mirrors: _MirrorPositions,
        slits: _SlitPoses,
        normals: _SurfaceNormals,
        genome: CzernyGenome,
        parts: CzernyParts,
    ) -> list[ElementPlacement]:
        elements = [
            ElementPlacement(
                label="entrance_slit", kind="slit",
                position=slits.entrance_slit, axis=normals.n_entrance_slit,
                params={"width_um": parts.slit_width_um,
                        "height_mm": parts.slit_height_mm,
                        "optic_center_height_mm": parts.optic_center_height_mm,
                        **({"hex_half_mm": parts.slit_mount["hex_half_mm"],
                            "bore_radius_mm": parts.slit_mount["bore_radius_mm"],
                            "slit_length_mm": parts.slit_mount["length_mm"]}
                           if parts.slit_mount else {})},
            ),
            ElementPlacement(
                label="M1", kind="mirror",
                position=mirrors.m1,
                axis=self.mirror_axis("M1", mirrors, normals),
                params={
                    "focal_length_mm": parts.m1_focal_length_mm,
                    "diameter_mm": parts.m1_diameter_mm,
                    "edge_thickness_mm": parts.m1_edge_thickness_mm,
                    "center_thickness_mm": parts.m1_center_thickness_mm,
                    "reflectance": parts.m1_reflectance,
                    "mirror_type": parts.m1_mirror_type,
                    "off_axis_angle_deg": parts.m1_off_axis_angle_deg,
                    **self.mirror_params("M1", mirrors),
                    **({"reflectance_file": parts.m1_reflectance_file}
                       if parts.m1_reflectance_file else {}),
                    **({"cylindrical_orientation":
                        parts.m1_cylindrical_orientation}
                       if parts.m1_cylindrical_orientation else {}),
                },
            ),
            ElementPlacement(
                label="grating", kind="grating",
                position=(0.0, 0.0, 0.0), axis=normals.n_grating,
                params={
                    "size_mm": parts.grating_size_mm,
                    "edge_thickness_mm": parts.grating_thickness_mm,
                    "center_thickness_mm": parts.grating_thickness_mm,
                    "groove_density_per_mm": parts.grating_groove_density_per_mm,
                    "blaze_nm": parts.grating_blaze_nm,
                    "blaze_angle_deg": parts.grating_blaze_angle_deg,
                    "peak_efficiency": parts.grating_peak_efficiency,
                    **({"efficiency_file": parts.grating_efficiency_file}
                       if parts.grating_efficiency_file else {}),
                },
            ),
            ElementPlacement(
                label="M2", kind="mirror",
                position=mirrors.m2,
                axis=self.mirror_axis("M2", mirrors, normals),
                params={
                    "focal_length_mm": parts.m2_focal_length_mm,
                    "diameter_mm": parts.m2_diameter_mm,
                    "edge_thickness_mm": parts.m2_edge_thickness_mm,
                    "center_thickness_mm": parts.m2_center_thickness_mm,
                    "reflectance": parts.m2_reflectance,
                    "mirror_type": parts.m2_mirror_type,
                    "off_axis_angle_deg": parts.m2_off_axis_angle_deg,
                    **self.mirror_params("M2", mirrors),
                    **({"reflectance_file": parts.m2_reflectance_file}
                       if parts.m2_reflectance_file else {}),
                },
            ),
        ]

        det_axis = normals.n_exit_slit
        if genome.theta_d_deg != 0.0:
            det_axis = _rotate_xy(normals.n_exit_slit,
                                  -radians(genome.theta_d_deg))

        elements.append(
            ElementPlacement(
                label="detector", kind="detector",
                position=slits.exit_slit, axis=det_axis,
                params={"width_um": parts.slit_width_um,
                        "height_mm": parts.slit_height_mm,
                        "optic_center_height_mm": parts.optic_center_height_mm,
                        "board_width_mm": parts.detector_board_width_mm,
                        "board_height_mm": parts.detector_board_height_mm,
                        "board_behind_die_mm": parts.detector_board_behind_die_mm,
                        "glass_to_die_mm": parts.detector_glass_to_die_mm,
                        "boundary_behind_die_mm": parts.detector_boundary_behind_die_mm,
                        "package_width_mm": parts.detector_package_width_mm,
                        "package_height_mm": parts.detector_package_height_mm,
                        "insert_spacing_width_mm": parts.detector_insert_spacing_width_mm,
                        "insert_spacing_height_mm": parts.detector_insert_spacing_height_mm,
                        "solder_clearance_depth_mm": parts.detector_solder_clearance_depth_mm,
                        "corner_boss_radius_mm": parts.detector_corner_boss_radius_mm,
                        "array_length_mm": parts.detector_array_length_mm},
            )
        )

        use_f1 = slits.f1 is not None
        use_f2 = slits.f2 is not None
        if use_f1:
            f1_params = {
                "diameter_mm": parts.f1_diameter_mm,
                "edge_thickness_mm": parts.f1_edge_thickness_mm,
                "center_thickness_mm": parts.f1_center_thickness_mm,
                "reflectance": parts.f1_reflectance,
                "mirror_type": parts.f1_mirror_type,
                "off_axis_angle_deg": 0.0,
            }
            if parts.f1_reflectance_file:
                f1_params["reflectance_file"] = parts.f1_reflectance_file
            if parts.f1_focal_length_mm is not None:
                f1_params["focal_length_mm"] = parts.f1_focal_length_mm
            if parts.f1_cylindrical_orientation is not None:
                f1_params["cylindrical_orientation"] = (
                    parts.f1_cylindrical_orientation)
            elements.insert(1, ElementPlacement(
                label="F1", kind="mirror",
                position=slits.f1[0], axis=slits.f1[1],
                params=f1_params,
            ))
        if use_f2:
            f2_params = {
                "diameter_mm": parts.f2_diameter_mm,
                "edge_thickness_mm": parts.f2_edge_thickness_mm,
                "center_thickness_mm": parts.f2_center_thickness_mm,
                "reflectance": parts.f2_reflectance,
                "mirror_type": parts.f2_mirror_type,
                "off_axis_angle_deg": 0.0,
            }
            if parts.f2_reflectance_file:
                f2_params["reflectance_file"] = parts.f2_reflectance_file
            if parts.f2_focal_length_mm is not None:
                f2_params["focal_length_mm"] = parts.f2_focal_length_mm
            if parts.f2_cylindrical_orientation is not None:
                f2_params["cylindrical_orientation"] = (
                    parts.f2_cylindrical_orientation)
            m2_idx = 4 if use_f1 else 3
            elements.insert(m2_idx + 1, ElementPlacement(
                label="F2", kind="mirror",
                position=slits.f2[0], axis=slits.f2[1],
                params=f2_params,
            ))

        return elements

    # ── Public builder ─────────────────────────────────────────────────

    def build_optics_only_scene(
        self, genome: CzernyGenome, parts: CzernyParts,
    ) -> Scene:
        """Lay out an asymmetric Czerny-Turner from the genome and parts."""
        mirrors = self._mirror_positions(genome)
        slits = self._slit_poses(genome, mirrors, parts)
        normals = self._surface_normals(genome, mirrors, slits)
        elements = self._build_element_list(mirrors, slits, normals, genome, parts)
        return Scene(elements=elements)


# ─── Fold-mirror aperture geometry ───────────────────────────────────────

def fold_fraction_lower_bound(
    mirror_diameter_mm: float,
    slit_half_height_mm: float,
    fold_mirror_diameter_mm: float,
    max_incidence_deg: float = 60.0,
) -> float:
    """Lower bound on fold_fraction such that the beam at the fold
    position fits within the fold mirror's projected aperture."""
    h_fold_proj = 0.5 * fold_mirror_diameter_mm * cos(
        radians(max_incidence_deg))
    h_slit = float(slit_half_height_mm)
    h_mirr = 0.5 * float(mirror_diameter_mm)
    if h_mirr <= h_slit:
        raise ValueError(
            f"fold_fraction_lower_bound: curved mirror half-extent "
            f"{h_mirr:.3f} mm ≤ slit half-height {h_slit:.3f} mm; "
            f"beam envelope doesn't expand along the arm."
        )
    if h_fold_proj <= h_slit:
        raise ValueError(
            f"fold_fraction_lower_bound: slit half-height "
            f"{h_slit:.3f} mm exceeds projected fold aperture "
            f"{h_fold_proj:.3f} mm (D_fold={fold_mirror_diameter_mm:.3f} "
            f"mm × cos({max_incidence_deg:.1f}°)/2). Fold mirror "
            f"is too small for this product class."
        )
    return 1.0 - (h_fold_proj - h_slit) / (h_mirr - h_slit)


def fold_fraction_upper_bound(
    L_arm_mm: float,
    min_slit_to_fold_mm: float,
) -> float:
    """Upper bound on fold_fraction such that the fold mount does
    not clash with the slit bulkhead assembly."""
    if L_arm_mm <= min_slit_to_fold_mm:
        raise ValueError(
            f"fold_fraction_upper_bound: arm length {L_arm_mm:.3f} mm "
            f"≤ min_slit_to_fold {min_slit_to_fold_mm:.3f} mm. "
            f"Fold cannot be placed on this arm without clashing "
            f"with the slit bulkhead."
        )
    return 1.0 - min_slit_to_fold_mm / L_arm_mm


# ─── Grating-to-mirror distance (L_m1, L_m2) derivation ───────────────

def _r2_from_beam_walk(kw: dict, parts: CzernyParts,
                       *, band_nm: tuple[float, float]) -> float:
    """Maximum L_m2 at which the grating image fills M2 across the bandpass.

    L_m2 = (D_M2/2 - D_beam/2) / max|tan(β_edge − β_center)|

    The walk on M2's surface (perpendicular to the center beam) is
    L_m2·|tan(β_edge − β_center)|, NOT L_m2·|tan β_edge − tan β_center|.

    Uses D_M1 as the collimated beam diameter (conservative: ignores
    fiber-NA underfill).

    Grating equation: mλ = d(sinα + sinβ), m = −1, so
    sinβ_edge = mλ/d − sinα.
    """
    from math import tan

    m = -1
    alpha = radians(float(kw["alpha_deg"]))
    beta_center = radians(float(kw["beta_deg"]))
    d_nm = 1.0e6 / parts.grating_groove_density_per_mm
    sin_alpha = sin(alpha)

    lam_min, lam_max = band_nm
    max_walk = 0.0
    for lam in (lam_min, lam_max):
        arg = m * lam / d_nm - sin_alpha
        if abs(arg) > 1.0:
            return -1.0  # band edge unreachable — infeasible
        beta_edge = asin(arg)
        walk = abs(tan(beta_edge - beta_center))
        if walk > max_walk:
            max_walk = walk

    D_beam = parts.m1_diameter_mm
    D_M2 = parts.m2_diameter_mm
    margin = (D_M2 - D_beam) / 2.0

    if margin <= 0:
        return -1.0  # M2 too small for the beam
    if max_walk < 1e-12:
        return 200.0  # near-zero dispersion; use a reasonable default
    return margin / max_walk


def _r1_from_clearance(kw: dict, parts: CzernyParts) -> float:
    """Minimum L_m1 clearing optic envelopes along the M1 arm.

    Conservative geometric estimate: half-widths of M1 and grating
    projected along the arm, plus a clearance margin. The exact mount
    clearance check runs downstream in the assembly feasibility gates.
    """
    t1 = radians(float(kw["theta_m1_deg"]))
    D_M1_half = parts.m1_diameter_mm / 2.0
    grating_half = parts.grating_size_mm / 2.0
    clearance = 5.0  # mm, conservative margin for mounts
    return D_M1_half / sin(max(t1, radians(1.0))) + grating_half + clearance


# ─── Coma cancellation ──────────────────────────────────────────────────


def shafer_theta2(
    theta_m1_deg: float,
    R1_mm: float,
    R2_mm: float,
    alpha_deg: float,
    beta_deg: float,
) -> float:
    """Derive θ₂ from the coma-cancellation condition.

    Balances third-order coma between the two mirrors and the grating's
    anamorphic magnification:
        sin(θ₂)/sin(θ₁) = (R₂²·cos³θ₂)/(R₁²·cos³θ₁) · (cos³α/cos³β)

    For equal-R mirrors the R² terms cancel, leaving a pure angle
    constraint.  Solved numerically via Brent's method.

    Parameters
    ----------
    theta_m1_deg : M1 off-axis angle in degrees.
    R1_mm, R2_mm : Mirror radii of curvature (R = 2f for spherical).
    alpha_deg, beta_deg : Grating incidence / diffraction angles.

    Returns
    -------
    theta_m2_deg : M2 off-axis angle in degrees.

    Raises
    ------
    ValueError
        If no solution exists in [0.01°, 80°].
    """
    if abs(theta_m1_deg) < 0.01:
        return 0.0

    from scipy.optimize import brentq

    t1 = radians(theta_m1_deg)
    a = radians(alpha_deg)
    b = radians(beta_deg)

    R_ratio_sq = (R2_mm / R1_mm) ** 2
    cos_a_cubed = cos(a) ** 3
    cos_b_cubed = cos(b) ** 3
    anamorphic = cos_a_cubed / cos_b_cubed
    sin_t1 = sin(t1)
    cos_t1_cubed = cos(t1) ** 3
    rhs = sin_t1 * R_ratio_sq * anamorphic / cos_t1_cubed

    def f(t2_rad: float) -> float:
        return sin(t2_rad) / cos(t2_rad) ** 3 - rhs

    lo = radians(0.01)
    hi = radians(80.0)
    if f(lo) * f(hi) > 0:
        raise ValueError(
            f"shafer_theta2: no sign change on [{0.01}, {80.0}] deg "
            f"(theta1={theta_m1_deg}, R1={R1_mm}, R2={R2_mm}, "
            f"alpha={alpha_deg}, beta={beta_deg})"
        )
    t2_rad = brentq(f, lo, hi, xtol=1e-12)
    return degrees(t2_rad)


# ─── Astigmatism-free fold position ─────────────────────────────────────

def astigmatism_free_l_SM1(
    R_fold: float,
    R_coll: float,
    R_focus: float,
    phi_fold_deg: float,
    theta_m1_deg: float,
    theta_m2_deg: float,
) -> float:
    """Slit-to-fold distance for astigmatism-free modified CT.

    l_SM1 = R₁R₂R₃ / [2(R₁R₃ sec θ₁ + R₂R₃ cos φ₁ − R₁R₂ cos θ₂)]

    Parameters
    ----------
    R_fold : Radius of curvature of the sagittal cylindrical fold (F1).
    R_coll : Radius of curvature of the tangential cylindrical collimator (M1).
    R_focus : Radius of curvature of the spherical focusing mirror (M2).
    phi_fold_deg : Angle of incidence on the fold mirror.
    theta_m1_deg : Angle of incidence on M1.
    theta_m2_deg : Angle of incidence on M2.
    """
    phi1 = radians(phi_fold_deg)
    phi2 = radians(theta_m1_deg)
    phi3 = radians(theta_m2_deg)
    num = R_fold * R_coll * R_focus
    denom = 2.0 * (
        R_fold * R_focus / cos(phi2)
        + R_coll * R_focus * cos(phi1)
        - R_fold * R_coll * cos(phi3)
    )
    if denom <= 0:
        raise ValueError("astigmatism-free condition infeasible (denom <= 0)")
    return num / denom


# ─── Grating rotation (pure optics math) ────────────────────────────────

def grating_angles(
    Dv_deg: float,
    lambda_center_nm: float,
    groove_period_nm: float,
    diffraction_order: int = -1,
) -> tuple[float, float]:
    """Solve the grating equation mλ = d(sinα + sinβ) for α, β.

    First-order diffraction. Grating normal is fixed at +y.
    |α| is the incidence angle, |β| is the diffraction angle.
    """
    dev_rad = radians(Dv_deg)
    half_cos = cos(0.5 * dev_rad)
    arg = diffraction_order * lambda_center_nm / (2.0 * groove_period_nm * half_cos)
    if abs(arg) > 1.0:
        raise ValueError(
            f"grating equation has no solution: deviation {Dv_deg}° cannot route "
            f"λ_c={lambda_center_nm} nm with groove period {groove_period_nm:.2f} nm; "
            f"|mλ/(2d cos(dev/2))| = {abs(arg):.3f} > 1"
        )
    phi_rad = asin(arg)
    alpha_rad = 0.5 * dev_rad + phi_rad
    beta_rad = phi_rad - 0.5 * dev_rad
    return degrees(alpha_rad), degrees(beta_rad)
