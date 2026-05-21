"""Layout-independent scene assembly pipeline.

Subclasses provide label-to-BOM mappings and beam-path order;
this base class handles mount building, 2-D polygon collision
checks (via ``optics.collision``), footprint/beam-cone validation,
and solid housing construction.
"""

import math
from dataclasses import replace
from typing import Optional

from optics.scene import InfeasibleGeometry, Mount, Scene, ElementPlacement
from optics.mounts import build_detector_bbox, build_detector_wall_mount, build_slit_bbox, build_slit_wall_mount, dispatch_mount, parse_mount_params


def element_half_extent_mm(element) -> float:
    """Half-extent of the beam in the dispersion (xy) plane at the element."""
    if element.kind == "mirror":
        return 0.5 * element.params["diameter_mm"]
    if element.kind == "grating":
        return 0.5 * element.params["size_mm"]
    if element.kind == "slit":
        return 0.5 * element.params["width_um"] * 1e-3
    return 0.0


def _point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def min_distance_to_polygon(px, py, verts) -> float:
    n = len(verts)
    min_d = float("inf")
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        d = _point_to_segment_distance(px, py, x0, y0, x1, y1)
        if d < min_d:
            min_d = d
    return min_d




def _convex_hull_contains(px: float, py: float,
                          pts: list[tuple[float, float]]) -> bool:
    """True if (px, py) is strictly inside the convex hull of *pts*."""
    from math import atan2
    n = len(pts)
    if n < 3:
        return False
    # Sort points by angle from centroid to get convex hull order.
    cx = sum(x for x, _ in pts) / n
    cy = sum(y for _, y in pts) / n
    ordered = sorted(pts, key=lambda p: atan2(p[1] - cy, p[0] - cx))
    # Point-in-convex-polygon via cross-product winding.
    for i in range(len(ordered)):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % len(ordered)]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < 0:
            return False
    return True



def _union_bbox(bboxes: list[tuple[float, float, float, float]]
                ) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    xs_lo = min(b[0] for b in bboxes)
    xs_hi = max(b[1] for b in bboxes)
    ys_lo = min(b[2] for b in bboxes)
    ys_hi = max(b[3] for b in bboxes)
    return (xs_lo, xs_hi, ys_lo, ys_hi)


_SLIT_WALL_BORE_RADIUS_MM = 6.5


def _slit_mount_housing_dims(slit_mount: dict) -> tuple[float, float]:
    """Extract (widest_cross_section_mm, front_extent_mm) from the slit mount dict."""
    return 2.0 * slit_mount["hex_half_mm"], 0.0


class SpectrographAssembly:
    """Base class for topology-specific scene assemblers.

    Subclasses must set:
      - ``mount_bom_key_by_label``: maps element labels to BOM mount dict names
      - ``beam_path_canonical``: ordered list of beam-path element labels
    """

    mount_bom_key_by_label: dict[str, str] = {}
    beam_path_canonical: list[str] = []

    #: After ``assemble()`` completes with housing, this holds the
    #: pure-geometry spec for downstream CAD export / feasibility checks.
    solid_housing_spec = None

    def build_mounts(self, scene: Scene, parts) -> list[Mount]:
        mounts: list[Mount] = []
        for el in scene.elements:
            if el.label == "detector":
                mounts.append(build_detector_bbox(el, albedo=parts.wall_albedo))
                mounts.append(build_detector_wall_mount(el, albedo=parts.wall_albedo))
                continue
            if el.label == "entrance_slit":
                mounts.append(build_slit_bbox(el, parts.slit_mount,
                                              albedo=parts.wall_albedo))
                mounts.append(build_slit_wall_mount(el, parts.slit_mount,
                                                    albedo=parts.wall_albedo))
                continue
            bom_key = self.mount_bom_key_by_label.get(el.label)
            if bom_key is None:
                continue
            mount_dict = getattr(parts, bom_key, None)
            if not mount_dict:
                continue
            try:
                mount_type = str(mount_dict["type"])
                mount_params = parse_mount_params(mount_type, mount_dict)
            except ValueError:
                continue
            except (KeyError, TypeError) as exc:
                raise InfeasibleGeometry(
                    f"mount params parse failure: {exc}"
                ) from exc
            el_for_build = el
            if el.kind == "slit":
                el_for_build = replace(
                    el, params={**el.params, "height_mm": parts.slit_height_mm}
                )
            mounts.append(dispatch_mount(
                el_for_build, mount_type, mount_params,
                albedo=parts.wall_albedo,
            ))
        return mounts

    def check_slit_on_perimeter(self, scene: Scene) -> None:
        """Reject layouts where the entrance slit is inside the convex hull
        of the other elements. The HASMA must be on the housing perimeter."""
        slit = scene.by_label("entrance_slit")
        if slit is None:
            return
        others = [(el.position[0], el.position[1])
                  for el in scene.elements if el.label != "entrance_slit"]
        if _convex_hull_contains(slit.position[0], slit.position[1], others):
            raise InfeasibleGeometry(
                f"entrance slit at ({slit.position[0]:.1f}, "
                f"{slit.position[1]:.1f}) is inside the element "
                f"convex hull — no housing wall for HASMA access"
            )

    def check_footprint(
        self,
        mounts: list[Mount],
        elements: list[ElementPlacement],
        max_xy_mm: Optional[tuple[float, float]],
    ) -> None:
        if max_xy_mm is None:
            return
        bboxes = [m.bbox_xy_mm for m in mounts]
        for el in elements:
            x, y, _ = el.position
            bboxes.append((x, x, y, y))
        x_lo, x_hi, y_lo, y_hi = _union_bbox(bboxes)
        width_mm = x_hi - x_lo
        height_mm = y_hi - y_lo
        max_w, max_h = max_xy_mm
        if width_mm > max_w or height_mm > max_h:
            raise InfeasibleGeometry(
                f"footprint {width_mm:.1f}×{height_mm:.1f} mm exceeds "
                f"envelope {max_w:.1f}×{max_h:.1f} mm"
            )

    def resolve_beam_path(self, scene: Scene) -> list[ElementPlacement]:
        present = {el.label: el for el in scene.elements}
        return [present[lbl] for lbl in self.beam_path_canonical
                if lbl in present]

    def check_beam_cone(
        self,
        scene: Scene,
        mounts: list[Mount],
        *,
        solid_housing_spec=None,
        n_samples: int = 10,
    ) -> None:
        """Check beam cones against the housing outline.

        Mount-vs-beam-cone checks are handled by
        ``check_assembly_collisions`` (SAT on slab polygons).
        This method only checks the housing outline polygon when
        a solid housing spec is provided.
        """
        if solid_housing_spec is None:
            return
        poly = solid_housing_spec.housing_outline

        beam_elements = self.resolve_beam_path(scene)

        for seg_idx in range(len(beam_elements) - 1):
            el_a = beam_elements[seg_idx]
            el_b = beam_elements[seg_idx + 1]
            seg_name = f"{el_a.label}→{el_b.label}"

            ax, ay = el_a.position[0], el_a.position[1]
            bx, by = el_b.position[0], el_b.position[1]
            r_a = element_half_extent_mm(el_a)
            r_b = element_half_extent_mm(el_b)

            for k in range(n_samples + 1):
                t = k / n_samples
                px = ax + t * (bx - ax)
                py = ay + t * (by - ay)
                r = r_a + t * (r_b - r_a)

                in_aperture = False
                if el_a.kind == "slit":
                    if ((px - ax) ** 2 + (py - ay) ** 2
                            < _SLIT_WALL_BORE_RADIUS_MM ** 2):
                        in_aperture = True
                if not in_aperture and el_b.kind == "slit":
                    if ((px - bx) ** 2 + (py - by) ** 2
                            < _SLIT_WALL_BORE_RADIUS_MM ** 2):
                        in_aperture = True
                if in_aperture:
                    continue
                dist = min_distance_to_polygon(px, py, poly)
                if dist < r:
                    raise InfeasibleGeometry(
                        f"beam-cone clip: {seg_name} clips wall "
                        f"at t={t:.2f} (dist={dist:.1f}mm, "
                        f"need {r:.1f}mm)"
                    )

    def assemble(
        self,
        optics_scene: Scene,
        parts,
        *,
        max_footprint_xy_mm: tuple[float, float] | None = None,
    ) -> Scene:
        """Run the full assembly pipeline on a pre-built optics scene.

        Builds mounts, runs collision checks, and constructs the solid
        housing spec (stored on ``self.solid_housing_spec`` for CAD
        export).
        """
        self.solid_housing_spec = None

        from optics.collision import check_optic_collisions, check_assembly_collisions

        beam_path = self.resolve_beam_path(optics_scene)
        check_optic_collisions(optics_scene, beam_path)

        mounts = self.build_mounts(optics_scene, parts)
        check_assembly_collisions(optics_scene, mounts, beam_path)
        self.check_slit_on_perimeter(optics_scene)
        self.check_footprint(mounts, optics_scene.elements, max_footprint_xy_mm)

        from optics.housing import build_solid_housing_spec
        try:
            solid_spec = build_solid_housing_spec(
                optics_scene, mounts, parts, beam_path,
            )
        except (ValueError, StopIteration) as exc:
            raise InfeasibleGeometry(
                f"solid housing spec: {exc}"
            ) from exc
        self.solid_housing_spec = solid_spec

        fixtures = list(mounts)

        return Scene(
            elements=optics_scene.elements,
            fixtures=tuple(fixtures),
        )
