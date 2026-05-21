"""Czerny-Turner scene assembler.

Subclass of SpectrographAssembly with CT-specific label mappings
and beam-path order. The public API is ``assemble_scene()``.
"""

from designs.czerny_base import (
    CzernyGenome,
    CzernyParts,
)
from optics.assembly import SpectrographAssembly
from optics.scene import InfeasibleGeometry


class CzernyAssembly(SpectrographAssembly):

    mount_bom_key_by_label = {
        # HASMA threads into housing wall — no mount for slits.
        "F1":            "f1_mount",
        "M1":            "m1_mount",
        "grating":       "grating_mount",
        "M2":            "m2_mount",
        "F2":            "f2_mount",
    }

    beam_path_canonical = [
        "entrance_slit", "F1", "M1", "grating", "M2", "F2", "detector",
    ]


_ASSEMBLY = CzernyAssembly()


def assemble_scene(
    genome: CzernyGenome,
    parts: CzernyParts,
    *,
    max_footprint_xy_mm: tuple[float, float] | None = None,
    scene_builder,
) -> "Scene":
    """Assemble the full Czerny-Turner product scene.

    ``scene_builder`` is the geometry variant's ``build_optics_only_scene``
    method (e.g. ``CzernyGeometry().build_optics_only_scene``).

    Builds mounts, collision checks, and solid housing.
    Raises ``InfeasibleGeometry`` on any pre-trace validation failure.
    """
    optics_scene = scene_builder(genome, parts)
    return _ASSEMBLY.assemble(
        optics_scene, parts,
        max_footprint_xy_mm=max_footprint_xy_mm,
    )
