"""Design registry — maps problem_name to design-specific modules."""

import importlib

from designs.czerny import CzernyGeometry

GEOMETRY_REGISTRY = {
    "czerny": CzernyGeometry,
}

_REGISTRY = {
    "czerny": "designs.czerny_bom",
}


def get_design(problem_name: str):
    """Return the design module for *problem_name*.

    Each design module exposes:
      - ``assemble_scene(genome, parts, ...)``
      - ``build_optics_only_scene(genome, parts)``
      - ``reconstruct_genome(full_params, best_point, bom_snapshot) -> (genome, parts)``
      - ``add_genome_args(ap)``
      - ``resolve_genome_from_cli(args) -> (genome, m1, m2, grating, label)``
    """
    module_path = _REGISTRY.get(problem_name)
    if module_path is None:
        raise ValueError(f"Unknown design {problem_name!r}; "
                         f"registered: {list(_REGISTRY)}")
    return importlib.import_module(module_path)
