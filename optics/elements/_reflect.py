"""Specular reflection utility shared by all mirror materials.

Raysect's specular math lives in Cython ``cdef`` methods inside
``Conductor`` and is not callable from Python.  This is the only
copy of the geometric formula used by our mirror materials.
"""


from raysect.core.math import Vector3D


def specular_reflect(d_local: Vector3D, normal: Vector3D) -> Vector3D:
    """Standard specular reflection: r = d - 2(d.n)n, normalised."""
    dx, dy, dz = d_local.x, d_local.y, d_local.z
    nx, ny, nz = normal.x, normal.y, normal.z
    dot = dx * nx + dy * ny + dz * nz
    return Vector3D(
        dx - 2.0 * dot * nx,
        dy - 2.0 * dot * ny,
        dz - 2.0 * dot * nz,
    ).normalise()
