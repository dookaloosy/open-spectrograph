"""Scene rendering utilities for raysect worlds.

Camera placement, checkerboard room, and PNG export. Used by
scripts (``export_scene.py`` etc.) to render assembled scenes;
generic enough for any raysect world.
"""


import math

from raysect.core.math import Point3D, Vector3D, rotate_basis, translate
from raysect.optical import ConstantSF
from raysect.optical.material.emitter.checkerboard import Checkerboard
from raysect.optical.observer.imaging.orthographic import OrthographicCamera
from raysect.optical.observer.imaging.pinhole import PinholeCamera
from raysect.optical.observer.pipeline.rgb import RGBPipeline2D
from raysect.primitive import Box

from optics._raysect_wrapper import safe_observe
from optics.world_builder import MM_TO_M


def camera_transform(distance_m: float, azimuth_deg: float,
                     elevation_deg: float,
                     target: Point3D = Point3D(0.0, 0.0, 0.0)):
    """Orbit camera at (distance, azimuth, elevation) aimed at *target*."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = target.x + distance_m * math.cos(el) * math.cos(az)
    y = target.y + distance_m * math.cos(el) * math.sin(az)
    z = target.z + distance_m * math.sin(el)
    forward = Vector3D(target.x - x, target.y - y, target.z - z).normalise()
    up = Vector3D(0.0, 0.0, 1.0)
    return translate(x, y, z) * rotate_basis(forward, up)


def camera_look_from_to(position: Point3D, aim: Point3D):
    """Camera transform looking from *position* toward *aim*."""
    forward = Vector3D(aim.x - position.x, aim.y - position.y,
                       aim.z - position.z).normalise()
    if abs(forward.z) < 0.99:
        up = Vector3D(0.0, 0.0, 1.0)
    else:
        up = Vector3D(0.0, 1.0, 0.0)
    return translate(position.x, position.y, position.z) * rotate_basis(
        forward, up)


def element_position_m(scene, label: str) -> Point3D:
    """Return the scene element's position in metres."""
    for element in scene.elements:
        if element.label == label:
            return Point3D(
                element.position[0] * 1e-3,
                element.position[1] * 1e-3,
                element.position[2] * 1e-3,
            )
    raise KeyError(f"element {label!r} not in scene")


def element_normal_vec(scene, label: str) -> Vector3D:
    """Return the scene element's outward normal as a raysect Vector3D."""
    for element in scene.elements:
        if element.label == label:
            n = element.axis
            return Vector3D(n[0], n[1], n[2]).normalise()
    raise KeyError(f"element {label!r} not in scene")


def build_checkerboard_room(world, side_m: float = 4.0,
                            check_width_m: float = 0.1):
    """Add a checkerboard-textured cube around the world for visual context."""
    half = 0.5 * side_m
    material = Checkerboard(
        check_width_m, ConstantSF(1.0), ConstantSF(1.0), 0.4, 0.8,
    )
    Box(
        lower=Point3D(-half, -half, -half),
        upper=Point3D(+half, +half, +half),
        parent=world,
        material=material,
    )


def render_scene_png(
    world,
    output_path,
    *,
    pixels: tuple[int, int] = (1280, 960),
    samples: int = 100,
    spectral_bins: int = 24,
    spectral_rays: int | None = None,
    fov_deg: float = 40.0,
    camera_transform_matrix=None,
    unsaturated_fraction: float = 0.995,
    orthographic: bool = False,
    ortho_width_mm: float = 30.0,
    ray_max_depth: int = 15,
) -> None:
    """Render a raysect world to a PNG file.

    *camera_transform_matrix* is a raysect AffineMatrix3D (from
    ``camera_transform`` or ``camera_look_from_to``). If None, a
    default orbit camera is used.
    """
    from pathlib import Path

    pipeline = RGBPipeline2D(display_progress=False)
    pipeline.display_unsaturated_fraction = unsaturated_fraction

    if orthographic:
        camera = OrthographicCamera(
            pixels,
            width=ortho_width_mm * 1e-3,
            parent=world,
            pipelines=[pipeline],
        )
    else:
        camera = PinholeCamera(
            pixels,
            fov=fov_deg,
            parent=world,
            pipelines=[pipeline],
        )
    camera.spectral_bins = spectral_bins
    camera.spectral_rays = spectral_rays or spectral_bins
    camera.pixel_samples = samples
    camera.ray_max_depth = ray_max_depth

    if camera_transform_matrix is not None:
        camera.transform = camera_transform_matrix

    safe_observe(camera)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(out))
