"""Transparent material that records every ray intersection.

The ray is logged and continues through the surface unattenuated,
without incrementing ``ray.depth``.  Used as an overlay on top of
real optical elements to capture beam footprints without altering
the physics of the scene.
"""


from dataclasses import dataclass

from raysect.optical.material import Material


@dataclass
class Hit:
    """One recorded ray-detector intersection."""

    point: tuple[float, float, float]     # world-frame hit coordinates (metres)
    direction: tuple[float, float, float] # world-frame ray direction at hit
    wavelength_nm: float                  # midpoint of ray's spectral range
    ray_depth: int                        # raysect's depth counter (bounces so far)


class HitRecorder(Material):
    """Transparent material that logs every hit."""

    def __init__(self):
        super().__init__()
        self.hits: list[Hit] = []

    def reset(self) -> None:
        self.hits.clear()

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if not exiting:
            w_point = hit_point.transform(primitive_to_world)
            d = ray.direction
            self.hits.append(Hit(
                point=(w_point.x, w_point.y, w_point.z),
                direction=(d.x, d.y, d.z),
                wavelength_nm=0.5 * (ray.min_wavelength + ray.max_wavelength),
                ray_depth=ray.depth,
            ))
            origin = inside_point.transform(primitive_to_world)
        else:
            origin = outside_point.transform(primitive_to_world)
        daughter = ray.spawn_daughter(origin, ray.direction.normalise())
        daughter.depth = ray.depth
        return daughter.trace(world)

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum
