"""Translate a `Scene` into a raysect `World`.

Takes a geometry-level `Scene` (positions and normals in millimetres) and
wires the elements into a raysect `World` with custom materials:
`SphericalMirror` on the mirrors (Sphere∩Cylinder CSG with true curved
surface — see `optics/elements/spherical_mirror.py`),
`TabulatedBlazedGrating` on the grating, and `FaceDiscriminatingEmitter`
on the entrance slit (f/#-matched cone source with built-in aperture mask).

The entrance slit uses a thin Box with `FaceDiscriminatingEmitter` material.
Only the front face (local +z) emits; side and back faces return zero
spectrum, acting as a built-in aperture mask without needing a jaw plate.

Unit conversion: the `Scene` is in millimetres; raysect is SI (metres).
All mm values cross the boundary here and are multiplied by `MM_TO_M`.

Coordinate convention: the `Scene` lays elements out in the xy plane with
z = 0. Raysect uses its own right-handed frame; we embed the Scene's xy
plane directly as raysect's xy plane and keep z as the out-of-plane axis.

Primitive convention: each disk/box is placed so that its flat hit face
lands on the element's nominal Scene position. Cylinders are backed off
by `_DISK_THICKNESS` along the local -z axis before the element transform
is applied, so the face at local z=+thickness (the one the beam hits)
coincides with the nominal position rather than sitting one thickness
behind it.
"""


import math
from dataclasses import dataclass

from raysect.core.math import Point3D, Vector3D, rotate_basis, rotate_x, rotate_y, rotate_z, translate
from raysect.optical import World
from raysect.optical.observer.nonimaging import TargetedPixel
from raysect.optical.observer.pipeline.mono import PowerPipeline0D
from raysect.primitive import Box, Cylinder, Sphere, Subtract

from optics.scene import (
    ElementPlacement,
    Scene,
)
from optics.elements import (
    BlazedGrating,
    CylindricalMirror,
    FlatMirror,
    ParaboloidalMirror,
    TabulatedBlazedGrating,
    TabulatedCylindricalMirror,
    TabulatedFlatMirror,
    TabulatedParaboloidalMirror,
    SphericalMirror,
    TabulatedSphericalMirror,
)
from raysect.optical.material import Material

MM_TO_M = 1.0e-3


class FaceDiscriminatingEmitter(Material):
    """Cone-limited emitter that only radiates from the front face.

    Subclasses ``Material`` directly (not ``AnisotropicSurfaceEmitter``)
    to get access to the surface ``normal`` in ``evaluate_surface``.
    Only the front face (local +z outward normal) emits; side and back
    faces return zero spectrum, acting as a built-in aperture mask.

    This eliminates the need for a separate jaw plate to block
    backward rays from clipping side faces of the emitter box.
    """

    def __init__(self, radiance: float, cos_min: float):
        super().__init__()
        if cos_min < 0.1:
            raise ValueError(
                f"cos_min={cos_min:.4f} is too small (half-angle "
                f">{math.degrees(math.acos(0.1)):.0f}°); "
                f"1/cos divergence will cause extreme variance"
            )
        self._radiance = float(radiance)
        self._cos_min = float(cos_min)

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting,
                         inside_point, outside_point, normal,
                         world_to_primitive, primitive_to_world, intersection):
        if exiting:
            return ray.new_spectrum()
        # Front face of the Box has outward normal ≈ (0, 0, +1) in local frame.
        if normal.z < 0.5:
            return ray.new_spectrum()
        d_local = ray.direction.transform(world_to_primitive).normalise()
        cosine = abs(d_local.z)
        if cosine < self._cos_min:
            return ray.new_spectrum()
        spectrum = ray.new_spectrum()
        r = self._radiance / cosine
        for i in range(len(spectrum.samples)):
            spectrum.samples[i] = r
        return spectrum

    def evaluate_volume(self, spectrum, world, ray, primitive,
                        start_point, end_point,
                        world_to_primitive, primitive_to_world):
        return spectrum


def _orthogonal_up(forward: Vector3D) -> Vector3D:
    """Return an up vector orthogonal to `forward` via Gram-Schmidt.

    Prefers global +z as the raw up direction; falls back to +y when
    `forward` is near-vertical to avoid collinearity.
    """
    if abs(forward.z) < 0.99:
        raw_up = Vector3D(0.0, 0.0, 1.0)
    else:
        raw_up = Vector3D(0.0, 1.0, 0.0)
    dot = raw_up.dot(forward)
    return Vector3D(
        raw_up.x - dot * forward.x,
        raw_up.y - dot * forward.y,
        raw_up.z - dot * forward.z,
    ).normalise()


# Default samples per pixel for the backward slit-plane observer.
# Callers override this (typically 1M–100M) via the observer's
# `pixel_samples` attribute before calling `observe()`.
_BACKWARD_PIXEL_SAMPLES = 1_000_000

# Targeting probability for M2.  95 % of rays are importance-sampled
# toward M2's bounding sphere (the imaging path); the remaining 5 %
# sample the full hemisphere, picking up wall-bounce stray paths.
_BACKWARD_TARGETED_PATH_PROB = 0.95

# Raysect render engine configuration via environment variable:
#   OPTICS_SIM_WORKERS=N  → MulticoreEngine with N processes
#   OPTICS_SIM_WORKERS=1  → SerialEngine (safe, no child processes)
#   unset                 → SerialEngine (GA parallelises at candidate level)
#
# Start method is 'fork' because raysect's Cython internals (memoryview
# slices) are not picklable, which 'spawn' requires. The real safety net
# against orphaned workers on interrupt is `safe_observe()` in
# `_raysect_wrapper.py`, not the start method.
import os as _os
import warnings as _warnings
_env_workers = _os.environ.get("OPTICS_SIM_WORKERS")
if _env_workers is not None:
    try:
        _N_WORKERS = int(_env_workers)
    except ValueError:
        _N_WORKERS = 1
        _warnings.warn(
            f"OPTICS_SIM_WORKERS={_env_workers!r} is not an integer, "
            f"falling back to SerialEngine",
            stacklevel=1,
        )
else:
    _N_WORKERS = 1
_START_METHOD = _os.environ.get("OPTICS_SIM_START_METHOD", "fork")


@dataclass
class BuiltWorld:
    """Container for a raysect World and its element-primitive map.

    `build_world` creates the optical primitives (mirrors, grating,
    entrance emitter) and parents design-injected geometry (mounts,
    baseplate, housing). `attach_observer` adds the backward-mode
    measurement endpoint (TargetedPixel observer at the exit slit).

    ``set_grating_rotation_deg`` rotates the grating primitive to an
    absolute angle from its home position, avoiding rotation drift.
    """

    world: World
    primitives: dict[str, object]
    observer: TargetedPixel | None = None
    observer_pipeline: PowerPipeline0D | None = None
    _grating_home: object = None

    def set_grating_rotation_deg(self, theta_geom_deg: float) -> None:
        """Rotate the grating to an absolute angle.

        Sign convention: positive toward M2 (geometry-side). Raysect's
        rotate_y is right-handed (opposite sign), so we negate.
        """
        self.primitives["grating"].transform = (
            self._grating_home * rotate_y(-theta_geom_deg)
        )


def build_world(
    scene: Scene,
    *,
    input_fnum: float,
) -> BuiltWorld:
    """Translate a geometry-level Scene into a raysect World.

    Builds mirrors, grating, fold mirrors, and entrance emitter from
    ``scene.elements``. Parents design-injected geometry (mounts,
    baseplate, housing) to the world.

    Does NOT set up measurement endpoints — call ``attach_observer``
    after this for backward tracing.
    """
    world = World()
    primitives: dict[str, object] = {}
    grating_home = None

    for element in scene.elements:
        if element.label == "entrance_slit":
            emitter = _build_entrance_slit(element, input_fnum)
            emitter.parent = world
            primitives["entrance_slit"] = emitter
            continue

        if element.kind == "detector":
            continue

        primitive, material = _build_element(element)
        primitive.material = material
        primitive.parent = world
        primitives[element.label] = primitive
        if element.label == "grating":
            grating_home = primitive.transform

    for fixture in scene.fixtures:
        if fixture.label in ("detector_bbox", "slit_bbox"):
            continue
        fixture.csg.parent = world
        primitives[fixture.label] = fixture.csg

    if grating_home is None:
        raise ValueError("scene has no 'grating' element")

    return BuiltWorld(
        world=world,
        primitives=primitives,
        _grating_home=grating_home,
    )


def attach_observer(
    built: BuiltWorld,
    scene: Scene,
    input_fnum: float,
) -> None:
    """Add backward-mode measurement endpoints to a built world.

    Places a ``TargetedPixel`` observer at the detector position,
    sized to the slit aperture, importance-sampled toward M2 (or F2
    if present). Also sets ``built.observer`` and
    ``built.observer_pipeline``.
    """
    det = next((e for e in scene.elements if e.kind == "detector"), None)
    if det is None:
        raise ValueError("scene has no detector element")

    observer_target = built.primitives.get("F2", built.primitives.get("M2"))
    if observer_target is None:
        raise ValueError(
            "attach_observer requires an 'M2' or 'F2' primitive for targeting"
        )

    pipeline = PowerPipeline0D(accumulate=False)
    observer = TargetedPixel(
        targets=[observer_target],
        x_width=det.params["width_um"] * 1.0e-6,
        y_width=det.params["height_mm"] * MM_TO_M,
        pipelines=[pipeline],
    )
    observer.pixel_samples = _BACKWARD_PIXEL_SAMPLES
    observer.targeted_path_prob = _BACKWARD_TARGETED_PATH_PROB
    observer.ray_max_depth = 15
    observer.quiet = True
    observer.transform = _placement_transform(det)
    observer.name = "detector_observer"

    if _N_WORKERS <= 1:
        from raysect.core.workflow import SerialEngine
        observer.render_engine = SerialEngine()
    else:
        from raysect.core.workflow import MulticoreEngine
        observer.render_engine = MulticoreEngine(
            processes=_N_WORKERS,
            start_method=_START_METHOD,
        )

    observer.parent = built.world
    built.primitives["detector_observer"] = observer
    built.observer = observer
    built.observer_pipeline = pipeline


# ─── Per-element builders ────────────────────────────────────────────────────

def _build_element(element: ElementPlacement):
    if element.kind == "mirror":
        return _build_mirror(element)
    if element.kind == "grating":
        return _build_grating(element)
    raise ValueError(f"unsupported element kind: {element.kind!r}")


def _build_mirror(element: ElementPlacement):
    """Concave focusing mirror primitive + material.

    Dispatches on ``element.params["mirror_type"]``:
    - ``"spherical"``: ``Subtract(Cylinder, Sphere)`` — a cylinder
      blank with the concave sphere (R=2f, centre at local (0,0,R))
      carved out.  Raysect delivers the true curved-surface normal;
      the material does standard specular reflection off it.
    - ``"paraboloidal"``: truncated cylinder (CSG) + virtual off-axis paraboloid.

    Spherical mirrors use ``_placement_transform`` (local +z = surface
    normal).  Paraboloidal mirrors orient local +z along the cylinder axis
    (toward grating) with local +x toward the parent focal axis, so
    the paraboloid equation ``(x-a)² + y² = 4f(z-v)`` works directly
    in the primitive's local frame.
    """
    diameter_mm = element.params["diameter_mm"]
    mirror_type = element.params["mirror_type"]
    refl_file = element.params.get("reflectance_file")

    if mirror_type == "spherical":
        focal_length_mm = element.params["focal_length_mm"]
        r_mirror_m = diameter_mm * 0.5 * MM_TO_M
        R_m = 2.0 * focal_length_mm * MM_TO_M
        sag_m = R_m - (R_m * R_m - r_mirror_m * r_mirror_m) ** 0.5
        substrate_m = element.params["center_thickness_mm"] * MM_TO_M

        cyl = Cylinder(
            radius=r_mirror_m,
            height=substrate_m + sag_m,
        )
        cyl.transform = translate(0.0, 0.0, -substrate_m)
        sphere = Sphere(radius=R_m)
        sphere.transform = translate(0.0, 0.0, R_m)

        mirror_prim = Subtract(cyl, sphere)
        mirror_prim.transform = _placement_transform(element)
        mirror_prim.name = element.label
        disk = mirror_prim

        if refl_file:
            material = TabulatedSphericalMirror.from_csv(
                focal_length_m=focal_length_mm * MM_TO_M,
                csv_path=refl_file,
            )
        else:
            material = SphericalMirror(
                focal_length_m=focal_length_mm * MM_TO_M,
                reflectance=element.params["reflectance"],
            )

    elif mirror_type == "cylindrical":
        focal_length_mm = element.params["focal_length_mm"]
        r_mirror_m = diameter_mm * 0.5 * MM_TO_M
        R_m = 2.0 * focal_length_mm * MM_TO_M
        sag_m = R_m - (R_m * R_m - r_mirror_m * r_mirror_m) ** 0.5
        substrate_m = element.params["center_thickness_mm"] * MM_TO_M

        aperture_shape = element.params.get("aperture_shape", "circular")
        if aperture_shape == "square":
            blank = Box(
                lower=Point3D(-r_mirror_m, -r_mirror_m, -substrate_m),
                upper=Point3D(+r_mirror_m, +r_mirror_m, sag_m),
            )
        else:
            blank = Cylinder(
                radius=r_mirror_m,
                height=substrate_m + sag_m,
            )
            blank.transform = translate(0.0, 0.0, -substrate_m)
        # Cut cylinder centre at (0, 0, R).  Axis orientation selects
        # which plane gets curvature:
        #   "tangential" (default) — axis along local y, curvature in xz
        #   "sagittal"             — axis along local x, curvature in yz
        # Uses rotate_basis (not rotate_x/rotate_y) — raysect CSG
        # requires rotate_basis for correct child-frame intersection.
        cyl_orient = element.params.get("cylindrical_orientation", "tangential")
        axis_rotation_deg = element.params.get("cylindrical_axis_rotation_deg", 0.0)
        cut_len = 2.0 * r_mirror_m + 0.01
        cut = Cylinder(radius=R_m, height=cut_len)
        if cyl_orient == "sagittal":
            cut.transform = (rotate_z(axis_rotation_deg)
                             * translate(-cut_len / 2.0, 0.0, R_m)
                             * rotate_basis(Vector3D(1, 0, 0),
                                            Vector3D(0, 0, 1)))
        else:
            cut.transform = (rotate_z(axis_rotation_deg)
                             * translate(0.0, -cut_len / 2.0, R_m)
                             * rotate_basis(Vector3D(0, 1, 0),
                                            Vector3D(0, 0, 1)))

        mirror_prim = Subtract(blank, cut)
        mirror_prim.transform = _placement_transform(element)
        mirror_prim.name = element.label
        disk = mirror_prim

        if refl_file:
            material = TabulatedCylindricalMirror.from_csv(
                focal_length_m=focal_length_mm * MM_TO_M,
                csv_path=refl_file,
                orientation=cyl_orient,
                axis_rotation_deg=axis_rotation_deg,
            )
        else:
            material = CylindricalMirror(
                focal_length_m=focal_length_mm * MM_TO_M,
                reflectance=element.params["reflectance"],
                orientation=cyl_orient,
                axis_rotation_deg=axis_rotation_deg,
            )

    elif mirror_type == "flat":
        r_mirror_m = diameter_mm * 0.5 * MM_TO_M
        thickness_m = element.params["edge_thickness_mm"] * MM_TO_M
        disk = Cylinder(radius=r_mirror_m, height=thickness_m)
        disk.transform = (_placement_transform(element)
                          * translate(0.0, 0.0, -thickness_m))
        disk.name = element.label

        if refl_file:
            material = TabulatedFlatMirror.from_csv(csv_path=refl_file)
        else:
            material = FlatMirror(
                reflectance=element.params["reflectance"])

    elif mirror_type == "paraboloidal":
        disk, material = _build_paraboloidal_mirror(element)

    else:
        raise ValueError(f"unknown mirror_type {mirror_type!r}")
    return disk, material


def _build_paraboloidal_mirror(element: ElementPlacement):
    """Paraboloidal mirror: ``Subtract(Cylinder, Parabola)`` — true curved surface.

    The parent paraboloid in the cylinder's local frame is::

        (x − a)² + y² = 4f(z − v)

    with axis at ``(a, 0)`` parallel to +z, vertex at ``(a, 0, v)``.
    Raysect's ``Parabola(radius, height)`` primitive is on-axis with
    vertex at ``z=height``, opening downward.  We rotate it 180° around
    x to open upward, then translate so its vertex lands at ``(a, 0, v)``.
    """
    from raysect.primitive import Parabola
    from optics.elements.oap_mirror import _paraboloid_params

    diameter_mm = element.params["diameter_mm"]
    focal_length_mm = element.params["focal_length_mm"]
    oaa = element.params["off_axis_angle_deg"]
    refl_file = element.params.get("reflectance_file")

    r_m = diameter_mm * 0.5 * MM_TO_M
    phi = math.radians(oaa)

    ct_mm = element.params["center_thickness_mm"]
    pa, pf, pv = _paraboloid_params(
        focal_length_mm * MM_TO_M, oaa,
        focus_height_m=ct_mm * MM_TO_M,
    )
    z_optical_centre = pv + pa * pa / (4.0 * pf)

    # BOM thickness_mm is the cylindrical blank height from the vendor
    # drawing. Subtract(Cylinder, Parabola) mirrors the manufacturing
    # process: start with this blank, cut the concave surface.
    h_cyl = element.params["edge_thickness_mm"] * MM_TO_M

    body = Cylinder(radius=r_m, height=h_cyl)

    # Parabola sizing: radius must cover the farthest cylinder point
    # from the paraboloid axis at (a, 0).
    parab_r = math.sqrt((pa + r_m) ** 2 + r_m ** 2) * 1.1
    parab_h = parab_r ** 2 / (4.0 * pf)

    parab = Parabola(radius=parab_r, height=parab_h)
    # Flip to open upward (vertex at bottom), then translate vertex to (a, 0, v).
    parab.transform = (
        translate(pa, 0.0, pv + parab_h)
        * rotate_basis(Vector3D(0.0, 0.0, -1.0), Vector3D(0.0, -1.0, 0.0))
    )

    prim = Subtract(body, parab)

    focus_dir = element.params["paraboloidal_focus_dir"]
    gx, gy, _ = element.axis
    fx, fy, _ = focus_dir

    forward = Vector3D(gx, gy, 0.0).normalise()
    right = Vector3D(fx, fy, 0.0).normalise()
    up = forward.cross(right).normalise()

    x, y, z = element.position
    prim.transform = (
        translate(x * MM_TO_M, y * MM_TO_M, z * MM_TO_M)
        * rotate_basis(forward, up)
        * translate(0.0, 0.0, -z_optical_centre)
    )
    prim.name = element.label

    ct_m = ct_mm * MM_TO_M
    if refl_file:
        material = TabulatedParaboloidalMirror.from_csv(
            reflected_focal_length_m=focal_length_mm * MM_TO_M,
            off_axis_angle_deg=oaa,
            csv_path=refl_file,
            center_thickness_m=ct_m,
        )
    else:
        material = ParaboloidalMirror(
            reflected_focal_length_m=focal_length_mm * MM_TO_M,
            off_axis_angle_deg=oaa,
            reflectance=element.params["reflectance"],
            center_thickness_m=ct_m,
        )
    return prim, material




def _build_grating(element: ElementPlacement):
    """Flat box with grating material.

    Wavelength-scan rotation is applied at trace time by
    `BuiltWorld.set_grating_for_wavelength()`, not at scene-build time.

    Uses `TabulatedBlazedGrating` (measured efficiency CSV) when the
    BOM provides an efficiency_file, otherwise falls back to
    `BlazedGrating` (sinc² model). Both are multi-order (m=0,±1,±2)
    with importance sampling — the same physics for forward and
    backward traces.
    """
    half = 0.5 * element.params["size_mm"] * MM_TO_M
    thickness = element.params["edge_thickness_mm"] * MM_TO_M
    grating_disk = Box(
        lower=Point3D(-half, -half, -thickness),
        upper=Point3D(+half, +half, 0.0),
    )
    grating_disk.transform = _placement_transform(element)
    grating_disk.name = element.label
    density = element.params["groove_density_per_mm"]
    eff_file = element.params.get("efficiency_file")
    if eff_file is not None:
        material = TabulatedBlazedGrating.from_csv(
            groove_density_per_mm=density,
            csv_path=eff_file,
        )
    else:
        material = BlazedGrating(
            groove_density_per_mm=density,
            blaze_angle_deg=element.params["blaze_angle_deg"],
            peak_efficiency=element.params["peak_efficiency"],
        )
    return grating_disk, material


def _build_entrance_slit(element: ElementPlacement, input_fnum: float):
    """Entrance-slit emitter with face-discriminating material.

    A thin Box at the slit plane with ``FaceDiscriminatingEmitter``
    material. Only the front face (local +z) emits; side and back
    faces return zero, acting as a built-in aperture mask. No jaw
    plate is needed.
    """
    half_w = 0.5 * element.params["width_um"] * 1.0e-6
    half_h = 0.5 * element.params["height_mm"] * MM_TO_M
    thickness = 1.0e-3 * MM_TO_M  # 1 μm — thin enough to not matter

    half_angle_rad = math.atan(1.0 / (2.0 * input_fnum))
    emitter = Box(
        lower=Point3D(-half_w, -half_h, -thickness),
        upper=Point3D(+half_w, +half_h, 0.0),
    )
    emitter.transform = _placement_transform(element)
    emitter.name = element.label
    emitter.material = FaceDiscriminatingEmitter(
        radiance=1.0, cos_min=math.cos(half_angle_rad))
    return emitter



# ─── Transform helpers ───────────────────────────────────────────────────────

def _point_mm_to_m(pos_mm) -> Point3D:
    return Point3D(pos_mm[0] * MM_TO_M, pos_mm[1] * MM_TO_M, pos_mm[2] * MM_TO_M)


def _placement_transform(element: ElementPlacement):
    """Compose `translate(world_pos) * rotate_basis(forward, up)`.

    Returns the transform that places the *placement origin* — body
    coordinates (0, 0, 0) — at the element's nominal world position
    with local +z pointing along the element normal. Each builder is
    responsible for any additional pre-translation needed to align its
    primitive's hit face with body z = 0 (or equivalently, the
    placement origin).

    `rotate_basis(forward, up)` builds an affine mapping local +z onto
    `forward` and local +y onto `up` (orthogonalised). The 'up' vector
    is arbitrary for rotationally-symmetric disks but pins the groove
    direction for the grating (its material treats local +y as the
    groove axis). We pick global +z if the normal has no z component,
    otherwise global +y, and orthogonalise.
    """
    x, y, z = element.position
    nx, ny, nz = element.axis

    forward = Vector3D(nx, ny, nz).normalise()
    up = _orthogonal_up(forward)

    return (
        translate(x * MM_TO_M, y * MM_TO_M, z * MM_TO_M)
        * rotate_basis(forward, up)
    )
