# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-06-23

Roll flexure for cylindrical axis alignment, foot bolt layout rework.

### Added
- 2-bladed leaf-spring roll flexure section on cylindrical mirror mounts
  (F1, M1), grating mount, and OAP mount.  Virtual pivot at optic
  centre; trapezoidal pedestals with heat-set insert bores; centre
  stiffening block between blades.
- Roll setscrew actuation: pair of M2×8 setscrews through heat-set
  inserts in the foot, pushing against pedestal lower surface.
- Hardware in grating flexure assembly (retention setscrew, pitch and
  roll pusher setscrews, heat-set inserts — matching mirror assemblies).
- Roll flexure clearance holes in housing floor for roll setscrew access.
- `tongue_half_mm` and `u_front_mm` fields on `Mount` dataclass;
  housing reads these instead of re-deriving tongue dimensions.
- Roll flexure BOM parameters: `roll_flexure_height_mm`,
  `roll_blade_thickness_mm`, `roll_insert_spacing_mm`,
  `roll_blade_spacing_mm`, `roll_pedestal_gap_mm`.

### Changed
- Foot bolt layout flipped: 2 forward bolts at `±foot_bolt_spacing_mm`,
  1 aft bolt at `u_wall_rear + 0.5 * boss_width`.  Wide tongue spans
  both forward bolts with radiused front corners.
- `foot_length_mm` simplified: `front_bolt_offset + 0.5 * boss_width`
  (removed `max()` guard that was pushing the tongue 3 mm past needed).
- `bolt_safety_mm` reduced from 6.0 to 2.0 (wide tongue provides
  adequate material around bolt heads).
- `front_bolt_offset_mm` reduced from 8.0 to 5.0 for all mounts.
- Aft foot bolt position changed from slab midpoint to
  `u_wall_rear + 0.5 * boss_width` (closer to rear wall).
- Pitch flexure parameters renamed: `flexure_thickness_mm` →
  `pitch_flexure_thickness_mm`, `flexure_gap_mm` →
  `pitch_flexure_gap_mm`.
- Face mill split into two bands when roll flexure is active (optic
  pocket + pitch shelf region), preserving the roll band at slab depth.
- Housing `_WALL_THICKNESS_MM` kept at 10.0 (lateral walls only).
- New `_LID_CLEARANCE_MM = 2.0` decouples lid gap from lateral wall
  thickness; z_top = tallest mount + lid clearance + cover depth.
- Housing `controller_cavity_wall_mm` reduced from 5.0 to 3.0.
- `_BOTTOM_COVER_MIN_WALL_MM` split: new `_FLARE_CEILING_WALL_MM = 3.0`
  for HASMA flare ceiling (was sharing `_BOTTOM_COVER_MIN_WALL_MM = 5.0`).
- HASMA tap fixture cone length reduced from 25.4 to 20.0 mm.
- `tongue_half_mm` and `u_front_mm` on `Mount` default to `None`
  (was `0.0`), consistent with other optional fields.
- Setscrew BOM: retention M2×4 (92605A043), pitch pushers M2×5
  (92605A044), roll pushers M2×8 (92605A912); removed M2×6
  (92605A047).
- Paper: added assembly procedure section (4.5), print settings in
  4.4, updated BOM cost table, ILF EE76 column headers, assembly
  dimensions, version references to v0.5.0.
- Mount body color (`_MOUNT_COLOR`) now set on all three mount types
  via `Part(result.wrapped)` cast.
- Assembly fixtures updated for wide tongue foot shape, roll flexure
  band hull, and corrected w-stacking.
- Housing locating ridges read `tongue_half_mm` from `Mount` object.
- Alignment screen tongue notch width updated for wide tongue.
- Raysect CSG mount builders synced with CAD: wide tongue foot,
  roll flexure band solid block (when `roll_flexure_height_mm > 0`).
- OAP BOM (`czerny_bom_tl_oap.toml`): switched foot bolts to M2,
  added `pusher_shelf_mm`, standardised bolt spacing to match
  regular mirror pattern (5/10 mm for 25/50 mm class).

## [0.4.0] — 2026-06-20

Flexure stiffening, mount foot locating ridges, laser alignment screen.

### Changed
- `flexure_thickness_mm` increased from 0.80 mm to 1.60 mm across all
  mounts (8× stiffer).  Back-of-envelope cantilever analysis predicts
  < 0.11 nm spectral shift at 45° box tilt, down from ~1 nm.
- `head_clearance_mm` increased from 5.0 mm to 6.0 mm.  Heat-set
  insert bottom now clears the top TPU contact bump by 0.4 mm (was
  overlapping by 0.6 mm).
- `bolt_safety_mm` increased from 2.0 mm to 6.0 mm, widening the
  foot tongue from 6 mm to 10 mm (3.3 mm wall per side of the 2.4 mm
  clearance hole).  Prevents tongue splitting under bolt clamping load
  due to weak interlayer adhesion.
- `insert_bore_dia_mm` reduced from 3.0 mm to 2.8 mm for tighter
  heat-set insert retention.

### Added
- Mount foot locating ridges on the housing cavity floor: L-shaped
  raised tabs (0.8 mm tall, 1.0 mm wide) at each mount foot position.
  Two L-ridges per mount follow the corner fillet through a continuous
  8-point arc (slicer-friendly toolpath), plus one back ridge spanning
  the crossbar width.  0.15 mm clearance constrains foot position to
  ±0.15 mm during screw tightening.
- `foot_outline_xy` field on `Mount` dataclass: actual T-shaped foot
  outline (not convex hull) for housing ridge placement.
- `mount_locating_ridges` field on `SolidHousingSpec`: pre-computed
  ridge rectangles from mount parameters.
- Laser alignment screen (`build_laser_alignment_screen` in
  `mounts_cad.py`): three-body multi-color print (frame, white disk,
  black reticle) with crosshair marks.  Base has tongue notch with
  corner-filleted corners, shallow ridge avoidance cuts (ridge height
  + clearance), and corner fillets matching both the foot fillet and
  L-ridge inner arc.
- Alignment screen exported as `alignment_screen.step` by `--cad`
  (standalone three-body STEP with tongue notch and ridge avoidance).
- `build_laser_alignment_holder` in `mounts_cad.py`: cuboid holder
  with coaxial bores for laser pointer and HASMA adapter alignment.

## [0.3.0] — 2026-06-12

Assembly fixtures, OAT tolerance overhaul, 0.86 nm first light.

### Changed
- ILF metric switched from FWHM to EE76 (encircled-energy width
  containing 76% of hits) in `optics/forward_trace.py`. Robust
  against split spots and skew tails.
- Tolerance analysis now uses true OAT perturbation for all axes:
  each element is perturbed independently without recomputing
  downstream positions. Layout distances shift only the target
  element; angular tilts rotate only the target element's axis.
- Rotation axes converted from one-sided perturbation to full ±
  sweep with plots.
- Dropped redundant θ\_F1 layout axis (identical to F1 normal tilt
  under OAT).
- Table 5 (v0 performance) and Table 6 (tolerance budget)
  recalculated. Table 6 uses paper variable names (θ, φ, α) with
  descriptions matching Table 4.
- Default `--max_rotation` changed from 5° to 2°.
- Mount construction steps reordered: outer contour (tombstone, side
  cuts, face mill, foot shape, pusher shelf) before internal features
  (bore, setscrew bore, bolt holes, chamfers).  Enables extracting the
  outer hull for fixture generation.
- Contact bumps separated from mount body as captive TPU inserts
  (3 mm dia cylinders in bore pockets). Three-point retention: two
  bottom bumps + one top bump, setscrew preloads optic against bottom
  pair.
- Contact bump parameters moved to BOM: `contact_radius_mm`,
  `contact_offset_mm`, `contact_separation_mm` per mount.
- `optic_clearance_mm` added to mount params (diametral bore/pocket
  clearance beyond assembly + print tolerance).  Default 2.0 mm.
- Bore sketch simplified: `upper_r`/`lower_r` unified to `bore_r`.
- Heat-set insert bore tightened from 3.1 mm to 3.0 mm.
- Mount STEP exports now include captive TPU contact bumps.
- Mirror builders (`_build_generic_spherical_mirror`,
  `_build_generic_cylindrical_mirror`) now return `Part` (via
  `Part.cast`) instead of `Compound`, fixing color export in
  assemblies.
- `_lookup_mirror_in_bom` unified: searches all four mirror groups
  (m1/m2/f1/f2), returns `focal_length_mm = inf` for flat mirrors.
  `_lookup_f_mirror_in_bom` removed.
- HASMA printed thread removed; replaced with tap drill bore
  (`tap_drill_dia_mm = 5.5`, formerly `pilot_bore_dia_mm`).  Thread
  cut post-print with tap guide fixture.
- Thread construction code, `Helix`/`sweep` imports, and
  `hasma_thread_tpi`/`hasma_thread_major_dia_mm` spec fields deleted.
- Setscrew and foot bolt holes use fixed-plane sketches with absolute
  coordinates instead of face-relative offsets (robust after mount
  step reordering).

### Added
- CFL spectrum figure (`fig_6_cfl_spectrum.png`) and §4.5 Results in
  paper. Measured 0.50 nm FWHM on isolated Hg 404.7 nm line
  (quadratic calibration, 6-line, 0.17 nm RMS).
- Normal tilt axes (θ\_M1, θ\_M2, θ\_F1, α) and cylinder axis
  rotation axes (φ\_M1, φ\_F1) in `scripts/tolerance.py`.
- Per-wavelength ILF histogram debug panels for all tolerance axes.
- Threshold-based tolerance budget: reports max deviation keeping
  ILF below a configurable target (default 1 nm), replacing the
  previous linear slope metric.
- `--ilf_target`, `--max_tilt` CLI flags for tolerance script.
- `build_mirror_assembly_fixture()`: envelope − mount hull (steps
  1–5) − mirror, with lip pocket (through-hole, 1 mm rim contact),
  foot/shelf relief through-cuts, and setscrew access hole.  Works
  for M1, M2, and F1.
- `build_grating_assembly_fixture()`: same pattern for grating mounts,
  rectangular lip pocket (full v-width, 1 mm w-lip), foot/shelf
  relief through-cuts.
- `build_hasma_tap_fixture()`: conical plug matching housing flare
  (45° half-angle) with 1/4" through bore to guide a 1/4"-36 tap.
- Fixture export integrated into `export.py --cad`: m1/m2/f1/grating
  fixtures + HASMA tap fixture alongside housing and mounts.
- `_FIXTURE_COLOR` (red) for fixture STEP exports.
- Paper section 4.4 "Printable parts": subsection 4.4.1 lists housing,
  covers, mounts, and TPU bumps; subsection 4.4.2 lists assembly
  fixtures.  All table captions moved above tables for consistency.
- Section 4.3 now shows `tolerance.py` command and references
  assembly fixtures for axis-rotation alignment.
- Section 4.2 performance table promoted to proper `table` environment
  with caption.
- F1 described as "sagittal cylindrical collimator" (not merely fold).
- `--` flags in `\texttt{}` escaped as `{-}{-}` for correct rendering.
- `\texttt{--}` dash fix applied to `--bom` reference in section 3.

### Fixed
- FWHM interpolation: added linear interpolation between bins for
  sub-pixel half-max crossings (before switching to EE76).

## [0.2.0] — 2026-05-24

Housing dimensional corrections for print fit.

### Changed
- Bottom cover plate thickness reduced from 5 mm to 3 mm; counterbore
  depth reduced from 4 mm to 2 mm to match.
- Bottom cover ledge constant split into `_BOTTOM_COVER_LEDGE_WIDTH_MM`
  (6 mm, horizontal shelf) and `_BOTTOM_COVER_EXTENSION_MM` (13 mm,
  vertical depth below z_bottom).
- Controller board placement shifted −3 mm toward z_bottom_ext
  (`_CONTROLLER_Z_INSET_MM`); pocket/solder/insert geometry unchanged.
- HASMA conical flare half-angle increased from 30° to 45°; cone now
  clipped to z-bounds (deepest cavity floor / top ledge − 5 mm wall).
- HASMA thread `_pitch_r` computed from pitch diameter (D − 0.6495P)
  instead of major diameter; internal minor now matches UNS-2B spec.
- HASMA pilot bore diameter moved from hardcoded 5.5 mm to BOM field
  `pilot_bore_dia_mm`.
- All fasteners switched from M2.5 to M2 self-tapping screws
  (99461A929 → 99461A915); cover screw, mount bolt, and insert pilot
  dimensions updated accordingly.
- Removed `print_tolerance_mm` adjustments from HASMA thread and
  detector package depth (covers still use print_tol for clearance fit).
- Assembly size updated from 157×172×85 mm to 157×172×99 mm in paper.
- Paper figure 4 updated to v0.2.0 CAD model screenshot.
- BOM cost language corrected: $500 is a target, not achieved ($912).
- Raysect reference corrected to Meakins and Carr (2014).
- Setscrew bores made blind (0.5 mm floor) so flat-tip setscrews can
  replace expensive nylon-tipped ones.
- Nylon-tip setscrew (93285A009, M2×3.8, $3.08) replaced with flat-tip
  (92605A044, M2×5, $0.41) for optic retention in all mounts.
- Pusher shelf added to all flexure mounts: semicircular lip above
  flexure gap prevents setscrew slipping past the flexure lip.
  Height configurable per mount via `pusher_shelf_mm` in BOM.
- Heat-set insert bore tightened from 3.3 mm to 3.1 mm.

### Added
- Bottom-side cavity glossary in housing_cad.py constants section.
- UNS-2B thread spec comments in BOM (`[slits.mount]`).

## [0.1.0] — 2026-05-21

Initial public release: v0 design (Thorlabs COTS optics).

### Added
- Czerny-Turner spectrograph simulation with raysect ray tracing
  (forward trace spot diagrams, throughput, ILF FWHM).
- Genetic algorithm optimizer via `evolutionary-solver` (coarse/fine
  grid sweep, Nelder-Mead basin refinement, collision detection).
- BOM-driven design: all part specs from TOML catalogs, no hardcoded
  optics. `--bom` selects which catalog to optimize against.
- Procedural CAD generation: all optics, mounts, and hardware solids
  built from BOM parameters (build123d). No vendor STEP files required.
- Unibody 3D-printed housing with flexure mounts, stepped cavity floor,
  top/detector/bottom covers, HASMA threaded bore, embossed ray path
  and text on top cover.
- STEP export: full assembly, individual parts, costed BOM CSV.
- Validation against Xia et al. (2017) published Zemax results.
- Diagnostic scripts: sensitivity sweep, spot panel, beam profile,
  stray light measurement, tolerance budget.
- Paper (LaTeX): design principles, simulation architecture, v0 design
  with performance and tolerance analysis.

[0.4.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.4.0
[0.3.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.3.0
[0.2.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.2.0
[0.1.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.1.0
