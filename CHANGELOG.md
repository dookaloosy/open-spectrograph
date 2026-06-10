# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  paper. Measured 0.86 nm FWHM on isolated 405 nm Hg line.
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

[0.3.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.3.0
[0.2.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.2.0
[0.1.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.1.0
