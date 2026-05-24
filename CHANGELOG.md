# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.2.0
[0.1.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.1.0
