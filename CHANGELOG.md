# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/dookaloosy/open-spectrograph/releases/tag/v0.1.0
