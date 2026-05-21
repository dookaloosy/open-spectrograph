# CLAUDE.md

## Repository Purpose

Open-source fiber spectrograph: simulation, optimization, CAD export,
documentation, and paper — all in one repo. Inspired by the OpenFlexure
Microscope: printable STEP files + BOM listings for anyone to build their
own spectrograph.

Depends on `evolutionary-solver` (sibling repo) for the domain-agnostic
sweep + optimizer engine.

## Setup

```bash
# 1. Sibling solver repo
git clone https://github.com/dookaloosy/evolutionary-solver.git ../evolutionary-solver

# 2. Bootstrap dev environment (creates .venv/, installs raysect +
#    scientific Python deps + build123d CAD, pip-installs
#    evolutionary-solver and . in editable mode)
./bootstrap.sh

# 3. Activate
source .venv/bin/activate
```

## Product definition

Fixed-grating Czerny-Turner fiber spectrograph. Same class of instrument
as Ocean Optics USB4000 / Flame series.

**Optical train (v0 — Thorlabs COTS):**
25 µm SMA-905 fiber (NA 0.12) → HASMA bulkhead → F1 (sagittal
cylindrical fold, f=25mm) → M1 (tangential cylindrical, f=100mm) →
Grating (600 g/mm, fixed) → M2 (spherical, f=100mm, D=50.8mm) →
TCD1304 SPI sensor board + controller (3648 px, 8 µm pitch,
29.2 mm array, 16-bit, USB) → spectral data.

**Key numbers:** ~15 nm/mm dispersion, ~440 nm per frame, 400-700 nm
band, ~0.38 nm geometric bandpass, <$500 BOM target. Design wavelength
550 nm.

## Repo layout

```
optics/                 — core library (scene, ray tracing, fitness, housing)
optics/elements/        — custom raysect materials (mirrors, grating, recorder)
designs/                — topology modules (Czerny-Turner variants)
data/                   — targets TOML, BOM TOMLs, baselines, efficiency CSVs
  BOMs (--bom flag selects which catalog):
    czerny_bom_v0_design.toml         Thorlabs COTS design (default)
    czerny_bom_tl_corrected.toml      Thorlabs COTS corrected CT (test fixture)
    czerny_bom_tl_standard.toml       Thorlabs COTS standard CT
    czerny_bom_tl_oap.toml            Thorlabs COTS OAP CT (procedural CAD tests)
    czerny_bom_xia2017_standard.toml  Xia 2017 validation — standard CT
    czerny_bom_xia2017_corrected.toml Xia 2017 validation — corrected CT
  Baselines:
    czerny_baseline_v0_design.toml    v0 design genome + part selections
    czerny_baseline_xia2017.toml      Xia 2017 validation baseline
  Other:
    czerny_targets.toml               design targets (center λ, BW, RLD, f/#)
    defaults.toml                     default sweep/optimizer params (CLI overrides)
    thorlabs_catalog.toml             Thorlabs optics catalog
  data/efficiency/                    grating/mirror efficiency curves (CSV, images)
  data/step/                          vendor STEP files (not redistributable)
run_optimizer.py        — GA optimizer entry point
export.py               — render / STEP / CAD export
tests/                  — unit tests (assert-based validation)
scripts/                — diagnostic and analysis utilities
open-spectrograph.tex   — paper (LaTeX, self-contained)
docs/                   — paper section sources (01-introduction,
                          02-design-principles, 03-simulation, 04-design-v0)
```

## Quick Reference

```bash
source .venv/bin/activate

# Optimizer — targets from czerny_targets.toml, params from defaults.toml
python3 run_optimizer.py czerny --n_workers 12
python3 run_optimizer.py --resume output/optim_<run_dir>
python3 run_optimizer.py --winners
python3 run_optimizer.py --best

# Export — from a GA run
python3 export.py --run output/optim_<run_dir> --layout
python3 export.py --run output/optim_<run_dir> --step
python3 export.py --run output/optim_<run_dir> --render

# Export — from baseline (no GA run needed; BOM resolved from baseline)
python3 export.py --baseline data/czerny_baseline_v0_design.toml --layout
python3 export.py --baseline data/czerny_baseline_v0_design.toml --cad --step
# --cad generates bom.csv (tallied from placed CAD assembly labels)

# Tests
python3 -m pytest tests/

# Build paper
./build_paper.sh

# Scripts
python3 scripts/sensitivity.py --run output/optim_<run_dir>
python3 scripts/spot_panel.py --run output/optim_<run_dir>
python3 scripts/beam_profile.py --run output/optim_<run_dir>
```

## Conventions

- **Units**: `Scene` and geometry modules use millimetres. raysect uses SI
  (metres). The mm→m boundary is in `optics/world_builder.py`.
- **Coordinate frame**: elements laid out in xy plane (z = 0).
- **No silent fallbacks**: every parameter traces to a TOML file or CLI flag.
  Missing keys raise errors, not defaults. `dict[key]` (strict), not `.get()`.
- **Grating equation**: `mλ = d(sin α + sin β)`, first order (|m| = 1).
  Code uses `m = −1` (sign convention artifact, not physical order).
- **BOM thickness**: `center_thickness_mm` = back face to optical vertex.
  `edge_thickness_mm` = blank height at rim. Both required, no fallback.
- **Design f/#**: encoded in the BOM via M1 `diameter_mm = f / f#`.
  No hardcoded f/# constants — the BOM is the single source of truth.
- **Cylindrical orientation**: every cylindrical mirror in the BOM must
  declare `cylindrical_orientation` explicitly (`"tangential"` for M1,
  `"sagittal"` for F1). No implicit defaults.
- **Procedural CAD**: all solids generated from BOM parameters — no vendor
  STEP files shipped. See `tests/test_procedural_cad.py` for validation.

## Where to find deeper context

- Optical theory and design principles: `docs/02-design-principles.md`
- Simulation architecture: `docs/03-simulation.md`
- v0 design (Thorlabs COTS): `docs/04-design-v0.md`

## Sibling repos

| Repo | Purpose |
|------|---------|
| `evolutionary-solver` | Domain-agnostic sweep + optimizer engine |
