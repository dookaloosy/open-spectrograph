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
band, ~0.38 nm geometric (theoretical) bandpass, 0.50 nm measured FWHM
at 404.7 nm, <$500 BOM target. Design wavelength 550 nm.

## Repo layout

```
optics/                 — core library (scene, ray tracing, fitness, housing)
optics/elements/        — custom raysect materials (mirrors, grating, recorder)
designs/                — topology modules (Czerny-Turner variants)
controller/             — TCD1304 detector client (desktop app + CLI;
                          `controller` launches the app)
data/                   — targets TOML, BOM TOMLs, baselines, efficiency CSVs
  BOMs (--bom flag selects which catalog):
    czerny_bom_v0_design.toml         Thorlabs COTS design (default)
    czerny_bom_v0_asbuilt.toml        Author's build: 25.0/50.0 mm mirror
                                      blanks, clearances keep OCH floor
                                      at 29.5 mm (same part keys as design)
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
                          02-design-principles, 03-simulation,
                          04-software, 05-design-v0)
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
# --cad generates bom.csv, alignment_screen.step, fixtures, mounts

# Instrument control (desktop app; subcommands for scripting)
controller
controller capture -e 25ms -n 4          # -> output/capture.<stamp>.tcd1304
controller calibrate <capture>.tcd1304

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
- **Mount flexures**: each mount has a pitch flexure (living-hinge web,
  `pitch_flexure_thickness_mm`) and optionally a roll flexure
  (2-bladed leaf-spring cross-pivot, `roll_flexure_height_mm > 0`).
  Roll flexure is active on cylindrical mirrors (F1, M1) and the
  grating; spherical M2 has `roll_flexure_height_mm = 0`.
  The w-stacking is: foot → pitch gap → pitch shelf → roll blades →
  optic pocket. The `Mount` object carries `tongue_half_mm` and
  `u_front_mm` — the housing reads these rather than re-deriving.
- **Foot bolt layout**: 2 forward bolts at `±foot_bolt_spacing_mm`,
  1 aft bolt at `u_wall_rear + 0.5 * boss_width`. Wide tongue spans
  both forward bolts. `foot_length_mm = front_bolt_offset + 0.5 *
  boss_width` (no `max()` guard).
- **Detector orientation**: pin 1 (notch) of the TCD1304 is the
  short-wavelength (blue) end. In the KiCad STEP, pin 1 is at +X;
  `place_in_scene_frame` maps STEP +X to raysect local +x.

- **Acquisition modes**: PIT + 20 clearing pulses at >=16 ms exposure
  (quantitative default; firmware validates its own floor), automatic
  PLM below with the interval floored at the ~15 ms readout.  The
  recommended sensor clock is set on every connect.  PLM is
  bright-line only (sub-knee signals read low; background not flat).
- **Calibration is stateless**: `data/cfl_lines.toml` holds reference
  wavelengths + nominal dispersion only; the pattern locator finds
  line positions from scratch each run and nothing writes back.

## Where to find deeper context

- Optical theory and design principles: `docs/02-design-principles.md`
- Simulation architecture: `docs/03-simulation.md`
- Instrument control software (`controller` app + CLI): `docs/04-software.md`
- v0 design (Thorlabs COTS): `docs/05-design-v0.md`

## Sibling repos

| Repo | Purpose |
|------|---------|
| `evolutionary-solver` | Domain-agnostic sweep + optimizer engine |
