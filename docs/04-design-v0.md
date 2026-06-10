# 4. The v0 Design

*We present a catalog-parts build that validates the simulation pipeline end-to-end and provides the fastest path to a physical prototype.*

## 4.1 Optical train

The v0 optical train is an aberration-corrected Czerny-Turner with a
tangential cylindrical collimator (M1), sagittal cylindrical collimator
(F1), and spherical focuser (M2):

| Element | Specification | Part | Cost |
|---------|---------------|------|------|
| Fiber | 25 µm core, NA 0.12 | SMA-905 | |
| Adapter | SMA bulkhead, through housing wall | HASMA | $10 |
| F1 | f = 25 mm, D = 25.4 mm, cyl. sagittal | CCM254-025-G01 | $201 |
| M1 | f = 100 mm, D = 25.4 mm, cyl. tangential | CCM254-100-G01 | $201 |
| Grating | 600 g/mm, 25×25 mm, blaze 500 nm | GR25-0605 | $137 |
| M2 | f = 100 mm, D = 50.8 mm, spherical | CM508-100-G01 | $138 |
| Detector board | 3648 px, 8 µm pitch, 16-bit ADC | TCD1304 SPI Rev2EB | ~$110 |
| Controller board | Teensy 4.0, FlexPWM, USB 2.0 HS | Instr. Controller | ~$90 |
| Housing | FDM PLA / SLA resin | 3D printed | ~$5 |
| | | **Total** | **~$892** |

All optics are Thorlabs catalog parts; the GA was run with:

```
python run_optimizer.py czerny \
    --bom data/czerny_bom_v0_design.toml \
    --max_rld 17 --min_bw 300 --max_fnum 4 --center 550 \
    --fold_mode F1 --pop_size 128
```

The GA optimizer (point source, 128 population, 7 generations)
converged to the following layout, exported as
`czerny_baseline_v0_design.toml`:

| Parameter | Value |
|-----------|-------|
| α | 3.49° |
| β | −23.01° |
| θ\_M1 | 12.5° |
| θ\_M2 | 15.41° |
| θ\_F1 | 24.4° |
| θ\_D | 9.2° |
| L\_A | 86.6 mm |
| L\_F1 | 59.5 mm |
| L\_A − L\_F1 | 27.1 mm |
| L\_B | 109.3 mm |
| L\_M1 | 86.9 mm |
| L\_M2 | 83.8 mm |

Thorlabs is the only Western manufacturer stocking concave cylindrical
mirrors, and the shortest available focal length is f = 25 mm
(CCM254-025-G01). This is twice the ideal F1 focal length for this
geometry, forcing the optimizer into steeper incidence angles and a
wider layout with higher residual aberrations at the band edges.

## 4.2 Performance

|  | Point source | | 25 µm fiber | |
|---|---|---|---|---|
| λ (nm) | RMS (µm) | ILF (nm) | RMS (µm) | ILF (nm) |
| 400 | 41.1 | 0.68 | 46.8 | 0.74 |
| 550 | 20.1 | 0.36 | 30.9 | 0.51 |
| 700 | 58.5 | 0.73 | 63.3 | 0.85 |
| Mean | 39.9 | 0.59 | 47.0 | 0.70 |

Figure 4 shows the v0 spot diagrams: point source (top) and 25 µm
fiber source (bottom).

## 4.3 Assembly tolerance budget

A true one-at-a-time (OAT) sweep perturbs each physical dimension
independently — angular tilts ±2°, layout distances ±5% — without
recomputing downstream positions, and records the ILF (encircled-energy
width containing 76% of hits, equivalent to Gaussian FWHM) at each
point. For each axis the table reports the maximum perturbation that
keeps the ILF below 1 nm:

```
python scripts/tolerance.py --rays 100000 --n_points 11 --ilf_target 1.0
```

| Rank | Symbol | Description | Nominal | Budget | Category |
|------|--------|-------------|---------|--------|----------|
| 1 | L\_F1 | M1 to F1 distance | 59.5 mm | ±0.12 mm | tight |
| 2 | L\_A | Slit to M1 distance | 86.6 mm | ±0.21 mm | tight |
| 3 | L\_M1 | Grating to M1 distance | 86.9 mm | ±0.22 mm | tight |
| 4 | L\_B | M2 to detector distance | 109.3 mm | ±0.24 mm | tight |
| 5 | L\_M2 | Grating to M2 distance | 83.8 mm | ±0.27 mm | tight |
| 6 | θ\_M2 | M2 incidence angle | 15.4° | ±0.28° | tight |
| 7 | θ\_M1 | M1 incidence angle | 12.5° | ±0.41° | moderate |
| 8 | φ\_F1 | F1 cylinder axis orientation | 90° | ±0.48° | moderate |
| 9 | φ\_M1 | M1 cylinder axis orientation | 0° | ±0.57° | moderate |
| 10 | α | Grating incidence angle | 3.5° | ±1.94° | relaxed |
| 11 | θ\_F1 | F1 fold incidence angle | 24.4° | > ±2° | relaxed |
| 12 | θ\_D | Detector tilt | 9.2° | > ±1° | relaxed |

Layout distances (ranks 1–5) have the tightest budgets: L\_F1 must
be held to ±0.12 mm, and all arm lengths to ~±0.25 mm. The M2
incidence angle θ\_M2 is comparably tight at ±0.28°. The cylinder
axis orientations φ\_M1 and φ\_F1 (ranks 8–9) tolerate ~±0.5°
deviation. The grating incidence angle α, fold angle θ\_F1, and
detector tilt θ\_D are relaxed and do not require special attention
during assembly.

## 4.4 Printable parts

All printable parts are generated procedurally from BOM parameters
and exported with:

```
python export.py --baseline data/czerny_baseline_v0_design.toml --cad
```

### 4.4.1 Housing and mounts

| Part | File | Material | Notes |
|------|------|----------|-------|
| Housing | `housing.step` | PLA/PETG | Unibody chassis + light-tight enclosure |
| Top cover | `top_cover.step` | PLA/PETG | Snap-fit lid with embossed ray path |
| Detector cover | `detector_cover.step` | PLA/PETG | Shields detector pocket |
| Bottom cover | `bottom_cover.step` | PLA/PETG | Base plate with screw holes |
| F1 mount | `f1_mount.step` | PLA/PETG | Sagittal cylindrical collimator mount |
| M1 mount | `m1_mount.step` | PLA/PETG | Tangential cylindrical collimator mount |
| M2 mount | `m2_mount.step` | PLA/PETG | Spherical focuser mount |
| Grating mount | `grating_mount.step` | PLA/PETG | Ruled grating mount |
| Contact bumps | (included in mounts) | TPU | Captive cylinders, three per mount |

Each optic mount includes captive TPU contact bumps — three cylinders
seated in pockets in the bore wall.  The top setscrew preloads the
optic against the two bottom bumps for three-point retention.  The
HASMA bore prints as a 5.5 mm tap drill hole; the 1/4"-36 thread is
tapped post-print.

### 4.4.2 Assembly fixtures

| Fixture | File | Purpose |
|---------|------|---------|
| M1 fixture | `m1_fixture.step` | Holds mirror during assembly |
| M2 fixture | `m2_fixture.step` | Holds mirror during assembly |
| F1 fixture | `f1_fixture.step` | Holds mirror during assembly |
| Grating fixture | `grating_fixture.step` | Holds grating during assembly |
| HASMA tap guide | `hasma_tap_fixture.step` | Guides 1/4"-36 tap through bore |

Each optic mount fixture is an envelope around the mount's outer
contour and the optic solid, with a 1 mm contact lip that supports
the optic face during assembly.  Through-cuts beneath the foot tongue
and pusher shelf (from the optic plane to the fixture floor) prevent
the mount from bottoming out before the body is fully seated.  A
setscrew access hole lets the operator tighten the retention setscrew
while the mount and optic are held in the fixture.

The HASMA tap guide is a 1" cone matching the housing's 45° conical
flare, with a 1/4" (6.35 mm) through bore that keeps the tap aligned
to the bore axis.

## 4.5 Results

A CFL spectrum captured with the assembled v0.3.0 instrument confirms
sub-1 nm resolution: a Gaussian fit to the isolated 405 nm Hg line
gives FWHM = 0.86 nm.

![CFL spectrum — v0.3.0](figures/fig_6_cfl_spectrum.png)

## 4.6 Summary

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Resolution (ILF) | <1 nm EE76 | 0.86 nm FWHM (405 nm) | Measured from CFL spectrum |
| Spectral range | 400-700 nm | 350-750 nm | Grating eff. & mirror refl. |
| Spectral coverage | ≥300 nm | 420 nm | TCD1304 at 14.5 nm/mm |
| Throughput | >0.5% | 22-41% | All losses, λ-dependent |
| f-number | ≤f/4 | f/3.9 | M1: 25.4 mm, f = 100 mm |
| RLD | ≤17 nm/mm | 14.5 nm/mm | 600 g/mm, f = 100 mm |
| BOM cost | <$500 | ~$892 | **Exceeds target by 80%** |

The v0 design meets the resolution target (0.86 nm FWHM measured at
405 nm) and the spectral coverage target (≥300 nm) but misses the BOM
cost target. The primary bottleneck is the F1 focal length: Thorlabs'
shortest cylindrical mirror is f = 25 mm, which limits the astigmatism
correction and forces the optimizer into a wider layout (87×118 mm)
with higher residual aberrations at the band edges. Every part can be
ordered from thorlabs.com with next-day shipping, making v0 the
fastest path to a physical prototype.
