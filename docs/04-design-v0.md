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
| Hardware | inserts, setscrews, screws | McMaster-Carr | ~$12 |
| Housing | FDM PETG | 3D printed | ~$5 |
| | | **Total** | **~$904** |

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

All parts were printed using a multimaterial fused deposition modelling
(FDM) printer with a 0.4 mm nozzle at 0.24 mm layer thickness with 15%
gyroid infill.  Optic mounts were printed on their backs (foot pointing up).
Supports were enabled for the alignment screen, housing, and bottom cover;
a PLA support interface was used for PETG parts for ease of support removal.

### 4.4.1 Housing and mounts

| Part | File | Material | Notes |
|------|------|----------|-------|
| Housing | `housing.step` | PETG | Unibody chassis + light-tight enclosure |
| Top cover | `top_cover.step` | PETG | Snap-fit lid with embossed ray path |
| Detector cover | `detector_cover.step` | PETG | Shields detector pocket |
| Bottom cover | `bottom_cover.step` | PETG | Base plate with screw holes |
| F1 mount | `f1_mount.step` | PETG | Sagittal cylindrical collimator mount |
| M1 mount | `m1_mount.step` | PETG | Tangential cylindrical collimator mount |
| M2 mount | `m2_mount.step` | PETG | Spherical focuser mount |
| Grating mount | `grating_mount.step` | PETG | Ruled grating mount |
| Contact bumps | (included in mounts) | TPU | Captive cylinders, three per mount |

Each mount slots into locating ridges on the housing cavity floor, and
includes captive TPU contact bumps (three per mount) for
setscrew-preloaded three-point retention.  The HASMA bore prints as a
5.5 mm tap drill hole; the 1/4"-36 thread is tapped post-print.

Cylindrical mirror mounts (F1, M1) and the grating mount include a
roll flexure section — a 2-bladed leaf-spring cross-pivot with a
virtual pivot at the optic centre, allowing fine adjustment of the
cylindrical axis orientation relative to the grating dispersion plane.
The roll flexure band sits between the pitch flexure shelf and the
optic pocket, with trapezoidal pedestals and a centre stiffening block
between the blades.  Roll adjustment is actuated by a pair of M2
setscrews through heat-set inserts in the foot, pushing against the
pedestal lower surface.

### 4.4.2 Assembly fixtures

| Fixture | File | Purpose |
|---------|------|---------|
| M1 fixture | `m1_fixture.step` | Holds mirror during assembly |
| M2 fixture | `m2_fixture.step` | Holds mirror during assembly |
| F1 fixture | `f1_fixture.step` | Holds mirror during assembly |
| Grating fixture | `grating_fixture.step` | Holds grating during assembly |
| HASMA tap guide | `hasma_tap_fixture.step` | Guides 1/4"-36 tap through bore |
| Alignment screen | `alignment_screen.step` | Beam centering target |

Each optic mount fixture is an envelope around the mount's outer
contour and the optic solid, with a 1 mm contact lip that supports
the optic face during assembly.  Through-cuts beneath the foot tongue
and pusher shelf (from the optic plane to the fixture floor) prevent
the mount from bottoming out before the body is fully seated.  A
setscrew access hole lets the operator tighten the retention setscrew
while the mount and optic are held in the fixture.

The HASMA tap guide is a 20 mm cone matching the housing's 45° conical
flare, with a 1/4" (6.35 mm) through bore that keeps the tap aligned
to the bore axis.  The laser alignment screen is a three-body
multi-color print (frame, white disk, black reticle) that straddles the
mount foot tongue and sits at the optic vertex plane, providing a
centering target for aligning each optic before the housing is sealed.

## 4.5 Assembly procedure

### 4.5.1 Prepare housing

Use the HASMA tap fixture to guide a 1/4"-36 tap during the tapping
process.  Slowly rotate the tap in the bore until the thread is
completely cut through the housing sidewall.  The tap must be held
perpendicular to the bore hole during this process; any misalignment
causes gross tilting and translation of the input beam.  Thread the
HASMA adapter into the tapped hole until the back surface is flush
with the inner wall of the housing, and secure in place with the
locking nut.  When a fiber ferrule is inserted, it should be flush
with the inner wall of the housing; fine-tune the adapter position
accordingly to achieve this.

### 4.5.2 Prepare mounts

Install the heat-set inserts into the optic mounts using a hot iron,
pushing each insert until it is slightly inset into the plastic; exact
position does not matter.  Install a 4 mm set screw into the optic
retention insert at the top of each mount, 5 mm set screws for the
pitch flexure pusher inserts, and 8 mm set screws for the roll flexure
pusher inserts.

### 4.5.3 Install boards

Install the detector board so that the notch on the TCD1304 package
(pin 1) is on the far (blue) side, away from the grating.  Insert the
cables on the detector board and thread them through the cable channel
before securing the board with self-tapping screws.  Then install the
controller board with the USB connector facing the outside edge; also
insert the cables before securing with self-tapping screws.

### 4.5.4 Mount optics

Place each optic into its mount and tighten the retaining set screw to
secure the optic, paying particular attention to the axial orientations
of the cylindrical mirrors.  The assembly fixtures can be used to orient
the mirrors during this process.  The grating should be installed with
the blaze direction pointing toward the detector; this is usually
indicated with an arrow on the side of the grating.

### 4.5.5 Install optics

Install each mounted optic in the housing, taking care to seat each
mount foot in the locating ridges before securing with self-tapping
screws.  Shine a bright light source such as a diode laser down the
input fiber and use the alignment screen to set the pitch flexure of
each optic in turn; the laser spot should be centred vertically on the
screen after reflecting from each optical element.

### 4.5.6 Align optics

With the top lid on, capture a live spectrum of a light source with
sharp spectral lines such as a compact fluorescent lamp (CFL).  Tune
the pitch flexures of each optic until maximum intensity is achieved;
set the roll flexures
of F1, M1, and the grating until the lines are sharpest.  Once
satisfied, calibrate the spectrum by mapping known peak wavelengths to
their detector channel positions; a linear or quadratic fit can be used
to extract fit coefficients that can be saved in the TCD1304 controller
memory.

### 4.5.7 Finish assembly

Secure the top, bottom, and detector covers using self-tapping screws.
Note that the housing can deform under load, so refrain from excessive
tightening to avoid spectral shifts.  A spectrum should still be
acquired after closing up, to validate the assembly and recalibrate if
desired.

## 4.6 Results

A CFL spectrum captured with the assembled v0.5.0 instrument after
fine alignment of all four optics.  The mercury emission lines
are used to perform a quadratic wavelength calibration; a Gaussian fit
to the isolated Hg 404.7 nm line gives FWHM = 0.50 nm, exceeding the
sub-1 nm design target.

![CFL spectrum — v0.5.0](figures/fig_6_cfl_spectrum.png)

## 4.7 Summary

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Resolution (ILF) | <1 nm EE76 | 0.50 nm FWHM (404.7 nm) | Measured from CFL spectrum |
| Spectral range | 400-700 nm | 350-750 nm | Grating eff. & mirror refl. |
| Spectral coverage | ≥300 nm | 420 nm | TCD1304 at 14.5 nm/mm |
| Throughput | >0.5% | 22-41% | All losses, λ-dependent |
| f-number | ≤f/4 | f/3.9 | M1: 25.4 mm, f = 100 mm |
| RLD | ≤17 nm/mm | 14.5 nm/mm | 600 g/mm, f = 100 mm |
| BOM cost | <$500 | ~$904 | **Exceeds target by 81%** |

The v0 design meets the resolution target (0.50 nm FWHM measured at
404.7 nm) and the spectral coverage target (≥300 nm) but misses the BOM
cost target. The primary bottleneck is the F1 focal length: Thorlabs'
shortest cylindrical mirror is f = 25 mm, which limits the astigmatism
correction and forces the optimizer into a wider layout (87×118 mm)
with higher residual aberrations at the band edges. Every part can be
ordered from thorlabs.com with next-day shipping, making v0 the
fastest path to a physical prototype.
