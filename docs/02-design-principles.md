# 2. Optical Design Principles

*We develop the optical relations --- grating equation, dispersion, and aberration theory --- that constrain the design space and define what the optimizer must satisfy.*

## 2.1 Standard vs. crossed Czerny-Turner

In a standard Czerny-Turner, the entrance
slit sits on the same side of the grating as the collimating mirror
(M1), and the detector sits on the same side as the focusing mirror
(M2). The beam path does not cross between the mirrors. In a crossed
Czerny-Turner, slit and detector swap sides so the beam path crosses
itself between the two mirrors. Commercial spectrometers (Ocean Optics,
Avantes) almost universally use the crossed configuration for its
compact form factor, though it is somewhat more susceptible to stray
light scattering due to the crossing beam paths.

We use the standard configuration throughout. The remainder of this
section develops the detector choice (§2.2), the instrument line
function (§2.3), dispersion relations (§2.4, §2.5), geometrical
factors (§2.6), and aberration theory (§2.7).

## 2.2 Detector

After the choice of instrument geometry, the detector is the most
consequential design decision: its pixel pitch, pixel count, and
active length jointly determine the achievable spectral sampling,
bandpass, and spatial extent of the focal plane.

We use the TCD1304DG (Toshiba): 3648 pixels at 8 µm pitch, 29.2 mm
active length, 200 µm pixel height, 300-1000 nm response. This is the
same sensor used in the Ocean Optics USB4000; its pixel height captures
the full sagittal fiber image without vignetting in the standard CT
configuration.

The readout is drmcnelson's open-source SPI instrumentation platform: a
sensor board (Rev2EB; dual-stage differential front-end, 16-bit 1 MSPS
ADC) connected via ribbon cable to a Teensy 4.0 controller (Rev3;
hardware-locked FlexPWM timing, USB 2.0 high-speed output). The
two-board architecture isolates the USB connector from the sensor,
eliminating board-flex alignment drift. Readout performance: <0.2% INL,
16-bit quantization, >140,000:1 dynamic range.

Following Toshiba's exit from the CCD market, the TCD1304DG is
end-of-life (secondary-market only, ~$15). The mitigation path is the
Hamamatsu S11639-01 (2048 pixels, 14 µm pitch, 28.7 mm active length,
Nelson 2024c); the optical design is agnostic to which sensor sits at
the focal plane.

## 2.3 Instrument line function

The instrument line function (ILF) is the spectral response of the
spectrograph to a monochromatic input; its FWHM defines the spectral
resolution. The ILF width has two contributions: a geometric minimum
(the fiber image width projected through the dispersion) and aberration
broadening from off-axis mirror use. We target mean RMS spot radius as
the fitness metric because it captures both in a single scalar;
minimizing the RMS spot directly minimizes the ILF width.

## 2.4 The grating equation

For a reflection grating with groove spacing d, the grating equation
relates the incidence angle α, diffraction angle β, and wavelength λ:

    m · λ = d · (sin α + sin β)

where |m| = 1 for first-order diffraction. The deviation angle
Dv = |α − β| is the total angle between the incident and diffracted
beams. For a fixed-grating spectrograph, all wavelengths disperse
simultaneously; different wavelengths exit at different β, forming a
spectrum across the focal plane.

### Blaze condition

Ruled gratings have a sawtooth groove profile with blaze angle γ_B.
Peak efficiency occurs at λ_B where the specular facet reflection
coincides with the first diffraction order:

    λ_B = 2 · d · sin(γ_B)

Efficiency rolls off as sinc² away from blaze. The simulation uses
vendor-measured efficiency curves when available, falling back to the
analytical sinc² model. We stress that vendor data are almost always
measured at Littrow (α = β), overstating performance at the non-Littrow
angles of a fixed-grating instrument. Only first-order diffraction is
modeled; non-design orders are zeroed.

## 2.5 Dispersion, resolution, and bandpass

The reciprocal linear dispersion (RLD) maps wavelength to focal-plane
position:

    R_D = d · cos β / (|m| · f_cam)    [nm/mm]

Two constraints bound it. The **resolution constraint** requires the
geometric ILF width Δλ_geo = R_D × W_fiber (where W_fiber = 25 µm) to
be well below 1 nm, giving R_D ≤ 20 nm/mm. The **bandpass constraint**
requires the array to capture 300 nm: R_D ≥ 10 nm/mm.

The joint window is 10 ≤ R_D ≤ 20 nm/mm. For a 600 g/mm grating
(d = 1667 nm), this maps to f_cam ∈ [83, 162] mm. A 100 mm focal
length gives R_D ≈ 16.7 nm/mm at the center wavelength, delivering
~487 nm bandwidth and a geometric ILF of Δλ_geo = 0.42 nm.

### Detector sampling

At R_D = 16.7 nm/mm the per-pixel width is
δλ = R_D × W_pixel = 16.7 × 0.008 = 0.13 nm/pixel.
Nyquist sampling of a 1 nm ILF requires 0.5 nm/pixel; the detector
oversamples by ~8×. The detector is not the resolution-limiting
element; the optics are.

### Aberration budget

The geometric ILF (0.42 nm) consumes less than half the 1 nm target.
The remaining ~0.6 nm (in quadrature) is the budget for aberration
broadening and geometrical magnification. Whether this suffices depends
on mirror geometry, off-axis angles, and fold configuration; the
following sections (§2.6, §2.7) develop these contributions.

## 2.6 Geometrical factors

Several geometrical factors affect both throughput (vignetting) and
resolution (magnification). These are wavelength-dependent and interact
non-linearly; the simulation evaluates them by ray tracing.

### 2.6.1 f/# matching

The fiber (NA 0.12, f/4.2) requires M1 to be at least f/4.2; a slower
mirror vignettes the input cone irreversibly. The 0.12 NA is well
matched to f/4 mirrors (25 mm at 100 mm), avoiding the severe
aberrations of faster optics.

### 2.6.2 Grating overfill

The beam footprint on the grating is D_beam / cos α. For a 25 mm
grating with a 25 mm beam at α = 5°, the footprint is 25.1 mm;
overfill is negligible. Rays that miss the grating do not reach the
detector.

### 2.6.3 Beam walk at M2

As wavelength varies, β changes and the beam shifts on M2 by
Δ_w = L_M2 · |tan(β_edge − β_center)|. The BOM filter enforces
D_M2 ≥ D_beam + 2Δ_w; partial band-edge vignetting is scored through
reduced hit counts, giving the GA a continuous gradient.

### 2.6.4 Detector pixel height overfill

The TCD1304 pixel height is 200 µm. For a 25 µm fiber at unity
sagittal magnification the image is well contained. The aberration-corrected CT
with a cylindrical fold magnifies the sagittal fiber image by the
ratio of M2's sagittal focal length to F1's; if the magnified image
exceeds the pixel height, light is lost above and below the active
area.

## 2.7 Aberrations in Czerny-Turner spectrographs

Spherical mirrors used off-axis introduce four aberrations that degrade
spectral resolution. Two — coma and astigmatism — have analytical
correction conditions that we enforce when seeding each candidate. The
other two — spherical aberration and field curvature — have no
closed-form correction in this geometry; the optimizer manages them
through f-number selection and detector tilt. The Nelder-Mead
refinement is then free to relax the analytically seeded corrections
to minimize the overall RMS spot across the band.

### 2.7.1 Coma is analytically corrected

Coma is the dominant off-axis aberration in CT instruments, producing
an asymmetric one-sided tail that broadens the ILF. The coma
contributions from M1 and M2 cancel when (Shafer, 1964):

    sin(θ_M2)/sin(θ_M1) = (R_M2²·cos³θ_M2)/(R_M1²·cos³θ_M1) · (cos³α/cos³β)

For equal-R mirrors the R² terms cancel and this becomes a pure angle
constraint (the Shafer condition). We enforce it by construction: the
GA evolves θ_M1, and θ_M2 is derived from the equation above. Coma
cancellation is maintained throughout Nelder-Mead refinement because
neither mirror angle is a refinement variable.

### 2.7.2 Astigmatism is analytically corrected

Off-axis spherical mirrors produce astigmatism: the tangential and
sagittal focal lengths differ. The Coddington equations quantify the
splitting:

    f_t = (R/2) cos θ,    f_s = R / (2 cos θ)

The spectrograph images the slit in the tangential (dispersion) plane,
so the arm lengths are seeded at the tangential focus:
L_A = (R_M1/2) cos θ_M1 and L_B = (R_M2/2) cos θ_M2.
Since f_t < f_s, the sagittal image is defocused at these distances,
producing an elongated spot perpendicular to the dispersion axis.

Following Xia *et al.* (2017), the sagittal defocus can be eliminated
by using a sagittal cylindrical collimating fold mirror (F1) and a tangential
cylindrical collimator (M1). Astigmatism vanishes when the M1-to-fold
distance satisfies:

    L_F1 = L_A − R_F1·R_M1·R_M2 / [2(R_F1·R_M2·sec θ_M1 + R_M1·R_M2·cos θ_F1 − R_F1·R_M1·cos θ_M2)]

Unlike coma, the astigmatism correction is not held fixed: the
Nelder-Mead refinement perturbs L_A, L_B, and θ_F1 away from their
analytical seeds, allowing the optimizer to trade astigmatism
correction against overall spot quality.

### 2.7.3 Field curvature is mitigated by detector tilt

The Petzval field curvature produces a curved focal surface. For
symmetric mirrors (f1 = f2 = 100 mm):

    1/R_p = -2/f1 - 2/f2  →  R_p = -25 mm

This curvature cannot be corrected with spherical optics, but tilting
the detector (θ_D) varies the effective focus distance across the
array, partially compensating the curved field. In practice, detector
tilt is the dominant Nelder-Mead degree of freedom, with arm lengths
changing only marginally from their Coddington seeds.

### 2.7.4 Spherical aberration is mitigated by f/# choice

Spherical aberration scales as the cube of the aperture angle; at f/2
it is 8× worse than at f/4. There is no closed-form correction in
the CT geometry. By selecting a low-NA fiber (0.12 NA, f/4.2), we
operate at f/4 where spherical aberration is manageable.
