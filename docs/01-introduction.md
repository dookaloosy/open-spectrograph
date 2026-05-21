# 1. Introduction

*We identify the gap between commercial miniature spectrometers and existing open-source efforts, and describe a simulation-driven approach that bridges optical design to fabrication-ready output in a single toolchain.*

## 1.1 The miniature spectrometer landscape

Compact fiber-coupled spectrometers have become ubiquitous analytical
instruments. Since Ocean Optics (now Ocean Insight) introduced the first miniature
fiber spectrometer S1000 in 1992, the market has consolidated around
a common architecture: a Czerny-Turner optical train with a fixed
diffraction grating and a linear CCD or CMOS detector array, packaged
in a machined aluminum enclosure roughly the size of a deck of cards.

| Manufacturer | Model(s) | Notes |
|-------------|----------|-------|
| Ocean Insight | USB4000, Flame, HDX | TCD1304 CCD, 200-1100 nm, ~1.5 nm FWHM. $2,000-4,000 new. |
| Avantes | AvaSpec series | Comparable crossed CT architecture; interchangeable gratings. |
| Hamamatsu | C12666MA, C14384MA | Integrated mini-spectrometer modules; $250-500. |
| Thorlabs | CCS series | Compact CCD spectrometers. |

These instruments share proprietary enclosure designs, closed-source
firmware, and price points that reflect the specialized nature of the
market. A typical visible-band fiber spectrometer costs $2,000-4,000
new; even discontinued units like the USB4000 trade at $700-1,300 on
the secondary market.

## 1.2 The open-source gap

Open-source spectrometer projects exist, but occupy a different
performance tier. The Public Lab spectrometer (a webcam pointed at a
transmission diffraction grating) achieves qualitative spectral
identification but lacks the quantitative resolution and calibration
stability needed for analytical work. The Theremino spectrometer uses a
TCD1304 CCD with a transmission grating; better quantitative
performance, but still no simulation-driven design. Various academic
designs published in *Review of Scientific Instruments* and *Applied
Optics* are typically one-off instruments rather than reproducible
open-source builds.

The gap is straightforward: no open-source spectrometer combines
(a) the same optical architecture used in commercial instruments
(reflection-grating Czerny-Turner with matched f/# fiber coupling),
(b) simulation-driven design optimization, and (c) a complete,
reproducible build using off-the-shelf optics and open-source
electronics. Our target for Open Spectrograph is ~1 nm spectral
resolution over a 400-700 nm bandpass, fiber-coupled at f/4 (NA 0.12),
with a total bill of materials under $500.

## 1.3 Design approach

We address this gap with three decisions:

**Same proven architecture.** We use a Czerny-Turner layout with a
fixed reflection grating and linear CCD array, the same class of
instrument as the Ocean Optics USB4000 and Flame series. This is not
a new optical topology; it is a well-understood architecture executed
with open tooling and simulation-driven optimization.

**Simulation-driven optimization.** We employ a non-sequential forward
ray-tracing simulation built on raysect (Sherwood and Sherwood, 2022)
to evaluate candidate designs against an RMS spot-size fitness function.
A genetic algorithm searches over the deviation angle, mirror incidence
angles, fold mirror placement, and discrete BOM part selections. The
same code path that evaluates fitness exports the final design as STEP
and STL files, eliminating drift between simulated and manufactured
geometry.

**Off-the-shelf optics, open-source electronics.** All optical elements
are catalog parts (spherical and cylindrical mirrors, ruled diffraction
gratings, SMA fiber bulkheads). The detector readout uses drmcnelson's
open-source 16-bit CCD sensor board (Nelson, 2024a) and instrumentation
controller (Nelson, 2024b); the housing and optical mounts are
3D-printed as a unibody enclosure with flexure mounts.

## 1.4 Document structure

Section 2 presents the optical design principles that motivate and
constrain the optimization: the instrument line function, dispersion
relations, geometrical throughput factors, and aberration theory.
Section 3 describes the simulation framework: scene representation,
forward ray tracing, and the evolutionary optimization loop.
Subsequent sections present concrete designs --- the specific optical
trains, part selections, and engineering tradeoffs --- and evaluate
each against the design targets.
