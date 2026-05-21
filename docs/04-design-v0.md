# 4. The v0 Design

*We present a catalog-parts build that validates the simulation pipeline end-to-end and provides the fastest path to a physical prototype.*

## 4.1 Optical train

The v0 optical train is an aberration-corrected Czerny-Turner with a
cylindrical collimator (M1), sagittal cylindrical fold mirror (F1), and
spherical focuser (M2):

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
| 400 | 41.0 | 0.65 | 46.6 | 0.78 |
| 550 | 19.9 | 0.25 | 31.3 | 0.25 |
| 700 | 58.4 | 0.12 | 63.1 | 0.48 |
| Mean | 39.8 | 0.34 | 47.0 | 0.50 |

Figure 4 shows the v0 spot diagrams: point source (top) and 25 µm
fiber source (bottom).

## 4.3 Assembly tolerance budget

A one-at-a-time tolerance sweep perturbs each physical dimension ±5%
around nominal and measures both mean RMS spot size and mean ILF FWHM
(400, 550, 700 nm; point source; 5,000 rays per wavelength). Axis
rotation slopes are computed from one-sided perturbation (the
sensitivity is symmetric).

| Rank | Parameter | Nominal | RMS slope | ILF slope |
|------|-----------|---------|-----------|-----------|
| 1 | F1 axis rotation | 90° | 44.5 µm/deg | 1.476 nm/deg |
| 2 | M1 axis rotation | 0° | 32.0 µm/deg | 1.154 nm/deg |
| 3 | θ\_F1 | 24.4° | 2.8 µm/deg | 0.001 nm/deg |
| 4 | L\_A | 86.6 mm | 2.0 µm/mm | 0.100 nm/mm |
| 5 | L\_F1 | 59.5 mm | 1.7 µm/mm | 0.018 nm/mm |
| 6 | L\_B | 109.3 mm | 1.1 µm/mm | 0.210 nm/mm |
| 7 | θ\_D | 9.2° | 0.5 µm/deg | 0.129 nm/deg |
| 8 | L\_M1 | 86.9 mm | 0.1 µm/mm | 0.012 nm/mm |
| 9 | L\_M2 | 83.8 mm | ~0 | 0.050 nm/mm |

Cylinder axis rotation (ranks 1–2) is the only assembly-critical
parameter: F1 degrades ILF by 1.5 nm per degree of axis error,
consuming the full resolution budget at ~0.3°. Round Thorlabs blanks
require an alignment procedure to achieve this. All other parameters
(ranks 3–9) are comfortably within FDM print accuracy and do not
require special attention during assembly.

## 4.4 Summary

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Resolution (ILF) | <1 nm FWHM | 0.25-0.78 nm | 25 µm fiber source |
| Spectral range | 400-700 nm | 350-750 nm | Grating eff. & mirror refl. |
| Spectral coverage | ≥300 nm | 420 nm | TCD1304 at 14.5 nm/mm |
| Throughput | >0.5% | 22-41% | All losses, λ-dependent |
| f-number | ≤f/4 | f/3.9 | M1: 25.4 mm, f = 100 mm |
| RLD | ≤17 nm/mm | 14.5 nm/mm | 600 g/mm, f = 100 mm |
| BOM cost | <$500 | ~$892 | **Exceeds target by 80%** |

The v0 design meets the resolution target (ILF < 1 nm) and the spectral
coverage target (≥300 nm) but misses the BOM cost target. The primary
bottleneck is the F1 focal length: Thorlabs' shortest cylindrical mirror
is f = 25 mm, which limits the astigmatism correction and forces the
optimizer into a wider layout (87×118 mm) with higher residual
aberrations at the band edges. Every part can be ordered from
thorlabs.com with next-day shipping, making v0 the fastest path to a
physical prototype.
