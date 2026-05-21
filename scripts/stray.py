"""Run the industry-standard 8-bandpass stray-light test.

Measures flux at the test wavelength (grating tuned to λ₀) and at
λ₀ ± 8×bandpass (grating detuned). The ratio I(off)/I(on) is the
stray-light figure by the HORIBA / Jobin Yvon protocol.

Usage::

    python scripts/stray.py
    python scripts/stray.py --run output/optimizer_... --candidate gen01_cand75
    python scripts/stray.py --spp 1000000
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from designs.czerny_base import CzernyGenome, CzernyParts
from optics.diagnostics import stray_light_test

from designs.czerny_assembly import assemble_scene
from designs.czerny_bom import BASELINE, load_baseline_parts, build_optics_only_scene

SPP_DEFAULT = 10_000_000


def _run_one(label: str, genome: CzernyGenome, parts: CzernyParts,
             spp: int) -> None:
    input_fnum = parts.m1_focal_length_mm / parts.m1_diameter_mm
    scene = assemble_scene(genome, parts,
                           scene_builder=build_optics_only_scene)
    print(f"\n{'='*60}")
    print(f"  8-bandpass stray test: {label}  ({spp:,} spp, 3 traces)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    result = stray_light_test(
        scene, genome, parts,
        wavelength_nm=550.0,
        backward_spp=spp,
        target_fnum=input_fnum,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f} s\n")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.6e}")
        else:
            print(f"  {k:30s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None,
                    help="Optimizer run dir (loads winner genome)")
    ap.add_argument("--candidate", type=str, default=None,
                    help="Candidate ID within --run (default: best winner)")
    ap.add_argument("--spp", type=int, default=SPP_DEFAULT,
                    help=f"Backward-trace rays per pixel (default: {SPP_DEFAULT:,})")
    ap.add_argument("--baseline", action="store_true",
                    help="Also run the baseline for comparison")
    args = ap.parse_args()

    baseline_parts = load_baseline_parts()

    if args.run:
        from export import _load_winner
        _, genome, parts, label = _load_winner(args.run, args.candidate)
        if args.baseline:
            _run_one("BASELINE", BASELINE, baseline_parts, args.spp)
        _run_one(label, genome, parts, args.spp)
    else:
        _run_one("BASELINE", BASELINE, baseline_parts, args.spp)

    print("\nDone.")


if __name__ == "__main__":
    main()
