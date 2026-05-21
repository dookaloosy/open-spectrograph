"""SpectrographProblem — layout-independent evolutionary-solver adapter.

Generic base class for all spectrometer topology Problem adapters. The
sibling sweep/optimizer engine expects a Problem subclass that

1. Exposes a set of *searched axes* (grid-swept DoF) and a fixed
   params dict (everything else).
2. Implements ``evaluate(axis_values, seed, timeout)`` returning a
   result tuple for one grid point.
3. Implements ``fitness(stats)`` to pick the best grid point from the
   result ndarrays attached to the sweep's stats dict.

Topology-specific subclasses (e.g. ``CzernyProblem``) override
``_genome_kwargs()`` to expand virtual axes into physical genome fields
and set ``name``, ``_ROUND_MM``.
"""


import hashlib
import json as json_mod
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from sweep_engine import Problem, extract_basins

from optics.scene import InfeasibleGeometry
from optics.metrics import raw_metrics


def _unit_for(axis_name: str) -> str:
    if axis_name.endswith("_mm"):
        return "[mm]"
    if axis_name.endswith("_deg"):
        return "[deg]"
    return ""


class SpectrographProblem(Problem):
    """Layout-independent sweep/optimizer adapter for spectrometer fitness.

    Subclasses must set ``name`` and override ``_genome_kwargs()`` to
    translate virtual search axes (e.g. Dv_deg, r_total_mm) into
    physical genome constructor arguments.
    """

    name = "spectrograph"
    acceptance_field = "min_throughput"

    _ROUND_MM: dict[str, int] = {}

    def __init__(
        self,
        parts,
        scalarizer: Callable[[dict[str, float]], float],
        searched_axis_names: Iterable[str],
        assembler,
        bounds: dict[str, tuple[float, float]],
        *,
        genome_class: type,
        genome_field_names: set[str] | frozenset[str],
        fitness_wavelengths_nm: Iterable[float],
        parts_builder: "Callable[[dict], object] | None" = None,
        target_fnum: float,
        design_wavelength_nm: float,
        forward_rays: int = 5_000,
        coarse_forward_rays: int = 1_000,
        point_source: bool = False,
        project_root: Path | str | None = None,
        acceptance_threshold: float = 0.005,
    ):
        self._parts = parts
        self._parts_builder = parts_builder
        self._active_parts = parts
        self._scalarizer = scalarizer
        self._searched = list(searched_axis_names)
        self._assembler = assembler
        self._genome_class = genome_class
        self._genome_field_names = set(genome_field_names)
        self._design_wavelength_nm = design_wavelength_nm
        self._forward_rays = forward_rays
        self._coarse_forward_rays = coarse_forward_rays
        self._active_forward_rays = forward_rays
        self._target_fnum = target_fnum
        self._point_source = point_source
        self._project_root = Path(project_root).resolve() if project_root else None
        self._acceptance_threshold = acceptance_threshold

        allowed = set(bounds.keys())
        bad = set(self._searched) - allowed
        if bad:
            raise ValueError(
                f"searched_axis_names contains unknown axes: {sorted(bad)}; "
                f"allowed: {sorted(allowed)}"
            )

        self._bounds = dict(bounds)
        self._axis_metadata_list = [
            {"name": n, "unit": _unit_for(n)} for n in self._searched
        ]
        self._params: dict = {}

        self._fitness_wavelengths = tuple(float(l) for l in fitness_wavelengths_nm)
        self._throughput_keys = tuple(
            f"throughput_{int(l)}_nm" for l in self._fitness_wavelengths
        )
        self._rms_spot_keys = tuple(
            f"rms_spot_{int(l)}_um" for l in self._fitness_wavelengths
        )
        self._sigma_x_keys = tuple(
            f"sigma_x_{int(l)}_um" for l in self._fitness_wavelengths
        )
        self._sigma_y_keys = tuple(
            f"sigma_y_{int(l)}_um" for l in self._fitness_wavelengths
        )
        self._sag_tan_ratio_keys = tuple(
            f"sag_tan_ratio_{int(l)}" for l in self._fitness_wavelengths
        )
        self._tan_skewness_keys = tuple(
            f"tan_skewness_{int(l)}" for l in self._fitness_wavelengths
        )
        self._geometry_cols = (
            "L_f1_mm", "L_a_mm", "L_b_mm", "L_f2_mm",
            "theta_f1_deg", "theta_m1_deg", "theta_m2_deg",
            "theta_d_deg",
        )
        self.result_names = (
            *self._geometry_cols,
            "throughput",
            *self._throughput_keys,
            "min_throughput",
            "rms_spot_um",
            *self._rms_spot_keys,
            *self._sigma_x_keys,
            *self._sigma_y_keys,
            *self._sag_tan_ratio_keys,
            *self._tan_skewness_keys,
            "footprint_width_mm",
            "footprint_height_mm",
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def coarsen_params(self, factor):
        params = dict(self._params)
        params['_coarse_forward_rays'] = self._coarse_forward_rays
        return params

    def prepare(self, params):
        self._params = dict(params) if params else {}
        coarse_fwd = self._params.pop('_coarse_forward_rays', None)
        if coarse_fwd is not None:
            self._active_forward_rays = coarse_fwd
        else:
            self._active_forward_rays = self._forward_rays

        if self._parts_builder is not None and 'm1_part' in self._params:
            try:
                self._active_parts = self._parts_builder(self._params)
            except (KeyError, ValueError, TypeError):
                self._active_parts = None
        else:
            self._active_parts = self._parts

    # ── Genome construction ─────────────────────────────────────────────

    def _merge_params(self, axis_values) -> dict:
        """Merge prepare-set params with per-call searched-axis values."""
        kw = dict(self._params)
        for name, val in zip(self._searched, axis_values):
            kw[name] = float(val)
        return kw

    def _genome_kwargs(self, axis_values) -> dict:
        """Build genome constructor kwargs from params + axis values.

        Default: simple merge + filter to genome field names. Subclasses
        override to expand virtual axes (e.g. deviation → α,β).
        """
        kw = self._merge_params(axis_values)
        return {k: v for k, v in kw.items() if k in self._genome_field_names}

    def _build_genome(self, axis_values):
        kw = self._genome_kwargs(axis_values)
        return self._genome_class(**kw)

    def _try_assemble(self, axis_values):
        try:
            genome = self._build_genome(axis_values)
        except (ValueError, TypeError):
            return None
        try:
            return self._assembler(genome, self._active_parts)
        except InfeasibleGeometry:
            return None

    def evaluate(self, axis_values, seed, timeout):
        try:
            genome = self._build_genome(axis_values)
        except (ValueError, TypeError) as exc:
            return None, f"infeasible:{type(exc).__name__}"
        active = self._active_parts

        # Tier 1: analytical pre-filter — reject if any fitness
        # wavelength falls off the CCD.
        from optics.grating_math import wavelengths_on_detector
        if not wavelengths_on_detector(
            self._fitness_wavelengths,
            self._design_wavelength_nm,
            active.grating_groove_density_per_mm,
            active.m2_focal_length_mm,
        ):
            return None, "infeasible:off_detector"

        try:
            scene = self._assembler(genome, active)
        except InfeasibleGeometry as exc:
            return None, f"infeasible:{type(exc).__name__}"

        metrics = raw_metrics(
            scene, genome, active,
            fitness_wavelengths_nm=self._fitness_wavelengths,
            target_fnum=self._target_fnum,
            design_wavelength_nm=self._design_wavelength_nm,
            forward_rays=self._active_forward_rays,
            point_source=self._point_source,
        )
        for col in self._geometry_cols:
            v = getattr(genome, col, None)
            metrics[col] = float(v) if v is not None else float('nan')
        return (
            tuple(float(metrics[k]) for k in self.result_names),
            "ok",
        )

    # ── Validation ───────────────────────────────────────────────────────

    def validate_point(self, ax_vals):
        for name, v in zip(self._searched, ax_vals):
            lo, hi = self._bounds[name]
            if v < lo or v > hi:
                return -1
        return 0 if self._try_assemble(ax_vals) is not None else -1

    def valid_range(self):
        out = {}
        for n in self._searched:
            lo, hi = self._bounds[n]
            out[f"{n}_min"] = float(lo)
            out[f"{n}_max"] = float(hi)
        return out

    # ── Fitness extraction ───────────────────────────────────────────────

    def _build_fitness_map(self, stats, acceptance_threshold):
        throughput = stats["throughput_map"]
        min_tp = stats["min_throughput_map"]
        width = stats["footprint_width_mm_map"]
        height = stats["footprint_height_mm_map"]
        throughput_per_lambda = {k: stats[f"{k}_map"] for k in self._throughput_keys}
        rms_spot_per_lambda = {k: stats[f"{k}_map"] for k in self._rms_spot_keys}
        sigma_x_per_lambda = {k: stats[f"{k}_map"] for k in self._sigma_x_keys}
        sigma_y_per_lambda = {k: stats[f"{k}_map"] for k in self._sigma_y_keys}
        sag_tan_per_lambda = {k: stats[f"{k}_map"] for k in self._sag_tan_ratio_keys}
        skewness_per_lambda = {k: stats[f"{k}_map"] for k in self._tan_skewness_keys}

        converged = ~np.isnan(min_tp)
        accepted = converged & (min_tp >= acceptance_threshold)

        fmap = np.full(min_tp.shape, np.inf, dtype=float)
        maps = (throughput, throughput_per_lambda, width, height)
        if not np.any(accepted):
            return fmap, accepted, maps

        it = np.nditer(accepted, flags=["multi_index"])
        while not it.finished:
            if it[0]:
                idx = it.multi_index
                raw = {
                    "throughput": float(throughput[idx]),
                    "min_throughput": float(min_tp[idx]),
                    "footprint_width_mm": float(width[idx]),
                    "footprint_height_mm": float(height[idx]),
                }
                for k, arr in throughput_per_lambda.items():
                    raw[k] = float(arr[idx])
                for k, arr in rms_spot_per_lambda.items():
                    raw[k] = float(arr[idx])
                for k, arr in sigma_x_per_lambda.items():
                    raw[k] = float(arr[idx])
                for k, arr in sigma_y_per_lambda.items():
                    raw[k] = float(arr[idx])
                for k, arr in sag_tan_per_lambda.items():
                    raw[k] = float(arr[idx])
                for k, arr in skewness_per_lambda.items():
                    raw[k] = float(arr[idx])
                fmap[idx] = float(self._scalarizer(raw))
            it.iternext()
        return fmap, accepted, maps

    def _extract_best_point(self, idx, maps, axes):
        throughput, throughput_per_lambda, _w, _h = maps
        axis_names_list = list(axes.keys())
        axis_arrays = list(axes.values())
        bp = {
            axis_names_list[d]: float(axis_arrays[d][idx[d]])
            for d in range(len(axes))
        }
        bp["throughput"] = float(throughput[idx])
        for k, arr in throughput_per_lambda.items():
            bp[k] = float(arr[idx])
        return bp

    def fitness(self, stats, acceptance_threshold=1e-15):
        fmap, accepted, maps = self._build_fitness_map(
            stats, acceptance_threshold
        )
        if not np.any(np.isfinite(fmap)):
            return None, None

        idx = np.unravel_index(np.argmin(fmap), fmap.shape)
        best_val = float(fmap[idx])
        if not np.isfinite(best_val):
            return None, None

        best_point = self._extract_best_point(idx, maps, stats["searched_axes"])
        return best_val, best_point

    def fitness_basins(self, stats, acceptance_threshold=1e-15,
                       n_basins=1, min_separation=2):
        fmap, _, maps = self._build_fitness_map(
            stats, acceptance_threshold
        )
        if not np.any(np.isfinite(fmap)):
            return []

        basin_indices = extract_basins(
            fmap, k=n_basins, min_separation=min_separation,
        )

        axes = stats["searched_axes"]
        results = []
        for bidx in basin_indices:
            bp = self._extract_best_point(bidx, maps, axes)
            results.append((float(fmap[bidx]), bp))
        return results

    # ── Seeding ──────────────────────────────────────────────────────────

    def compute_seed_from_neighbors(self, neighbor_results, weights,
                                    default_seed):
        return default_seed

    # ── Axis metadata ────────────────────────────────────────────────────

    def axis_names(self):
        return list(self._searched)

    def axis_metadata(self):
        return list(self._axis_metadata_list)

    # ── Parameter rounding ───────────────────────────────────────────────

    def round_param(self, name, value):
        digits = self._ROUND_MM.get(name)
        if digits is not None:
            return float(round(value, digits))
        return float(value) if hasattr(value, 'item') else value

    # ── Config / hashing ─────────────────────────────────────────────────

    def physics_params(self):
        def _serialise(v):
            if isinstance(v, bool) or v is None:
                return v
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                return v
            if hasattr(v, "__dataclass_fields__"):
                return asdict(v)
            if isinstance(v, (list, tuple)):
                return [_serialise(x) for x in v]
            return str(v)

        optics_genome = {k: _serialise(v) for k, v in self._params.items()}

        if self._active_parts is None:
            parts_dict = {}
        else:
            parts_dict = asdict(self._active_parts)
        for k in list(parts_dict.keys()):
            if k.endswith("_mount"):
                parts_dict.pop(k, None)
            elif k.endswith("_file") and self._project_root and parts_dict[k]:
                try:
                    parts_dict[k] = str(
                        Path(parts_dict[k]).resolve().relative_to(self._project_root)
                    )
                except ValueError:
                    pass

        return {
            "parts": parts_dict,
            "optics_genome": optics_genome,
        }

    def run_config(self):
        return {
            "searched": list(self._searched),
            "fitness_wavelengths_nm": list(self._fitness_wavelengths),
            "forward_rays": self._forward_rays,
            "coarse_forward_rays": self._coarse_forward_rays,
        }

    _REFINED_DISPLAY = (
        ("L_m1_mm", "L_m1", ".1f"),
        ("L_m2_mm", "L_m2", ".1f"),
        ("L_a_mm", "L_a", ".1f"),
        ("L_b_mm", "L_b", ".1f"),
        ("theta_f1_deg", "θ_f1", ".1f°"),
        ("theta_d_deg", "θ_d", ".1f°"),
    )

    def format_best_point(self, best_point):
        parts = []
        for key, label, fmt in self._REFINED_DISPLAY:
            if key in best_point:
                v = best_point[key]
                if fmt.endswith("°"):
                    parts.append(f"{label}={v:{fmt[:-1]}}{fmt[-1]}")
                else:
                    parts.append(f"{label}={v:{fmt}}")
        return f"({', '.join(parts)})" if parts else ''

    def config_hash(self):
        meta = self.physics_params()
        return hashlib.sha256(
            json_mod.dumps(meta, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

    def save_config(self, output_dir, config_hash, searched_axes,
                    acceptance_threshold, max_attempts, grid_resolution):
        import os
        grid = {}
        for name, arr in searched_axes.items():
            grid[f"{name}_start"] = float(arr[0])
            grid[f"{name}_stop"] = (
                float(arr[-1]) + float(arr[1] - arr[0])
                if len(arr) > 1 else float(arr[0])
            )
            grid[f"{name}_step"] = (
                float(arr[1] - arr[0]) if len(arr) > 1 else 0.0
            )
        units = {m["name"]: m["unit"] for m in self._axis_metadata_list}
        cfg = {
            "problem": self.name,
            "config_hash": config_hash,
            "physics": self.physics_params(),
            "run_config": self.run_config(),
            "grid": grid,
            "sweep": {
                "acceptance_threshold": acceptance_threshold,
                "max_attempts": max_attempts,
                "grid_resolution": grid_resolution,
            },
            "units": units,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json_mod.dump(cfg, f, indent=2, default=str)

    def load_config(self, output_dir):
        import os
        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json_mod.load(f)
        return cfg["problem"], cfg["physics"], cfg["grid"], cfg["sweep"]

    # ── Display formatting ───────────────────────────────────────────────

    def format_fitness(self, value):
        return f"{value:.4f}"

    def format_point(self, name, value):
        unit = _unit_for(name).strip("[]")
        if unit:
            return f"{name}={value:.3f} {unit}"
        return f"{name}={value:.3f}"

    def format_bounds(self, name, lo, hi, unit=""):
        u = unit or _unit_for(name).strip("[]")
        tail = f" {u}" if u else ""
        return f"{name}=[{lo:.3f}, {hi:.3f}]{tail}"
