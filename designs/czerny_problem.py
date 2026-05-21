"""CzernyProblem — Czerny-Turner topology adapter.

Thin adapter between the evolutionary-solver Problem protocol and the
CzernyGeometry module.  Genome expansion is delegated to
geometry.expand_genome(); this class handles axis merging, center-
wavelength extraction, and genome field filtering.

The GA evolves a combo_id (pre-validated BOM combo) plus continuous
params (Dv, theta_m1, theta_f1).  prepare() maps combo_id to
part names and loads parts via parts_builder.

refine_basin() polishes each coarse basin with Nelder-Mead over
(L_m1, L_m2, L_a, [theta_f1], L_b, theta_d), seeded from the coarse
grid center and physics-derived initial values.  For cylindrical F1, NM varies
theta_f1 and re-derives L_f1 from the astigmatism-free condition
each evaluation, keeping the condition locked.
Dimensionality depends on fold_mode and astigmatism feasibility: 6 vars
when the fold is active and the condition is satisfied, 5 vars otherwise.
If fold_mode requests a fold but the astigmatism-free condition rejects
it, the basin is assigned penalty fitness (the foldless scene the coarse
grid evaluated is not a valid design).  A throughput guard inside the NM
objective prevents drift into zero-throughput configurations.
"""


from designs.czerny_base import (
    CzernyGenome,
    GENOME_FIELD_NAMES,
)
from optics.problem import SpectrographProblem


class CzernyProblem(SpectrographProblem):

    name = "czerny"

    _ROUND_MM: dict[str, int] = {
        "L_a_mm": 0,
        "L_b_mm": 0,
        "theta_m1_deg": 1,
        "theta_m2_deg": 1,
        "Dv_deg": 1,
        "theta_d_deg": 1,
        "f1_fraction": 3,
        "f2_fraction": 3,
        "theta_f1_deg": 1,
        "theta_f2_deg": 1,
    }

    def describe_candidate(self, cand):
        ev = cand['evolved_values']
        parts = []
        combo_id = ev.get('combo_id')
        if combo_id is not None and self._combo_list:
            combo = self._combo_list[int(combo_id)]
            parts.append(f"{combo['m1']}+{combo['m2']}+{combo['grating']}")
        Dv = ev.get('Dv_deg')
        if Dv is not None:
            parts.append(f"Dv={Dv:.1f}°")
        t1 = ev.get('theta_m1_deg')
        if t1 is not None:
            parts.append(f"θ₁={t1:.1f}°")
        tf = ev.get('theta_f1_deg')
        if tf is not None:
            parts.append(f"θ_f={tf:.1f}°")
        return '  '.join(parts)

    def __init__(self, parts, scalarizer, searched_axis_names, assembler,
                 bounds, *, geometry, combo_list=None, fold_mode="none",
                 **kw):
        self._geometry = geometry
        self._combo_list = combo_list or []
        self._fold_mode = fold_mode
        super().__init__(
            parts, scalarizer, searched_axis_names, assembler, bounds,
            genome_class=CzernyGenome,
            genome_field_names=GENOME_FIELD_NAMES,
            **kw,
        )
        geo_cols = ["L_a_mm", "L_b_mm", "theta_m1_deg", "theta_m2_deg",
                     "theta_d_deg"]
        if fold_mode in ("F1", "both"):
            geo_cols[0:0] = ["L_f1_mm", "theta_f1_deg"]
        if fold_mode == "both":
            geo_cols.insert(geo_cols.index("L_b_mm") + 1, "L_f2_mm")
        self._geometry_cols = tuple(geo_cols)
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

    def prepare(self, params):
        """Resolve combo_id to part names in-place, then delegate.

        Writes m1_part/m2_part/grating_part/f1_part/f2_part into the
        params dict so the optimizer state always has resolved names.
        The optimizer engine stores ``cand['full_params']`` and passes
        the same dict here — mutations propagate to the saved state.
        """
        if not params:
            params = {}
        combo_id = params.get("combo_id")
        if combo_id is not None and self._combo_list:
            combo = self._combo_list[int(combo_id)]
            params["m1_part"] = combo["m1"]
            params["m2_part"] = combo["m2"]
            params["grating_part"] = combo["grating"]
            if "f1" in combo:
                params["f1_part"] = combo["f1"]
            if "f2" in combo:
                params["f2_part"] = combo["f2"]
        super().prepare(params)
        self._coarse_mode = bool(params.get('_coarse_forward_rays'))

    def _genome_kwargs(self, axis_values):
        kw = self._merge_params(axis_values)
        kw.setdefault("theta_d_deg", 0.0)
        parts = self._active_parts
        lambdas = sorted(self._fitness_wavelengths)
        lam_c = lambdas[len(lambdas) // 2]
        band = (lambdas[0], lambdas[-1])
        self._geometry.expand_genome(kw, parts, lam_c, band_nm=band)
        return {k: v for k, v in kw.items() if k in self._genome_field_names}

    def valid_range(self):
        if self._active_parts is None:
            return None
        vr = super().valid_range()
        if vr is None:
            return None

        # Dynamic L_m1/L_m2 bounds from the cascade.
        if "L_m1_mm" in self._searched or "L_m2_mm" in self._searched:
            from designs.czerny_base import _r2_from_beam_walk
            parts = self._active_parts
            kw = self._merge_params(())
            lambdas = sorted(self._fitness_wavelengths)
            lam_c = lambdas[len(lambdas) // 2]
            band = (lambdas[0], lambdas[-1])
            self._geometry.expand_genome(kw, parts, lam_c, band_nm=band)
            if "alpha_deg" not in kw:
                return None
            L_a = kw.get("L_a_mm", parts.m1_focal_length_mm)
            if "L_m1_mm" in self._searched:
                vr["L_m1_mm_max"] = L_a
            if "L_m2_mm" in self._searched:
                r2_max = _r2_from_beam_walk(kw, parts, band_nm=band)
                if r2_max <= 0:
                    return None
                vr["L_m2_mm_max"] = r2_max

        return vr

    def validate_point(self, ax_vals):
        return 0

    def refine_basin(self, center, bounds, output_dir):
        """Nelder-Mead basin refinement.

        Dimensionality depends on fold_mode and F1 mirror type:
          fold_mode=none:              (L_m1, L_m2, L_a, L_b, θ_d) — 5 vars
          fold_mode=F1, flat F1:       (L_m1, L_m2, L_a, L_f1, L_b, θ_d) — 6 vars
          fold_mode=F1, cylindrical:   (L_m1, L_m2, L_a, θ_f1, L_b, θ_d) — 6 vars
          fold_mode=F1, astigmatism infeasible:  penalty (no NM run)

        For cylindrical F1, NM varies θ_f1 and re-derives L_f1 from
        the astigmatism-free condition each evaluation.
        For flat F1, NM varies L_f1 directly.

        Returns (1e6, center) for astigmatism-infeasible folds — the
        coarse grid evaluated a foldless fallback that is not a valid
        design.

        A throughput guard rejects NM evaluations where min_throughput
        drops below acceptance_threshold, preventing drift into
        zero-throughput configurations.
        """
        from scipy.optimize import minimize
        from optics.scene import InfeasibleGeometry
        from optics.metrics import raw_metrics
        from designs.czerny_base import (
            astigmatism_free_l_SM1,
            fold_fraction_lower_bound,
            fold_fraction_upper_bound,
        )

        parts = self._active_parts
        if parts is None:
            return None

        center_axis_values = tuple(center[ax] for ax in self._searched)
        try:
            gk = self._genome_kwargs(center_axis_values)
        except (ValueError, TypeError, KeyError):
            return None

        r1 = gk.get("L_m1_mm")
        r2 = gk.get("L_m2_mm")
        L_a = gk.get("L_a_mm")
        L_b = gk.get("L_b_mm")
        if r1 is None or r2 is None or L_a is None or L_b is None:
            return None

        wants_fold = self._fold_mode in ("F1", "both")
        has_fold = wants_fold and gk.get("L_f1_mm") is not None
        if wants_fold and not has_fold:
            return (1e6, dict(center))

        astigmatism_corrected = (has_fold
                     and parts.f1_focal_length_mm is not None
                     and parts.f1_mirror_type == "cylindrical")

        if astigmatism_corrected:
            _R_fold = 2.0 * parts.f1_focal_length_mm
            _R_coll = 2.0 * parts.m1_focal_length_mm
            _R_focus = 2.0 * parts.m2_focal_length_mm
            _theta_m1 = float(gk["theta_m1_deg"])
            _theta_m2 = float(gk["theta_m2_deg"])

        if has_fold:
            tf1_0 = float(gk["theta_f1_deg"])
            x0 = [r1, r2, L_a, tf1_0, L_b, 0.0]
        else:
            x0 = [r1, r2, L_a, L_b, 0.0]

        best = [None, float('inf')]
        nm_history: list[dict] = []
        forward_rays = self._active_forward_rays
        def _obj(x):
            if has_fold:
                r1_v, r2_v, L_a_v, tf1_v, L_b_v, tilt = x
            else:
                r1_v, r2_v, L_a_v, L_b_v, tilt = x
                tf1_v = None

            if (r1_v <= 0 or r2_v <= 0 or L_a_v <= 0 or L_b_v <= 0
                    or (tf1_v is not None and tf1_v <= 0)):
                row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                       "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                       "theta_d_deg": tilt,
                       "fitness": 1e6, "reject": "physical"}
                nm_history.append(row)
                return 1e6

            L_f1_v = None
            if astigmatism_corrected:
                try:
                    l_SM1 = astigmatism_free_l_SM1(
                        _R_fold, _R_coll, _R_focus,
                        tf1_v, _theta_m1, _theta_m2)
                    L_f1_v = L_a_v - l_SM1
                    if L_f1_v <= 0 or l_SM1 <= 0 or l_SM1 >= L_a_v:
                        row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                               "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                               "theta_f1_deg": tf1_v,
                               "theta_d_deg": tilt,
                               "fitness": 1e6, "reject": "astigmatism"}
                        nm_history.append(row)
                        return 1e6
                except ValueError:
                    row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                           "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                           "theta_f1_deg": tf1_v,
                           "theta_d_deg": tilt,
                           "fitness": 1e6, "reject": "astigmatism"}
                    nm_history.append(row)
                    return 1e6
            elif has_fold:
                slit_half_h = parts.slit_height_mm / 2.0
                min_stf = (0.5 * parts.f1_diameter_mm
                           + parts.f1_mount["wall_margin_mm"])
                try:
                    frac_lo = fold_fraction_lower_bound(
                        parts.m1_diameter_mm, slit_half_h,
                        parts.f1_diameter_mm, tf1_v)
                    frac_hi = fold_fraction_upper_bound(
                        L_a_v, min_stf)
                except ValueError:
                    row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                           "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                           "theta_f1_deg": tf1_v,
                           "theta_d_deg": tilt,
                           "fitness": 1e6, "reject": "fold_infeasible"}
                    nm_history.append(row)
                    return 1e6
                if frac_lo >= frac_hi:
                    row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                           "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                           "theta_f1_deg": tf1_v,
                           "theta_d_deg": tilt,
                           "fitness": 1e6, "reject": "fold_infeasible"}
                    nm_history.append(row)
                    return 1e6
                L_f1_v = (frac_lo + frac_hi) / 2.0 * L_a_v

            kw = dict(gk)
            kw["L_m1_mm"] = r1_v
            kw["L_m2_mm"] = r2_v
            kw["L_a_mm"] = L_a_v
            kw["L_b_mm"] = L_b_v
            kw["theta_d_deg"] = tilt
            if tf1_v is not None:
                kw["theta_f1_deg"] = tf1_v
            if L_f1_v is not None:
                kw["L_f1_mm"] = L_f1_v

            try:
                genome = self._genome_class(**{
                    k: v for k, v in kw.items()
                    if k in self._genome_field_names
                })
            except (ValueError, TypeError):
                row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                       "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                       "theta_d_deg": tilt,
                       "fitness": 1e6, "reject": "genome"}
                if L_f1_v is not None:
                    row["L_f1_mm"] = L_f1_v
                nm_history.append(row)
                return 1e6

            try:
                scene = self._assembler(genome, parts)
            except InfeasibleGeometry as exc:
                row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                       "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                       "theta_d_deg": tilt,
                       "fitness": 1e6, "reject": str(exc)}
                if L_f1_v is not None:
                    row["L_f1_mm"] = L_f1_v
                nm_history.append(row)
                return 1e6

            try:
                metrics = raw_metrics(
                    scene, genome, parts,
                    fitness_wavelengths_nm=self._fitness_wavelengths,
                    target_fnum=self._target_fnum,
                    design_wavelength_nm=self._design_wavelength_nm,
                    forward_rays=forward_rays,
                    point_source=self._point_source,
                )
            except Exception:
                row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                       "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                       "theta_d_deg": tilt,
                       "fitness": 1e6, "reject": "trace"}
                if L_f1_v is not None:
                    row["L_f1_mm"] = L_f1_v
                nm_history.append(row)
                return 1e6

            min_tp = metrics.get("min_throughput", 0.0)
            if min_tp <= 0 or min_tp < self._acceptance_threshold:
                row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                       "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                       "theta_d_deg": tilt,
                       "fitness": 1e6, "reject": "throughput"}
                if L_f1_v is not None:
                    row["L_f1_mm"] = L_f1_v
                row.update({k: metrics[k] for k in self.result_names
                            if k in metrics})
                nm_history.append(row)
                return 1e6

            fitness = self._scalarizer(metrics)
            if fitness is None:
                return 1e6

            row = {"L_m1_mm": r1_v, "L_m2_mm": r2_v,
                   "L_a_mm": L_a_v, "L_b_mm": L_b_v,
                   "theta_d_deg": tilt, "fitness": fitness}
            if L_f1_v is not None:
                row["L_f1_mm"] = L_f1_v
            for gc in self._geometry_cols:
                if gc not in row and gc in kw:
                    row[gc] = kw[gc]
            row.update({k: metrics[k] for k in self.result_names
                        if k in metrics})
            nm_history.append(row)

            if fitness < best[1]:
                best[1] = fitness
                bp = dict(center)
                bp["L_m1_mm"] = r1_v
                bp["L_m2_mm"] = r2_v
                bp["theta_d_deg"] = tilt
                bp["L_a_mm"] = L_a_v
                bp["L_b_mm"] = L_b_v
                if tf1_v is not None:
                    bp["theta_f1_deg"] = tf1_v
                if L_f1_v is not None:
                    bp["L_f1_mm"] = L_f1_v
                for k in self.result_names:
                    if k in metrics:
                        bp[k] = metrics[k]
                best[0] = bp

            return fitness

        n = len(x0)
        simplex = [list(x0)]
        for i in range(n):
            vertex = list(x0)
            vertex[i] += max(abs(vertex[i]) * 0.05, 1.0)
            simplex.append(vertex)

        minimize(
            _obj, x0,
            method="Nelder-Mead",
            options={"xatol": 0.1, "fatol": 0.5,
                     "maxiter": 200, "adaptive": True,
                     "initial_simplex": simplex},
        )

        if best[0] is not None:
            self._save_refine_results(output_dir, best[0], best[1],
                                      nm_history)
        if best[0] is not None:
            return (best[1], best[0])
        return None

    _REFINE_COLS = ("L_m1_mm", "L_m2_mm", "L_f1_mm", "L_a_mm", "L_b_mm",
                     "L_f2_mm", "theta_f1_deg", "theta_m1_deg",
                     "theta_m2_deg", "theta_d_deg")

    def _save_refine_results(self, output_dir, best_point, fitness,
                             nm_history=None):
        import csv, os, json, tempfile, shutil
        os.makedirs(output_dir, exist_ok=True)

        # ── config.json (mirrors coarse dir structure) ──────────────────
        refine_cols = [c for c in self._REFINE_COLS if c in best_point]
        cfg = {
            "problem": self.name,
            "physics": self.physics_params(),
            "run_config": self.run_config(),
            "refined_params": {c: best_point[c] for c in refine_cols},
        }
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        # ── results.csv — unified format matching the coarse CSV ───────
        # Columns: L_m1_mm, L_m2_mm, <result_names...>
        # result_names already includes L_a_mm, L_b_mm, L_f1_mm,
        # theta_d_deg plus all metric columns.
        axis_cols = ("L_m1_mm", "L_m2_mm")
        rnames = self.result_names
        csv_cols = list(axis_cols) + list(rnames)

        results_file = os.path.join(output_dir, "results.csv")
        fd, tmp_path = tempfile.mkstemp(
            suffix=".csv", dir=os.path.dirname(results_file) or ".")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_cols)

                def _fmt(v):
                    if isinstance(v, float):
                        return f"{v:.6g}"
                    return str(v)

                # Best result first
                writer.writerow([_fmt(best_point.get(c, ""))
                                 for c in csv_cols])

                # All NM evaluations
                if nm_history:
                    for row in nm_history:
                        writer.writerow([_fmt(row.get(c, ""))
                                         for c in csv_cols])

                f.flush()
                os.fsync(f.fileno())
            shutil.move(tmp_path, results_file)
        except BaseException:
            os.unlink(tmp_path)
            raise
