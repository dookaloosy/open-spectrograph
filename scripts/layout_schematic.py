"""Publication-quality 2-D layout schematic for Section 3.3.

Produces a clean top-down diagram of the Czerny-Turner geometry with
parameter symbols on each dimension and angle arcs — no numeric values,
no gridlines, no legend.

Usage:
    python scripts/layout_schematic.py
    python scripts/layout_schematic.py --output output/layout_schematic.png
    python scripts/layout_schematic.py --baseline data/czerny_baseline_v0_design.toml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from designs.czerny_assembly import CzernyAssembly
from designs.czerny_base import CzernyGenome
from designs.czerny_bom import (
    build_optics_only_scene,
    load_baseline,
    load_bom,
    load_parts,
)
from optics._config import read_toml
from optics.collision import _axis_2d, polygon_for_element


def _angle_of(vx, vy):
    return math.degrees(math.atan2(vy, vx))


def _draw_angle_arc(ax, center, v_from, v_to, radius, label, *,
                    color="#333333", fontsize=10, label_radius_factor=1.5,
                    acute=True):
    if acute:
        dot = v_from[0] * v_to[0] + v_from[1] * v_to[1]
        if dot < 0:
            v_to = (-v_to[0], -v_to[1])

    a1 = _angle_of(*v_from)
    a2 = _angle_of(*v_to)

    sweep = (a2 - a1) % 360
    if sweep > 180:
        a1, a2 = a2, a1
        sweep = 360 - sweep
    start = a1

    arc = mpatches.Arc(center, 2 * radius, 2 * radius,
                       angle=0, theta1=start, theta2=start + sweep,
                       color=color, linewidth=1.0)
    ax.add_patch(arc)

    mid_angle = math.radians(start + sweep / 2)
    lr = radius * label_radius_factor
    lx = center[0] + lr * math.cos(mid_angle)
    ly = center[1] + lr * math.sin(mid_angle)
    ax.text(lx, ly, label, fontsize=fontsize, color=color,
            ha="center", va="center", style="italic")


def _beam_dir(el_from, el_to):
    dx = el_to.position[0] - el_from.position[0]
    dy = el_to.position[1] - el_from.position[1]
    ln = math.hypot(dx, dy)
    return (dx / ln, dy / ln)


def _neg(v):
    return (-v[0], -v[1])


def generate_schematic(genome, parts, output_path):
    scene = build_optics_only_scene(genome, parts)
    ct = CzernyAssembly()
    beam_path = ct.resolve_beam_path(scene)
    bp_map = {el.label: (i, el) for i, el in enumerate(beam_path)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")

    _ELEM_COLORS = {
        "slit": "#5588BB",
        "mirror": "#CC5566",
        "grating": "#338844",
        "detector": "#BB9933",
    }
    _ELEM_LABELS = {
        "entrance_slit": "Fiber input",
        "detector": "Detector",
    }
    _LABEL_OFFSETS = {
        "entrance_slit": (-50, 30),
        "F1": (10, -16),
    }

    # ── Element bodies ─────────────────────────────────────────────────

    for el in scene.elements:
        poly = polygon_for_element(el)
        if poly is None:
            continue
        color = _ELEM_COLORS.get(el.kind, "#999999")
        patch = mpatches.Polygon(
            poly, closed=True,
            facecolor=color, edgecolor="black",
            alpha=0.45, linewidth=1.0, zorder=2,
        )
        ax.add_patch(patch)

        x, y = el.position[0], el.position[1]
        display = _ELEM_LABELS.get(el.label, el.label)
        ofs = _LABEL_OFFSETS.get(el.label, (9, 9))
        ax.annotate(
            display, (x, y),
            textcoords="offset points", xytext=ofs,
            fontsize=9, fontweight="bold", color=color, zorder=5,
        )

    # ── Beam path lines ────────────────────────────────────────────────

    for i in range(len(beam_path) - 1):
        ea, eb = beam_path[i], beam_path[i + 1]
        ax.plot(
            [ea.position[0], eb.position[0]],
            [ea.position[1], eb.position[1]],
            color="#666666", linewidth=0.9, linestyle="--", zorder=1,
        )

    # ── Dimension labels (symbols only) ────────────────────────────────

    _SEG_SYMBOL = {
        ("entrance_slit", "F1"): "$L_A - L_{F1}$",
        ("F1", "M1"): "$L_{F1}$",
        ("entrance_slit", "M1"): "$L_A$",
        ("M1", "grating"): "$L_{M1}$",
        ("grating", "M2"): "$L_{M2}$",
        ("M2", "F2"): "$L_B - L_{F2}$",
        ("F2", "detector"): "$L_{F2}$",
        ("M2", "detector"): "$L_B$",
    }

    for i in range(len(beam_path) - 1):
        ea, eb = beam_path[i], beam_path[i + 1]
        key = (ea.label, eb.label)
        sym = _SEG_SYMBOL.get(key)
        if sym is None:
            continue
        x0, y0 = ea.position[0], ea.position[1]
        x1, y1 = eb.position[0], eb.position[1]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            continue
        nx, ny = -dy / ln, dx / ln
        off = 6.0
        if key == ("entrance_slit", "F1"):
            off = -8.0
        rot = math.degrees(math.atan2(dy, dx))
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180
        ax.text(
            mx + off * nx, my + off * ny, sym,
            fontsize=11, color="#333333", ha="center", va="center",
            rotation=rot, rotation_mode="anchor",
        )

    # ── Surface normal arrows (thin, gray) ─────────────────────────────

    arrow_len = 20.0
    normal_color = "#AAAAAA"

    for el_label in ("grating", "M1", "M2", "F1", "F2", "detector"):
        if el_label not in {e.label for e in scene.elements}:
            continue
        el = scene.by_label(el_label)
        fwd, _ = _axis_2d(el)
        ex, ey = el.position[0], el.position[1]
        ax.annotate(
            "", xy=(ex + fwd[0] * arrow_len, ey + fwd[1] * arrow_len),
            xytext=(ex, ey),
            arrowprops=dict(arrowstyle="-|>", color=normal_color,
                            lw=0.8, mutation_scale=7),
            zorder=3,
        )

    # ── Angle arcs ─────────────────────────────────────────────────────

    arc_color = "#2255AA"
    arc_r = 13.0

    grating_el = scene.by_label("grating")
    gx, gy = grating_el.position[0], grating_el.position[1]
    gn = grating_el.axis[:2]

    m1_idx, m1_el = bp_map["M1"]
    m2_idx, m2_el = bp_map["M2"]

    d_m1_to_g = _beam_dir(m1_el, grating_el)
    d_g_to_m2 = _beam_dir(grating_el, m2_el)

    # α: grating normal ↔ incoming beam
    _draw_angle_arc(ax, (gx, gy), gn, d_m1_to_g, arc_r,
                    r"$\alpha$", color=arc_color)

    # β: grating normal ↔ outgoing beam
    _draw_angle_arc(ax, (gx, gy), gn, d_g_to_m2, arc_r + 5,
                    r"$\beta$", color=arc_color)

    # θ_M1: incoming beam ↔ mirror normal at M1
    m1x, m1y = m1_el.position[0], m1_el.position[1]
    m1_fwd, _ = _axis_2d(m1_el)
    prev_m1 = beam_path[m1_idx - 1]
    d_to_m1 = _beam_dir(prev_m1, m1_el)
    _draw_angle_arc(ax, (m1x, m1y), m1_fwd, d_to_m1, arc_r,
                    r"$\theta_{M1}$", color=arc_color)

    # θ_M2: incoming beam ↔ mirror normal at M2
    m2x, m2y = m2_el.position[0], m2_el.position[1]
    m2_fwd, _ = _axis_2d(m2_el)
    d_into_m2 = _neg(d_g_to_m2)
    _draw_angle_arc(ax, (m2x, m2y), m2_fwd, d_into_m2, arc_r,
                    r"$\theta_{M2}$", color=arc_color)

    # θ_F1
    if "F1" in bp_map:
        f1_idx, f1_el = bp_map["F1"]
        f1x, f1y = f1_el.position[0], f1_el.position[1]
        f1_fwd, _ = _axis_2d(f1_el)
        prev_f1 = beam_path[f1_idx - 1]
        d_to_f1 = _beam_dir(prev_f1, f1_el)
        _draw_angle_arc(ax, (f1x, f1y), f1_fwd, d_to_f1, arc_r * 0.7,
                        r"$\theta_{F1}$", color=arc_color, fontsize=9)

    # θ_F2
    if "F2" in bp_map:
        f2_idx, f2_el = bp_map["F2"]
        f2x, f2y = f2_el.position[0], f2_el.position[1]
        f2_fwd, _ = _axis_2d(f2_el)
        prev_f2 = beam_path[f2_idx - 1]
        d_to_f2 = _beam_dir(prev_f2, f2_el)
        _draw_angle_arc(ax, (f2x, f2y), f2_fwd, d_to_f2, arc_r,
                        r"$\theta_{F2}$", color=arc_color)

    # θ_D: detector tilt from the beam axis
    if genome.theta_d_deg != 0.0 and "detector" in bp_map:
        det_idx, det_el = bp_map["detector"]
        det_x, det_y = det_el.position[0], det_el.position[1]
        det_fwd, _ = _axis_2d(det_el)
        prev_det = beam_path[det_idx - 1]
        d_to_det = _beam_dir(prev_det, det_el)
        _draw_angle_arc(ax, (det_x, det_y), det_fwd, d_to_det, arc_r * 0.8,
                        r"$\theta_D$", color=arc_color, fontsize=9)

    # ── Finalize ───────────────────────────────────────────────────────

    ax.autoscale()
    pad = 8
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    ax.set_xlim(xl - pad, xr + pad)
    ax.set_ylim(yl - pad, yr + pad)

    fig.savefig(str(output_path), dpi=250, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _load_genome_parts(baseline_path, bom_path):
    data = read_toml(Path(baseline_path or "data/czerny_baseline_xia2017.toml"))
    genome_keys = {f.name for f in __import__("dataclasses").fields(CzernyGenome)}
    genome = CzernyGenome(**{k: v for k, v in data.items() if k in genome_keys})
    bom_path = bom_path or data.get("bom_path")
    m1 = data["m1_part"]
    bom = load_bom(bom_path)
    optic_size = float(bom["mirrors"]["m1_options"][m1]["diameter_mm"])
    parts = load_parts(
        m1_part=m1, m2_part=data["m2_part"],
        grating_part=data["grating_part"],
        f1_part=data.get("f1_part"), f2_part=data.get("f2_part"),
        optic_size_mm=optic_size, bom_path=bom_path,
    )
    return genome, parts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=Path("output/layout_schematic.png"))
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--bom", type=str, default=None)
    ap.add_argument("--L_m1", type=float, default=None,
                    help="Override L_m1_mm for display")
    ap.add_argument("--L_m2", type=float, default=None,
                    help="Override L_m2_mm for display")
    args = ap.parse_args()

    genome, parts = _load_genome_parts(args.baseline, args.bom)
    if args.L_m1 is not None:
        genome = __import__("dataclasses").replace(genome, L_m1_mm=args.L_m1)
    if args.L_m2 is not None:
        genome = __import__("dataclasses").replace(genome, L_m2_mm=args.L_m2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_schematic(genome, parts, args.output)


if __name__ == "__main__":
    main()
