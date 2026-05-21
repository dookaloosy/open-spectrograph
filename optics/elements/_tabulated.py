"""Shared helpers for tabulated material data (CSV loading + interpolation)."""


import bisect


def load_two_column_csv(csv_path: str) -> tuple[list[float], list[float]]:
    """Load a two-column CSV (wavelength_nm, value) skipping comments/headers."""
    try:
        f = open(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"tabulated data file not found: {csv_path}")
    xs, ys = [], []
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("wavelength"):
                continue
            parts = line.split(",")
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"{csv_path}:{lineno}: expected two numeric columns, "
                    f"got {line!r}"
                ) from exc
    if len(xs) < 2:
        raise ValueError(f"{csv_path}: need at least 2 data rows, got {len(xs)}")
    return xs, ys


def interp_table(x: float, xs: list[float], ys: list[float]) -> float:
    """Linearly interpolate a sorted table, clamping at boundaries."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])
