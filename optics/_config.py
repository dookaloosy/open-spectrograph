"""Layout-independent configuration loaders.

Loads TOML data files from ``data/``:
  - ``load_defaults()`` — sweep/optimizer defaults

Design-specific loaders (``load_parts``, ``load_bom``, ``BASELINE``)
live in their respective ``designs/<name>.py`` modules.
"""


import tomllib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict that's `base` with `overlay` overlaid recursively."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_path(p: str) -> str:
    """Resolve a BOM file path against the project root."""
    if not p:
        return ""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_defaults() -> dict:
    """Load ``defaults.toml`` from the data directory."""
    path = DATA_DIR / "defaults.toml"
    if not path.exists():
        raise FileNotFoundError(f"defaults.toml not found at {path}")
    return read_toml(path)
