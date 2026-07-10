"""Version consistency check for the top-cover emboss.

The lid embosses the installed package version; an editable install
freezes that metadata at install time.  `_package_version` must fail
loudly when the venv is stale rather than carve the wrong version.
"""

from unittest import mock

import pytest

from optics.housing_cad import _package_version


def test_version_matches_pyproject():
    # Raises RuntimeError if the venv is stale — a failure here means
    # pyproject.toml was bumped without `pip install -e . --no-deps`.
    assert _package_version()


def test_stale_install_raises():
    with mock.patch("importlib.metadata.version", return_value="0.0.0"):
        with pytest.raises(RuntimeError, match="stale editable install"):
            _package_version()
