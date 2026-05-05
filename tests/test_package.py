"""Package-level smoke tests."""

import subprocess
import sys
import tomllib
from pathlib import Path

import nebulift


def test_package_version_matches_pyproject():
    """Package version should match project metadata."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert nebulift.__version__ == pyproject["project"]["version"]


def test_module_entrypoint_help():
    """`python -m nebulift --help` should invoke the CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "nebulift", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "Nebulift: Astrophotography Quality Assessment" in result.stdout
    assert "batch" in result.stdout
