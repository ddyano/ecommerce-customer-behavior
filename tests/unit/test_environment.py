"""Environment tests. Data here is tiny and SYNTHETIC; no real dataset is used."""
import importlib
import sys

import pytest

from scripts.check_environment import MIN_PYTHON, PACKAGES, run_all_checks


def test_python_version_supported():
    assert sys.version_info[:2] >= MIN_PYTHON


@pytest.mark.parametrize(
    "import_name", [name for name, _ in PACKAGES if name != "pytest"]
)
def test_package_imports(import_name):
    importlib.import_module(import_name)


def test_smoke_checks_all_pass():
    results = run_all_checks()
    failed = [r for r in results if not r.ok]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)